"""
file: analyzer.py
"""
import numpy as np
import networkx as nx
from ase.neighborlist import natural_cutoffs, neighbor_list
from collections import Counter
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import hashlib
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Optional imports handled safely
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# --- Helper Functions ---

# --- Global Variables for Fork-based Multiprocessing Memory Sharing ---
# By storing the trajectory globally before forking, we prevent ProcessPoolExecutor 
# from pickling and duplicating gigabytes of Atoms objects over IPC queues.
_GLOBAL_TRAJ = None
_GLOBAL_CLEANED_BONDS = None

import zlib
import pickle
import gc
import multiprocessing as mp


def _can_inherit_worker_globals():
    """Whether ProcessPool workers inherit the in-memory trajectory globals."""
    try:
        return mp.get_context().get_start_method() == "fork"
    except (RuntimeError, ValueError):
        return False

class CompressedList:
    def __init__(self):
        self._data = []
        
    def append(self, item):
        self._data.append(zlib.compress(pickle.dumps(item, protocol=-1)))
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [pickle.loads(zlib.decompress(b)) for b in self._data[idx]]
        return pickle.loads(zlib.decompress(self._data[idx]))
        
    def __len__(self):
        return len(self._data)
        
    def __iter__(self):
        for b in self._data:
            yield pickle.loads(zlib.decompress(b))

    def __bool__(self):
        return bool(self._data)


if HAS_NUMBA:
    @jit(nopython=True)
    def viterbi_fast_numba(obs_sequence, trans_prob, emit_prob):
        n_obs = len(obs_sequence)
        dp = np.zeros((n_obs, 2))
        path = np.zeros((n_obs, 2), dtype=np.int32)
        dp[0, 0] = emit_prob[0, obs_sequence[0]]
        dp[0, 1] = emit_prob[1, obs_sequence[0]]
        for t in range(1, n_obs):
            for s in range(2):
                p0 = dp[t-1, 0] + trans_prob[0, s]
                p1 = dp[t-1, 1] + trans_prob[1, s]
                if p0 > p1: dp[t, s] = p0 + emit_prob[s, obs_sequence[t]]; path[t, s] = 0
                else: dp[t, s] = p1 + emit_prob[s, obs_sequence[t]]; path[t, s] = 1
        best = np.zeros(n_obs, dtype=np.int32)
        best[-1] = 0 if dp[-1, 0] > dp[-1, 1] else 1
        for t in range(n_obs - 2, -1, -1): best[t] = path[t+1, best[t+1]]
        return best

def compute_geometric_edges_chunk(chunk_data):
    start_idx, end_idx, bond_scale, sub_indices_set, sub_elements_set, pre_cutoffs, is_sub_mask, implicit_cell = chunk_data
    edges_list = []
    
    has_substrate_def = is_sub_mask is not None
    mask_len = len(is_sub_mask) if has_substrate_def else 0
    
    # Access the globally shared trajectory instead of receiving it via IPC
    global _GLOBAL_TRAJ
    if _GLOBAL_TRAJ is not None:
        if hasattr(_GLOBAL_TRAJ, 'get_chunk'):
            atoms_list = _GLOBAL_TRAJ.get_chunk(start_idx, end_idx)
        else:
            atoms_list = _GLOBAL_TRAJ[start_idx:end_idx]
    else:
        atoms_list = []
        
    # For surface systems the substrate is deliberately excluded from molecular topology. 
    mobile_indices = None
    if has_substrate_def:
        mobile_indices = np.flatnonzero(~np.asarray(is_sub_mask, dtype=bool))

    # ensure we only loop over the actual returned atoms length, not exactly the requested slice
    for atoms in atoms_list:
        if implicit_cell is not None and not any(atoms.get_pbc()):
            atoms = atoms.copy()
            atoms.set_cell(implicit_cell)
            atoms.set_pbc([True, True, True])

        frame_edges = set()

        if (
            mobile_indices is not None
            and mask_len == len(atoms)
            and len(mobile_indices) < len(atoms)
        ):
            if len(mobile_indices) > 1:
                mobile_atoms = atoms[mobile_indices]
                if pre_cutoffs is not None and len(atoms) == len(pre_cutoffs):
                    full_cutoffs = np.asarray(pre_cutoffs, dtype=float)
                    cutoffs = full_cutoffs[mobile_indices]
                else:
                    cutoffs = natural_cutoffs(
                        mobile_atoms, mult=bond_scale
                    )
                i_list, j_list = neighbor_list(
                    'ij', mobile_atoms, cutoffs
                )
                for u_local, v_local in zip(i_list, j_list):
                    if u_local < v_local:
                        frame_edges.add((
                            int(mobile_indices[u_local]),
                            int(mobile_indices[v_local]),
                        ))
        else:
            # No valid substrate mask: preserve the general all-atom path.
            if pre_cutoffs is not None and len(atoms) == len(pre_cutoffs):
                cutoffs = pre_cutoffs
            else:
                cutoffs = natural_cutoffs(atoms, mult=bond_scale)
            i_list, j_list = neighbor_list('ij', atoms, cutoffs)
            for u, v in zip(i_list, j_list):
                if u < v:
                    if has_substrate_def:
                        u_sub = is_sub_mask[u] if u < mask_len else False
                        v_sub = is_sub_mask[v] if v < mask_len else False
                        if u_sub or v_sub:
                            continue
                    frame_edges.add((int(u), int(v)))
        edges_list.append(frame_edges)
    return edges_list

def _connectivity_key(atoms, comp_indices, G_sub, formula):
    """Return a deterministic element-labelled connectivity identifier.

    This graph key is independent of atom ordering and of the particular SMILES
    spelling. It contains connectivity only; the current trajectory analysis
    does not infer electronic state, charge, stereochemistry, or bond order.
    """
    local_graph = nx.Graph()
    for ase_idx in sorted(int(i) for i in comp_indices):
        local_graph.add_node(
            int(ase_idx),
            element=str(atoms[int(ase_idx)].symbol),
        )
    for u, v in G_sub.edges():
        u, v = int(u), int(v)
        if u in local_graph and v in local_graph:
            local_graph.add_edge(u, v)

    try:
        wl_hash = nx.weisfeiler_lehman_graph_hash(
            local_graph,
            node_attr='element',
            iterations=4,
        )
    except Exception:
        nodes = sorted(
            (int(i), str(local_graph.nodes[i].get('element', '')))
            for i in local_graph.nodes
        )
        edges = sorted(
            tuple(sorted((int(u), int(v))))
            for u, v in local_graph.edges
        )
        payload = repr((nodes, edges)).encode('utf-8')
        wl_hash = hashlib.sha256(payload).hexdigest()[:32]

    degree_signature = ','.join(
        str(value)
        for value in sorted(
            int(local_graph.degree(i)) for i in local_graph.nodes
        )
    )
    payload = f"{formula}|{len(local_graph)}|{degree_signature}|{wl_hash}"
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]
    return f"graph:{formula}:{digest}"


def generate_species_identity(atoms, comp_indices, G_sub, formula):
    """Build an auditable connectivity identity.

    The trajectory supplies geometry/connectivity but no authoritative bond
    orders, charges, spin state, aromaticity, or stereochemistry.  RDKit is
    therefore given an explicit all-single-bond graph.  The resulting SMILES is
    a deterministic connectivity label, not a chemical bond-order assignment.
    Sanitization failures remain audit-only valence-ambiguous observations.
    """
    connectivity_key = _connectivity_key(
        atoms,
        comp_indices,
        G_sub,
        formula,
    )
    result = {
        'smiles': str(formula),
        'canonical_smiles': '',
        'raw_smiles': '',
        'species_id': connectivity_key,
        'connectivity_key': connectivity_key,
        'smiles_valid': False,
        'valence_warning': '',
        'topology_evidence': 'connectivity_only_no_rdkit',
        'bond_order_model': 'all_single',
        'clean_eligible': False,
    }

    if not HAS_RDKIT:
        result['valence_warning'] = 'rdkit_unavailable'
        return result

    try:
        mol = Chem.RWMol()
        idx_map = {}
        for ase_idx in sorted(int(i) for i in comp_indices):
            atom = Chem.Atom(int(atoms[ase_idx].number))
            atom.SetNoImplicit(True)
            idx_map[ase_idx] = mol.AddAtom(atom)

        for u, v in G_sub.edges():
            u, v = int(u), int(v)
            if u in idx_map and v in idx_map:
                mol.AddBond(
                    idx_map[u],
                    idx_map[v],
                    Chem.BondType.SINGLE,
                )

        try:
            Chem.SanitizeMol(mol)
            canonical = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
            result.update({
                'smiles': canonical,
                'canonical_smiles': canonical,
                'raw_smiles': canonical,
                'species_id': f"smi:{canonical}",
                'smiles_valid': True,
                'valence_warning': '',
                'topology_evidence': (
                    'rdkit_sanitized_connectivity_smiles_all_single_bonds'
                ),
                'bond_order_model': 'all_single',
                'clean_eligible': True,
            })
            return result
        except Exception as exc:
            warning = f"{type(exc).__name__}: {exc}"
            raw_smiles = ''
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
                raw_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True,
                    allBondsExplicit=False,
                )
            except Exception:
                raw_smiles = ''

            result.update({
                'smiles': raw_smiles or str(formula),
                'raw_smiles': raw_smiles,
                'valence_warning': warning,
                'topology_evidence': (
                    'connectivity_only_valence_ambiguous'
                ),
                'clean_eligible': False,
            })
            return result
    except Exception as exc:
        result['valence_warning'] = f"{type(exc).__name__}: {exc}"
        result['topology_evidence'] = (
            'connectivity_identity_generation_failed'
        )
        result['clean_eligible'] = False
        return result


def generate_robust_smiles(atoms, comp_indices, G_sub):
    """Backward-compatible wrapper returning only the display SMILES."""
    formula = atoms[list(comp_indices)].get_chemical_formula(mode='hill')
    return generate_species_identity(
        atoms,
        comp_indices,
        G_sub,
        formula,
    ).get('smiles')

def rebuild_molecules_chunk(chunk_data):
    import tempfile
    import struct
    start_idx, end_idx, use_rdkit, do_ring, max_size, sub_elements_set, is_sub_mask, implicit_cell = chunk_data
    # Memory optimization: process in micro-batches and spill to disk to strictly limit peak RAM
    
    global _GLOBAL_TRAJ
    global _GLOBAL_CLEANED_BONDS
    
    if _GLOBAL_TRAJ is not None:
        if hasattr(_GLOBAL_TRAJ, 'get_chunk'):
            atoms_list = _GLOBAL_TRAJ.get_chunk(start_idx, end_idx)
        else:
            atoms_list = _GLOBAL_TRAJ[start_idx:end_idx]
    else:
        atoms_list = []
        
    actual_len = len(atoms_list)
    edges_list = _GLOBAL_CLEANED_BONDS[start_idx:start_idx+actual_len] if _GLOBAL_CLEANED_BONDS is not None else []
    
    fd, tmp_path = tempfile.mkstemp(prefix=f"rxn_chunk_{start_idx}_{end_idx}_", suffix=".tmp")
    micro_batch = []
    
    with os.fdopen(fd, 'wb') as f:
        for atoms, stable_edges in zip(atoms_list, edges_list):
            if implicit_cell is not None and not any(atoms.get_pbc()):
                atoms = atoms.copy()
                atoms.set_cell(implicit_cell)
                atoms.set_pbc([True, True, True])

            n_atoms = len(atoms)
            symbols = atoms.get_chemical_symbols()

            if is_sub_mask is not None and len(is_sub_mask) == n_atoms:
                mobile_indices = [
                    idx for idx, is_sub in enumerate(is_sub_mask)
                    if not is_sub
                ]
            else:
                mobile_indices = list(range(n_atoms))
            original_to_mobile = {
                original: local
                for local, original in enumerate(mobile_indices)
            }
            
            # 1. Build Graph & Components
            if not stable_edges:
                comps = [[k] for k in mobile_indices]
            else:
                mobile_edges = [
                    (int(u), int(v))
                    for u, v in stable_edges
                    if int(u) in original_to_mobile
                    and int(v) in original_to_mobile
                ]
                if not mobile_edges:
                    comps = [[k] for k in mobile_indices]
                else:
                    rows = [
                        original_to_mobile[u] for u, _ in mobile_edges
                    ]
                    cols = [
                        original_to_mobile[v] for _, v in mobile_edges
                    ]
                    adj = csr_matrix(
                        ([1] * len(rows), (rows, cols)),
                        shape=(
                            len(mobile_indices), len(mobile_indices)
                        ),
                    )
                    n_comps, labels = connected_components(
                        adj, directed=False, return_labels=True
                    )
                    comps = [[] for _ in range(n_comps)]
                    for local_idx, label in enumerate(labels):
                        comps[label].append(mobile_indices[local_idx])

            G = None 
            if use_rdkit or do_ring:
                G = nx.Graph()
                G.add_nodes_from(mobile_indices)
                G.add_edges_from(
                    (u, v) for u, v in stable_edges
                    if u in original_to_mobile and v in original_to_mobile
                )

            # 2. Process Molecules
            mols_in_frame, formulas = [], []
            
            for comp_indices in comps:
                # Substrate Filtering
                is_sub_comp = True
                for idx in comp_indices:
                    if is_sub_mask is not None:
                        if idx < len(is_sub_mask) and not is_sub_mask[idx]:
                            is_sub_comp = False
                            break
                        elif idx >= len(is_sub_mask):
                            is_sub_comp = False
                            break
                    else:
                        if sub_elements_set and symbols[idx] not in sub_elements_set:
                            is_sub_comp = False
                            break
                        elif not sub_elements_set:
                            is_sub_comp = False
                            break
                
                if is_sub_comp:
                    continue

                sub = atoms[comp_indices]
                formula = sub.get_chemical_formula(mode='hill')

                if use_rdkit and G and (len(comp_indices) <= max_size):
                    identity = generate_species_identity(
                        atoms,
                        comp_indices,
                        G.subgraph(comp_indices),
                        formula,
                    )
                else:
                    graph = (
                        G.subgraph(comp_indices)
                        if G is not None
                        else nx.Graph()
                    )
                    if G is None:
                        graph.add_nodes_from(comp_indices)
                    connectivity_key = _connectivity_key(
                        atoms,
                        comp_indices,
                        graph,
                        formula,
                    )
                    identity = {
                        # Keep large connectivity-distinct species separate
                        # in tables and networks instead of merging by formula.
                        'smiles': connectivity_key,
                        'canonical_smiles': '',
                        'raw_smiles': '',
                        'species_id': connectivity_key,
                        'connectivity_key': connectivity_key,
                        'smiles_valid': False,
                        'valence_warning': (
                            'rdkit_size_limit_exceeded'
                            if len(comp_indices) > max_size
                            else 'rdkit_identity_unavailable'
                        ),
                        'topology_evidence': (
                            'connectivity_only_size_limited'
                        ),
                        'bond_order_model': 'all_single',
                        'clean_eligible': True,
                    }

                molecule = {
                    'formula': formula,
                    'indices': frozenset(comp_indices),
                    **identity,
                }
                mols_in_frame.append(molecule)
                formulas.append(formula)

            # 3. Process Bonds (FIXED: Now actually counting them)
            bond_strs = []
            if stable_edges:
                for u, v in stable_edges:
                    # Sort elements alphabetically (e.g., C-H instead of H-C)
                    s1, s2 = symbols[u], symbols[v]
                    b_type = f"{min(s1, s2)}-{max(s1, s2)}"
                    bond_strs.append(b_type)

            # 4. Process Rings
            ring_counts = []
            if do_ring and G: 
                try: ring_counts = [len(r) for r in nx.cycle_basis(G)]
                except: pass
                
            micro_batch.append((mols_in_frame, Counter(formulas), Counter(bond_strs), Counter(ring_counts)))
            
            # Spill to disk every 100 frames to strictly bound RAM and prevent IPC bloat
            if len(micro_batch) >= 100:
                blob = zlib.compress(pickle.dumps(micro_batch, protocol=-1))
                f.write(struct.pack('>I', len(blob)))
                f.write(blob)
                micro_batch.clear()
                
        # Flush remaining
        if micro_batch:
            blob = zlib.compress(pickle.dumps(micro_batch, protocol=-1))
            f.write(struct.pack('>I', len(blob)))
            f.write(blob)
            micro_batch.clear()
            
    return tmp_path

class HMMBondFilter:
    def __init__(self, persistence=0.99, noise_tolerance=0.1):
        p = np.array([[persistence, 1-persistence], [1-persistence, persistence]])
        e = np.array([[1-noise_tolerance, noise_tolerance], [noise_tolerance, 1-noise_tolerance]])
        self.trans_prob = np.log(np.clip(p, 1e-10, 1.0))
        self.emit_prob = np.log(np.clip(e, 1e-10, 1.0))

    def viterbi(self, obs_sequence):
        if HAS_NUMBA: 
            return viterbi_fast_numba(obs_sequence.astype(np.int32), self.trans_prob, self.emit_prob)
        return obs_sequence

class TrajectoryAnalyzer:
    def __init__(self, atoms_list):
        self.trajectory = atoms_list
        self.n_frames = len(atoms_list)
        self.frame_molecules = CompressedList()
        self.molecule_counts, self.bond_history, self.ring_history = [], [], []

    def analyze(self, bond_scale=1.2, hmm_persistence=0.99, n_workers=None, do_ring_stat=False, substrate_config=None, max_size=60):
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = int(n_workers)
        if n_workers <= 0:
            raise ValueError(f"n_workers must be > 0, got {n_workers}")
        sub_indices_set = set(substrate_config.get('indices', [])) if substrate_config else set()
        sub_elements_set = set(substrate_config.get('elements', [])) if substrate_config else set()

        if sub_elements_set:
            print(f"🧱 Substrate Masking Active: {sub_elements_set}")

        print(f"🚀 [Step 1] Computing Topology (Cores: {n_workers})...")
        
        # Pre-compute cutoffs from the first frame to avoid ASE internal concurrency issues and speed up
        pre_cutoffs = None
        is_sub_mask = None
        self.implicit_cell = None
        if self.n_frames > 0:
            from ase.neighborlist import natural_cutoffs
            # Use first_frame if available from LazyTrajectory to avoid parsing overhead here
            first_frame = getattr(self.trajectory, 'first_frame', None)
            if first_frame is None:
                first_frame = self.trajectory[0]
            
            if not any(first_frame.get_pbc()):
                # Explicitly disable auto-detected PBC boxes for non-periodic XYZ files to prevent 
                # gas-phase radical fragments from being artificially wrapped back via Minimum Image Convention.
                self.implicit_cell = None

            pre_cutoffs = natural_cutoffs(first_frame, mult=bond_scale)
            
            # Pre-compute substrate mask for faster loop filtering
            has_sub = len(sub_indices_set) > 0 or len(sub_elements_set) > 0
            if has_sub:
                symbols = first_frame.get_chemical_symbols()
                is_sub_mask = [False] * len(symbols)
                for idx, sym in enumerate(symbols):
                    if idx in sub_indices_set or sym in sub_elements_set:
                        is_sub_mask[idx] = True
            
        # Bind trajectory to global variable for fork-based memory sharing
        global _GLOBAL_TRAJ
        _GLOBAL_TRAJ = self.trajectory
            
        chunk_size = max(1, self.n_frames // (n_workers * 4))
        # Hard cap chunk size for large text files to ensure responsive progress bar and avoid massive memory spikes per chunk
        if getattr(self.trajectory, 'is_xyz', False):
            chunk_size = min(chunk_size, 50)
            
        chunks = [(i, i + chunk_size, bond_scale, sub_indices_set, sub_elements_set, pre_cutoffs, is_sub_mask, self.implicit_cell) for i in range(0, self.n_frames, chunk_size)]
        
        all_geometric_edges = []
        
        # OOM Prevention: Freeze garbage collector to prevent CoW over global shared variables
        gc.collect()
        if hasattr(gc, 'freeze'):
            gc.freeze()
            
        use_fork_pool = n_workers > 1 and _can_inherit_worker_globals()
        if n_workers > 1 and not use_fork_pool:
            print(
                "⚠️ Multiprocessing start method does not inherit trajectory "
                "globals; topology analysis is running serially for correctness."
            )
        if use_fork_pool:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                iterator = executor.map(compute_geometric_edges_chunk, chunks)
                for res in tqdm(iterator, total=len(chunks), unit="chunk"):
                    all_geometric_edges.extend(res)
        else:
            for chunk in tqdm(chunks, total=len(chunks), unit="chunk"):
                all_geometric_edges.extend(compute_geometric_edges_chunk(chunk))
                
        if hasattr(gc, 'unfreeze'):
            gc.unfreeze()
            
        print("🧠 [Step 2] HMM Filtering...")
        all_potential_bonds = list(set().union(*all_geometric_edges))
                
        cleaned_bonds_per_frame = [[] for _ in range(self.n_frames)]
        
        if HAS_NUMBA and hmm_persistence > 0:
            hmm = HMMBondFilter(persistence=hmm_persistence)
            
            # Smart sparse edge building to save memory
            edge_presence = {bond: [] for bond in all_potential_bonds}
            for f, edges in enumerate(all_geometric_edges):
                for bond in edges:
                    edge_presence[bond].append(f)
                    
            # Free the raw memory-hog
            all_geometric_edges.clear()
            
            for bond in tqdm(all_potential_bonds, desc="HMM Smoothing"):
                obs_seq = np.zeros(self.n_frames, dtype=np.int8)
                obs_seq[edge_presence[bond]] = 1
                
                # Optimization: Skip if constant
                if np.all(obs_seq == 0) or np.all(obs_seq == 1): 
                    clean_seq = obs_seq
                else: 
                    clean_seq = hmm.viterbi(obs_seq)
                
                for f in np.flatnonzero(clean_seq): cleaned_bonds_per_frame[f].append(bond)
                
            # Cleanup presence map
            edge_presence.clear()
        else:
            cleaned_bonds_per_frame = [list(f) for f in all_geometric_edges]
            all_geometric_edges.clear()
        
        print(f"🏗️  [Step 3] Rebuilding Molecules & Stats...")
        
        global _GLOBAL_CLEANED_BONDS
        _GLOBAL_CLEANED_BONDS = cleaned_bonds_per_frame
        
        step3_chunks = []
        for i in range(0, self.n_frames, chunk_size):
            step3_chunks.append((i, i + chunk_size, HAS_RDKIT, do_ring_stat, max_size, sub_elements_set, is_sub_mask, self.implicit_cell))
        
        # OOM Prevention: Freeze garbage collector to prevent CoW over global shared variables
        gc.collect()
        if hasattr(gc, 'freeze'):
            gc.freeze()
            
        import struct
        if use_fork_pool:
            executor_context = ProcessPoolExecutor(max_workers=n_workers)
            tmp_paths = executor_context.map(
                rebuild_molecules_chunk, step3_chunks
            )
        else:
            executor_context = None
            tmp_paths = map(rebuild_molecules_chunk, step3_chunks)
        try:
            for tmp_path in tqdm(tmp_paths, total=len(step3_chunks), unit="chunk"):
                try:
                    with open(tmp_path, 'rb') as f:
                        while True:
                            size_data = f.read(4)
                            if not size_data:
                                break
                            size = struct.unpack('>I', size_data)[0]
                            blob = f.read(size)
                            micro_batch = pickle.loads(zlib.decompress(blob))
                            for (mols, f_count, b_count, r_count) in micro_batch:
                                self.frame_molecules.append(mols)
                                self.molecule_counts.append(f_count)
                                self.bond_history.append(b_count)
                                if do_ring_stat: self.ring_history.append(Counter(r_count))
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        finally:
            if executor_context is not None:
                executor_context.shutdown()
                    
        if hasattr(gc, 'unfreeze'):
            gc.unfreeze()

    @staticmethod
    def _stable_count_table(records):
        """Return a deterministic wide frame-by-count table.

        Rows retain trajectory-frame order, the CSV index is explicitly named
        ``Frame``, absent counts are zero, and data columns are sorted
        case-insensitively for stable output across runs.
        """
        table = pd.DataFrame(records).fillna(0)
        if not table.empty:
            table = table.reindex(
                sorted(
                    table.columns,
                    key=lambda value: [
                        (1, int(token))
                        if token.isdigit()
                        else (0, token.lower())
                        for token in re.split(r'(\d+)', str(value))
                    ],
                ),
                axis=1,
            )
        table = table.astype(int)
        table.index = pd.RangeIndex(
            start=0,
            stop=len(table),
            step=1,
            name="Frame",
        )
        return table

    def get_molecule_table(self):
        return self._stable_count_table(self.molecule_counts)

    def get_bond_table(self):
        return self._stable_count_table(self.bond_history)
    def get_ring_table(self): return pd.DataFrame(self.ring_history).fillna(0).astype(int)
