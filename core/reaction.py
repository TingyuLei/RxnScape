import pandas as pd
import os
import numpy as np
from ase.io import write
from ase.neighborlist import natural_cutoffs, neighbor_list
from collections import Counter, defaultdict
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import ast
import json
import multiprocessing as mp
import time
logger = logging.getLogger(__name__)

try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:  # pragma: no cover
    BrokenProcessPool = RuntimeError

# 🚀 Intelligent import of the ASE NEB module (compatible with different versions)
try:
    from ase.mep import NEB  
except ImportError:
    try:
        from ase.neb import NEB 
    except ImportError:
        print("❌ Critical: Cannot import NEB from ase.mep or ase.neb. Please check ASE installation.")
        NEB = None

# --- Global Variables for Fork-based Multiprocessing Memory Sharing ---
_GLOBAL_RXN_TRAJ = None
_GLOBAL_RXN_FRAMES = None


def _adsorbed_indices_from_geometry(
    atoms,
    is_sub_mask,
    pre_cutoffs,
    pair_cutoffs,
    adsorption_scale,
):
    """Pure worker-safe adsorption contact calculation for one frame."""
    curr_len = len(atoms)
    if pre_cutoffs is not None and curr_len == len(pre_cutoffs):
        cutoffs = pre_cutoffs
    else:
        cutoffs = natural_cutoffs(atoms, mult=adsorption_scale)

    try:
        if pair_cutoffs is not None:
            ii, jj = neighbor_list('ij', atoms, pair_cutoffs)
        else:
            ii, jj = neighbor_list('ij', atoms, cutoffs)
    except Exception:
        ii, jj = neighbor_list('ij', atoms, cutoffs)

    ads_idx = set()
    if is_sub_mask is not None and curr_len == len(is_sub_mask):
        for u, v in zip(ii, jj):
            u = int(u)
            v = int(v)
            if is_sub_mask[v] and not is_sub_mask[u]:
                ads_idx.add(u)
            if is_sub_mask[u] and not is_sub_mask[v]:
                ads_idx.add(v)
    return ads_idx


def _compute_adsorption_indices_chunk(args):
    """Compute adsorption contacts for a frame range in a fork worker."""
    (
        start_idx,
        end_idx,
        is_sub_mask,
        pre_cutoffs,
        pair_cutoffs,
        adsorption_scale,
    ) = args
    global _GLOBAL_RXN_TRAJ
    if _GLOBAL_RXN_TRAJ is None:
        return []
    atoms_list = _GLOBAL_RXN_TRAJ[start_idx:end_idx]
    return [
        _adsorbed_indices_from_geometry(
            atoms,
            is_sub_mask,
            pre_cutoffs,
            pair_cutoffs,
            adsorption_scale,
        )
        for atoms in atoms_list
    ]


def _truthy_value(value):
    """Interpret serialized boolean-like values consistently across stages."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_missing_scalar(value):
    """Return True for scalar missing values, including pandas/NumPy NaN."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "<na>"}
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _normalize_atom_groups(value):
    """Return deterministic per-species atom groups."""
    if _is_missing_scalar(value):
        return []

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return []

    if isinstance(parsed, dict):
        parsed = parsed.get("groups", parsed.get("atom_groups", []))
    if not isinstance(parsed, (list, tuple)):
        return []

    groups = []
    for group in parsed:
        if isinstance(group, dict):
            group = group.get(
                "atom_indices",
                group.get("indices", group.get("atoms", [])),
            )
        if isinstance(group, (int, np.integer)):
            group = [group]
        if not isinstance(group, (list, tuple, set)):
            continue
        indices = []
        for index in group:
            try:
                indices.append(int(index))
            except (TypeError, ValueError):
                continue
        groups.append(tuple(sorted(set(indices))))
    return groups


def _serialize_atom_groups(value):
    """Serialize per-species atom groups as compact, stable JSON."""
    return json.dumps(
        [list(group) for group in _normalize_atom_groups(value)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _reaction_id_first(df):
    """Move Reaction_ID to the first column whenever it exists."""
    if df is None or not hasattr(df, 'columns') or 'Reaction_ID' not in df.columns:
        return df
    return df.loc[:, ['Reaction_ID'] + [c for c in df.columns if c != 'Reaction_ID']]


def _can_inherit_reaction_globals():
    """Process workers can use module globals only with a fork context."""
    try:
        return mp.get_context().get_start_method() == "fork"
    except (RuntimeError, ValueError):
        return False

def _sanitize_for_filename(s):
    """Remove shell-unsafe characters from path components."""
    import re
    s = s.replace(" + ", "_")
    s = s.replace("*", "surf")
    s = s.replace("(g)", "gas")
    s = s.replace("(s)", "solid")
    s = re.sub(r'[^\w\-.]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def _process_single_event_task(args):
    (row_data, local_start, local_end, abs_start, abs_end, sub_elements, sub_indices, neb_dir, output_dir, bond_scale, adsorption_scale, r_smi, p_smi, rxn_id) = args

    try:
        global _GLOBAL_RXN_TRAJ, _GLOBAL_RXN_FRAMES
        traj_subset = _GLOBAL_RXN_TRAJ[abs_start:abs_end + 1] if _GLOBAL_RXN_TRAJ is not None else []
        frames_subset = _GLOBAL_RXN_FRAMES[abs_start:abs_end + 1] if _GLOBAL_RXN_FRAMES is not None else []

        detector = ReactionDetector(
            frames_subset,
            trajectory=traj_subset,
            output_dir=output_dir,
            neb_dir=neb_dir,
            bond_scale=bond_scale,
            substrate_elements=sub_elements,
            substrate_indices=sub_indices,
            adsorption_scale=adsorption_scale,
            frame_offset=abs_start,
        )

        # Use the true reactant/product representative frames local to traj_subset.
        try:
            f_start = int(row_data.get('Reactant_Frame'))
            f_end = int(row_data.get('Product_Frame'))
        except Exception:
            f_start, f_end = int(local_start), int(local_end)
        f_start = max(0, min(f_start, len(traj_subset) - 1)) if traj_subset else 0
        f_end = max(0, min(f_end, len(traj_subset) - 1)) if traj_subset else 0
        if f_end < f_start:
            f_start, f_end = f_end, f_start

        indices = row_data['Atom_Indices']
        if isinstance(indices, str):
            try:
                import ast
                cleaned_indices = indices.replace('"', '').replace("'", "")
                indices = ast.literal_eval(cleaned_indices)
            except Exception:
                indices = []

        r_safe = _sanitize_for_filename(str(row_data['Reactants_Formula']))
        p_safe = _sanitize_for_filename(str(row_data['Products_Formula']))

        MAX_SUFFIX_LEN = 150
        suffix_str = f"_{r_safe}_to_{p_safe}"
        if len(suffix_str) > MAX_SUFFIX_LEN:
            suffix_str = suffix_str[:MAX_SUFFIX_LEN] + "_etc"

        errors = []
        try:
            neb_path = detector._generate_neb_inputs(
                f_start, f_end, indices, rxn_id, r_safe, p_safe
            )
        except Exception as exc:
            import traceback
            neb_path = f"Structure_Error: {type(exc).__name__}: {exc}"
            errors.append("NEB generation failed:\n" + traceback.format_exc())

        try:
            snap_path = detector._save_trajectory(
                f_start, f_end, indices, rxn_id, suffix_str
            )
        except Exception as exc:
            import traceback
            snap_path = f"Structure_Error: {type(exc).__name__}: {exc}"
            errors.append("Snapshot generation failed:\n" + traceback.format_exc())

        return rxn_id, neb_path, snap_path, "\n".join(errors)
    except Exception as exc:
        import traceback
        msg = f"{type(exc).__name__}: {exc}"
        return rxn_id, f"Structure_Error: {msg}", f"Structure_Error: {msg}", traceback.format_exc()

class ReactionDetector:
    def __init__(self, frame_molecules, trajectory=None, output_dir='./reaction_snapshots', neb_dir='./auto_neb_inputs', bond_scale=1.2, substrate_elements=None, substrate_indices=None, adsorption_scale=1.25, frame_offset=0):
        self.frames = frame_molecules
        self.traj = trajectory
        self.output_dir = output_dir
        self.neb_dir = neb_dir 
        self.bond_scale = bond_scale
        self.adsorption_scale = adsorption_scale
        self.sub_elements = set(substrate_elements) if substrate_elements else set()
        self.sub_indices = set(substrate_indices) if substrate_indices else set()
        self.frame_offset = int(frame_offset or 0)
        self.unresolved_components = []
        self._unresolved_component_key_to_row = {}
        self._unresolved_duplicate_count = 0
        self._single_frame_recovery_enabled = True
        # Bounded horizon used by invalid-topology corridor recovery.  It is
        # reset for each detect() run from the requested global stability.
        self.recrossing_window = 40
        self._requested_local_stability = None
        self._segmentation_local_stability = None
        self._recovery_scan_local_stability = 1
        self._surface_minimum_dwell = 2
        self._single_frame_recovery_stats = Counter()
        self._reset_runtime_caches()

    def _reset_runtime_caches(self):
        self._ads_idx_by_frame = {}
        self._ads_idx_by_obj = {}
        self._traj_obj_to_frame = {}
        self._local_bonds_cache = {}
        self._local_mols_cache = {}
        self._local_sig_cache = {}
        self._local_topology_sig_cache = {}
        self._frame_bond_adj_cache = {}
        self._frame_atom_mol_index_cache = {}
        self._lineage_family_presence_cache = {}
        # Compact, exact presence indexes populated while Mapping States is
        # already visiting every decompressed frame.  
        self._lineage_family_intervals = {}
        self._species_presence_index = {}
        self._presence_index_complete = False
        self._bond_cutoffs = None
        self._ads_pair_cutoffs = None

        # Endpoint-filter audit is lightweight. 
        self._last_endpoint_filter_rejections = []
        self._endpoint_rejection_audit = []

    @staticmethod
    def _append_presence_bounds(bounds, frame_idx):
        """Append one frame to one flat inclusive interval list."""
        frame_idx = int(frame_idx)
        if not bounds:
            bounds.extend((frame_idx, frame_idx))
        elif bounds[-1] == frame_idx - 1:
            bounds[-1] = frame_idx
        elif bounds[-1] != frame_idx:
            bounds.extend((frame_idx, frame_idx))

    @classmethod
    def _append_presence_interval(cls, index, key, frame_idx):
        """Append one frame to a keyed inclusive run-length interval index."""
        bounds = index.get(key)
        if bounds is None:
            bounds = []
            index[key] = bounds
        cls._append_presence_bounds(bounds, frame_idx)

    @staticmethod
    def _frame_in_presence_intervals(bounds, frame_idx):
        """Return whether ``frame_idx`` lies in flat inclusive intervals."""
        if not bounds:
            return False
        frame_idx = int(frame_idx)
        lo = 0
        hi = len(bounds) // 2
        while lo < hi:
            mid = (lo + hi) // 2
            start = int(bounds[2 * mid])
            end = int(bounds[2 * mid + 1])
            if frame_idx < start:
                hi = mid
            elif frame_idx > end:
                lo = mid + 1
            else:
                return True
        return False

    def _index_frame_presence(
        self,
        frame_idx,
        molecules,
        adsorbed_indices=None,
        surface_system=False,
    ):
        """Record exact lineage/species presence during the existing frame scan."""
        frame_idx = int(frame_idx)
        adsorbed_indices = adsorbed_indices or set()
        for mol in molecules:
            indices = tuple(sorted(int(i) for i in mol.get("indices", []) or []))
            identity = self._molecule_identity(mol)
            family = self._lineage_family_key(
                identity,
                mol.get("formula", ""),
            )
            if indices and family:
                self._append_presence_interval(
                    self._lineage_family_intervals,
                    (indices, family),
                    frame_idx,
                )

            smiles = str(
                mol.get("smiles")
                or mol.get("canonical_smiles")
                or ""
            ).strip()
            if not smiles:
                continue
            formula = str(mol.get("formula", "") or "").strip()
            if surface_system and formula:
                base_formula = self._strip_surface_suffix(formula)
                is_adsorbed = any(i in adsorbed_indices for i in indices)
                formula = base_formula + ("*" if is_adsorbed else "(g)")
            species_key = (smiles, formula)
            entry = self._species_presence_index.get(species_key)
            if entry is None:
                entry = {
                    "intervals": [],
                    "representative": (frame_idx, indices),
                }
                self._species_presence_index[species_key] = entry
            self._append_presence_bounds(entry["intervals"], frame_idx)

    def _clear_heavy_scan_caches(self):
        """Clear memory-heavy caches during long transition scans."""
        for name in [
            "_ads_idx_by_obj",
            "_local_bonds_cache",
            "_local_mols_cache",
            "_local_sig_cache",
            "_local_topology_sig_cache",
            "_frame_bond_adj_cache",
            "_frame_atom_mol_index_cache",
            "_lineage_family_presence_cache",
        ]:
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()

        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _frame_index_for_atoms(self, atoms):
        try:
            return self._traj_obj_to_frame.get(id(atoms))
        except Exception:
            return None

    def _get_frame_bond_adjacency(self, atoms):
        """Build non-substrate bond adjacency once per trajectory frame."""
        frame_idx = self._frame_index_for_atoms(atoms)
        cache_key = frame_idx if frame_idx is not None else id(atoms)

        cached = self._frame_bond_adj_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._bond_cutoffs is not None and len(self._bond_cutoffs) == len(atoms):
            cutoffs = self._bond_cutoffs
        else:
            cutoffs = natural_cutoffs(atoms, mult=self.bond_scale)

        ii, jj = neighbor_list('ij', atoms, cutoffs)
        symbols = atoms.get_chemical_symbols()

        adj = defaultdict(set)

        for u, v in zip(ii, jj):
            u = int(u)
            v = int(v)
            if u >= v:
                continue

            sym_u, sym_v = symbols[u], symbols[v]

            if self.sub_elements or self.sub_indices:
                if (
                    u in self.sub_indices or v in self.sub_indices
                    or sym_u in self.sub_elements or sym_v in self.sub_elements
                ):
                    continue

            bond_type = f"{min(sym_u, sym_v)}-{max(sym_u, sym_v)}"
            item = ((u, v), bond_type)
            adj[u].add(item)
            adj[v].add(item)

        self._frame_bond_adj_cache[cache_key] = adj
        return adj

    def _get_local_bonds(self, atoms, indices):
        target_tuple = tuple(sorted(int(i) for i in indices))
        if not target_tuple:
            return set()
        target_set = set(target_tuple)

        frame_idx = self._frame_index_for_atoms(atoms)
        cache_key = (frame_idx if frame_idx is not None else id(atoms), target_tuple)

        cached = self._local_bonds_cache.get(cache_key)
        if cached is not None:
            return cached

        adj = self._get_frame_bond_adjacency(atoms)

        bonds = set()
        for i in target_tuple:
            for item in adj.get(i, set()):
                pair, _bond_type = item
                if int(pair[0]) in target_set and int(pair[1]) in target_set:
                    bonds.add(item)

        self._local_bonds_cache[cache_key] = bonds
        return bonds

    @staticmethod
    def _molecule_identity(mol):
        """Return the formal species identity produced by topology analysis."""
        return str(
            mol.get('species_id')
            or mol.get('canonical_smiles')
            or mol.get('smiles')
            or mol.get('formula')
            or ''
        ).strip()

    @staticmethod
    def _molecule_connectivity(mol):
        return str(
            mol.get('connectivity_key')
            or mol.get('species_id')
            or mol.get('canonical_smiles')
            or mol.get('smiles')
            or mol.get('formula')
            or ''
        ).strip()

    def _molecule_atom_group(self, mol):
        """Return the non-substrate atoms owned by one molecule instance."""
        return tuple(sorted({
            int(index)
            for index in mol.get('indices', []) or []
            if int(index) not in self.sub_indices
        }))

    def _molecule_instance_key(self, mol):
        """Atom-resolved molecule key used for plateau and event detection."""
        return (
            self._molecule_identity(mol),
            self._molecule_atom_group(mol),
        )

    @staticmethod
    def _molecule_topology_valid(mol):
        value = mol.get('smiles_valid', True)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}

    @staticmethod
    def _molecule_clean_eligible(mol):
        value = mol.get('clean_eligible', True)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}

    def _disentangle_events(self, r_frame, p_frame, r_idx=None, p_idx=None):
        r_map = {idx: i for i, m in enumerate(r_frame) for idx in m['indices'] if idx not in self.sub_indices}
        p_map = {idx: i for i, m in enumerate(p_frame) for idx in m['indices'] if idx not in self.sub_indices}
        G = nx.Graph()
        for atom_idx in r_map:
            if atom_idx in p_map:
                G.add_edge(f"R_{r_map[atom_idx]}", f"P_{p_map[atom_idx]}")
        events = []
        
        r_atoms = self.traj[r_idx] if r_idx is not None and self.traj else None
        p_atoms = self.traj[p_idx] if p_idx is not None and self.traj else None
        
        for comp in nx.connected_components(G):
            sub_r = [r_frame[int(n.split('_')[1])] for n in comp if n.startswith('R_')]
            sub_p = [p_frame[int(n.split('_')[1])] for n in comp if n.startswith('P_')]
            r_s = sorted([self._molecule_identity(m) for m in sub_r])
            p_s = sorted([self._molecule_identity(m) for m in sub_p])

            r_instances = sorted(
                self._molecule_instance_key(m) for m in sub_r
            )
            p_instances = sorted(
                self._molecule_instance_key(m) for m in sub_p
            )
            is_different = (
                r_s != p_s
                or r_instances != p_instances
            )
            
            if not is_different and r_atoms is not None and p_atoms is not None and self.sub_elements:
                r_state = sorted([(self._molecule_identity(m), self._is_adsorbed(r_atoms, m['indices'])) for m in sub_r])
                p_state = sorted([(self._molecule_identity(m), self._is_adsorbed(p_atoms, m['indices'])) for m in sub_p])
                if r_state != p_state:
                    is_different = True
                    
            if is_different:
                events.append((sub_r, sub_p))
        return events

    # =====================================================================
    # PBC-aware local geometry reconstruction for Snapshot / NEB outputs
    # =====================================================================
    def _cell_array(self, atoms):
        try:
            return np.asarray(atoms.get_cell().array, dtype=float)
        except Exception:
            try:
                return np.asarray(atoms.get_cell(), dtype=float).reshape(3, 3)
            except Exception:
                return None

    def _has_valid_cell(self, atoms):
        """Return True only when the input really carries a usable periodic cell."""
        if atoms is None:
            return False
        
        try:
            pbc = np.asarray(atoms.get_pbc(), dtype=bool)
            if not np.any(pbc):
                return False
        except Exception:
            return False

        cell = self._cell_array(atoms)
        if cell is None or cell.shape != (3, 3):
            return False
            
        try:
            lengths = np.linalg.norm(cell, axis=1)
            vol = abs(float(np.linalg.det(cell)))
        except Exception:
            return False
            
        return vol > 1.0e-8 and np.all(lengths > 1.0e-8)

    def _infer_hidden_orthorhombic_cell(self):
        """
        If the user provides a true periodic XYZ, ASE
        will read the Lattice naturally; otherwise, leave it as non-periodic.
        """
        if hasattr(self, '_hidden_cell_cache'):
            return self._hidden_cell_cache
        self._hidden_cell_cache = (None, None, 'none')
        if self.traj is None or len(self.traj) == 0:
            return self._hidden_cell_cache

        # If any frame has a real cell, prefer that over inference.
        for atoms in self.traj[: min(len(self.traj), 5)]:
            if self._has_valid_cell(atoms):
                self._hidden_cell_cache = (self._cell_array(atoms), np.asarray(atoms.get_pbc(), dtype=bool), 'explicit')
                return self._hidden_cell_cache

        # Disabled the aggressive coordinate-span bounding box inference
        return self._hidden_cell_cache

    def _reconstruction_cell(self, atoms):
        # Only apply explicit PBC unwrapping if there is a valid cell AND 
        # the system is defined as a surface/periodic one via sub_elements/sub_indices.
        is_periodic_system = bool(hasattr(self, 'sub_elements') and (self.sub_elements or self.sub_indices))
        if is_periodic_system and self._has_valid_cell(atoms):
            return self._cell_array(atoms), np.asarray(atoms.get_pbc(), dtype=bool), 'explicit'
        return self._infer_hidden_orthorhombic_cell()

    def _mic_delta(self, delta, cell, pbc):
        """Minimum-image delta for ASE row-vector cell convention."""
        delta = np.asarray(delta, dtype=float)
        if cell is None or pbc is None or not np.any(pbc):
            return delta
        try:
            inv_cell = np.linalg.inv(np.asarray(cell, dtype=float))
            frac = np.dot(delta, inv_cell)
            shift = np.zeros(3)
            pbc = np.asarray(pbc, dtype=bool)
            shift[pbc] = np.round(frac[pbc])
            return delta - np.dot(shift, cell)
        except Exception:
            return delta

    def _molecular_components_for_indices(self, frame_idx, idx_list):
        """Use analyzer molecule membership to split local atoms into true fragments."""
        idx_set = set(int(i) for i in idx_list)
        comps, seen = [], set()
        if self.frames is not None and 0 <= frame_idx < len(self.frames):
            for mol in self.frames[frame_idx]:
                inds = set(int(i) for i in mol.get('indices', []))
                comp = sorted(idx_set.intersection(inds))
                if comp:
                    comps.append(comp)
                    seen.update(comp)
        for i in sorted(idx_set - seen):
            comps.append([i])
        return comps if comps else [sorted(idx_set)]

    def _pair_distance_mic(self, pos_i, pos_j, cell, pbc):
        return float(np.linalg.norm(self._mic_delta(np.asarray(pos_j) - np.asarray(pos_i), cell, pbc)))

    def _connectivity_edges_for_component(self, full_atoms, comp, cell, pbc):
        """
        Build an unwrap graph for one molecular component.
        """
        comp = [int(i) for i in comp]
        if len(comp) <= 1:
            return []
        positions = full_atoms.get_positions()
        try:
            cutoffs = natural_cutoffs(full_atoms, mult=self.bond_scale)
        except Exception:
            cutoffs = [1.0] * len(full_atoms)

        g = nx.Graph()
        g.add_nodes_from(comp)
        complete = nx.Graph()
        complete.add_nodes_from(comp)
        for a_i, i in enumerate(comp):
            for j in comp[a_i + 1:]:
                d = self._pair_distance_mic(positions[i], positions[j], cell, pbc)
                complete.add_edge(i, j, weight=d)
                try:
                    cutoff = float(cutoffs[i] + cutoffs[j])
                except Exception:
                    cutoff = 1.8
                # Slight slack is intentional for inferred hidden XYZ cells.
                if d <= cutoff * 1.20:
                    g.add_edge(i, j, weight=d)

        if g.number_of_edges() == 0 or not nx.is_connected(g):
            try:
                mst = nx.minimum_spanning_tree(complete, weight='weight')
                g.add_edges_from((u, v, data) for u, v, data in mst.edges(data=True))
            except Exception:
                for i, j in zip(comp[:-1], comp[1:]):
                    g.add_edge(i, j, weight=0.0)
        return list(g.edges())

    def _unwrap_component(self, full_atoms, comp, cell, pbc):
        comp = [int(i) for i in comp]
        raw = {i: full_atoms.positions[i].copy() for i in comp}
        if len(comp) <= 1:
            return raw
        edges = self._connectivity_edges_for_component(full_atoms, comp, cell, pbc)
        graph = nx.Graph()
        graph.add_nodes_from(comp)
        graph.add_edges_from(edges)
        root = comp[0]
        out = {root: raw[root]}
        queue = [root]
        seen = {root}
        while queue:
            u = queue.pop(0)
            for v in graph.neighbors(u):
                if v in seen:
                    continue
                out[v] = out[u] + self._mic_delta(raw[v] - raw[u], cell, pbc)
                seen.add(v)
                queue.append(v)
        # Should only happen for pathological components; keep remaining atoms near root.
        for v in comp:
            if v not in out:
                out[v] = out[root] + self._mic_delta(raw[v] - raw[root], cell, pbc)
        return out

    def _align_components_to_reference(self, pos_map, comps, ref_pos_map, cell, pbc):
        if not ref_pos_map:
            return pos_map
        aligned = dict(pos_map)
        for comp in comps:
            common = [i for i in comp if i in ref_pos_map and i in aligned]
            if not common:
                continue
            curr_com = np.mean([aligned[i] for i in common], axis=0)
            ref_com = np.mean([ref_pos_map[i] for i in common], axis=0)
            nearest_delta = self._mic_delta(curr_com - ref_com, cell, pbc)
            shift = (ref_com + nearest_delta) - curr_com
            for i in comp:
                if i in aligned:
                    aligned[i] = aligned[i] + shift
        return aligned

    @staticmethod
    def _sanitize_local_structure_metadata(atoms, idx_list):
        """Keep local structures self-contained and calculation-ready."""
        atoms.info.clear()
        for name in list(atoms.arrays):
            if name not in {"numbers", "positions"}:
                del atoms.arrays[name]
        atoms.set_array("orig_index", np.asarray(idx_list, dtype=int))
        return atoms

    def _reconstruct_local_atoms(self, frame_idx, idx_list, reference_positions=None):
        """
        Return a local ASE Atoms object with bonded atoms reconstructed into a
        chemically continuous image.  Original trajectory atom IDs are stored in
        ``atoms.arrays['orig_index']`` and ``atoms.info['Atom_Indices']``.
        """
        if self.traj is None or frame_idx < 0 or frame_idx >= len(self.traj):
            return None, {}, 'no_traj'
        idx_list = sorted(int(i) for i in idx_list)
        full = self.traj[frame_idx]
        if not idx_list:
            local = self._sanitize_local_structure_metadata(
                full[:0].copy(),
                [],
            )
            local.info['Atom_Indices'] = ''
            local.info['PBC_Reconstruction'] = 'empty'
            return local, {}, 'empty'

        cell, pbc, status = self._reconstruction_cell(full)
        comps = self._molecular_components_for_indices(frame_idx, idx_list)
        pos_map = {}
        for comp in comps:
            pos_map.update(self._unwrap_component(full, comp, cell, pbc))
        pos_map = self._align_components_to_reference(pos_map, comps, reference_positions, cell, pbc)

        local = self._sanitize_local_structure_metadata(
            full[idx_list].copy(),
            idx_list,
        )
        local.set_positions(np.array([pos_map[i] for i in idx_list], dtype=float))
        local.info['Atom_Indices'] = ','.join(map(str, idx_list))
        local.info['PBC_Reconstruction'] = status
        return local, {i: pos_map[i] for i in idx_list}, status

    def _preserve_periodic_output(self, frame_idx):
        """
        Preserve the true periodic cell for surface/catalytic outputs; otherwise
        export local gas/cluster reactions in a common vacuum cell after PBC repair.
        """
        if self.traj is None or frame_idx < 0 or frame_idx >= len(self.traj):
            return False
        return bool(self.sub_elements or self.sub_indices) and self._has_valid_cell(self.traj[frame_idx])

    def _put_pair_in_common_cell(self, ini_atoms, fin_atoms, source_atoms=None, vacuum=10.0, preserve_periodic=False):
        ini_atoms = ini_atoms.copy()
        fin_atoms = fin_atoms.copy()
        if preserve_periodic and source_atoms is not None and self._has_valid_cell(source_atoms):
            cell = source_atoms.get_cell()
            pbc = source_atoms.get_pbc()
            for atoms in (ini_atoms, fin_atoms):
                atoms.set_cell(cell)
                atoms.set_pbc(pbc)
                atoms.info['Structure_Output_Mode'] = 'periodic_cell'
            return ini_atoms, fin_atoms, True

        all_pos = np.vstack([ini_atoms.get_positions(), fin_atoms.get_positions()])
        lo = np.min(all_pos, axis=0)
        hi = np.max(all_pos, axis=0)
        dims = np.maximum(hi - lo + 2.0 * vacuum, np.array([12.0, 12.0, 12.0]))
        shift = vacuum - lo
        for atoms in (ini_atoms, fin_atoms):
            atoms.set_positions(atoms.get_positions() + shift)
            atoms.set_cell(np.diag(dims))
            atoms.set_pbc([False, False, False])
            atoms.info['Structure_Output_Mode'] = 'cluster_vacuum_after_pbc_repair'
        return ini_atoms, fin_atoms, False

    def _put_sequence_in_common_cell(self, images, source_atoms=None, vacuum=10.0, preserve_periodic=False):
        if not images:
            return images, False
        out = [img.copy() for img in images]
        if preserve_periodic and source_atoms is not None and self._has_valid_cell(source_atoms):
            cell = source_atoms.get_cell()
            pbc = source_atoms.get_pbc()
            for atoms in out:
                atoms.set_cell(cell)
                atoms.set_pbc(pbc)
                atoms.info['Structure_Output_Mode'] = 'periodic_cell'
            return out, True
        all_pos = np.vstack([img.get_positions() for img in out if len(img) > 0])
        lo = np.min(all_pos, axis=0)
        hi = np.max(all_pos, axis=0)
        dims = np.maximum(hi - lo + 2.0 * vacuum, np.array([12.0, 12.0, 12.0]))
        shift = vacuum - lo
        for atoms in out:
            atoms.set_positions(atoms.get_positions() + shift)
            atoms.set_cell(np.diag(dims))
            atoms.set_pbc([False, False, False])
            atoms.info['Structure_Output_Mode'] = 'cluster_vacuum_after_pbc_repair'
        return out, False

    # Backward-compatible wrapper. 
    def _make_contiguous(self, atoms):
        if atoms is None or len(atoms) <= 1:
            return atoms
        if self._has_valid_cell(atoms):
            # No frame/molecule context is available here, so keep this as a safe
            # no-op wrapper for legacy callers.  Real repair happens in
            # _reconstruct_local_atoms where frame molecules are available.
            return atoms
        return atoms

    def _standardize_cell(self, atoms, vacuum=10.0):
        """Legacy wrapper: put an already reconstructed local cluster in vacuum."""
        if len(atoms) == 0:
            return atoms
        atoms = atoms.copy()
        pos = atoms.get_positions()
        lo = np.min(pos, axis=0)
        hi = np.max(pos, axis=0)
        dims = np.maximum(hi - lo + 2.0 * vacuum, np.array([12.0, 12.0, 12.0]))
        atoms.set_positions(pos + (vacuum - lo))
        atoms.set_cell(np.diag(dims))
        atoms.set_pbc([False, False, False])
        atoms.info['Structure_Output_Mode'] = 'cluster_vacuum'
        return atoms

    def _generate_neb_inputs(self, ini_frame_idx, fin_frame_idx, atom_indices, rxn_id_str, r_str, p_str):
        if (self.sub_elements or self.sub_indices) and self.traj:
            return "N/A (surface system)"

        if not self.traj or self.neb_dir == "DISABLED" or ini_frame_idx < 0 or fin_frame_idx >= len(self.traj):
            return "N/A"

        idx_list = sorted(int(i) for i in atom_indices)
        ini_atoms, ini_map, status_i = self._reconstruct_local_atoms(ini_frame_idx, idx_list)
        fin_atoms, fin_map, status_f = self._reconstruct_local_atoms(fin_frame_idx, idx_list, reference_positions=ini_map)
        if ini_atoms is None or fin_atoms is None or len(ini_atoms) == 0 or len(fin_atoms) == 0:
            return "N/A"

        preserve_periodic = self._preserve_periodic_output(ini_frame_idx)
        source_atoms = self.traj[ini_frame_idx] if self.traj is not None else None
        ini_atoms, fin_atoms, use_mic = self._put_pair_in_common_cell(
            ini_atoms, fin_atoms, source_atoms=source_atoms, vacuum=10.0, preserve_periodic=preserve_periodic
        )
        ini_atoms.info['Reactant_Frame'] = int(ini_frame_idx) + self.frame_offset
        ini_atoms.info['Product_Frame'] = int(fin_frame_idx) + self.frame_offset
        ini_atoms.info['Reactant_Local_Frame'] = int(ini_frame_idx)
        ini_atoms.info['Product_Local_Frame'] = int(fin_frame_idx)
        ini_atoms.info['PBC_Reconstruction'] = f"reactant:{status_i}; product:{status_f}"
        fin_atoms.info.update(ini_atoms.info)

        full_path = os.path.join(self.neb_dir, f"{rxn_id_str}_NEB")
        n_interp = 5
        images = [ini_atoms.copy()]
        for _ in range(n_interp):
            images.append(ini_atoms.copy())
        images.append(fin_atoms.copy())

        if NEB is None:
            raise RuntimeError("ASE NEB support is unavailable")
        neb = NEB(images)
        try:
            neb.interpolate(method='idpp', mic=bool(use_mic))
        except Exception:
            try:
                neb.interpolate(method='linear', mic=bool(use_mic))
            except Exception as linear_exc:
                raise RuntimeError(
                    "both IDPP and linear NEB interpolation failed"
                ) from linear_exc

        os.makedirs(full_path, exist_ok=True)
        for k, img in enumerate(images):
            img.info['NEB_Image'] = k
            img.info['Reaction_ID'] = rxn_id_str
            write(os.path.join(full_path, f"{k:02d}.xyz"), img)
        return full_path

    def _save_trajectory(self, start_frame, end_frame, atom_indices, rxn_id_str, suffix=""):
        if not self.traj or not self.output_dir:
            return "N/A"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        clean_indices = list(atom_indices)
        if self.sub_elements or self.sub_indices:
            atoms = self.traj[start_frame]
            syms = atoms.get_chemical_symbols()
            clean_indices = [idx for idx in atom_indices if syms[idx] not in self.sub_elements and idx not in self.sub_indices]
            if not clean_indices:
                clean_indices = list(atom_indices)
        idx_list = sorted(int(i) for i in clean_indices)

        images = []
        step = 1
        duration = int(end_frame) - int(start_frame)
        if duration > 200:
            step = max(1, duration // 100)

        prev_map = None
        for f in range(int(start_frame), int(end_frame) + 1, step):
            if f < 0 or f >= len(self.traj):
                break
            atoms, pos_map, status = self._reconstruct_local_atoms(f, idx_list, reference_positions=prev_map)
            if atoms is None:
                continue
            atoms.info['frame'] = int(f) + self.frame_offset
            atoms.info['local_frame'] = int(f)
            atoms.info['rxn_id'] = rxn_id_str
            atoms.info['PBC_Reconstruction'] = status
            images.append(atoms)
            prev_map = pos_map

        if not images:
            return "N/A"
        preserve_periodic = self._preserve_periodic_output(start_frame)
        source_atoms = self.traj[start_frame] if self.traj is not None else None
        images, _use_mic = self._put_sequence_in_common_cell(
            images, source_atoms=source_atoms, vacuum=10.0, preserve_periodic=preserve_periodic
        )

        fname = f"{rxn_id_str}_traj{suffix}.xyz"
        fpath = os.path.join(self.output_dir, fname)
        write(fpath, images)
        return fpath

    def _calculate_compactness(self, frame_idx, atom_indices):
        if not self.traj or frame_idx >= len(self.traj):
            return 9999.0
        clean_indices = list(atom_indices)
        if self.sub_elements or self.sub_indices:
            atoms = self.traj[frame_idx]
            syms = atoms.get_chemical_symbols()
            clean_indices = [idx for idx in atom_indices if syms[idx] not in self.sub_elements and idx not in self.sub_indices]
            if not clean_indices:
                clean_indices = list(atom_indices)

        try:
            atoms, _pos_map, _status = self._reconstruct_local_atoms(frame_idx, clean_indices)
            positions = atoms.get_positions() if atoms is not None else self.traj[frame_idx][clean_indices].get_positions()
        except Exception:
            positions = self.traj[frame_idx][clean_indices].get_positions()
        if len(positions) <= 1:
            return 0.0
        center = np.mean(positions, axis=0)
        rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
        return rg

    def _resolve_structure_workers(self, n_workers, n_tasks):
        """Resolve worker count for structural output generation."""
        if n_tasks <= 0:
            return 0
        cpu_count = os.cpu_count() or 1

        if n_workers is None:
            env_value = os.environ.get("RXNSCAPE_STRUCTURE_WORKERS")
            if env_value:
                try:
                    n_workers = int(env_value)
                except Exception:
                    print(f"⚠️ Ignoring invalid RXNSCAPE_STRUCTURE_WORKERS={env_value!r}")
                    n_workers = None

        if n_workers is None:
            n_workers = min(4, max(1, cpu_count - 1))

        try:
            n_workers = int(n_workers)
        except Exception:
            n_workers = 1

        return max(1, min(n_workers, n_tasks, cpu_count))

    def _record_structure_error(self, rxn_id, error_text):
        """Append structural output errors without interrupting the full run."""
        try:
            log_path = os.path.join(os.getcwd(), "structure_generation_errors.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {rxn_id} =====\n")
                f.write(str(error_text).rstrip() + "\n")
        except Exception:
            pass

    def _run_structure_tasks_serial(self, tasks, desc="Writing Structures"):
        results_map = {}
        for task in tqdm(tasks, total=len(tasks), unit="rxn", desc=desc):
            r_id, neb, snap, err = _process_single_event_task(task)
            results_map[r_id] = (neb, snap)
            if err:
                self._record_structure_error(r_id, err)
        return results_map

    def generate_output_files(self, df, snap_duration=10, n_workers=None):
        if df.empty:
            return df
        print("📸 Generating Structural Files...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if self.neb_dir != "DISABLED" and not os.path.exists(self.neb_dir):
            os.makedirs(self.neb_dir)

        if self.traj is None or len(self.traj) == 0:
            print("⚠️ No trajectory is available. Structural files will be skipped.")
            df_results = df.copy()
            df_results['NEB_Input'] = "Skipped (No Trajectory)"
            df_results['Snapshot'] = "Skipped (No Trajectory)"
            return _reaction_id_first(df_results)

        # 1. select representative reactions
        representative_keys = [
            'Reactants_SMILES',
            'Products_SMILES',
            'Reactants_Formula',
            'Products_Formula',
        ]
        selection_df = df.copy()
        for key in representative_keys:
            selection_df[key] = selection_df.get(
                key, pd.Series('', index=selection_df.index)
            ).map(
                lambda value: ''
                if _is_missing_scalar(value)
                else str(value)
            )
        grouped = selection_df.groupby(representative_keys, dropna=False)
        best_events_map = {}

        for group_key, group in tqdm(grouped, desc="Selecting Representatives"):
            candidates = group.sort_values(
                ['Frame', 'Reaction_ID'],
                kind='mergesort',
            ).head(5)
            scored_candidates = []
            for _, row in candidates.iterrows():
                frame = int(row['Frame'])
                indices = row['Atom_Indices']
                if isinstance(indices, str):
                    try:
                        import ast
                        indices = ast.literal_eval(indices)
                    except Exception:
                        indices = []
                score = self._calculate_compactness(frame, indices)
                scored_candidates.append((score, row['Reaction_ID']))

            if not scored_candidates:
                continue
            scored_candidates.sort(key=lambda x: x[0])
            mid_idx = len(scored_candidates) // 2
            best_rxn_id = scored_candidates[mid_idx][1]
            best_events_map[group_key] = best_rxn_id

        # 2. prepare parallel tasks
        tasks = []
        df_results = df.copy()
        df_results['NEB_Input'] = "Skipped (Representative Exists)"
        df_results['Snapshot'] = "Skipped (Representative Exists)"
        df_results['Structure_Representative_ID'] = ""

        calc_indices = []
        traj_len = len(self.traj)

        for idx, row in df.iterrows():
            r_smi = row['Reactants_SMILES']
            p_smi = row['Products_SMILES']
            rxn_id = row['Reaction_ID']
            group_key = tuple(
                '' if _is_missing_scalar(row.get(key, '')) else str(row.get(key, ''))
                for key in representative_keys
            )
            representative_id = best_events_map.get(group_key, '')
            df_results.at[idx, 'Structure_Representative_ID'] = representative_id

            if representative_id == rxn_id:
                f_start = int(row.get('Reactant_Frame', row['Frame'] - 5))
                f_end = int(row.get('Product_Frame', row['Frame'] + 5))

                # Expand the padding significantly to ensure we don't truncate the trajectory
                # subset before the molecules have physically separated in the visualization.
                safe_start = max(0, f_start - 30)
                safe_end = min(traj_len, f_end + 30)
                if safe_end <= safe_start:
                    safe_end = min(traj_len, safe_start + 1)

                row_data = row.to_dict()
                row_data['Reactant_Frame'] = f_start - safe_start
                row_data['Product_Frame'] = f_end - safe_start

                # Zero-copy memory sharing task definition
                task = (
                    row_data, f_start - safe_start, f_end - safe_start,
                    safe_start, safe_end - 1,
                    self.sub_elements, self.sub_indices, self.neb_dir, self.output_dir,
                    self.bond_scale, self.adsorption_scale, r_smi, p_smi, rxn_id
                )
                tasks.append(task)
                calc_indices.append(idx)

        if not tasks:
            return _reaction_id_first(df_results)

        # 3. run in parallel.
        global _GLOBAL_RXN_TRAJ, _GLOBAL_RXN_FRAMES
        _GLOBAL_RXN_TRAJ = self.traj
        _GLOBAL_RXN_FRAMES = self.frames

        n_workers = self._resolve_structure_workers(n_workers, len(tasks))
        print(f"   🚀 Launching {len(tasks)} structural tasks on {n_workers} cores...")
        if n_workers == 1 or not _can_inherit_reaction_globals():
            if n_workers > 1:
                print(
                    "⚠️ Current multiprocessing start method cannot inherit "
                    "trajectory globals; writing structures serially."
                )
            results_map = self._run_structure_tasks_serial(tasks)
        else:
            results_map = {}
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_task = {executor.submit(_process_single_event_task, task): task for task in tasks}
                    for future in tqdm(as_completed(future_to_task), total=len(future_to_task), unit="rxn", desc="Writing Structures"):
                        task = future_to_task[future]
                        rxn_id = task[-1]
                        try:
                            r_id, neb, snap, err = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as exc:
                            r_id = rxn_id
                            msg = f"{type(exc).__name__}: {exc}"
                            neb = f"Structure_Error: {msg}"
                            snap = f"Structure_Error: {msg}"
                            err = msg
                        results_map[r_id] = (neb, snap)
                        if err:
                            self._record_structure_error(r_id, err)
            except BrokenProcessPool:
                print("⚠️ Structural worker pool crashed, likely due to memory pressure.")
                print("   Retrying unfinished structural tasks serially...")
                remaining_tasks = [task for task in tasks if task[-1] not in results_map]
                retry_results = self._run_structure_tasks_serial(remaining_tasks, desc="Retry Structures")
                results_map.update(retry_results)

        for idx in calc_indices:
            rid = df_results.at[idx, 'Reaction_ID']
            if rid in results_map:
                df_results.at[idx, 'NEB_Input'] = results_map[rid][0]
                df_results.at[idx, 'Snapshot'] = results_map[rid][1]

        return _reaction_id_first(df_results)


    def _is_radical(self, formula, size):
        if size == 1: return True
        import re
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        val_map = {'C': 4, 'H': 1, 'O': 6, 'Cl': 7, 'F': 7, 'Br': 7, 'N': 5, 'S': 6, 'P': 5}
        total_val = 0
        for sym, count in matches:
            c = int(count) if count else 1
            total_val += val_map.get(sym, 0) * c
        return total_val % 2 != 0
        
    def _make_adsorption_pair_cutoffs(self, atoms, is_sub_mask, pre_cutoffs):
        """Only search substrate-mobile neighbor pairs for adsorption detection."""
        if atoms is None or is_sub_mask is None or pre_cutoffs is None:
            return None

        symbols = atoms.get_chemical_symbols()
        if len(symbols) != len(is_sub_mask) or len(symbols) != len(pre_cutoffs):
            return None

        sub_elems = {symbols[i] for i, is_sub in enumerate(is_sub_mask) if is_sub}
        mob_elems = {symbols[i] for i, is_sub in enumerate(is_sub_mask) if not is_sub}

        if not sub_elems or not mob_elems:
            return None

        if sub_elems.intersection(mob_elems):
            return None

        radius = {}
        for sym, r in zip(symbols, pre_cutoffs):
            radius.setdefault(sym, float(r))

        pair_cutoffs = {}
        for s in sub_elems:
            for m in mob_elems:
                pair_cutoffs[(s, m)] = radius[s] + radius[m]

        return pair_cutoffs or None

    def _adsorbed_indices_for_atoms(self, atoms, frame_idx=None, is_sub_mask=None, pre_cutoffs=None):
        """Return all non-substrate atom indices touching the substrate in one frame."""
        if not (self.sub_elements or self.sub_indices) or atoms is None:
            return set()

        if frame_idx is None:
            frame_idx = self._frame_index_for_atoms(atoms)
        if frame_idx is not None and frame_idx in self._ads_idx_by_frame:
            return self._ads_idx_by_frame[frame_idx]

        obj_key = id(atoms)
        cached = self._ads_idx_by_obj.get(obj_key)
        if cached is not None:
            return cached

        curr_len = len(atoms)
        if is_sub_mask is not None and curr_len == len(is_sub_mask):
            ads_idx = _adsorbed_indices_from_geometry(
                atoms,
                is_sub_mask,
                pre_cutoffs,
                self._ads_pair_cutoffs,
                self.adsorption_scale,
            )
        else:
            if pre_cutoffs is not None and curr_len == len(pre_cutoffs):
                cutoffs = pre_cutoffs
            else:
                cutoffs = natural_cutoffs(
                    atoms, mult=self.adsorption_scale
                )
            try:
                if self._ads_pair_cutoffs is not None:
                    nl = neighbor_list(
                        'ij', atoms, self._ads_pair_cutoffs
                    )
                else:
                    nl = neighbor_list('ij', atoms, cutoffs)
            except Exception:
                nl = neighbor_list('ij', atoms, cutoffs)
            ads_idx = set()
            symbols = atoms.get_chemical_symbols()
            for u, v in zip(nl[0], nl[1]):
                is_u_sub = (u in self.sub_indices) or (symbols[u] in self.sub_elements)
                is_v_sub = (v in self.sub_indices) or (symbols[v] in self.sub_elements)
                if is_v_sub and not is_u_sub:
                    ads_idx.add(int(u))
                if is_u_sub and not is_v_sub:
                    ads_idx.add(int(v))

        if frame_idx is not None:
            self._ads_idx_by_frame[frame_idx] = ads_idx
        self._ads_idx_by_obj[obj_key] = ads_idx
        return ads_idx

    def _precompute_adsorption_indices(
        self,
        is_sub_mask,
        pre_cutoffs,
        n_workers=1,
    ):
        """Populate the compact per-frame adsorption cache once per run."""
        if self.traj is None or not is_sub_mask or pre_cutoffs is None:
            return

        n_frames = len(self.traj)
        if n_frames == 0:
            return

        try:
            n_workers = max(1, int(n_workers or 1))
        except Exception:
            n_workers = 1

        can_parallel = (
            n_workers > 1
            and n_frames >= n_workers * 8
            and _can_inherit_reaction_globals()
        )
        chunk_size = max(1, n_frames // (n_workers * 4))
        chunks = [
            (
                start,
                min(n_frames, start + chunk_size),
                is_sub_mask,
                pre_cutoffs,
                self._ads_pair_cutoffs,
                self.adsorption_scale,
            )
            for start in range(0, n_frames, chunk_size)
        ]

        global _GLOBAL_RXN_TRAJ
        _GLOBAL_RXN_TRAJ = self.traj

        if can_parallel:
            print(
                f"   - Pre-computing adsorption contacts "
                f"({n_workers} workers)..."
            )
            try:
                with ProcessPoolExecutor(
                    max_workers=n_workers
                ) as executor:
                    iterator = executor.map(
                        _compute_adsorption_indices_chunk, chunks
                    )
                    frame_idx = 0
                    for result in tqdm(
                        iterator,
                        total=len(chunks),
                        desc="   Adsorption Contacts",
                        unit="chunk",
                    ):
                        for ads_idx in result:
                            self._ads_idx_by_frame[frame_idx] = ads_idx
                            frame_idx += 1
                if frame_idx == n_frames:
                    return
                self._ads_idx_by_frame.clear()
            except (BrokenProcessPool, Exception) as exc:
                self._ads_idx_by_frame.clear()
                print(
                    "   ⚠️ Parallel adsorption mapping failed; "
                    f"falling back to serial ({type(exc).__name__})."
                )

        print("   - Pre-computing adsorption contacts (serial)...")
        for frame_idx, atoms in tqdm(
            enumerate(self.traj),
            total=n_frames,
            desc="   Adsorption Contacts",
        ):
            self._ads_idx_by_frame[frame_idx] = (
                _adsorbed_indices_from_geometry(
                    atoms,
                    is_sub_mask,
                    pre_cutoffs,
                    self._ads_pair_cutoffs,
                    self.adsorption_scale,
                )
            )

    def _is_adsorbed(self, atoms, mol_indices, is_sub_mask=None, pre_cutoffs=None):
        if not (self.sub_elements or self.sub_indices) or not mol_indices:
            return False
        ads_idx = self._adsorbed_indices_for_atoms(atoms, None, is_sub_mask, pre_cutoffs)
        if not ads_idx:
            return False
        return any(int(i) in ads_idx for i in mol_indices)

    def _classify_reaction(self, sub_r, sub_p, b_types, f_types, f_react_atoms=None, f_prod_atoms=None, is_sub_mask=None, pre_cutoffs=None):
        base_type = self._get_base_reaction_type(sub_r, sub_p, b_types, f_types)
        
        # Surface Chemistry Mapping
        if (self.sub_elements or self.sub_indices) and f_react_atoms is not None and f_prod_atoms is not None:
            # Reuse the single atom-mapped spectator cancellation policy.
            r_net, p_net = self._net_molecules(sub_r, sub_p)

            r_adsorbed = [self._is_adsorbed(f_react_atoms, m.get('indices', []), is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs) for m in r_net]
            p_adsorbed = [self._is_adsorbed(f_prod_atoms, m.get('indices', []), is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs) for m in p_net]

            r_gas_count = r_adsorbed.count(False)
            r_ads_count = r_adsorbed.count(True)
            p_gas_count = p_adsorbed.count(False)
            p_ads_count = p_adsorbed.count(True)

            r_formulas = [m.get('formula', '') for m in r_net]

            # LH and ER
            mechanism_prefix = ""
            if r_ads_count > 0 and r_gas_count == 0:
                mechanism_prefix = "L-H "
            elif r_ads_count > 0 and r_gas_count > 0:
                mechanism_prefix = "E-R "
            elif r_ads_count == 0:
                mechanism_prefix = "Gas-phase "

            if base_type in ["Radical Recombination", "Association", "Radical Addition"]:
                if r_ads_count == 0:
                    if p_ads_count > 0: return "Associative Adsorption"
                    return "Gas-phase " + base_type
                if p_ads_count == 0:
                    return "Associative Desorption"
                if any(f == 'H' for f in r_formulas):
                    return f"Surface {mechanism_prefix}Hydrogenation"
                return f"Surface {mechanism_prefix}Coupling"
                
            elif base_type in ["Homolysis", "Radical Dissociation", "Fragmentation", "Elimination"]:
                if r_ads_count == 0:
                    if p_ads_count > 0: return "Dissociative Adsorption"
                    return "Gas-phase " + base_type
                if p_ads_count == 0:
                    return "Desorptive Dissociation"
                return "Surface Dissociation"

            elif (
                base_type in [
                    "Atom Abstraction",
                    "Substitution",
                    "Group Transfer",
                    "Atom Transfer",
                ]
                or base_type.endswith("-Atom Transfer")
            ):
                if r_ads_count == 0:
                    return "Gas-phase " + base_type
                return f"Surface {mechanism_prefix}Transfer"
                
            elif base_type in ["Conformational Change", "Physical Collision", "Intramolecular Rearrangement", "Intramolecular Ring Closure", "Intramolecular Ring Opening"]:
                if r_ads_count == 0 and p_ads_count > 0:
                    return "Molecular Adsorption"
                if r_ads_count > 0 and p_ads_count == 0:
                    return "Molecular Desorption"
                if r_ads_count == 0:
                    return "Gas-phase " + base_type

            return f"Surface Reaction" if r_ads_count > 0 else f"Gas-phase {base_type}"
                
        return base_type

    def _get_base_reaction_type(self, sub_r, sub_p, b_types, f_types):
        """
        Chemistry-first classification.
        """
        # --- 1) Remove unchanged atom-mapped spectators once, centrally. ---
        r_net, p_net = self._net_molecules(sub_r, sub_p)

        # --- 2) basic metrics ---
        b_total = sum(b_types.values()) if isinstance(b_types, dict) else len(b_types)
        f_total = sum(f_types.values()) if isinstance(f_types, dict) else len(f_types)
        if not r_net and not p_net:
            return "Physical Collision" if (b_total == 0 and f_total == 0) else "Conformational Change"

        get_size = lambda m: len(m.get('indices', [1])) or 1
        r_sizes = sorted([get_size(m) for m in r_net])
        p_sizes = sorted([get_size(m) for m in p_net])
        r_count, p_count = len(r_net), len(p_net)
        r_formulas = [m.get('formula', '') for m in r_net]
        p_formulas = [m.get('formula', '') for m in p_net]
        has_radical_r = any(self._is_radical(m.get('formula', ''), len(m.get('indices', [1])) or 1) for m in r_net)
        has_radical_p = any(self._is_radical(m.get('formula', ''), len(m.get('indices', [1])) or 1) for m in p_net)
        all_radical_r = all(self._is_radical(m.get('formula', ''), len(m.get('indices', [1])) or 1) for m in r_net)
        all_radical_p = all(self._is_radical(m.get('formula', ''), len(m.get('indices', [1])) or 1) for m in p_net)

        # classical small-molecule elimination
        small_gases = {'HCl', 'H2', 'Cl2', 'H2O', 'HF', 'HBr', 'NH3', 'CO', 'CO2', 'N2', 'O2', 'H2S', 'SO2'}

        # 1 → 1 intramolecular reaction
        if r_count == 1 and p_count == 1:
            if b_total == 0 and f_total == 0:
                return "Conformational Change"
            if b_total == 0 and f_total > 0:
                return "Intramolecular Ring Closure"
            if b_total > 0 and f_total == 0:
                return "Intramolecular Ring Opening"
            return "Intramolecular Rearrangement"

        # 1 → N decomposition
        if r_count == 1 and p_count >= 2:
            if p_count == 2:
                if all_radical_p:
                    return "Homolysis"
                if has_radical_p:
                    return "Radical Dissociation"
                if any(f in small_gases for f in p_formulas):
                    return "Elimination"
                return "Fragmentation"
            else:
                return "Complex Fragmentation"

        # N → 1 addition / recombination
        if r_count >= 2 and p_count == 1:
            if all_radical_r:
                return "Radical Recombination"
            if has_radical_r:
                return "Radical Addition"
            return "Association"

        # 2 → 2 exchange / abstraction
        if r_count == 2 and p_count == 2:
            a, b = r_sizes[0], r_sizes[1]
            c, d = p_sizes[0], p_sizes[1]

            transferred_elements = self._mapped_transfer_elements(
                r_net,
                p_net,
            )
            if transferred_elements:
                if transferred_elements == ["H"]:
                    return "Atom Abstraction"
                if len(transferred_elements) == 1:
                    return f"{transferred_elements[0]}-Atom Transfer"
                return "Group Transfer"

            if has_radical_r:
                return "Atom Abstraction"

            shift = min(abs(a - c), abs(a - d))
            if shift == 0:
                return "Physical Collision" if (b_total == 0 and f_total == 0) else "Substitution"
            if shift in (1, 2):
                return "Atom Abstraction"
            if min(p_sizes) <= 4 and max(p_sizes) > max(r_sizes):
                return "Condensation"
            return "Group Transfer"

        # multimolecular complex cases
        if r_count >= 2 and p_count >= 2:
            if r_count < p_count:
                return "Complex Fragmentation"
            if r_count > p_count:
                return "Complex Association"
            return "Complex Exchange"

        return "Complex Reaction"

    def _mapped_transfer_elements(self, reactants, products):
        """Identify atoms transferred between two mapped molecules."""
        if len(reactants) != 2 or len(products) != 2:
            return []

        reactant_sets = [
            set(int(index) for index in molecule.get("indices", []))
            for molecule in reactants
        ]
        product_sets = [
            set(int(index) for index in molecule.get("indices", []))
            for molecule in products
        ]
        if (
            not all(reactant_sets)
            or not all(product_sets)
            or set().union(*reactant_sets) != set().union(*product_sets)
        ):
            return []

        pairings = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
        pairing = max(
            pairings,
            key=lambda pairs: sum(
                len(reactant_sets[r_index] & product_sets[p_index])
                for r_index, p_index in pairs
            ),
        )

        gained = Counter()
        lost = Counter()
        for reactant_index, product_index in pairing:
            reactant_formula = self._formula_counter_one(
                reactants[reactant_index].get("formula", "")
            )
            product_formula = self._formula_counter_one(
                products[product_index].get("formula", "")
            )
            gained.update(product_formula - reactant_formula)
            lost.update(reactant_formula - product_formula)

        if not gained or gained != lost:
            return []

        return sorted(
            element
            for element, count in gained.items()
            for _ in range(int(count))
        )

    def _strip_surface_suffix(self, formula):
        """Remove adsorption labels used only for reporting, e.g. CH3* / CH3(g)."""
        if formula is None:
            return ""
        s = str(formula).strip()
        if s.endswith('*'):
            s = s[:-1]
        if s.endswith('(g)'):
            s = s[:-3]
        return s.strip()

    def _formula_counter_one(self, formula):
        """Parse a Hill-like formula into an element counter. Robust for CH4, O2, HO, etc."""
        import re
        formula = self._strip_surface_suffix(formula)
        c = Counter()
        if not formula:
            return c
        # Drop simple charge/radical annotations if they appear in future outputs.
        formula = formula.replace('+', '').replace('-', '').replace('.', '')
        for sym, num in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            c[sym] += int(num) if num else 1
        return c

    def _formula_counter_many(self, formula_text):
        total = Counter()
        if not isinstance(formula_text, str):
            return total
        for part in formula_text.split(' + '):
            part = part.strip()
            if part:
                total.update(self._formula_counter_one(part))
        return total

    def _is_mass_balanced_formula_text(self, r_formula, p_formula):
        return self._formula_counter_many(r_formula) == self._formula_counter_many(p_formula)

    def _species_count_from_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return 0
        return len([x for x in text.split(' + ') if x.strip()])

    def _event_species_count(self, record):
        # Backward-compatible helper retained for external callers.
        return max(
            self._species_count_from_text(record.get('Reactants_Formula', '')),
            self._species_count_from_text(record.get('Products_Formula', '')),
            self._species_count_from_text(record.get('Reactants_SMILES', '')),
            self._species_count_from_text(record.get('Products_SMILES', '')),
        )

    def _elementary_side_counts(self, record):
        r_count = max(
            self._species_count_from_text(record.get('Reactants_Formula', '')),
            self._species_count_from_text(record.get('Reactants_SMILES', '')),
        )
        p_count = max(
            self._species_count_from_text(record.get('Products_Formula', '')),
            self._species_count_from_text(record.get('Products_SMILES', '')),
        )
        return r_count, p_count

    def _is_elementary_stoichiometry(self, record):
        """Chemistry-facing elementary guard: 1->1, 1->2, 2->1, or 2->2 (and up to 3 for termolecular MD events).
        """
        r_count, p_count = self._elementary_side_counts(record)
        return 1 <= r_count <= 3 and 1 <= p_count <= 3

    def _mol_equiv_key(self, mol):
        """Key used for cancelling unchanged spectators."""
        return (self._molecule_identity(mol), tuple(sorted(int(i) for i in mol.get('indices', []))))

    def _net_molecules(self, sub_r, sub_p):
        """Remove unchanged spectator molecules by species + size, preserving atom-mapped changes."""
        net_r = []
        p_pool = list(sub_p)
        for r in sub_r:
            kr = self._mol_equiv_key(r)
            found = -1
            for i, p in enumerate(p_pool):
                if self._mol_equiv_key(p) == kr:
                    found = i
                    break
            if found >= 0:
                p_pool.pop(found)
            else:
                net_r.append(r)
        net_p = p_pool
        if not net_r and not net_p:
            return list(sub_r), list(sub_p)
        return net_r, net_p

    def _build_formula_list(self, mols, frame_idx, is_sub_mask=None, pre_cutoffs=None):
        out = []
        atoms = None
        if self.traj is not None and frame_idx is not None and 0 <= frame_idx < len(self.traj):
            atoms = self.traj[frame_idx]
        for m in mols:
            f = m.get('formula', '')
            if (self.sub_elements or self.sub_indices) and atoms is not None:
                f += '*' if self._is_adsorbed(atoms, m.get('indices', []), is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs) else '(g)'
            out.append(f)
        return out

    def _focus_key(self, focus_indices):
        return tuple(sorted(int(i) for i in focus_indices))

    def _atom_mol_index_for_frame(self, frame_idx):
        """Build atom -> molecule index once per frame for fast local molecule lookup."""
        frame_idx = int(frame_idx)

        cached = self._frame_atom_mol_index_cache.get(frame_idx)
        if cached is not None:
            return cached

        index = defaultdict(list)

        for mi, m in enumerate(self.frames[frame_idx]):
            inds = m.get('indices', [])

            # Keep original behavior: skip substrate-only molecules.
            if self.sub_indices and inds and all(int(i) in self.sub_indices for i in inds):
                continue

            for idx in inds:
                index[int(idx)].append(mi)

        self._frame_atom_mol_index_cache[frame_idx] = index
        return index

    def _local_mols(self, frame_idx, focus_indices):
        """Molecules in a frame intersecting focus atom indices, excluding substrate-only species."""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return []

        focus_key = self._focus_key(focus_indices)
        cache_key = (int(frame_idx), focus_key)

        cached = self._local_mols_cache.get(cache_key)
        if cached is not None:
            return cached

        atom_to_mols = self._atom_mol_index_for_frame(int(frame_idx))

        mol_ids = set()
        for idx in focus_key:
            mol_ids.update(atom_to_mols.get(int(idx), []))

        mols = [self.frames[frame_idx][i] for i in sorted(mol_ids)]

        self._local_mols_cache[cache_key] = mols
        return mols

    def _local_state_signature(self, mols, frame_idx=None):
        """Return an atom-mapped local state consistent with the global state.Surface adsorption labels are included locally."""
        ads_idx = set()
        use_adsorption = (
            frame_idx is not None
            and self.traj is not None
            and 0 <= int(frame_idx) < len(self.traj)
            and bool(self.sub_elements or self.sub_indices)
        )
        if use_adsorption:
            ads_idx = self._adsorbed_indices_for_atoms(
                self.traj[int(frame_idx)], frame_idx=int(frame_idx)
            )

        sig = []
        for m in mols:
            inds = tuple(sorted(int(i) for i in m.get('indices', [])))
            label = self._molecule_identity(m)
            if use_adsorption:
                suffix = '*' if any(i in ads_idx for i in inds) else '(g)'
                label = f"{label}_{suffix}"
            sig.append((label, inds))
        return tuple(sorted(sig))

    def _local_signature_at(self, frame_idx, focus_indices):
        focus_key = self._focus_key(focus_indices)
        cache_key = (int(frame_idx), focus_key)
        cached = self._local_sig_cache.get(cache_key)
        if cached is not None:
            return cached
        sig = self._local_state_signature(
            self._local_mols(frame_idx, focus_key), frame_idx=frame_idx
        )
        self._local_sig_cache[cache_key] = sig
        return sig

    def _local_topology_signature_at(self, frame_idx, focus_indices):
        """Return an atom-mapped local topology signature without adsorption."""
        focus_key = self._focus_key(focus_indices)
        cache_key = (int(frame_idx), focus_key)
        cached = self._local_topology_sig_cache.get(cache_key)
        if cached is not None:
            return cached

        signature = tuple(sorted(
            self._molecule_instance_key(molecule)
            for molecule in self._local_mols(frame_idx, focus_key)
        ))
        self._local_topology_sig_cache[cache_key] = signature
        return signature

    def _atom_instance_memberships(self, molecules):
        """Map each non-substrate atom to its molecule-instance state."""
        memberships = {}
        for molecule in list(molecules):
            atom_group = self._molecule_atom_group(molecule)
            if not atom_group:
                continue
            instance = (
                self._molecule_identity(molecule),
                atom_group,
            )
            for atom in atom_group:
                memberships[int(atom)] = instance
        return memberships

    def _topology_boundary_components(
        self,
        reactant_molecules,
        product_molecules,
        reactant_memberships=None,
        product_memberships=None,
    ):
        """Return complete reaction-context components changed at one boundary."""
        if reactant_memberships is None:
            reactant_memberships = self._atom_instance_memberships(
                reactant_molecules
            )
        if product_memberships is None:
            product_memberships = self._atom_instance_memberships(
                product_molecules
            )

        all_atoms = (
            set(reactant_memberships)
            | set(product_memberships)
        )
        changed_atoms = {
            int(atom)
            for atom in all_atoms
            if reactant_memberships.get(atom)
            != product_memberships.get(atom)
        }
        if not changed_atoms:
            return []

        parent = {atom: atom for atom in changed_atoms}

        def find(atom):
            root = atom
            while parent[root] != root:
                root = parent[root]
            while parent[atom] != atom:
                next_atom = parent[atom]
                parent[atom] = root
                atom = next_atom
            return root

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for molecule in (
            list(reactant_molecules)
            + list(product_molecules)
        ):
            local_group = [
                int(atom)
                for atom in self._molecule_atom_group(molecule)
                if int(atom) in changed_atoms
            ]
            if len(local_group) < 2:
                continue
            anchor = local_group[0]
            for atom in local_group[1:]:
                union(anchor, atom)

        changed_components = defaultdict(set)
        for atom in changed_atoms:
            changed_components[find(atom)].add(int(atom))

        molecule_groups = []
        for molecule in list(reactant_molecules) + list(product_molecules):
            group = {
                int(atom)
                for atom in self._molecule_atom_group(molecule)
            }
            if group:
                molecule_groups.append(group)

        def close_over_endpoint_molecules(seed):
            """Return the full endpoint-molecule closure of ``seed``."""
            expanded = set(seed)
            changed = True
            while changed:
                changed = False
                for group in molecule_groups:
                    if group & expanded and not group <= expanded:
                        expanded.update(group)
                        changed = True
            return expanded

        closed_components = []
        for component in changed_components.values():
            expanded = close_over_endpoint_molecules(component)
            merged = True
            while merged:
                merged = False
                for index in range(len(closed_components) - 1, -1, -1):
                    if expanded & closed_components[index]:
                        expanded.update(closed_components.pop(index))
                        expanded = close_over_endpoint_molecules(expanded)
                        merged = True
            closed_components.append(expanded)

        return [
            tuple(sorted(component))
            for component in sorted(
                closed_components,
                key=lambda values: (
                    min(values) if values else -1,
                    len(values),
                ),
            )
            if component
        ]

    def _build_local_timeline(self, start, end, focus_indices, stable_frames):
        """Build stable local runs and ignore short flashes at both boundaries."""
        n_frames = len(self.frames)
        if n_frames == 0:
            return []
        start = max(0, int(start))
        end = min(n_frames - 1, int(end))
        if end < start:
            return []

        focus_key = self._focus_key(focus_indices)
        states = [self._local_signature_at(f, focus_key) for f in range(start, end + 1)]
        stable_frames = max(1, int(stable_frames))
        timeline = []

        i = 0
        n = len(states)
        while i < n:
            j = i + 1
            while j < n and states[j] == states[i]:
                j += 1
            if (j - i) >= stable_frames:
                run_start = start + i
                run_end = start + j - 1
                state = states[i]
                if not timeline or state != timeline[-1]['state']:
                    timeline.append({
                        'start': run_start,
                        'end': run_end,
                        'state': state,
                    })
                else:
                    timeline[-1]['end'] = run_end
            i = j
        return timeline

    def _isomerization_class(self, net_r, net_p):
        """Classify 1-to-1 same-formula changes by validated identity layers."""
        if len(net_r) != 1 or len(net_p) != 1:
            return ''

        r_formula = self._strip_surface_suffix(net_r[0].get('formula', ''))
        p_formula = self._strip_surface_suffix(net_p[0].get('formula', ''))
        if not r_formula or r_formula != p_formula:
            return ''

        if not (
            self._molecule_clean_eligible(net_r[0])
            and self._molecule_clean_eligible(net_p[0])
        ):
            return 'valence_ambiguous_topology_change'

        if not (
            self._molecule_topology_valid(net_r[0])
            and self._molecule_topology_valid(net_p[0])
        ):
            return 'connectivity_only_size_limited_isomerization'

        r_identity = self._molecule_identity(net_r[0])
        p_identity = self._molecule_identity(net_p[0])
        if r_identity == p_identity:
            return ''

        r_connectivity = self._molecule_connectivity(net_r[0])
        p_connectivity = self._molecule_connectivity(net_p[0])
        if r_connectivity != p_connectivity:
            return 'validated_connectivity_isomerization'
        return 'validated_electronic_state_isomerization'

    def _event_record(
        self,
        sub_r,
        sub_p,
        transition_frame,
        reactant_frame,
        product_frame,
        is_sub_mask=None,
        pre_cutoffs=None,
        allow_non_elementary=False,
    ):
        """Create one mass-balanced event record, or return ``None``."""
        sub_r = list(sub_r)
        sub_p = list(sub_p)
        if not sub_r and not sub_p:
            return None

        sub_r.sort(
            key=lambda x: (
                self._molecule_identity(x),
                len(x.get('indices', [])),
            )
        )
        sub_p.sort(
            key=lambda x: (
                self._molecule_identity(x),
                len(x.get('indices', [])),
            )
        )
        net_r, net_p = self._net_molecules(sub_r, sub_p)

        r_s = " + ".join([
            m.get('smiles', m.get('formula', '')) for m in net_r
        ])
        p_s = " + ".join([
            m.get('smiles', m.get('formula', '')) for m in net_p
        ])
        r_i = " + ".join(self._molecule_identity(m) for m in net_r)
        p_i = " + ".join(self._molecule_identity(m) for m in net_p)
        r_c = " + ".join(self._molecule_connectivity(m) for m in net_r)
        p_c = " + ".join(self._molecule_connectivity(m) for m in net_p)
        r_f = " + ".join(
            self._build_formula_list(
                net_r,
                reactant_frame,
                is_sub_mask,
                pre_cutoffs,
            )
        )
        p_f = " + ".join(
            self._build_formula_list(
                net_p,
                product_frame,
                is_sub_mask,
                pre_cutoffs,
            )
        )

        # Preserve atom identity per species instance.  
        def _mapped_group(mol):
            group = sorted(set(int(i) for i in mol.get("indices", []) or []))
            if self.sub_indices:
                group = [i for i in group if i not in self.sub_indices]
            return group

        reactant_atom_groups = [_mapped_group(m) for m in net_r]
        product_atom_groups = [_mapped_group(m) for m in net_p]

        if not r_s or not p_s or not r_f or not p_f or not r_i or not p_i:
            return None
        chemical_identity = (
            sorted(x.strip() for x in r_i.split(' + '))
            == sorted(x.strip() for x in p_i.split(' + '))
            and sorted(x.strip() for x in r_f.split(' + '))
            == sorted(x.strip() for x in p_f.split(' + '))
        )
        reactant_atom_union = {
            atom for group in reactant_atom_groups for atom in group
        }
        product_atom_union = {
            atom for group in product_atom_groups for atom in group
        }
        atom_mapping_exchange = (
            chemical_identity
            and bool(reactant_atom_union)
            and reactant_atom_union == product_atom_union
            and sorted(tuple(group) for group in reactant_atom_groups)
            != sorted(tuple(group) for group in product_atom_groups)
        )

        # A chemically identical state with unchanged molecule instances is a
        # true identity/flicker and remains excluded.  
        if chemical_identity and not atom_mapping_exchange:
            return None

        all_indices = sorted(
            set(
                idx
                for m in net_r + net_p
                for idx in m.get('indices', [])
            )
        )
        if self.sub_indices:
            all_indices = [
                idx for idx in all_indices if idx not in self.sub_indices
            ]
        if not all_indices:
            return None

        topology_validated = all(
            self._molecule_topology_valid(m) for m in net_r + net_p
        )
        clean_eligible = all(
            self._molecule_clean_eligible(m)
            for m in net_r + net_p
        )
        topology_evidence = sorted({
            str(m.get('topology_evidence', '') or '')
            for m in net_r + net_p
            if str(m.get('topology_evidence', '') or '')
        })
        valence_warnings = sorted({
            str(m.get('valence_warning', '') or '')
            for m in net_r + net_p
            if str(m.get('valence_warning', '') or '')
        })

        tmp = {
            'Frame': int(transition_frame),
            'Reactant_Frame': int(reactant_frame),
            'Product_Frame': int(product_frame),
            'Reactants_Formula': r_f,
            'Products_Formula': p_f,
            'Reactants_SMILES': r_s,
            'Products_SMILES': p_s,
            'Reactants_Identity': r_i,
            'Products_Identity': p_i,
            'Reactants_Connectivity': r_c,
            'Products_Connectivity': p_c,
            'Topology_Validated': bool(topology_validated),
            'Topology_Evidence': '; '.join(topology_evidence),
            'Valence_Warning': '; '.join(valence_warnings),
            'Isomerization_Class': self._isomerization_class(net_r, net_p),
            'Clean_Eligible': bool(clean_eligible),
            'Atom_Indices': all_indices,
            'Reactant_Atom_Groups': reactant_atom_groups,
            'Product_Atom_Groups': product_atom_groups,
            'Atom_Mapping_Exchange': bool(atom_mapping_exchange),
        }
        if atom_mapping_exchange:
            tmp.update({
                'Elementary_Mode': 'atom_mapping_exchange',
                'Elementary_Evidence': (
                    'chemically_identical_endpoints;atom_instance_transfer;'
                    'atom_union_conserved'
                ),
            })

        if not self._is_mass_balanced_formula_text(r_f, p_f):
            return None
        if (
            not allow_non_elementary
            and not self._is_elementary_stoichiometry(tmp)
        ):
            return None

        b_types = Counter()
        f_types = Counter()
        if (
            self.traj is not None
            and reactant_frame is not None
            and product_frame is not None
            and 0 <= reactant_frame < len(self.traj)
            and 0 <= product_frame < len(self.traj)
        ):
            b_react = self._get_local_bonds(
                self.traj[reactant_frame], all_indices
            )
            b_prod = self._get_local_bonds(
                self.traj[product_frame], all_indices
            )
            b_types = Counter([b[1] for b in (b_react - b_prod)])
            f_types = Counter([b[1] for b in (b_prod - b_react)])

        if (
            self.traj is not None
            and 0 <= reactant_frame < len(self.traj)
            and 0 <= product_frame < len(self.traj)
        ):
            rxn_type = self._classify_reaction(
                net_r,
                net_p,
                b_types,
                f_types,
                self.traj[reactant_frame],
                self.traj[product_frame],
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
        else:
            rxn_type = self._get_base_reaction_type(
                net_r,
                net_p,
                b_types,
                f_types,
            )

        if atom_mapping_exchange:
            rxn_type = "Atom-Mapping Exchange"

        tmp.update({
            'Type': rxn_type,
            'Snapshot': None,
            'NEB_Input': None,
        })
        return tmp

    def _changed_bond_edges(
        self,
        reactant_frame,
        product_frame,
        atom_indices,
    ):
        """Return the symmetric difference of endpoint local bond sets."""
        if (
            self.traj is None
            or reactant_frame is None
            or product_frame is None
        ):
            return set()
        if (
            reactant_frame < 0
            or product_frame < 0
            or reactant_frame >= len(self.traj)
            or product_frame >= len(self.traj)
        ):
            return set()

        atom_indices = set(int(i) for i in atom_indices)
        b_react = {
            tuple(sorted(int(x) for x in pair))
            for pair, _typ in self._get_local_bonds(
                self.traj[reactant_frame],
                atom_indices,
            )
        }
        b_prod = {
            tuple(sorted(int(x) for x in pair))
            for pair, _typ in self._get_local_bonds(
                self.traj[product_frame],
                atom_indices,
            )
        }
        return b_react ^ b_prod

    @staticmethod
    def _changed_bond_components_from_edges(changed_edges):
        if not changed_edges:
            return []
        graph = nx.Graph()
        graph.add_edges_from(changed_edges)
        return [
            set(component)
            for component in nx.connected_components(graph)
        ]

    def _stable_reaction_centre_bridges(
        self,
        reactant_frame,
        product_frame,
        atom_indices,
        changed_edges,
    ):
        """Connect two changed-edge centres through one unchanged endpoint bond."""
        changed = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in (changed_edges or [])
        }
        components = self._changed_bond_components_from_edges(changed)
        if len(components) != 2:
            return set()
        if (
            self.traj is None
            or reactant_frame is None
            or product_frame is None
        ):
            return set()
        try:
            reactant_frame = int(reactant_frame)
            product_frame = int(product_frame)
        except (TypeError, ValueError):
            return set()
        if (
            reactant_frame < 0
            or product_frame < 0
            or reactant_frame >= len(self.traj)
            or product_frame >= len(self.traj)
        ):
            return set()

        indices = sorted({int(index) for index in (atom_indices or [])})
        reactant_bonds = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _typ in self._get_local_bonds(
                self.traj[reactant_frame],
                indices,
            )
        }
        product_bonds = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _typ in self._get_local_bonds(
                self.traj[product_frame],
                indices,
            )
        }
        stable_bonds = (reactant_bonds & product_bonds) - changed
        left, right = components
        bridges = {
            edge
            for edge in stable_bonds
            if (
                (edge[0] in left and edge[1] in right)
                or (edge[1] in left and edge[0] in right)
            )
        }
        return bridges if len(bridges) == 1 else set()

    def _changed_bond_components(
        self,
        reactant_frame,
        product_frame,
        atom_indices,
    ):
        changed = self._changed_bond_edges(
            reactant_frame,
            product_frame,
            atom_indices,
        )
        return self._changed_bond_components_from_edges(changed)

    @staticmethod
    def _format_changed_bond_edges(edges):
        """Return a deterministic, compact audit representation of bond edges."""
        normalized = sorted({
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in (edges or [])
            if edge is not None and len(edge) == 2
        })
        return ";".join(f"{u}-{v}" for u, v in normalized)

    @staticmethod
    def _component_membership_maps(molecules):
        """Map mapped atoms to molecular-component owners and member sets."""
        owner = {}
        members_by_atom = {}
        ambiguous = set()
        for component_id, molecule in enumerate(list(molecules or [])):
            members = {
                int(index)
                for index in molecule.get("indices", [])
            }
            if not members:
                continue
            frozen_members = frozenset(members)
            for atom_index in members:
                # Molecule components should be disjoint.  If malformed topology
                # assigns an atom twice, omit that atom from owner-based
                # localization rather than making an arbitrary choice.
                if atom_index in ambiguous:
                    continue
                if atom_index in owner:
                    owner.pop(atom_index, None)
                    members_by_atom.pop(atom_index, None)
                    ambiguous.add(atom_index)
                    continue
                owner[atom_index] = component_id
                members_by_atom[atom_index] = frozen_members
        return owner, members_by_atom

    def _component_transfer_changed_bonds(
        self,
        reactant_frame,
        product_frame,
        atom_indices,
        raw_changed_edges,
    ):
        """Localize endpoint changes to component-transfer reaction centres."""
        raw_changed = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in (raw_changed_edges or [])
        }
        result = {
            "raw_changed": raw_changed,
            "validated_changed": set(raw_changed),
            "spectator_changed": set(),
            "ownership_transition_edges": set(),
            "transfer_atoms": set(),
            "localization_used": False,
        }
        if not raw_changed:
            return result

        reactant_mols = self._local_mols(
            int(reactant_frame),
            atom_indices,
        )
        product_mols = self._local_mols(
            int(product_frame),
            atom_indices,
        )
        if not reactant_mols or not product_mols:
            return result

        r_owner, r_members = self._component_membership_maps(reactant_mols)
        p_owner, p_members = self._component_membership_maps(product_mols)

        ownership_transition_edges = set()
        stable_intracomponent_edges = set()
        edge_membership_known = set()
        for u, v in raw_changed:
            if (
                u not in r_owner
                or v not in r_owner
                or u not in p_owner
                or v not in p_owner
            ):
                continue
            edge_membership_known.add((u, v))
            r_same = r_owner[u] == r_owner[v]
            p_same = p_owner[u] == p_owner[v]
            if r_same != p_same:
                ownership_transition_edges.add((u, v))
            elif r_same and p_same:
                stable_intracomponent_edges.add((u, v))

        if not ownership_transition_edges:
            return result

        transfer_atoms = set()
        centre_atoms = {
            atom
            for edge in ownership_transition_edges
            for atom in edge
        }
        for atom in centre_atoms:
            before = set(r_members.get(atom, ()))
            after = set(p_members.get(atom, ()))
            before.discard(atom)
            after.discard(atom)
            if (before - after) and (after - before):
                transfer_atoms.add(atom)

        # Directional endpoint bond sets let us decide whether a stable
        # intracomponent changed edge is structurally required. 
        reactant_bonds = {
            tuple(sorted(int(x) for x in pair))
            for pair, _typ in self._get_local_bonds(
                self.traj[int(reactant_frame)],
                atom_indices,
            )
        }
        product_bonds = {
            tuple(sorted(int(x) for x in pair))
            for pair, _typ in self._get_local_bonds(
                self.traj[int(product_frame)],
                atom_indices,
            )
        }

        def endpoint_bridge_edges(molecules, frame_index):
            bridges = set()
            atoms = self.traj[int(frame_index)]
            for molecule in molecules:
                members = {
                    int(index)
                    for index in molecule.get("indices", [])
                }
                if len(members) < 2:
                    continue
                graph = nx.Graph()
                graph.add_nodes_from(members)
                graph.add_edges_from(
                    tuple(sorted(int(x) for x in pair))
                    for pair, _typ in self._get_local_bonds(
                        atoms,
                        members,
                    )
                )
                bridges.update(
                    tuple(sorted((int(u), int(v))))
                    for u, v in nx.bridges(graph)
                )
            return bridges

        reactant_bridges = endpoint_bridge_edges(
            reactant_mols,
            reactant_frame,
        )
        product_bridges = endpoint_bridge_edges(
            product_mols,
            product_frame,
        )

        classifiable_edges = (
            ownership_transition_edges
            | stable_intracomponent_edges
        )
        if (
            edge_membership_known != raw_changed
            or classifiable_edges != raw_changed
        ):
            return result

        localized = set(ownership_transition_edges)
        reaction_centre_edges = set()
        for component_nodes in self._changed_bond_components_from_edges(
            raw_changed
        ):
            component_edges = {
                edge
                for edge in raw_changed
                if (
                    edge[0] in component_nodes
                    and edge[1] in component_nodes
                )
            }
            if component_edges.intersection(
                ownership_transition_edges
            ):
                reaction_centre_edges.update(component_edges)

        for edge in stable_intracomponent_edges:
            if edge not in reaction_centre_edges:
                continue
            incident_to_transfer = bool(
                set(edge).intersection(transfer_atoms)
            )
            endpoint_bridge = (
                (edge in reactant_bonds and edge in reactant_bridges)
                or (edge in product_bonds and edge in product_bridges)
            )
            if incident_to_transfer or endpoint_bridge:
                localized.add(edge)

        spectator = raw_changed - localized
        if not localized:
            return result

        result.update({
            "validated_changed": localized,
            "spectator_changed": spectator,
            "ownership_transition_edges": ownership_transition_edges,
            "transfer_atoms": transfer_atoms,
            "localization_used": True,
        })
        return result

    @staticmethod
    def _mapped_atom_set(molecules):
        return {
            int(index)
            for molecule in list(molecules)
            for index in molecule.get("indices", [])
        }

    def _whole_event_composite_candidate(
        self,
        sub_r,
        sub_p,
        reactant_frame,
        product_frame,
        transition_frame,
        changed_edges,
        changed_components,
        is_sub_mask=None,
        pre_cutoffs=None,
        max_changed_bonds=4,
        max_changed_components=3,
        max_species_per_side=4,
    ):
        """Build a conservative whole-event composite observation."""
        sub_r = list(sub_r)
        sub_p = list(sub_p)
        changed_edges = {
            tuple(sorted(int(x) for x in edge))
            for edge in changed_edges
        }
        changed_components = [
            set(int(x) for x in component)
            for component in changed_components
        ]

        if not sub_r or not sub_p:
            return None
        if not (
            2 <= len(changed_components) <= int(max_changed_components)
        ):
            return None
        if not (
            2 <= len(changed_edges) <= int(max_changed_bonds)
        ):
            return None
        if (
            len(sub_r) > int(max_species_per_side)
            or len(sub_p) > int(max_species_per_side)
        ):
            return None

        reactant_atoms = self._mapped_atom_set(sub_r)
        product_atoms = self._mapped_atom_set(sub_p)
        if (
            not reactant_atoms
            or reactant_atoms != product_atoms
        ):
            return None

        if not all(
            self._molecule_clean_eligible(molecule)
            for molecule in sub_r + sub_p
        ):
            return None

        record = self._event_record(
            sub_r,
            sub_p,
            transition_frame,
            reactant_frame,
            product_frame,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
            allow_non_elementary=True,
        )
        if record is None:
            return None

        stable_centre_bridges = set()
        if (
            len(changed_edges) <= 3
            and len(sub_r) <= 3
            and len(sub_p) <= 3
        ):
            stable_centre_bridges = self._stable_reaction_centre_bridges(
                reactant_frame,
                product_frame,
                reactant_atoms,
                changed_edges,
            )
        if stable_centre_bridges:
            record.update({
                "Whole_Event_Fallback": False,
                "Composite_Candidate": False,
                "Concerted_Multi_Bond": True,
                "Elementary_Mode": "concerted_multi_bond",
                "Elementary_Evidence": (
                    "stable_endpoints;atom_mapped;mass_balanced;"
                    "one_unchanged_backbone_bridge;"
                    "no_resolved_stable_intermediate"
                ),
                "Reaction_Centre_Connectivity": (
                    "one_stable_bond_bridge"
                ),
                "Reaction_Centre_Bridge_Bond_Count": 1,
                "Reaction_Centre_Bridge_Bonds": (
                    self._format_changed_bond_edges(
                        stable_centre_bridges
                    )
                ),
                "Changed_Bond_Count": len(changed_edges),
                "Changed_Bond_Components": len(changed_components),
                "Split_Elementary_Records": 0,
            })
            return record

        record.update({
            "Whole_Event_Fallback": True,
            "Composite_Candidate": True,
            "Composite_Evidence": (
                "stable_atom_mapped_mass_balanced_endpoints"
            ),
            "Changed_Bond_Count": len(changed_edges),
            "Changed_Bond_Components": len(changed_components),
            "Changed_Bond_Component_Sizes": ",".join(
                str(len(component))
                for component in sorted(
                    changed_components,
                    key=lambda component: (
                        min(component) if component else -1,
                        len(component),
                    ),
                )
            ),
            "Split_Elementary_Records": 0,
        })
        return record

    def _split_by_changed_bond_centers(
        self,
        sub_r,
        sub_p,
        reactant_frame,
        product_frame,
        is_sub_mask=None,
        pre_cutoffs=None,
        transition_frame=None,
    ):
        """Split independent centres, then preserve a safe whole event if needed."""
        all_indices = sorted(
            self._mapped_atom_set(list(sub_r) + list(sub_p))
        )
        changed_edges = self._changed_bond_edges(
            reactant_frame,
            product_frame,
            all_indices,
        )
        components = self._changed_bond_components_from_edges(
            changed_edges
        )

        event_frame = (
            transition_frame
            if transition_frame is not None
            else product_frame
        )

        if not components:
            record = self._event_record(
                sub_r,
                sub_p,
                event_frame,
                reactant_frame,
                product_frame,
                is_sub_mask,
                pre_cutoffs,
            )
            return [record] if record is not None else []

        records = []
        for component_atoms in components:
            reactant_part = [
                molecule
                for molecule in sub_r
                if not set(molecule.get("indices", [])).isdisjoint(
                    component_atoms
                )
            ]
            product_part = [
                molecule
                for molecule in sub_p
                if not set(molecule.get("indices", [])).isdisjoint(
                    component_atoms
                )
            ]
            if not reactant_part or not product_part:
                continue

            record = self._event_record(
                reactant_part,
                product_part,
                event_frame,
                reactant_frame,
                product_frame,
                is_sub_mask,
                pre_cutoffs,
            )
            if record is not None:
                records.append(record)

        if records:
            return records

        whole_event = self._whole_event_composite_candidate(
            sub_r,
            sub_p,
            reactant_frame,
            product_frame,
            event_frame,
            changed_edges,
            components,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        return [whole_event] if whole_event is not None else []

    def _raw_geometry_partition(self, frame_idx, focus_indices):
        """Connected-component grouping of focus atoms based on RAW geometry bonds."""
        focus = sorted(int(i) for i in focus_indices)
        if not focus or self.traj is None or frame_idx < 0 or frame_idx >= len(self.traj):
            return None
        focus_set = set(focus)
        g = nx.Graph()
        g.add_nodes_from(focus)
        for pair, _typ in self._get_local_bonds(self.traj[frame_idx], focus):
            u, v = pair
            if u in focus_set and v in focus_set:
                g.add_edge(u, v)
        return [set(c) for c in nx.connected_components(g)]

    def _hmm_partition_from_mols(self, mols, focus_indices):
        """Target grouping of focus atoms obtained from HMM-cleaned molecules."""
        focus = set(int(i) for i in focus_indices)
        parts, covered = [], set()
        for m in mols:
            grp = set(int(i) for i in m.get('indices', []) if int(i) in focus)
            if grp:
                parts.append(grp)
                covered |= grp
        for i in (focus - covered):
            parts.append({i})
        return parts

    @staticmethod
    def _partition_agreement(part_a, part_b, focus):
        """Consistency ratio of the two groupings on focus atom pairs (whether assignments are consistent: same group / different group)."""
        focus = sorted(int(i) for i in focus)
        if len(focus) < 2:
            return 1.0
        map_a, map_b = {}, {}
        for gi, grp in enumerate(part_a):
            for i in grp:
                map_a[i] = gi
        for gi, grp in enumerate(part_b):
            for i in grp:
                map_b[i] = gi
        agree = total = 0
        for x in range(len(focus)):
            for y in range(x + 1, len(focus)):
                i, j = focus[x], focus[y]
                total += 1
                if (map_a.get(i) == map_a.get(j)) == (map_b.get(i) == map_b.get(j)):
                    agree += 1
        return agree / total if total else 1.0

    def _best_geometry_frame(self, plateau_start, plateau_end, focus_indices,
                             target_partition, default_frame):
        if self.traj is None:
            return default_frame
        ps = max(0, int(plateau_start))
        pe = min(len(self.traj) - 1, int(plateau_end))
        if pe < ps:
            return default_frame
        mid = (ps + pe) // 2
        best_frame = default_frame
        best_score = None
        for f in range(ps, pe + 1):
            raw = self._raw_geometry_partition(f, focus_indices)
            if raw is None:
                continue
            agreement = self._partition_agreement(raw, target_partition, focus_indices)
            score = (agreement, -abs(f - mid))   
            if best_score is None or score > best_score:
                best_score = score
                best_frame = f
        return best_frame

    @staticmethod
    def _resolve_local_stability_policy(
        stability,
        local_stability,
        single_frame_recovery=True,
    ):
        """Resolve separate thresholds for segmentation and recovery."""
        stability = max(1, int(stability))
        automatic = max(2, stability // 2)

        if local_stability is None:
            requested = automatic
        else:
            requested = max(1, int(local_stability))

        normalized = False
        if bool(single_frame_recovery) and requested == 1:
            segmentation = automatic
            normalized = True
        else:
            segmentation = requested

        recovery_scan = 1 if bool(single_frame_recovery) else segmentation
        return {
            "requested": int(requested),
            "segmentation": int(segmentation),
            "recovery_scan": int(recovery_scan),
            "normalized_single_frame_request": bool(normalized),
        }

    def _local_signature_run_bounds(
        self,
        frame_idx,
        focus_indices,
        start=None,
        end=None,
    ):
        """Return contiguous local-signature run bounds around one frame."""
        if not self.frames:
            return None
        frame_idx = int(frame_idx)
        lower = 0 if start is None else max(0, int(start))
        upper = (
            len(self.frames) - 1
            if end is None
            else min(len(self.frames) - 1, int(end))
        )
        if frame_idx < lower or frame_idx > upper:
            return None

        signature = self._local_signature_at(frame_idx, focus_indices)
        run_start = frame_idx
        run_end = frame_idx

        while (
            run_start > lower
            and self._local_signature_at(
                run_start - 1,
                focus_indices,
            ) == signature
        ):
            run_start -= 1

        while (
            run_end < upper
            and self._local_signature_at(
                run_end + 1,
                focus_indices,
            ) == signature
        ):
            run_end += 1

        return {
            "start": int(run_start),
            "end": int(run_end),
            "length": int(run_end - run_start + 1),
            "signature": signature,
        }

    def _find_valid_reactant_run_before_invalid_corridor(
        self,
        record,
        product_start,
        focus_indices,
        scan_start,
        minimum_dwell,
        maximum_corridor,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Find a stable valid reactant run before a topology-invalid corridor."""
        product_start = int(product_start)
        scan_start = max(0, int(scan_start))
        minimum_dwell = max(1, int(minimum_dwell))
        maximum_corridor = max(1, int(maximum_corridor))
        upper = min(len(self.frames) - 1, product_start - 1)
        if upper < scan_start:
            return None

        target_state = self._record_side_state_multiset(
            record,
            "reactants",
        )
        target_formula_state = tuple(sorted(
            self._strip_surface_suffix(formula)
            for formula in self._parts_text(
                record.get("Reactants_Formula", "")
            )
            if self._strip_surface_suffix(formula)
        ))
        expected_atoms = set(int(i) for i in focus_indices)
        if (not target_formula_state and not target_state) or not expected_atoms:
            return None

        def valid_target(frame_idx):
            molecules = self._local_mols(frame_idx, focus_indices)
            if not molecules:
                return None
            if not all(
                self._molecule_clean_eligible(molecule)
                for molecule in molecules
            ):
                return None
            if self._mapped_atom_set(molecules) != expected_atoms:
                return None
            observed_formulas = tuple(sorted(
                self._strip_surface_suffix(formula)
                for formula in self._build_formula_list(
                    molecules,
                    frame_idx,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                if self._strip_surface_suffix(formula)
            ))
            if target_formula_state:
                if observed_formulas != target_formula_state:
                    return None
            elif self._molecule_state_multiset(
                molecules,
                frame_idx=frame_idx,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            ) != target_state:
                return None
            return molecules

        frame = upper
        while frame >= scan_start:
            molecules = valid_target(frame)
            if molecules is None:
                frame -= 1
                continue

            run_end = frame
            run_start = frame
            while run_start > scan_start:
                if valid_target(run_start - 1) is None:
                    break
                run_start -= 1

            corridor_length = product_start - run_end - 1
            if corridor_length < 0 or corridor_length > maximum_corridor:
                return None

            # Do not jump across a valid intermediate state.  Every frame
            # between the recovered reactant run and the product plateau must
            # contain at least one topology-ineligible mapped molecule.
            corridor_is_invalid = True
            for corridor_frame in range(run_end + 1, product_start):
                corridor_molecules = self._local_mols(
                    corridor_frame,
                    focus_indices,
                )
                if (
                    not corridor_molecules
                    or all(
                        self._molecule_clean_eligible(molecule)
                        for molecule in corridor_molecules
                    )
                ):
                    corridor_is_invalid = False
                    break
            if not corridor_is_invalid:
                return None

            if run_end - run_start + 1 < minimum_dwell:
                return None
            return {
                "start": int(run_start),
                "end": int(run_end),
                "length": int(run_end - run_start + 1),
                "signature": self._local_signature_at(
                    run_end,
                    focus_indices,
                ),
                "corridor_length": int(corridor_length),
            }

        return None

    def _surface_state_run_length(
        self,
        frame_idx,
        focus_indices,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Return adsorption state and its contiguous run length."""
        if self.traj is None:
            return None, 0

        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= len(self.traj):
            return None, 0

        try:
            indices = sorted({int(index) for index in focus_indices})
        except Exception:
            return None, 0
        if not indices:
            return None, 0

        def state_at(index):
            return bool(self._is_adsorbed(
                self.traj[int(index)],
                indices,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            ))

        target_state = state_at(frame_idx)
        matched = 1
        left = frame_idx - 1
        right = frame_idx + 1

        while left >= 0 and state_at(left) == target_state:
            matched += 1
            left -= 1
        while right < len(self.traj) and state_at(right) == target_state:
            matched += 1
            right += 1

        return target_state, int(matched)

    def _surface_state_has_minimum_dwell(
        self,
        frame_idx,
        focus_indices,
        minimum_dwell=2,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Return adsorption state and whether its local run is long enough."""
        state, run_length = self._surface_state_run_length(
            frame_idx,
            focus_indices,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        if state is None:
            return None, False
        minimum_dwell = max(1, int(minimum_dwell))
        return state, bool(run_length >= minimum_dwell)

    def _resolved_surface_minimum_dwell(self):
        """Use ordinary local segmentation stability for surface endpoints."""
        value = getattr(self, "_surface_minimum_dwell", None)
        if value is None:
            value = getattr(self, "_segmentation_local_stability", None)
        if value is None:
            value = 2
        return max(1, int(value))

    def _record_side_state_multiset(self, record, side):
        prefix = (
            "Reactants"
            if str(side).lower().startswith("r")
            else "Products"
        )
        identities = self._parts_text(
            record.get(
                f"{prefix}_Identity",
                record.get(f"{prefix}_SMILES", ""),
            )
        )
        formulas = self._parts_text(
            record.get(f"{prefix}_Formula", "")
        )
        size = max(len(identities), len(formulas))
        return tuple(sorted(
            (
                identities[index] if index < len(identities) else "",
                formulas[index] if index < len(formulas) else "",
            )
            for index in range(size)
        ))

    def _is_state_only_surface_transition(self, record):
        """Return True when only gas/adsorbed reporting state changes."""
        reactant_states = Counter(
            self._record_side_state_multiset(record, "reactants")
        )
        product_states = Counter(
            self._record_side_state_multiset(record, "products")
        )
        if not reactant_states or reactant_states == product_states:
            return False

        def base_states(states):
            result = Counter()
            for (identity, formula), count in states.items():
                result[
                    (
                    identity,
                    self._strip_surface_suffix(formula),
                    )
                ] += count
            return result

        return base_states(reactant_states) == base_states(product_states)

    def _record_side_expected_adsorbed(self, record, side):
        """Return the unambiguous */(g) state encoded for one event side."""
        states = self._record_side_state_multiset(record, side)
        if not states:
            return None

        flags = set()
        for _identity, formula in states:
            formula = str(formula or "").strip()
            if formula.endswith("*"):
                flags.add(True)
            elif formula.endswith("(g)"):
                flags.add(False)
            else:
                return None
        if len(flags) != 1:
            return None
        return flags.pop()

    def _surface_endpoint_dwell_evidence(
        self,
        record,
        reactant_frame,
        product_frame,
        focus_indices,
        minimum_dwell=2,
    ):
        """Build state-only endpoint evidence independent of covalent flicker."""
        minimum_dwell = max(1, int(minimum_dwell))
        r_expected = self._record_side_expected_adsorbed(
            record,
            "reactants",
        )
        p_expected = self._record_side_expected_adsorbed(
            record,
            "products",
        )
        r_observed, r_run_length = self._surface_state_run_length(
            reactant_frame,
            focus_indices,
        )
        p_observed, p_run_length = self._surface_state_run_length(
            product_frame,
            focus_indices,
        )
        r_dwell = bool(r_run_length >= minimum_dwell)
        p_dwell = bool(p_run_length >= minimum_dwell)
        evidence = {
            "Endpoint_Surface_Minimum_Dwell": int(minimum_dwell),
            "Endpoint_Surface_Reactant_Expected_Adsorbed": r_expected,
            "Endpoint_Surface_Reactant_Observed_Adsorbed": r_observed,
            "Endpoint_Surface_Reactant_Run_Length": int(r_run_length),
            "Endpoint_Surface_Reactant_Dwell_Validated": bool(r_dwell),
            "Endpoint_Surface_Product_Expected_Adsorbed": p_expected,
            "Endpoint_Surface_Product_Observed_Adsorbed": p_observed,
            "Endpoint_Surface_Product_Run_Length": int(p_run_length),
            "Endpoint_Surface_Product_Dwell_Validated": bool(p_dwell),
        }
        evidence[
            f"Endpoint_Surface_Reactant_Dwell_At_Least_{minimum_dwell}"
        ] = bool(r_dwell)
        evidence[
            f"Endpoint_Surface_Product_Dwell_At_Least_{minimum_dwell}"
        ] = bool(p_dwell)
        if minimum_dwell != 2:
            evidence[
                "Endpoint_Surface_Reactant_Dwell_At_Least_2"
            ] = bool(r_run_length >= 2)
            evidence[
                "Endpoint_Surface_Product_Dwell_At_Least_2"
            ] = bool(p_run_length >= 2)
        return evidence

    def _ensure_surface_endpoint_evidence(
        self,
        record,
        minimum_dwell=2,
    ):
        """Populate dwell evidence for any pure surface-state transition."""
        if not self._is_state_only_surface_transition(record):
            return record

        minimum_dwell = max(1, int(minimum_dwell))
        fields = (
            "Endpoint_Surface_Minimum_Dwell",
            "Endpoint_Surface_Reactant_Expected_Adsorbed",
            "Endpoint_Surface_Reactant_Observed_Adsorbed",
            "Endpoint_Surface_Reactant_Run_Length",
            "Endpoint_Surface_Reactant_Dwell_Validated",
            "Endpoint_Surface_Product_Expected_Adsorbed",
            "Endpoint_Surface_Product_Observed_Adsorbed",
            "Endpoint_Surface_Product_Run_Length",
            "Endpoint_Surface_Product_Dwell_Validated",
        )
        stored_minimum = None
        try:
            stored_minimum = int(
                record.get("Endpoint_Surface_Minimum_Dwell")
            )
        except (TypeError, ValueError, OverflowError):
            pass
        if all(
            field in record
            and not _is_missing_scalar(record.get(field))
            for field in fields
        ) and stored_minimum == minimum_dwell:
            return record

        legacy_fields = tuple(
            field
            for side in ("Reactant", "Product")
            for field in (
                f"Endpoint_Surface_{side}_Expected_Adsorbed",
                f"Endpoint_Surface_{side}_Observed_Adsorbed",
                f"Endpoint_Surface_{side}_Dwell_At_Least_2",
            )
        )
        if minimum_dwell == 2 and all(
            field in record
            and not _is_missing_scalar(record.get(field))
            for field in legacy_fields
        ):
            record["Endpoint_Surface_Minimum_Dwell"] = 2
            for side in ("Reactant", "Product"):
                legacy_dwell = _truthy_value(
                    record.get(
                        f"Endpoint_Surface_{side}_Dwell_At_Least_2"
                    )
                )
                record.setdefault(
                    f"Endpoint_Surface_{side}_Run_Length",
                    2 if legacy_dwell else 1,
                )
                record[
                    f"Endpoint_Surface_{side}_Dwell_Validated"
                ] = bool(legacy_dwell)
            return record

        def first_frame(*names):
            for name in names:
                value = record.get(name)
                if _is_missing_scalar(value):
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            return None

        reactant_frame = first_frame(
            "Representative_Reactant_Frame",
            "Reactant_Frame",
        )
        product_frame = first_frame(
            "Representative_Product_Frame",
            "Product_Frame",
        )
        if reactant_frame is None or product_frame is None:
            return record

        indices = record.get("Atom_Indices", [])
        if isinstance(indices, str):
            text = indices.strip()
            try:
                if "|" in text:
                    indices = [
                        int(value)
                        for value in text.split("|")
                        if value.strip()
                    ]
                else:
                    import ast
                    parsed = ast.literal_eval(text)
                    indices = (
                        list(parsed)
                        if isinstance(parsed, (list, tuple, set))
                        else []
                    )
            except (SyntaxError, ValueError, TypeError):
                indices = []

        try:
            indices = sorted({int(index) for index in indices})
        except (TypeError, ValueError):
            return record
        if not indices:
            return record

        record.update(
            self._surface_endpoint_dwell_evidence(
                record,
                reactant_frame,
                product_frame,
                indices,
                minimum_dwell=minimum_dwell,
            )
        )
        return record

    def _ensure_recovered_surface_endpoint_evidence(
        self,
        record,
        minimum_dwell=2,
    ):
        """Backward-compatible alias for the generalized evidence finalizer."""
        return self._ensure_surface_endpoint_evidence(
            record,
            minimum_dwell=minimum_dwell,
        )

    def _unstable_surface_state_transition(
        self,
        record,
        minimum_dwell=2,
    ):
        """Identify pure surface-state events with an unstable endpoint."""
        if not self._is_state_only_surface_transition(record):
            return False

        fields = tuple(
            field
            for side in ("Reactant", "Product")
            for field in (
                f"Endpoint_Surface_{side}_Expected_Adsorbed",
                f"Endpoint_Surface_{side}_Observed_Adsorbed",
                (
                    f"Endpoint_Surface_{side}_Dwell_At_Least_"
                    f"{int(minimum_dwell)}"
                ),
            )
        )
        if (
            not all(field in record for field in fields)
            or any(
                _is_missing_scalar(record.get(field))
                for field in fields
            )
        ):
            return False

        for side in ("Reactant", "Product"):
            expected_field = (
                f"Endpoint_Surface_{side}_Expected_Adsorbed"
            )
            observed_field = (
                f"Endpoint_Surface_{side}_Observed_Adsorbed"
            )
            dwell_field = (
                f"Endpoint_Surface_{side}_Dwell_At_Least_"
                f"{int(minimum_dwell)}"
            )
            values = [
                record.get(expected_field),
                record.get(observed_field),
                record.get(dwell_field),
            ]
            expected, observed, dwell = values
            if _truthy_value(expected) != _truthy_value(observed):
                return True
            if not _truthy_value(dwell):
                return True
        return False

    def _unstable_recovered_surface_transition(
        self,
        record,
        minimum_dwell=2,
    ):
        """Backward-compatible alias for the generalized surface-state guard."""
        return self._unstable_surface_state_transition(
            record,
            minimum_dwell=minimum_dwell,
        )

    def _molecule_state_multiset(
        self,
        molecules,
        frame_idx=None,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        molecules = list(molecules)
        identities = [
            self._molecule_identity(molecule)
            for molecule in molecules
        ]
        formulas = self._build_formula_list(
            molecules,
            frame_idx,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        size = max(len(identities), len(formulas))
        return tuple(sorted(
            (
                identities[index] if index < len(identities) else "",
                formulas[index] if index < len(formulas) else "",
            )
            for index in range(size)
        ))

    def _recover_topology_ineligible_event_corridor(
        self,
        record,
        segmentation_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Replace one topology-ineligible event with a valid short endpoint."""
        if not self._single_frame_recovery_enabled:
            return None
        if _truthy_value(record.get("Clean_Eligible", True)):
            return None
        if _truthy_value(record.get("Composite_Candidate", False)):
            return None

        def reject(reason, **audit):
            record["Endpoint_Recovery_Attempted"] = True
            record["Endpoint_Recovered"] = False
            record["Endpoint_Rejection_Reason"] = str(reason)
            for key, value in audit.items():
                record[key] = value
            return None

        record["Endpoint_Recovery_Attempted"] = True
        record["Endpoint_Rejection_Reason"] = ""

        try:
            event_frame = int(record.get("Frame", -1))
            original_reactant_frame = int(
                record.get("Reactant_Frame", -1)
            )
            original_product_frame = int(
                record.get("Product_Frame", -1)
            )
        except Exception:
            return reject("invalid_event_or_representative_frame")

        if (
            event_frame < 1
            or original_reactant_frame < 0
            or original_product_frame <= event_frame
            or original_product_frame >= len(self.frames)
        ):
            return reject("representative_frame_out_of_range")

        raw_indices = record.get("Atom_Indices", [])
        try:
            focus_indices = sorted({
                int(index) for index in raw_indices
            })
        except Exception:
            return reject("invalid_atom_indices")
        if not focus_indices:
            return reject("empty_atom_indices")

        target_products = self._record_side_state_multiset(
            record,
            "products",
        )
        if not target_products:
            return reject("missing_target_product_state")

        segmentation_stability = max(
            2,
            int(segmentation_stability),
        )
        # Use a finite window derived only from the segmentation scale.  
        search_margin = max(
            4 * segmentation_stability,
            8,
        )
        scan_start = max(
            0,
            min(
                original_reactant_frame,
                event_frame - search_margin,
            ),
        )
        scan_end = min(
            len(self.frames) - 1,
            max(
                original_product_frame,
                event_frame + segmentation_stability,
            ),
        )

        product_run = self._local_signature_run_bounds(
            original_product_frame,
            focus_indices,
            start=scan_start,
            end=scan_end,
        )
        if product_run is None:
            return reject("persistent_product_run_not_found")
        if product_run["length"] < segmentation_stability:
            return reject(
                "persistent_product_run_too_short",
                Endpoint_Product_Run_Length=int(product_run["length"]),
                Endpoint_Segmentation_Threshold=int(segmentation_stability),
            )

        timing_tolerance = max(
            1,
            segmentation_stability - 1,
        )
        if (
            product_run["start"] < event_frame
            or product_run["start"] - event_frame
            > timing_tolerance
        ):
            return reject(
                "product_boundary_not_aligned_with_event",
                Endpoint_Boundary_Product_Frame=int(product_run["start"]),
            )

        boundary_product_frame = int(product_run["start"])
        boundary_reactant_frame = boundary_product_frame - 1
        record["Endpoint_Boundary_Reactant_Frame"] = (
            boundary_reactant_frame
        )
        record["Endpoint_Boundary_Product_Frame"] = (
            boundary_product_frame
        )
        record["Representative_Reactant_Frame"] = (
            original_reactant_frame
        )
        record["Representative_Product_Frame"] = (
            original_product_frame
        )

        if boundary_reactant_frame < scan_start:
            return reject("reactant_boundary_out_of_scan_range")

        reactant_search_start = max(
            0,
            int(scan_start)
            - max(int(search_margin), int(segmentation_stability))
            - int(segmentation_stability),
        )
        reactant_run = (
            self._find_valid_reactant_run_before_invalid_corridor(
                record,
                boundary_product_frame,
                focus_indices,
                reactant_search_start,
                minimum_dwell=segmentation_stability,
                maximum_corridor=search_margin,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
        )
        if reactant_run is None:
            immediate_run = self._local_signature_run_bounds(
                boundary_reactant_frame,
                focus_indices,
                start=scan_start,
                end=boundary_product_frame - 1,
            )
            if (
                immediate_run is not None
                and immediate_run["length"] == 1
                and immediate_run["end"] + 1 == boundary_product_frame
            ):
                reactant_run = immediate_run
            else:
                return reject("no_valid_reactant_endpoint_before_corridor")

        boundary_reactant_frame = int(reactant_run["end"])
        record["Endpoint_Boundary_Reactant_Frame"] = (
            boundary_reactant_frame
        )
        if "corridor_length" in reactant_run:
            record["Endpoint_Recovery_Evidence"] = (
                "valid_reactant_plateau_before_invalid_topology_corridor"
            )

        reactant_mols = self._local_mols(
            boundary_reactant_frame,
            focus_indices,
        )
        product_mols = self._local_mols(
            boundary_product_frame,
            focus_indices,
        )
        if not reactant_mols or not product_mols:
            return reject("boundary_molecule_mapping_empty")
        if not all(
            self._molecule_clean_eligible(molecule)
            for molecule in reactant_mols + product_mols
        ):
            return reject("boundary_topology_not_clean_eligible")

        expected_atom_set = set(focus_indices)
        if (
            self._mapped_atom_set(reactant_mols)
            != self._mapped_atom_set(product_mols)
        ):
            return reject("boundary_atom_map_mismatch")
        if self._mapped_atom_set(reactant_mols) != expected_atom_set:
            return reject("boundary_atom_set_differs_from_event")

        accepted = []
        candidate_reasons = []
        for sub_r, sub_p in self._disentangle_events(
            reactant_mols,
            product_mols,
            boundary_reactant_frame,
            boundary_product_frame,
        ):
            candidate = self._event_record(
                sub_r,
                sub_p,
                event_frame,
                boundary_reactant_frame,
                boundary_product_frame,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if candidate is None:
                candidate_reasons.append("event_record_rejected")
                continue
            if not _truthy_value(
                candidate.get("Clean_Eligible", False)
            ):
                candidate_reasons.append("candidate_topology_not_clean")
                continue
            if _truthy_value(
                candidate.get("Composite_Candidate", False)
            ):
                candidate_reasons.append("candidate_is_composite")
                continue
            if set(candidate.get("Atom_Indices", [])) != expected_atom_set:
                candidate_reasons.append("candidate_atom_set_mismatch")
                continue
            if (
                self._record_side_state_multiset(
                    candidate,
                    "products",
                )
                != target_products
            ):
                candidate_reasons.append("candidate_product_state_mismatch")
                continue

            # Validate on the adjacent chemical boundary, but retain stable
            # representative frames for downstream structure generation.
            candidate.update({
                "Endpoint_Boundary_Reactant_Frame": (
                    boundary_reactant_frame
                ),
                "Endpoint_Boundary_Product_Frame": (
                    boundary_product_frame
                ),
                "Representative_Reactant_Frame": (
                    original_reactant_frame
                ),
                "Representative_Product_Frame": (
                    original_product_frame
                ),
                "Reactant_Frame": original_reactant_frame,
                "Product_Frame": original_product_frame,
            })

            filtered, _rejected = (
                self._filter_conservative_endpoint_records(
                    [candidate],
                    max_changed_bonds=3,
                )
            )
            if not filtered:
                if self._last_endpoint_filter_rejections:
                    audit = self._last_endpoint_filter_rejections[-1]
                    candidate_reasons.append(
                        str(
                            audit.get(
                                "Endpoint_Rejection_Reason",
                                "endpoint_filter_rejected",
                            )
                        )
                    )
                    for field in (
                        "Endpoint_Raw_Changed_Bond_Count",
                        "Endpoint_Validated_Changed_Bond_Count",
                        "Endpoint_Raw_Changed_Bonds",
                        "Endpoint_Validated_Changed_Bonds",
                        "Endpoint_Spectator_Changed_Bonds",
                        "Endpoint_Raw_Changed_Bond_Components",
                        "Endpoint_Validated_Changed_Bond_Components",
                        "Endpoint_Transfer_Atom_Count",
                        "Endpoint_Transfer_Atoms",
                        "Endpoint_Component_Localization_Used",
                    ):
                        if field in audit:
                            record[field] = audit[field]
                continue

            replacement = dict(filtered[0])
            replacement.update({
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Attempted": True,
                "Endpoint_Rejection_Reason": "",
                "Endpoint_Recovery_Mode": (
                    "topology_ineligible_event_corridor_refinement"
                ),
                "Endpoint_Recovery_Evidence": (
                    "adjacent_boundary_topology_validated_and_"
                    "component_transfer_localized"
                    if _truthy_value(
                        replacement.get(
                            "Endpoint_Component_Localization_Used",
                            False,
                        )
                    )
                    else "adjacent_boundary_topology_validated"
                ),
                "Endpoint_Reactant_Run_Length": int(
                    reactant_run["length"]
                ),
                "Endpoint_Product_Run_Length": int(
                    product_run["length"]
                ),
                "Endpoint_Segmentation_Threshold": int(
                    segmentation_stability
                ),
                "Endpoint_Requested_Local_Stability": int(
                    self._requested_local_stability
                ),
                "Endpoint_Original_Reactant_Frame": int(
                    original_reactant_frame
                ),
                "Endpoint_Original_Product_Frame": int(
                    original_product_frame
                ),
                "Endpoint_Replaced_Invalid_Event": True,
            })
            accepted.append(replacement)

        if not accepted:
            reason = (
                "no_boundary_candidate_passed:"
                + ",".join(sorted(set(candidate_reasons)))
                if candidate_reasons
                else "no_boundary_candidate_generated"
            )
            return reject(reason)

        accepted = self._dedupe_records(accepted)
        accepted.sort(
            key=lambda candidate: (
                abs(
                    int(
                        candidate.get(
                            "Endpoint_Boundary_Reactant_Frame",
                            event_frame - 1,
                        )
                    )
                    - (event_frame - 1)
                ),
                len(candidate.get("Atom_Indices", []) or []),
                str(candidate.get("Reactants_Identity", "")),
            )
        )
        return accepted[0]

    def _recover_stable_exits_from_ineligible_topology(
        self,
        topology_boundaries,
        segmentation_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Recover a net reaction across an invalid-topology corridor."""
        if not self._single_frame_recovery_enabled or not topology_boundaries:
            return []

        minimum_dwell = max(2, int(segmentation_stability))
        # A settled exit must outlive the short intermediate phase.
        settled_dwell = max(
            minimum_dwell,
            min(int(self.recrossing_window), 4 * minimum_dwell),
        )
        search_window = max(
            int(self.recrossing_window),
            6 * settled_dwell,
        )
        recovered = []
        seen = set()

        for boundary in topology_boundaries:
            boundary_frame = int(boundary.get("frame", -1))
            focus_indices = tuple(sorted({
                int(atom)
                for atom in boundary.get("focus_indices", ())
            }))
            if boundary_frame <= 0 or not focus_indices:
                continue

            # Start from every full invalid molecule, not just the atoms whose
            # ownership changed at this boundary.
            invalid_frame = boundary_frame - 1
            invalid_molecules = self._local_mols(
                invalid_frame,
                focus_indices,
            )
            invalid_molecules = [
                molecule for molecule in invalid_molecules
                if not self._molecule_clean_eligible(molecule)
            ]
            if not invalid_molecules:
                continue

            context_indices = set(focus_indices)
            for molecule in invalid_molecules:
                context_indices.update(
                    int(atom)
                    for atom in self._molecule_atom_group(molecule)
                )
            context_indices = tuple(sorted(context_indices))
            if not context_indices:
                continue

            # Recover the last valid state before the invalid corridor.
            reactant_search_start = max(
                settled_dwell - 1,
                boundary_frame - search_window,
            )
            settled_reactant = None
            for reactant_frame in range(
                invalid_frame, reactant_search_start - 1, -1
            ):
                reactant_molecules = self._local_mols(
                    reactant_frame, context_indices
                )
                if (
                    not reactant_molecules
                    or not all(
                        self._molecule_clean_eligible(molecule)
                        for molecule in reactant_molecules
                    )
                ):
                    continue
                signature = self._local_topology_signature_at(
                    reactant_frame, context_indices
                )
                if any(
                    self._local_topology_signature_at(
                        frame, context_indices
                    ) != signature
                    for frame in range(
                        reactant_frame - settled_dwell + 1,
                        reactant_frame,
                    )
                ):
                    continue
                settled_reactant = {
                    "reactant_frame": int(reactant_frame),
                    "reactant_molecules": reactant_molecules,
                }
                break

            # Search beyond short valid flashes. 
            search_stop = min(
                len(self.frames) - settled_dwell,
                boundary_frame + search_window,
            )
            settled = None
            for product_frame in range(boundary_frame, search_stop + 1):
                product_molecules = self._local_mols(
                    product_frame,
                    context_indices,
                )
                if (
                    not product_molecules
                    or not all(
                        self._molecule_clean_eligible(molecule)
                        for molecule in product_molecules
                    )
                ):
                    continue
                signature = self._local_topology_signature_at(
                    product_frame,
                    context_indices,
                )
                if any(
                    self._local_topology_signature_at(
                        frame,
                        context_indices,
                    ) != signature
                    for frame in range(
                        product_frame + 1,
                        product_frame + settled_dwell,
                    )
                ):
                    continue
                settled = {
                    "product_frame": int(product_frame),
                    "product_molecules": product_molecules,
                }
                break

            if settled is None or settled_reactant is None:
                continue

            reactant_frame = int(settled_reactant["reactant_frame"])
            reactant_molecules = list(settled_reactant["reactant_molecules"])
            product_frame = int(settled["product_frame"])
            product_molecules = list(settled["product_molecules"])
            settled_products = {
                int(atom)
                for molecule in product_molecules
                for atom in self._molecule_atom_group(molecule)
                if int(atom) not in self.sub_indices
            }
            candidate = None
            for sub_r, sub_p in self._disentangle_events(
                reactant_molecules,
                product_molecules,
                reactant_frame,
                product_frame,
            ):
                trial = self._event_record(
                    sub_r,
                    sub_p,
                    product_frame,
                    reactant_frame,
                    product_frame,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                    allow_non_elementary=True,
                )
                if trial is None:
                    continue
                # A bridge must carry the complete atom-mapped endpoint context. 
                trial_products = {
                    int(atom)
                    for group in trial.get("Product_Atom_Groups", []) or []
                    for atom in group
                }
                if settled_products and trial_products == settled_products:
                    candidate = trial
                    break

            # If splitting into elementary-looking subevents loses endpoint
            # context, retain the mass-balanced whole net transformation.
            if candidate is None:
                trial = self._event_record(
                    reactant_molecules,
                    product_molecules,
                    product_frame,
                    reactant_frame,
                    product_frame,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                    allow_non_elementary=True,
                )
                if trial is not None:
                    trial_products = {
                        int(atom)
                        for group in trial.get("Product_Atom_Groups", []) or []
                        for atom in group
                    }
                    if settled_products and trial_products == settled_products:
                        candidate = trial

            if candidate is None:
                continue

            if _truthy_value(candidate.get("Clean_Eligible", True)):
                candidate["Clean_Eligible"] = False

            exact_key = self._lineage_event_key(candidate, include_frame=True)
            if exact_key in seen:
                continue
            seen.add(exact_key)

            candidate.update({
                "Is_Elementary": False,
                "Elementary_Mode": "observed_corridor_net",
                "Observed_Corridor_Net_Elementary": False,
                "Elementary_Evidence": (
                    "valid_settled_reactant_to_valid_settled_product;"
                    "intervening_invalid_topology;atom_mapped;"
                    "both_endpoint_dwells_validated"
                ),
                "Observed_Stable_Exit_Source": False,
                "Observed_Stable_Exit_Product_Validated": True,
                "Observed_Corridor_Net_Validated": True,
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Attempted": True,
                "Endpoint_Recovery_Mode": (
                    "topology_ineligible_corridor_net"
                ),
                "Endpoint_Recovery_Evidence": (
                    "settled_valid_reactant;intervening_invalid_topology;"
                    "settled_valid_product;exact_atom_mapped_context"
                ),
                "Endpoint_Reactant_Run_Length": int(minimum_dwell),
                "Endpoint_Product_Run_Length": int(settled_dwell),
                "Endpoint_Segmentation_Threshold": int(minimum_dwell),
                "Endpoint_Requested_Local_Stability": int(
                    self._requested_local_stability
                ),
                "Endpoint_Original_Reactant_Frame": int(invalid_frame),
                "Endpoint_Original_Product_Frame": product_frame,
                "Observed_State_Source_Min_Dwell": int(settled_dwell),
                "Observed_Stable_Exit_Settled_After_Frames": int(
                    product_frame - boundary_frame
                ),
            })
            recovered.append(candidate)

        recovered = self._dedupe_records(recovered)
        self._single_frame_recovery_stats[
            "topology_ineligible_corridor_nets"
        ] += len(recovered)
        return recovered

    def _merge_topology_ineligible_corridor_pairs(
        self,
        records,
        segmentation_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Collapse two invalid records into one validated net event."""
        ordered = sorted(
            (dict(record) for record in records),
            key=lambda record: (
                int(record.get("Frame", 0) or 0),
                tuple(sorted(record.get("Atom_Indices", []) or [])),
            ),
        )
        if len(ordered) < 2:
            return ordered, 0

        minimum_run = max(2, int(segmentation_stability))
        maximum_gap = max(2, minimum_run)
        merged = []
        merged_count = 0
        index = 0

        while index < len(ordered):
            current = ordered[index]
            if index + 1 >= len(ordered):
                merged.append(current)
                break
            following = ordered[index + 1]

            try:
                current_frame = int(current.get("Frame", -1))
                following_frame = int(following.get("Frame", -1))
                reactant_frame = int(current.get("Reactant_Frame", -1))
                product_frame = int(following.get("Product_Frame", -1))
                current_indices = {
                    int(atom)
                    for atom in current.get("Atom_Indices", [])
                }
                following_indices = {
                    int(atom)
                    for atom in following.get("Atom_Indices", [])
                }
            except Exception:
                merged.append(current)
                index += 1
                continue

            eligible_pair = (
                not _truthy_value(current.get("Clean_Eligible", True))
                and not _truthy_value(following.get("Clean_Eligible", True))
                and not _truthy_value(
                    current.get("Composite_Candidate", False)
                )
                and not _truthy_value(
                    following.get("Composite_Candidate", False)
                )
                and bool(current_indices)
                and current_indices == following_indices
                and 0 < following_frame - current_frame <= maximum_gap
                and reactant_frame < product_frame
                and self._record_side_state_multiset(
                    current,
                    "products",
                )
                == self._record_side_state_multiset(
                    following,
                    "reactants",
                )
            )
            if not eligible_pair:
                merged.append(current)
                index += 1
                continue

            scan_margin = max(10, 4 * minimum_run)
            reactant_run = self._local_signature_run_bounds(
                reactant_frame,
                current_indices,
                start=max(0, reactant_frame - scan_margin),
                end=min(len(self.frames) - 1, current_frame),
            )
            product_run = self._local_signature_run_bounds(
                product_frame,
                current_indices,
                start=max(0, following_frame),
                end=min(len(self.frames) - 1, product_frame + scan_margin),
            )
            if (
                reactant_run is None
                or product_run is None
                or reactant_run["length"] < minimum_run
                or product_run["length"] < minimum_run
            ):
                merged.append(current)
                index += 1
                continue

            reactant_mols = self._local_mols(
                reactant_frame,
                current_indices,
            )
            product_mols = self._local_mols(
                product_frame,
                current_indices,
            )
            if (
                not reactant_mols
                or not product_mols
                or not all(
                    self._molecule_clean_eligible(molecule)
                    for molecule in reactant_mols + product_mols
                )
                or self._mapped_atom_set(reactant_mols) != current_indices
                or self._mapped_atom_set(product_mols) != current_indices
            ):
                merged.append(current)
                index += 1
                continue

            candidates = []
            for sub_r, sub_p in self._disentangle_events(
                reactant_mols,
                product_mols,
                reactant_frame,
                product_frame,
            ):
                candidate = self._event_record(
                    sub_r,
                    sub_p,
                    following_frame,
                    reactant_frame,
                    product_frame,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                if (
                    candidate is None
                    or not _truthy_value(
                        candidate.get("Clean_Eligible", False)
                    )
                    or _truthy_value(
                        candidate.get("Composite_Candidate", False)
                    )
                    or set(candidate.get("Atom_Indices", []))
                    != current_indices
                ):
                    continue
                candidate.update({
                    "Endpoint_Boundary_Reactant_Frame": reactant_frame,
                    "Endpoint_Boundary_Product_Frame": product_frame,
                    "Representative_Reactant_Frame": reactant_frame,
                    "Representative_Product_Frame": product_frame,
                    "Reactant_Frame": reactant_frame,
                    "Product_Frame": product_frame,
                })
                filtered, _rejected = (
                    self._filter_conservative_endpoint_records(
                        [candidate],
                        max_changed_bonds=3,
                        max_raw_changed_bonds=8,
                    )
                )
                candidates.extend(filtered)

            candidates = self._dedupe_records(candidates)
            if len(candidates) != 1:
                merged.append(current)
                index += 1
                continue

            replacement = dict(candidates[0])
            replacement.update({
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Attempted": True,
                "Endpoint_Rejection_Reason": "",
                "Endpoint_Recovery_Mode": (
                    "connected_topology_ineligible_corridor_merge"
                ),
                "Endpoint_Recovery_Evidence": (
                    "stable_clean_endpoints_across_short_invalid_corridor"
                ),
                "Endpoint_Reactant_Run_Length": int(
                    reactant_run["length"]
                ),
                "Endpoint_Product_Run_Length": int(
                    product_run["length"]
                ),
                "Endpoint_Replaced_Invalid_Event": True,
                "Recovery_Source_Frames": (
                    f"{current_frame},{following_frame}"
                ),
            })
            merged.append(replacement)
            merged_count += 1
            index += 2

        return merged, merged_count


    def _refine_topology_ineligible_event_corridors(
        self,
        records,
        segmentation_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Replace eligible invalid-endpoint records without adding duplicates."""
        if not records or not self._single_frame_recovery_enabled:
            return list(records)

        records, merged_count = (
            self._merge_topology_ineligible_corridor_pairs(
                records,
                segmentation_stability,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
        )
        refined = []
        recovered_count = 0
        for record in records:
            replacement = (
                self._recover_topology_ineligible_event_corridor(
                    record,
                    segmentation_stability,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
            )
            if replacement is None:
                refined.append(record)
                continue
            refined.append(replacement)
            recovered_count += 1

        if recovered_count:
            self._single_frame_recovery_stats[
                "corridor_endpoint_refinements"
            ] += recovered_count
        if merged_count:
            self._single_frame_recovery_stats[
                "corridor_pair_merges"
            ] += merged_count

        return self._dedupe_records(refined)

    def _strict_single_frame_endpoint_records(
        self,
        sub_r,
        sub_p,
        scan_start,
        scan_end,
        focus_indices,
        stable_frames,
        segmentation_stability,
        transition_frame,
        primary_records=None,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Refine a transition through a valid one-frame reactant endpoint."""
        if (
            not self._single_frame_recovery_enabled
            or int(segmentation_stability) <= 1
        ):
            return []

        expected_r, expected_p = self._net_molecules(
            list(sub_r),
            list(sub_p),
        )
        if not expected_p:
            expected_p = list(sub_p)

        expected_products = self._molecule_state_multiset(
            expected_p,
            frame_idx=int(transition_frame),
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        if not expected_products:
            return []

        for record in list(primary_records or []):
            if (
                _truthy_value(record.get("Clean_Eligible", True))
                and self._record_side_state_multiset(
                    record,
                    "products",
                ) == expected_products
            ):
                return []

        candidates = self._scan_local_elementary_events(
            scan_start,
            scan_end,
            focus_indices,
            stable_frames,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
            local_stability=self._recovery_scan_local_stability,
        )

        accepted = []
        for candidate in candidates:
            if not _truthy_value(
                candidate.get("Clean_Eligible", True)
            ):
                continue
            if _truthy_value(
                candidate.get("Composite_Candidate", False)
            ):
                continue
            if (
                self._record_side_state_multiset(
                    candidate,
                    "products",
                )
                != expected_products
            ):
                continue

            event_indices = candidate.get("Atom_Indices", [])
            reactant_frame = int(
                candidate.get(
                    "Reactant_Frame",
                    candidate.get("Frame", -1),
                )
                or -1
            )
            product_frame = int(
                candidate.get(
                    "Product_Frame",
                    candidate.get("Frame", -1),
                )
                or -1
            )
            if (
                reactant_frame < int(scan_start)
                or product_frame > int(scan_end)
                or reactant_frame >= product_frame
            ):
                continue

            reactant_run = self._local_signature_run_bounds(
                reactant_frame,
                event_indices,
                start=scan_start,
                end=scan_end,
            )
            product_run = self._local_signature_run_bounds(
                product_frame,
                event_indices,
                start=scan_start,
                end=scan_end,
            )
            if reactant_run is None or product_run is None:
                continue
            if reactant_run["length"] != 1:
                continue
            if (
                product_run["length"]
                < int(segmentation_stability)
            ):
                continue
            if (
                reactant_run["end"] + 1
                != product_run["start"]
            ):
                continue

            # Validate the chemistry on the true adjacent topology boundary.
            boundary_reactant_frame = int(reactant_run["end"])
            boundary_product_frame = int(product_run["start"])
            representative_reactant_frame = int(reactant_frame)
            representative_product_frame = int(product_frame)

            candidate = dict(candidate)
            candidate.update({
                "Endpoint_Boundary_Reactant_Frame": (
                    boundary_reactant_frame
                ),
                "Endpoint_Boundary_Product_Frame": (
                    boundary_product_frame
                ),
                "Representative_Reactant_Frame": (
                    representative_reactant_frame
                ),
                "Representative_Product_Frame": (
                    representative_product_frame
                ),
                # Preserve the stable geometry representatives used by downstream Snapshot/NEB generation.
                "Reactant_Frame": representative_reactant_frame,
                "Product_Frame": representative_product_frame,
                # Set recovery provenance before endpoint filtering.
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Mode": (
                    "single_frame_reactant_to_stable_product"
                ),
                "Endpoint_Reactant_Run_Length": int(
                    reactant_run["length"]
                ),
                "Endpoint_Product_Run_Length": int(
                    product_run["length"]
                ),
            })

            filtered, _rejected = (
                self._filter_conservative_endpoint_records(
                    [candidate],
                    max_changed_bonds=3,
                )
            )
            if not filtered:
                continue

            candidate = dict(filtered[0])
            candidate.update({
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Mode": (
                    "single_frame_reactant_to_stable_product"
                ),
                "Endpoint_Recovery_Evidence": (
                    "valid_atom_mapped_one_frame_reactant_"
                    "stable_global_product"
                ),
                "Endpoint_Reactant_Run_Length": int(
                    reactant_run["length"]
                ),
                "Endpoint_Product_Run_Length": int(
                    product_run["length"]
                ),
                "Endpoint_Segmentation_Threshold": int(
                    segmentation_stability
                ),
                "Endpoint_Requested_Local_Stability": int(
                    self._requested_local_stability
                ),
            })
            accepted.append(candidate)

        if not accepted:
            return []

        # Keep one best refinement per formal event. Prefer the transition
        # nearest the stable global boundary, then the longest product run.
        best = {}
        for candidate in self._dedupe_records(accepted):
            event_key = self._lineage_event_key(
                candidate,
                include_frame=False,
            )
            score = (
                abs(
                    int(candidate.get("Frame", 0) or 0)
                    - int(transition_frame)
                ),
                -int(
                    candidate.get(
                        "Endpoint_Product_Run_Length",
                        0,
                    )
                    or 0
                ),
                len(candidate.get("Atom_Indices", []) or []),
            )
            old = best.get(event_key)
            if old is None or score < old[0]:
                best[event_key] = (score, candidate)

        recovered = [entry[1] for entry in best.values()]
        self._single_frame_recovery_stats[
            "direct_endpoint_refinements"
        ] += len(recovered)
        return recovered

    def _scan_local_elementary_events(self, start, end, focus_indices, stable_frames,
                                      is_sub_mask=None, pre_cutoffs=None,
                                      local_stability=None):
        """Recover locally stable elementary steps around a global transition."""
        if local_stability is None:
            local_stability = stable_frames
        local_stability = max(1, int(local_stability))

        timeline = self._build_local_timeline(start, end, focus_indices, local_stability)
        if len(timeline) < 2:
            return []

        records = []
        half = (max(1, int(stable_frames)) - 1) // 2

        for k in range(len(timeline) - 1):
            prev = timeline[k]
            curr = timeline[k + 1]
            trans = int(curr['start'])

            f_react = max(int(prev['start']), int(prev['end']) - half)
            f_prod = min(int(curr['end']), int(curr['start']) + half)
            if f_react >= f_prod:
                continue

            react_mols = self._local_mols(f_react, focus_indices)
            prod_mols = self._local_mols(f_prod, focus_indices)

            for sub_r, sub_p in self._disentangle_events(
                react_mols, prod_mols, f_react, f_prod):
                event_indices = set(
                    idx for m in list(sub_r) + list(sub_p)
                    for idx in m.get('indices', [])
                )
                if event_indices.isdisjoint(focus_indices):
                    continue

                react_target = self._hmm_partition_from_mols(sub_r, event_indices)
                prod_target = self._hmm_partition_from_mols(sub_p, event_indices)
                rf = self._best_geometry_frame(
                    int(prev['start']), int(prev['end']), event_indices,
                    react_target, f_react)
                pf = self._best_geometry_frame(
                    int(curr['start']), int(curr['end']), event_indices,
                    prod_target, f_prod)

                records.extend(self._split_by_changed_bond_centers(
                    sub_r, sub_p, rf, pf,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                    transition_frame=trans
                ))
        return records

    def _persistent_local_topology_boundary(
        self,
        boundary_frame,
        focus_indices,
        minimum_dwell,
    ):
        """Validate adjacent atom-local topology dwell around one boundary."""
        boundary_frame = int(boundary_frame)
        minimum_dwell = max(1, int(minimum_dwell))
        n_frames = len(self.frames)
        reactant_start = boundary_frame - minimum_dwell
        reactant_end = boundary_frame - 1
        product_start = boundary_frame
        product_end = boundary_frame + minimum_dwell - 1

        if (
            reactant_start < 0
            or product_end >= n_frames
            or reactant_end >= product_start
        ):
            return None

        reactant_signature = self._local_topology_signature_at(
            reactant_end,
            focus_indices,
        )
        product_signature = self._local_topology_signature_at(
            product_start,
            focus_indices,
        )
        if reactant_signature == product_signature:
            return None

        if any(
            self._local_topology_signature_at(frame, focus_indices)
            != reactant_signature
            for frame in range(reactant_start, reactant_end + 1)
        ):
            return None
        if any(
            self._local_topology_signature_at(frame, focus_indices)
            != product_signature
            for frame in range(product_start, product_end + 1)
        ):
            return None

        return {
            "reactant_start": int(reactant_start),
            "reactant_end": int(reactant_end),
            "product_start": int(product_start),
            "product_end": int(product_end),
            "reactant_signature": reactant_signature,
            "product_signature": product_signature,
            "minimum_dwell": int(minimum_dwell),
        }

    def _expand_scoped_topology_boundaries(
        self,
        topology_boundaries,
        scoped_boundaries,
        recrossing_window,
    ):
        """Complete atom-local boundary clusters before event recovery."""
        recrossing_window = max(1, int(recrossing_window))

        normalized = {}
        for candidate in topology_boundaries:
            frame = int(candidate.get("frame", -1))
            focus = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            if frame <= 0 or not focus:
                continue
            normalized[(frame, focus)] = {
                "frame": frame,
                "focus_indices": focus,
            }

        selected_keys = set()
        for candidate in (
            scoped_boundaries.values()
            if isinstance(scoped_boundaries, dict)
            else scoped_boundaries
        ):
            frame = int(candidate.get("frame", -1))
            focus = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            if frame > 0 and focus:
                selected_keys.add((frame, focus))

        if not selected_keys:
            return {}

        by_focus = defaultdict(list)
        for key, candidate in normalized.items():
            by_focus[key[1]].append(candidate)

        expanded = {}
        for focus, candidates in by_focus.items():
            candidates.sort(key=lambda item: int(item["frame"]))
            cluster = []

            def retain_cluster(items):
                if not items:
                    return
                cluster_keys = {
                    (int(item["frame"]), focus)
                    for item in items
                }
                if cluster_keys.isdisjoint(selected_keys):
                    return
                for item in items:
                    key = (int(item["frame"]), focus)
                    expanded[key] = {
                        "frame": key[0],
                        "focus_indices": focus,
                        "closure_companion": (
                            key not in selected_keys
                        ),
                    }

            for candidate in candidates:
                if (
                    cluster
                    and int(candidate["frame"])
                    - int(cluster[-1]["frame"])
                    > recrossing_window
                ):
                    retain_cluster(cluster)
                    cluster = []
                cluster.append(candidate)
            retain_cluster(cluster)

        # Preserve a selected candidate even if an upstream caller supplied a
        # boundary absent from the full normalized collection.
        for key in selected_keys:
            expanded.setdefault(key, {
                "frame": key[0],
                "focus_indices": key[1],
                "closure_companion": False,
            })
        return expanded

    def _detected_event_topology_boundary_seeds(
        self,
        topology_boundaries,
        detected_records,
    ):
        """Return topology boundaries already represented by detected events."""
        detected_keys = set()
        for record in detected_records:
            frame = int(record.get("Frame", -1))
            focus = tuple(sorted({
                int(atom)
                for atom in record.get("Atom_Indices", ())
            }))
            if frame > 0 and focus:
                detected_keys.add((frame, focus))

        if not detected_keys:
            return {}

        seeds = {}
        for candidate in topology_boundaries:
            frame = int(candidate.get("frame", -1))
            focus = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            key = (frame, focus)
            if frame > 0 and focus and key in detected_keys:
                seeds[key] = {
                    "frame": frame,
                    "focus_indices": focus,
                }
        return seeds

    def _persistent_topology_boundary_seeds(
        self,
        topology_boundaries,
        minimum_dwell,
    ):
        """Return every atom-local topology boundary with stable dwell."""
        minimum_dwell = max(1, int(minimum_dwell))
        seeds = {}

        for candidate in topology_boundaries or ():
            frame = int(candidate.get("frame", -1))
            focus = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            if frame <= 0 or not focus:
                continue

            if self._persistent_local_topology_boundary(
                frame,
                focus,
                minimum_dwell,
            ) is None:
                continue

            seeds[(frame, focus)] = {
                "frame": frame,
                "focus_indices": focus,
            }

        self._single_frame_recovery_stats[
            "persistent_topology_boundaries_considered"
        ] = len(topology_boundaries or ())
        self._single_frame_recovery_stats[
            "persistent_topology_boundary_seeds"
        ] = len(seeds)
        return seeds

    def _invalid_raw_exit_boundary_seeds(self, records):
        """Create conservative seeds for invalid Raw events."""
        if not records or getattr(self, "frames", None) is None:
            return []
        try:
            frame_count = len(self.frames)
        except Exception:
            return []

        seeds = {}
        for record in records:
            if _truthy_value(record.get("Topology_Validated", True)):
                continue
            reason = str(
                record.get("Exclusion_Reason", "")
                or record.get("Endpoint_Rejection_Reason", "")
                or ""
            ).strip().lower()
            warning = record.get("Valence_Warning", "")
            if "valence" not in reason and _is_missing_scalar(warning):
                continue

            focus = set()
            for side in ("Reactants", "Products"):
                for group in _normalize_atom_groups(
                    record.get(f"{side}_Atom_Groups", "")
                ):
                    focus.update(int(atom) for atom in group)
            if not focus:
                try:
                    focus.update(
                        int(atom) for atom in record.get("Atom_Indices", [])
                    )
                except Exception:
                    pass
            if not focus:
                continue

            try:
                event_frame = int(record.get("Frame", -1))
                product_frame = int(
                    record.get("Product_Frame", event_frame)
                )
            except Exception:
                continue

            for boundary_frame in {
                event_frame + 1,
                product_frame + 1,
            }:
                if boundary_frame <= 0 or boundary_frame >= frame_count:
                    continue
                key = (int(boundary_frame), tuple(sorted(focus)))
                seeds[key] = {
                    "frame": int(boundary_frame),
                    "focus_indices": tuple(sorted(focus)),
                    "synthetic_invalid_raw_seed": True,
                    "synthetic_seed_raw_id": str(
                        record.get(
                            "Raw_Reaction_ID",
                            record.get("Reaction_ID", ""),
                        )
                        or ""
                    ),
                }
        return list(seeds.values())

    def _recover_persistent_atom_local_events(
        self,
        topology_boundaries,
        stable_frames,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Recover stable local reactions hidden by whole-system activity."""
        if not topology_boundaries or self.traj is None:
            return []

        stable_frames = max(1, int(stable_frames))
        local_stability = max(1, int(local_stability))
        half = (local_stability - 1) // 2
        recovered = []
        rejected = 0

        for candidate in topology_boundaries:
            boundary_frame = int(candidate.get("frame", -1))
            focus_indices = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            if boundary_frame <= 0 or not focus_indices:
                continue

            dwell = self._persistent_local_topology_boundary(
                boundary_frame,
                focus_indices,
                local_stability,
            )
            if dwell is None:
                continue

            nominal_reactant = max(
                int(dwell["reactant_start"]),
                int(dwell["reactant_end"]) - half,
            )
            nominal_product = min(
                int(dwell["product_end"]),
                int(dwell["product_start"]) + half,
            )

            reactant_molecules = self._local_mols(
                nominal_reactant,
                focus_indices,
            )
            product_molecules = self._local_mols(
                nominal_product,
                focus_indices,
            )

            boundary_records = []
            for sub_r, sub_p in self._disentangle_events(
                reactant_molecules,
                product_molecules,
                nominal_reactant,
                nominal_product,
            ):
                event_indices = {
                    int(atom)
                    for molecule in list(sub_r) + list(sub_p)
                    for atom in molecule.get("indices", [])
                    if int(atom) not in self.sub_indices
                }
                if (
                    not event_indices
                    or event_indices.isdisjoint(focus_indices)
                ):
                    continue

                reactant_target = self._hmm_partition_from_mols(
                    sub_r,
                    event_indices,
                )
                product_target = self._hmm_partition_from_mols(
                    sub_p,
                    event_indices,
                )
                representative_reactant = self._best_geometry_frame(
                    int(dwell["reactant_start"]),
                    int(dwell["reactant_end"]),
                    event_indices,
                    reactant_target,
                    nominal_reactant,
                )
                representative_product = self._best_geometry_frame(
                    int(dwell["product_start"]),
                    int(dwell["product_end"]),
                    event_indices,
                    product_target,
                    nominal_product,
                )
                if representative_reactant >= representative_product:
                    continue

                boundary_records.extend(
                    self._split_by_changed_bond_centers(
                        sub_r,
                        sub_p,
                        representative_reactant,
                        representative_product,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                        transition_frame=boundary_frame,
                    )
                )

            if not boundary_records:
                continue

            boundary_records, boundary_rejected = (
                self._filter_conservative_endpoint_records(
                    boundary_records,
                    max_changed_bonds=3,
                    max_raw_changed_bonds=8,
                )
            )
            rejected += int(boundary_rejected)

            for record in boundary_records:
                record.update({
                    "Endpoint_Recovered": True,
                    "Endpoint_Recovery_Mode": (
                        "persistent_atom_local_topology"
                    ),
                    "Endpoint_Recovery_Evidence": (
                        "atom_mapped_local_topology_changed;"
                        "both_endpoint_dwell_satisfied;"
                        "mass_balanced;"
                        "geometry_validated;"
                        "conservative_reaction_centre_validation"
                    ),
                    "Endpoint_Reactant_Run_Length": int(
                        dwell["minimum_dwell"]
                    ),
                    "Endpoint_Product_Run_Length": int(
                        dwell["minimum_dwell"]
                    ),
                    "Endpoint_Segmentation_Threshold": int(
                        local_stability
                    ),
                    "Endpoint_Requested_Local_Stability": int(
                        self._requested_local_stability
                    ),
                    "Endpoint_Original_Reactant_Frame": int(
                        boundary_frame - 1
                    ),
                    "Endpoint_Original_Product_Frame": int(
                        boundary_frame
                    ),
                    "Local_Topology_Recovery": True,
                    "Local_Topology_Closure_Companion": bool(
                        candidate.get("closure_companion", False)
                    ),
                })
                recovered.append(record)

        recovered = self._dedupe_records(recovered)
        self._single_frame_recovery_stats[
            "persistent_atom_local_recoveries"
        ] += len(recovered)
        self._single_frame_recovery_stats[
            "persistent_atom_local_rejections"
        ] += int(rejected)
        return recovered

    def _persistent_local_surface_boundary(
        self,
        boundary_frame,
        focus_indices,
        minimum_dwell,
        candidate=None,
    ):
        """Validate one exact mapped gas/adsorbed-state boundary."""
        boundary_frame = int(boundary_frame)
        minimum_dwell = max(1, int(minimum_dwell))
        focus_indices = tuple(sorted({
            int(atom) for atom in focus_indices
        }))
        if (
            boundary_frame <= 0
            or not focus_indices
            or boundary_frame >= len(self.frames)
        ):
            return None

        reactant_molecule = self._exact_mapped_molecule(
            boundary_frame - 1,
            focus_indices,
        )
        product_molecule = self._exact_mapped_molecule(
            boundary_frame,
            focus_indices,
        )
        if reactant_molecule is None or product_molecule is None:
            return None
        if (
            self._molecule_identity(reactant_molecule)
            != self._molecule_identity(product_molecule)
            or self._molecule_connectivity(reactant_molecule)
            != self._molecule_connectivity(product_molecule)
        ):
            return None

        reactant_end = boundary_frame - 1
        product_start = boundary_frame
        candidate = dict(candidate or {})
        try:
            reactant_run_length = int(
                candidate["reactant_run_length"]
            )
            product_run_length = int(
                candidate["product_run_length"]
            )
            reactant_state = bool(candidate["reactant_adsorbed"])
            product_state = bool(candidate["product_adsorbed"])
            has_precomputed_dwell = True
        except (KeyError, TypeError, ValueError, OverflowError):
            has_precomputed_dwell = False

        if not has_precomputed_dwell:
            reactant_signature = self._local_signature_at(
                boundary_frame - 1,
                focus_indices,
            )
            product_signature = self._local_signature_at(
                boundary_frame,
                focus_indices,
            )
            if reactant_signature == product_signature:
                return None

            reactant_start = boundary_frame - minimum_dwell
            product_end = boundary_frame + minimum_dwell - 1
            if reactant_start < 0 or product_end >= len(self.frames):
                return None
            if any(
                self._local_signature_at(frame, focus_indices)
                != reactant_signature
                for frame in range(reactant_start, reactant_end + 1)
            ):
                return None
            if any(
                self._local_signature_at(frame, focus_indices)
                != product_signature
                for frame in range(product_start, product_end + 1)
            ):
                return None

            reactant_state, reactant_run_length = (
                self._surface_state_run_length(
                    reactant_end,
                    focus_indices,
                )
            )
            product_state, product_run_length = (
                self._surface_state_run_length(
                    product_start,
                    focus_indices,
                )
            )
        if (
            reactant_state is None
            or product_state is None
            or reactant_state == product_state
            or reactant_run_length < minimum_dwell
            or product_run_length < minimum_dwell
        ):
            return None

        return {
            "reactant_molecule": reactant_molecule,
            "product_molecule": product_molecule,
            "reactant_frame": int(reactant_end),
            "product_frame": int(product_start),
            "reactant_run_length": int(reactant_run_length),
            "product_run_length": int(product_run_length),
            "reactant_adsorbed": bool(reactant_state),
            "product_adsorbed": bool(product_state),
            "minimum_dwell": int(minimum_dwell),
        }

    def _recover_persistent_atom_local_surface_events(
        self,
        surface_boundaries,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Recover persistent pure adsorption/desorption state changes."""
        if (
            not surface_boundaries
            or self.traj is None
            or not (self.sub_elements or self.sub_indices)
        ):
            return []

        local_stability = max(1, int(local_stability))
        recovered = []
        rejected = 0

        for candidate in surface_boundaries:
            boundary_frame = int(candidate.get("frame", -1))
            focus_indices = tuple(sorted({
                int(atom)
                for atom in candidate.get("focus_indices", ())
            }))
            dwell = self._persistent_local_surface_boundary(
                boundary_frame,
                focus_indices,
                local_stability,
                candidate=candidate,
            )
            if dwell is None:
                continue

            record = self._event_record(
                [dwell["reactant_molecule"]],
                [dwell["product_molecule"]],
                transition_frame=boundary_frame,
                reactant_frame=dwell["reactant_frame"],
                product_frame=dwell["product_frame"],
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if record is None:
                rejected += 1
                continue

            record.update({
                "Endpoint_Recovered": True,
                "Endpoint_Recovery_Attempted": True,
                "Endpoint_Recovery_Mode": (
                    "persistent_atom_local_surface_state"
                ),
                "Endpoint_Recovery_Evidence": (
                    "exact_atom_mapped_molecule;"
                    "covalent_topology_conserved;"
                    "gas_adsorbed_state_changed;"
                    "both_endpoint_dwell_satisfied"
                ),
                "Endpoint_Reactant_Run_Length": int(
                    dwell["reactant_run_length"]
                ),
                "Endpoint_Product_Run_Length": int(
                    dwell["product_run_length"]
                ),
                "Endpoint_Segmentation_Threshold": int(
                    local_stability
                ),
                "Endpoint_Requested_Local_Stability": int(
                    self._requested_local_stability
                ),
                "Endpoint_Original_Reactant_Frame": int(
                    boundary_frame - 1
                ),
                "Endpoint_Original_Product_Frame": int(
                    boundary_frame
                ),
                "Endpoint_Surface_Minimum_Dwell": int(
                    local_stability
                ),
                "Endpoint_Surface_Reactant_Expected_Adsorbed": bool(
                    dwell["reactant_adsorbed"]
                ),
                "Endpoint_Surface_Reactant_Observed_Adsorbed": bool(
                    dwell["reactant_adsorbed"]
                ),
                "Endpoint_Surface_Reactant_Run_Length": int(
                    dwell["reactant_run_length"]
                ),
                "Endpoint_Surface_Reactant_Dwell_Validated": True,
                "Endpoint_Surface_Product_Expected_Adsorbed": bool(
                    dwell["product_adsorbed"]
                ),
                "Endpoint_Surface_Product_Observed_Adsorbed": bool(
                    dwell["product_adsorbed"]
                ),
                "Endpoint_Surface_Product_Run_Length": int(
                    dwell["product_run_length"]
                ),
                "Endpoint_Surface_Product_Dwell_Validated": True,
                "Local_Surface_Recovery": True,
            })
            record[
                f"Endpoint_Surface_Reactant_Dwell_At_Least_"
                f"{local_stability}"
            ] = True
            record[
                f"Endpoint_Surface_Product_Dwell_At_Least_"
                f"{local_stability}"
            ] = True
            accepted, rejected_count = (
                self._filter_conservative_endpoint_records(
                    [record],
                    max_changed_bonds=3,
                    max_raw_changed_bonds=8,
                )
            )
            rejected += int(rejected_count)
            recovered.extend(accepted)

        recovered = self._dedupe_records(recovered)
        self._single_frame_recovery_stats[
            "persistent_atom_local_surface_recoveries"
        ] += len(recovered)
        self._single_frame_recovery_stats[
            "persistent_atom_local_surface_rejections"
        ] += int(rejected)
        return recovered

    def _find_stable_local_endpoint(self, focus_indices, target_signature,
                                    start, end, stable_frames, prefer='first'):
        """Return a near-transition frame inside a stable local endpoint run."""
        n_frames = len(self.frames)
        start = max(0, int(start))
        end = min(n_frames - 1, int(end))
        stable_frames = max(1, int(stable_frames))
        if end < start:
            return None

        runs = []
        i = start
        while i <= end:
            sig = self._local_signature_at(i, focus_indices)
            j = i + 1
            while j <= end and self._local_signature_at(j, focus_indices) == sig:
                j += 1
            if sig == target_signature and (j - i) >= stable_frames:
                runs.append((i, j - 1))
            i = j

        if not runs:
            return None
        run_start, run_end = runs[-1] if prefer == 'last' else runs[0]
        half = (stable_frames - 1) // 2
        if prefer == 'last':
            return max(run_start, run_end - half)
        return min(run_end, run_start + half)


    def _mols_to_smiles_text(self, mols):
        """Human-readable species text for unresolved-component review tables."""
        return " + ".join([
            str(m.get("smiles", m.get("formula", "")))
            for m in list(mols)
        ])

    def _mols_to_formula_text(self, mols, frame_idx=None, is_sub_mask=None, pre_cutoffs=None):
        """Human-readable formula text, including adsorption labels when available."""
        try:
            return " + ".join(self._build_formula_list(
                list(mols), frame_idx, is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
            ))
        except Exception:
            return " + ".join([
                str(m.get("formula", ""))
                for m in list(mols)
            ])


    def _unresolved_table_columns(self):
        return [
            "Unresolved_ID",
            "Reason",
            "Frame",
            "Reactant_Frame",
            "Product_Frame",
            "Reactants_Formula",
            "Products_Formula",
            "Reactants_SMILES",
            "Products_SMILES",
            "Type",
            "Reactants_Identity",
            "Products_Identity",
            "Topology_Validated",
            "Topology_Evidence",
            "Valence_Warning",
            "Isomerization_Class",
            "Clean_Eligible",
            "Endpoint_Recovery_Attempted",
            "Endpoint_Recovered",
            "Endpoint_Rejection_Reason",
            "Endpoint_Rejected_Count",
            "Endpoint_Recovery_Mode",
            "Endpoint_Recovery_Evidence",
            "Endpoint_Reactant_Run_Length",
            "Endpoint_Product_Run_Length",
            "Endpoint_Segmentation_Threshold",
            "Endpoint_Requested_Local_Stability",
            "Endpoint_Original_Reactant_Frame",
            "Endpoint_Original_Product_Frame",
            "Endpoint_Boundary_Reactant_Frame",
            "Endpoint_Boundary_Product_Frame",
            "Representative_Reactant_Frame",
            "Representative_Product_Frame",
            "Endpoint_Raw_Changed_Bond_Count",
            "Endpoint_Validated_Changed_Bond_Count",
            "Endpoint_Raw_Changed_Bonds",
            "Endpoint_Validated_Changed_Bonds",
            "Endpoint_Spectator_Changed_Bonds",
            "Endpoint_Raw_Changed_Bond_Components",
            "Endpoint_Validated_Changed_Bond_Components",
            "Endpoint_Transfer_Atom_Count",
            "Endpoint_Transfer_Atoms",
            "Endpoint_Component_Localization_Used",
            "Endpoint_Replaced_Invalid_Event",
            "Source_Raw_Reaction_ID",
            "Whole_Event_Fallback",
            "Composite_Candidate",
            "Composite_Evidence",
            "Changed_Bond_Count",
            "Changed_Bond_Components",
            "Changed_Bond_Component_Sizes",
            "Split_Elementary_Records",
            "Explained_By_Unresolved_ID",
            "Source_Explanation",
            "Atom_Indices",
            "Reactant_Atom_Groups",
            "Product_Atom_Groups",
            "Elementary_Mode",
            "Note",
        ]

    @classmethod
    def _freeze_unresolved_value(cls, value):
        """Return a deterministic, hashable representation of audit data."""
        if _is_missing_scalar(value):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return tuple(sorted(
                (
                    str(key),
                    cls._freeze_unresolved_value(item),
                )
                for key, item in value.items()
            ))
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(
                cls._freeze_unresolved_value(item)
                for item in value
            )
        if isinstance(value, (set, frozenset)):
            frozen = [
                cls._freeze_unresolved_value(item)
                for item in value
            ]
            return tuple(sorted(frozen, key=repr))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    def _unresolved_component_key(self, row):
        """Key an audit row by every published semantic field except its ID."""
        return tuple(
            (
                column,
                self._freeze_unresolved_value(row.get(column)),
            )
            for column in self._unresolved_table_columns()
            if column != "Unresolved_ID"
        )

    def _record_unresolved_component(
        self,
        reason,
        transition_index,
        prev,
        curr,
        sub_r,
        sub_p,
        focus_indices,
        scan_start=None,
        scan_end=None,
        reactant_frame=None,
        product_frame=None,
        is_sub_mask=None,
        pre_cutoffs=None,
        endpoint_rejected_count=0,
        note="",
        record_override=None
    ):
        """Record a mapped transition component that could not be accepted safely.

        This is an audit table only. It does not change raw/clean event schemas
        or the accepted-event algorithm.
        """
        sub_r = list(sub_r)
        sub_p = list(sub_p)

        all_indices = sorted(set(
            int(idx)
            for m in sub_r + sub_p
            for idx in m.get("indices", [])
        ))
        focus_indices = sorted(set(int(i) for i in focus_indices))

        if not all_indices:
            all_indices = list(focus_indices)

        override_frame = None
        if record_override is not None:
            try:
                override_frame = int(record_override.get("Frame"))
            except Exception:
                override_frame = None
        try:
            current_start = int(curr.get("start"))
        except Exception:
            current_start = None
        event_frame = (
            override_frame
            if override_frame is not None
            else current_start
            if current_start is not None and current_start >= 0
            else int(transition_index)
        )

        row = {
            "Unresolved_ID": f"UNRES-{len(self.unresolved_components) + 1:04d}",
            "Reason": str(reason),
            "Frame": int(event_frame),
            "Reactant_Frame": None if reactant_frame is None else int(reactant_frame),
            "Product_Frame": None if product_frame is None else int(product_frame),
            "Endpoint_Rejected_Count": int(endpoint_rejected_count or 0),
            "Reactants_Formula": self._mols_to_formula_text(
                sub_r, reactant_frame, is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
            ),
            "Products_Formula": self._mols_to_formula_text(
                sub_p, product_frame, is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
            ),
            "Reactants_SMILES": self._mols_to_smiles_text(sub_r),
            "Products_SMILES": self._mols_to_smiles_text(sub_p),
            "Atom_Indices": str(all_indices),
            "Reactant_Atom_Groups": [
                sorted(set(int(i) for i in m.get("indices", []) or []))
                for m in sub_r
            ],
            "Product_Atom_Groups": [
                sorted(set(int(i) for i in m.get("indices", []) or []))
                for m in sub_p
            ],
            "Note": str(note or ""),
        }

        if endpoint_rejected_count and self._last_endpoint_filter_rejections:
            rejection_rows = list(self._last_endpoint_filter_rejections)
            reasons = sorted({
                str(r.get("Endpoint_Rejection_Reason", "") or "")
                for r in rejection_rows
                if str(r.get("Endpoint_Rejection_Reason", "") or "")
            })
            if reasons:
                row["Endpoint_Rejection_Reason"] = ";".join(reasons)
            if len(rejection_rows) == 1:
                audit = rejection_rows[0]
                for field in (
                    "Endpoint_Recovery_Attempted",
                    "Endpoint_Raw_Changed_Bond_Count",
                    "Endpoint_Validated_Changed_Bond_Count",
                    "Endpoint_Raw_Changed_Bonds",
                    "Endpoint_Validated_Changed_Bonds",
                    "Endpoint_Spectator_Changed_Bonds",
                    "Endpoint_Raw_Changed_Bond_Components",
                    "Endpoint_Validated_Changed_Bond_Components",
                    "Endpoint_Transfer_Atom_Count",
                    "Endpoint_Transfer_Atoms",
                    "Endpoint_Component_Localization_Used",
                    "Endpoint_Boundary_Reactant_Frame",
                    "Endpoint_Boundary_Product_Frame",
                    "Representative_Reactant_Frame",
                    "Representative_Product_Frame",
                ):
                    if field in audit:
                        row[field] = audit.get(field)

        if record_override is not None:
            rec = record_override
            if rec.get('Reactants_Formula'):
                row["Reactants_Formula"] = rec.get('Reactants_Formula')
            if rec.get('Products_Formula'):
                row["Products_Formula"] = rec.get('Products_Formula')
            if rec.get('Reactants_SMILES'):
                row["Reactants_SMILES"] = rec.get('Reactants_SMILES')
            if rec.get('Products_SMILES'):
                row["Products_SMILES"] = rec.get('Products_SMILES')
            for field in (
                'Type',
                'Reactants_Identity',
                'Products_Identity',
                'Topology_Validated',
                'Topology_Evidence',
                'Valence_Warning',
                'Isomerization_Class',
                'Clean_Eligible',
                'Endpoint_Recovery_Attempted',
                'Endpoint_Recovered',
                'Endpoint_Rejection_Reason',
                'Endpoint_Rejected_Count',
                'Endpoint_Recovery_Mode',
                'Endpoint_Recovery_Evidence',
                'Endpoint_Reactant_Run_Length',
                'Endpoint_Product_Run_Length',
                'Endpoint_Segmentation_Threshold',
                'Endpoint_Requested_Local_Stability',
                'Endpoint_Original_Reactant_Frame',
                'Endpoint_Original_Product_Frame',
                'Endpoint_Boundary_Reactant_Frame',
                'Endpoint_Boundary_Product_Frame',
                'Representative_Reactant_Frame',
                'Representative_Product_Frame',
                'Endpoint_Raw_Changed_Bond_Count',
                'Endpoint_Validated_Changed_Bond_Count',
                'Endpoint_Raw_Changed_Bonds',
                'Endpoint_Validated_Changed_Bonds',
                'Endpoint_Spectator_Changed_Bonds',
                'Endpoint_Raw_Changed_Bond_Components',
                'Endpoint_Validated_Changed_Bond_Components',
                'Endpoint_Transfer_Atom_Count',
                'Endpoint_Transfer_Atoms',
                'Endpoint_Component_Localization_Used',
                'Endpoint_Replaced_Invalid_Event',
                'Raw_Reaction_ID',
                'Source_Raw_Reaction_ID',
                'Whole_Event_Fallback',
                'Composite_Candidate',
                'Composite_Evidence',
                'Changed_Bond_Count',
                'Changed_Bond_Components',
                'Changed_Bond_Component_Sizes',
                'Split_Elementary_Records',
                'Reactant_Atom_Groups',
                'Product_Atom_Groups',
                'Elementary_Mode',
            ):
                if field in rec:
                    row[field] = rec.get(field)
            if rec.get('Atom_Indices') is not None:
                row["Atom_Indices"] = str(rec.get('Atom_Indices'))
            source_raw_id = str(
                rec.get(
                    "Source_Raw_Reaction_ID",
                    rec.get("Raw_Reaction_ID", ""),
                ) or ""
            )
            if source_raw_id:
                row["Source_Raw_Reaction_ID"] = source_raw_id

        semantic_key = self._unresolved_component_key(row)
        existing = self._unresolved_component_key_to_row.get(semantic_key)
        if existing is not None:
            self._unresolved_duplicate_count += 1
            return existing

        self.unresolved_components.append(row)
        self._unresolved_component_key_to_row[semantic_key] = row
        return row

    def get_unresolved_components_table(self):
        """Return unresolved mapped transition components as a review DataFrame."""
        return pd.DataFrame(self.unresolved_components, columns=self._unresolved_table_columns())

    def get_dropped_candidates_table(self):
        """Backward-compatible alias for older calling code."""
        return self.get_unresolved_components_table()


    def _parts_text(self, text):
        if not isinstance(text, str):
            return []
        return [x.strip() for x in text.split(' + ') if x.strip()]

    def _filter_conservative_endpoint_records(
        self,
        records,
        max_changed_bonds=3,
        max_raw_changed_bonds=8,
    ):
        """Apply conservative endpoint validation with reaction-centre localization."""
        accepted = []
        rejected = 0
        self._last_endpoint_filter_rejections = []

        def reject(evaluated, reason):
            nonlocal rejected
            evaluated["Endpoint_Recovered"] = False
            evaluated["Endpoint_Rejection_Reason"] = str(reason)
            rejected += 1
            snapshot = dict(evaluated)
            self._last_endpoint_filter_rejections.append(snapshot)
            self._endpoint_rejection_audit.append(snapshot)

        for rec in records:
            evaluated = dict(rec)

            if _truthy_value(
                evaluated.get("Whole_Event_Fallback", False)
            ):
                accepted.append(evaluated)
                continue

            evaluated["Endpoint_Recovery_Attempted"] = True
            evaluated["Endpoint_Rejection_Reason"] = ""

            r_count, p_count = self._elementary_side_counts(evaluated)
            if not (1 <= r_count <= 3 and 1 <= p_count <= 3):
                reject(evaluated, "unsupported_side_count")
                continue

            try:
                representative_r = int(
                    evaluated.get("Reactant_Frame", -1)
                )
                representative_p = int(
                    evaluated.get("Product_Frame", -1)
                )
                r_frame = int(
                    evaluated.get(
                        "Endpoint_Boundary_Reactant_Frame",
                        representative_r,
                    )
                )
                p_frame = int(
                    evaluated.get(
                        "Endpoint_Boundary_Product_Frame",
                        representative_p,
                    )
                )
            except Exception:
                reject(evaluated, "invalid_endpoint_frame")
                continue

            evaluated.setdefault(
                "Representative_Reactant_Frame",
                representative_r,
            )
            evaluated.setdefault(
                "Representative_Product_Frame",
                representative_p,
            )
            evaluated["Endpoint_Boundary_Reactant_Frame"] = r_frame
            evaluated["Endpoint_Boundary_Product_Frame"] = p_frame

            if (
                self.traj is None
                or r_frame < 0
                or p_frame < 0
                or r_frame >= len(self.traj)
                or p_frame >= len(self.traj)
                or r_frame >= p_frame
            ):
                reject(evaluated, "endpoint_frame_out_of_range_or_reversed")
                continue

            indices = evaluated.get("Atom_Indices", [])
            try:
                indices = sorted({int(index) for index in indices})
            except Exception:
                reject(evaluated, "invalid_endpoint_atom_indices")
                continue
            if not indices:
                reject(evaluated, "empty_endpoint_atom_indices")
                continue

            raw_changed = self._changed_bond_edges(
                r_frame,
                p_frame,
                indices,
            )
            raw_components = self._changed_bond_components_from_edges(
                raw_changed
            )
            evaluated.update({
                "Endpoint_Raw_Changed_Bond_Count": len(raw_changed),
                "Endpoint_Raw_Changed_Bonds": (
                    self._format_changed_bond_edges(raw_changed)
                ),
                "Endpoint_Raw_Changed_Bond_Components": (
                    len(raw_components)
                ),
            })

            state_only_transition = (
                self._is_state_only_surface_transition(evaluated)
            )
            if state_only_transition:
                surface_minimum_dwell = (
                    self._resolved_surface_minimum_dwell()
                )
                self._ensure_surface_endpoint_evidence(
                    evaluated,
                    minimum_dwell=surface_minimum_dwell,
                )
                if self._unstable_surface_state_transition(
                    evaluated,
                    minimum_dwell=surface_minimum_dwell,
                ):
                    self._single_frame_recovery_stats[
                        "surface_state_insufficient_dwell_rejections"
                    ] += 1
                    reject(
                        evaluated,
                        "insufficient_surface_state_dwell",
                    )
                    continue
                evaluated.update({
                    "Endpoint_Validated_Changed_Bond_Count": 0,
                    "Endpoint_Validated_Changed_Bonds": "",
                    "Endpoint_Spectator_Changed_Bonds": (
                        self._format_changed_bond_edges(raw_changed)
                    ),
                    "Endpoint_Validated_Changed_Bond_Components": 0,
                    "Endpoint_Transfer_Atom_Count": 0,
                    "Endpoint_Transfer_Atoms": "",
                    "Endpoint_Component_Localization_Used": False,
                })
                accepted.append(evaluated)
                continue

            localization = self._component_transfer_changed_bonds(
                r_frame,
                p_frame,
                indices,
                raw_changed,
            )
            raw_limit_exceeded = (
                len(raw_changed) > int(max_raw_changed_bonds)
            )
            evaluated.update({
                "Endpoint_Raw_Changed_Bond_Limit": int(
                    max_raw_changed_bonds
                ),
                "Endpoint_Raw_Changed_Bond_Limit_Exceeded": bool(
                    raw_limit_exceeded
                ),
            })

            if (
                raw_limit_exceeded
                and not localization["localization_used"]
            ):
                reject(evaluated, "too_many_unlocalized_raw_changed_bonds")
                continue

            changed = set(localization["validated_changed"])
            changed_components = (
                self._changed_bond_components_from_edges(changed)
            )
            transfer_atoms = sorted(
                int(atom)
                for atom in localization["transfer_atoms"]
            )
            evaluated.update({
                "Endpoint_Validated_Changed_Bond_Count": len(changed),
                "Endpoint_Validated_Changed_Bonds": (
                    self._format_changed_bond_edges(changed)
                ),
                "Endpoint_Spectator_Changed_Bonds": (
                    self._format_changed_bond_edges(
                        localization["spectator_changed"]
                    )
                ),
                "Endpoint_Validated_Changed_Bond_Components": (
                    len(changed_components)
                ),
                "Endpoint_Transfer_Atom_Count": len(transfer_atoms),
                "Endpoint_Transfer_Atoms": ",".join(
                    str(atom) for atom in transfer_atoms
                ),
                "Endpoint_Component_Localization_Used": bool(
                    localization["localization_used"]
                ),
            })

            if not (1 <= len(changed) <= int(max_changed_bonds)):
                reject(
                    evaluated,
                    "validated_changed_bond_count_out_of_range",
                )
                continue

            graph = nx.Graph()
            graph.add_edges_from(changed)
            graph_connected = (
                graph.number_of_nodes() > 0
                and nx.is_connected(graph)
            )
            stable_centre_bridges = set()
            if not graph_connected:
                stable_centre_bridges = (
                    self._stable_reaction_centre_bridges(
                        r_frame,
                        p_frame,
                        indices,
                        changed,
                    )
                )
            if not graph_connected and not stable_centre_bridges:
                reject(
                    evaluated,
                    "multiple_validated_reaction_centres",
                )
                continue
            if stable_centre_bridges:
                evaluated.update({
                    "Reaction_Centre_Connectivity": (
                        "one_stable_bond_bridge"
                    ),
                    "Reaction_Centre_Bridge_Bond_Count": 1,
                    "Reaction_Centre_Bridge_Bonds": (
                        self._format_changed_bond_edges(
                            stable_centre_bridges
                        )
                    ),
                })
            else:
                evaluated.update({
                    "Reaction_Centre_Connectivity": (
                        "changed_bond_connected"
                    ),
                    "Reaction_Centre_Bridge_Bond_Count": 0,
                    "Reaction_Centre_Bridge_Bonds": "",
                })

            # A 1↔3, 2↔3, or 3↔3 observation is not automatically a
            # non-elementary artefact.  It can be retained as a concerted
            # multi-bond candidate only after stable endpoint, mass balance,
            # atom mapping, changed-bond-count, and single connected reaction
            # centre checks have all succeeded above.
            if r_count == 3 or p_count == 3:
                evaluated["Concerted_Multi_Bond"] = True
                evaluated["Elementary_Mode"] = "concerted_multi_bond"
                evaluated["Elementary_Evidence"] = (
                    "stable_endpoints;atom_mapped;mass_balanced;"
                    + (
                        "one_unchanged_backbone_bridge;"
                        if stable_centre_bridges
                        else "single_connected_reaction_centre;"
                    )
                    + "no_resolved_stable_intermediate"
                )
            else:
                evaluated.setdefault("Concerted_Multi_Bond", False)
                evaluated.setdefault("Elementary_Mode", "standard")

            accepted.append(evaluated)

        return accepted, rejected

    def _dedupe_records(self, records):
        out = []
        seen = set()
        for r in sorted(
            records,
            key=lambda x: (
                int(x.get('Frame', 0)),
                tuple(x.get('Atom_Indices', [])),
                x.get('Reactants_Identity', x.get('Reactants_SMILES', '')),
                x.get('Products_Identity', x.get('Products_SMILES', '')),
            ),
        ):
            key = (
                int(r.get('Frame', 0)),
                tuple(sorted(r.get('Atom_Indices', []))),
                tuple(sorted(self._parts_text(
                    r.get('Reactants_Identity', r.get('Reactants_SMILES', ''))
                ))),
                tuple(sorted(self._parts_text(
                    r.get('Products_Identity', r.get('Products_SMILES', ''))
                ))),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out


    # =====================================================================
    # State-aware reaction-lineage recovery and validation
    # =====================================================================
    def _lineage_species_key(self, smiles, formula, atom_indices=None):
        """Return an atom-instance-aware species-state identity."""
        atoms = tuple(sorted(set(
            int(i) for i in (atom_indices or [])
        )))
        return (
            str(smiles or "").strip(),
            str(formula or "").strip(),
            atoms,
        )

    def _lineage_family_key(self, smiles, formula=""):
        """Return connectivity identity used only during parent recovery."""
        smi = str(smiles or "").strip()
        base_formula = self._strip_surface_suffix(formula)
        if smi and base_formula:
            return f"{smi}||{base_formula}"
        return smi or base_formula

    def _lineage_side_entries(self, record, side):
        """Return paired SMILES/formula entries for one event side."""
        side = str(side).lower()
        if side.startswith("r"):
            smi_text = record.get("Reactants_SMILES", "")
            formula_text = record.get("Reactants_Formula", "")
            atom_groups = _normalize_atom_groups(
                record.get("Reactant_Atom_Groups", [])
            )
        else:
            smi_text = record.get("Products_SMILES", "")
            formula_text = record.get("Products_Formula", "")
            atom_groups = _normalize_atom_groups(
                record.get("Product_Atom_Groups", [])
            )

        identity_field = (
            "Reactants_Identity"
            if side.startswith("r")
            else "Products_Identity"
        )
        identity_text = record.get(identity_field, smi_text)
        smis = self._parts_text(smi_text)
        identities = self._parts_text(identity_text)
        formulas = self._parts_text(formula_text)
        n = max(len(smis), len(identities), len(formulas))
        out = []
        for i in range(n):
            smi = smis[i] if i < len(smis) else ""
            identity = (
                identities[i]
                if i < len(identities)
                else smi
            )
            formula = formulas[i] if i < len(formulas) else smi
            atoms = atom_groups[i] if i < len(atom_groups) else tuple()
            if not atoms and n == 1:
                atoms = tuple(sorted(set(
                    int(x) for x in record.get("Atom_Indices", []) or []
                )))
            out.append({
                "smiles": smi,
                "identity": identity,
                "formula": formula,
                "atom_indices": list(atoms),
                "key": self._lineage_species_key(identity, formula, atoms),
                "family": self._lineage_family_key(identity, formula),
            })
        return out

    def _lineage_mol_entry(self, mol, frame_idx, is_sub_mask=None, pre_cutoffs=None):
        smi = str(mol.get("smiles", mol.get("formula", "")) or "").strip()
        formulas = self._build_formula_list(
            [mol], frame_idx, is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
        )
        formula = str(
            formulas[0] if formulas else mol.get("formula", smi) or smi
        ).strip()
        identity = self._molecule_identity(mol)
        atoms = sorted(set(
            int(i) for i in mol.get("indices", []) or []
            if int(i) not in self.sub_indices
        ))
        return {
            "smiles": smi,
            "identity": identity,
            "formula": formula,
            "atom_indices": atoms,
            "key": self._lineage_species_key(identity, formula, atoms),
            "family": self._lineage_family_key(identity, formula),
        }

    @staticmethod
    def _lineage_payload(entries):
        import json
        payload = [
            {
                "smiles": str(e.get("smiles", "")),
                "identity": str(e.get("identity", e.get("smiles", ""))),
                "formula": str(e.get("formula", "")),
                "atom_indices": [
                    int(i) for i in e.get("atom_indices", []) or []
                ],
            }
            for e in entries
        ]
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _initial_lineage_entries(self, is_sub_mask=None, pre_cutoffs=None):
        """State-aware species present in the physical first trajectory frame."""
        out = []
        if self.frames is None or len(self.frames) == 0:
            return out

        seen = set()
        for mol in self.frames[0]:
            inds = [int(i) for i in mol.get("indices", [])]
            if inds and (
                (self.sub_indices and all(i in self.sub_indices for i in inds))
                or (
                    self.sub_elements
                    and self.traj is not None
                    and len(self.traj) > 0
                    and all(
                        self.traj[0].get_chemical_symbols()[i]
                        in self.sub_elements
                        for i in inds
                    )
                )
            ):
                continue

            entry = self._lineage_mol_entry(
                mol,
                0,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if entry["key"] not in seen:
                out.append(entry)
                seen.add(entry["key"])
        return out

    def final_stable_lineage_entries(self, minimum_dwell=None):
        """Return exact molecular states continuously present at the final frame."""
        if self.frames is None or len(self.frames) == 0:
            return []

        final_frame = len(self.frames) - 1
        dwell = max(
            1,
            int(
                minimum_dwell
                if minimum_dwell is not None
                else getattr(self, "_segmentation_local_stability", 1) or 1
            ),
        )
        start_frame = max(0, final_frame - dwell + 1)

        final_entries = [
            self._lineage_mol_entry(mol, final_frame)
            for mol in self.frames[final_frame]
        ]
        final_entries = [
            entry for entry in final_entries if entry.get("atom_indices")
        ]

        stable_keys = {entry["key"] for entry in final_entries}
        for frame_idx in range(start_frame, final_frame):
            present = {
                self._lineage_mol_entry(mol, frame_idx)["key"]
                for mol in self.frames[frame_idx]
            }
            stable_keys.intersection_update(present)
            if not stable_keys:
                break

        out = []
        for entry in final_entries:
            if entry["key"] not in stable_keys:
                continue
            item = dict(entry)
            item["frame"] = int(final_frame)
            item["stable_start_frame"] = int(start_frame)
            item["stable_dwell_frames"] = int(final_frame - start_frame + 1)
            out.append(item)
        return out

    def _lineage_gaps(self, records, is_sub_mask=None, pre_cutoffs=None, annotate=False):
        """Find reactant states with no initial or earlier detected source."""
        initial_entries = self._initial_lineage_entries(
            is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
        )
        initial_keys = {e["key"] for e in initial_entries}
        available = set(initial_keys)
        gaps = []

        ordered = sorted(
            records,
            key=lambda r: (
                int(r.get("Frame", r.get("Product_Frame", 0)) or 0),
                int(r.get("Reactant_Frame", r.get("Frame", 0)) or 0),
                int(r.get("Product_Frame", r.get("Frame", 0)) or 0),
            ),
        )

        for rec in ordered:
            reactants = self._lineage_side_entries(rec, "reactants")
            products = self._lineage_side_entries(rec, "products")

            initial_reactants = [
                e for e in reactants if e["key"] in initial_keys
            ]
            missing = [e for e in reactants if e["key"] not in available]

            if annotate:
                rec["Initial_Species_Keys"] = self._lineage_payload(
                    initial_entries
                )
                rec["Initial_Reactant_Keys"] = self._lineage_payload(
                    initial_reactants
                )
                rec["Lineage_Complete"] = not bool(missing)
                rec["Missing_Reactant_Keys"] = self._lineage_payload(missing)
                rec["Missing_Reactant_Species"] = " + ".join(
                    e["formula"] or e["smiles"] for e in missing
                )

            for entry in missing:
                gaps.append({
                    "record": rec,
                    "entry": entry,
                    "consume_frame": int(
                        rec.get("Reactant_Frame", rec.get("Frame", 0)) or 0
                    ),
                })

            # Event-state inventory: valid events consume their exact reactant
            # instances and create exact product instances.  This prevents a
            # molecule consumed earlier from being reused by a later event.
            if not missing:
                available.difference_update(e["key"] for e in reactants)
            available.update(e["key"] for e in products)

        return gaps, initial_entries

    def _lineage_event_key(self, record, include_frame=True):
        r = tuple(
            sorted(
                e["key"]
                for e in self._lineage_side_entries(record, "reactants")
            )
        )
        p = tuple(
            sorted(
                e["key"]
                for e in self._lineage_side_entries(record, "products")
            )
        )
        idx = tuple(
            sorted(int(i) for i in record.get("Atom_Indices", []) or [])
        )
        if include_frame:
            return (
                int(record.get("Frame", record.get("Product_Frame", 0)) or 0),
                r,
                p,
                idx,
            )
        return (r, p, idx)

    def _record_family_sets(self, record):
        reactants = {
            e.get("family", "")
            for e in self._lineage_side_entries(record, "reactants")
        }
        products = {
            e.get("family", "")
            for e in self._lineage_side_entries(record, "products")
        }
        reactants.discard("")
        products.discard("")
        return reactants, products

    def _record_forms_family(self, record, target_family):
        """True only when connectivity is formed, not merely state-switched."""
        reactants, products = self._record_family_sets(record)
        return target_family in products and target_family not in reactants

    def _reachable_lineage_state_keys(
        self,
        records,
        initial_entries,
        cutoff_frame=None,
    ):
        """Return exact molecule instances available at ``cutoff_frame``."""
        available = {entry["key"] for entry in initial_entries}
        ordered = sorted(
            records,
            key=lambda r: (
                int(r.get("Product_Frame", r.get("Frame", 0)) or 0),
                int(r.get("Frame", 0) or 0),
            ),
        )

        for rec in ordered:
            product_frame = int(
                rec.get("Product_Frame", rec.get("Frame", 0)) or 0
            )
            if cutoff_frame is not None and product_frame > int(cutoff_frame):
                continue

            reactants = {
                e["key"]
                for e in self._lineage_side_entries(rec, "reactants")
            }
            products = {
                e["key"]
                for e in self._lineage_side_entries(rec, "products")
            }
            if not reactants or not products:
                continue
            if not reactants.issubset(available):
                continue
            available.difference_update(reactants)
            available.update(products)
        return available

    def _family_has_earlier_source(
        self,
        target_family,
        consume_frame,
        records,
        initial_entries,
    ):
        reachable = self._reachable_lineage_state_keys(
            records,
            initial_entries,
            cutoff_frame=consume_frame,
        )
        return any(
            self._lineage_family_key(smiles, formula) == target_family
            for smiles, formula, _atoms in reachable
        )

    def _target_family_present_at(
        self,
        frame_idx,
        focus_indices,
        target_family,
    ):
        """Whether the exact mapped atoms form the requested connectivity."""
        focus = tuple(sorted(int(i) for i in focus_indices))
        if not focus:
            return False
        frame_idx = int(frame_idx)
        cache_key = (frame_idx, focus, str(target_family))
        cached = self._lineage_family_presence_cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        # The compact interval index is built from the same molecule records during Mapping States.
        if self._presence_index_complete:
            bounds = self._lineage_family_intervals.get(
                (focus, str(target_family)),
                (),
            )
            present = self._frame_in_presence_intervals(bounds, frame_idx)
            self._lineage_family_presence_cache[cache_key] = present
            return present

        present = False
        # Direct frame scan avoids constructing the much heavier atom->molecule
        # index cache for every historical frame during long backward searches.
        for mol in self.frames[frame_idx]:
            indices = tuple(sorted(int(i) for i in mol.get("indices", [])))
            if indices != focus:
                continue
            family = self._lineage_family_key(
                self._molecule_identity(mol),
                mol.get("formula", ""),
            )
            if family == target_family:
                present = True
                break

        self._lineage_family_presence_cache[cache_key] = present
        return present

    def _find_persistent_family_start(
        self,
        focus_indices,
        target_family,
        consume_frame,
        stable_frames,
    ):
        """Find the start of the current family episode, bridging short flashes."""
        stable_frames = max(1, int(stable_frames))
        consume_frame = max(0, min(int(consume_frame), len(self.frames) - 1))
        if not self._target_family_present_at(
            consume_frame, focus_indices, target_family
        ):
            return None

        episode_start = consume_frame
        absent_run = 0
        short_absence_frames = 0

        for frame_idx in range(consume_frame - 1, -1, -1):
            present = self._target_family_present_at(
                frame_idx, focus_indices, target_family
            )
            if present:
                if 0 < absent_run < stable_frames:
                    short_absence_frames += absent_run
                absent_run = 0
                episode_start = frame_idx
                continue

            absent_run += 1
            if absent_run >= stable_frames:
                break

        if episode_start <= 0 and absent_run < stable_frames:
            return {
                "start_frame": 0,
                "short_absence_frames": short_absence_frames,
                "gap_tolerance": stable_frames - 1,
                "present_from_initial": True,
            }

        return {
            "start_frame": int(episode_start),
            "short_absence_frames": int(short_absence_frames),
            "gap_tolerance": int(stable_frames - 1),
            "present_from_initial": False,
        }

    def _candidate_original_frame(self, frame_idx):
        try:
            atoms = self.traj[int(frame_idx)]
            return int(atoms.info.get("original_frame", frame_idx))
        except Exception:
            return int(frame_idx)

    def _family_candidates_near_boundary(
        self,
        focus_indices,
        target_family,
        boundary_frame,
        consume_frame,
        stability,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Collect mapped formation candidates near a persistent-family start."""
        margin = max(30, 6 * int(stability), 10 * int(local_stability))
        scan_start = max(0, int(boundary_frame) - margin)
        scan_end = min(
            len(self.frames) - 1,
            int(consume_frame),
            int(boundary_frame) + margin,
        )

        candidates = list(
            self._scan_local_elementary_events(
                scan_start,
                scan_end,
                focus_indices,
                max(1, int(stability)),
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
                local_stability=self._recovery_scan_local_stability,
            )
        )

        # Boundary-anchored fallback. 
        if int(boundary_frame) > 0:
            reactant_frame = int(boundary_frame) - 1
            half = max(1, (max(1, int(stability)) - 1) // 2)
            product_frame = min(
                int(consume_frame),
                len(self.frames) - 1,
                int(boundary_frame) + half,
            )
            if self._target_family_present_at(
                product_frame, focus_indices, target_family
            ):
                react_mols = self._local_mols(reactant_frame, focus_indices)
                product_mols = self._local_mols(product_frame, focus_indices)
                direct = []
                for sub_r, sub_p in self._disentangle_events(
                    react_mols,
                    product_mols,
                    reactant_frame,
                    product_frame,
                ):
                    direct.extend(
                        self._split_by_changed_bond_centers(
                            sub_r,
                            sub_p,
                            reactant_frame,
                            product_frame,
                            is_sub_mask=is_sub_mask,
                            pre_cutoffs=pre_cutoffs,
                            transition_frame=int(boundary_frame),
                        )
                    )
                direct, _rejected = self._filter_conservative_endpoint_records(
                    direct,
                    max_changed_bonds=3,
                )
                candidates.extend(direct)

        safe_candidates = []
        for candidate in candidates:
            if not _truthy_value(
                candidate.get("Clean_Eligible", True)
            ):
                continue
            if _truthy_value(
                candidate.get("Composite_Candidate", False)
            ):
                continue
            filtered, _rejected = (
                self._filter_conservative_endpoint_records(
                    [candidate],
                    max_changed_bonds=3,
                )
            )
            safe_candidates.extend(filtered)
        candidates = safe_candidates

        best_by_event = {}
        half = max(1, (max(1, int(stability)) - 1) // 2)
        ideal_reactant = max(0, int(boundary_frame) - 1)
        ideal_product = min(
            int(consume_frame),
            len(self.frames) - 1,
            int(boundary_frame) + half,
        )
        for candidate in candidates:
            if not self._record_forms_family(candidate, target_family):
                continue
            product_frame = int(
                candidate.get("Product_Frame", candidate.get("Frame", 0)) or 0
            )
            if product_frame > int(consume_frame):
                continue
            key = self._lineage_event_key(candidate, include_frame=True)
            reactant_frame = int(
                candidate.get("Reactant_Frame", candidate.get("Frame", 0)) or 0
            )
            score = (
                abs(int(candidate.get("Frame", 0) or 0) - int(boundary_frame)),
                abs(reactant_frame - ideal_reactant)
                + abs(product_frame - ideal_product),
                abs(product_frame - int(boundary_frame)),
            )
            previous = best_by_event.get(key)
            if previous is None or score < previous[0]:
                best_by_event[key] = (score, candidate)
        return [item[1] for item in best_by_event.values()]

    def _exact_state_candidates_near_first_run(
        self,
        focus_indices,
        target_key,
        consume_frame,
        stability,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        target_sig = self._local_signature_at(consume_frame, focus_indices)
        first_target_frame = consume_frame
        while (
            first_target_frame > 0
            and self._local_signature_at(
                first_target_frame - 1, focus_indices
            ) == target_sig
        ):
            first_target_frame -= 1

        if first_target_frame <= 0:
            return []

        margin = max(30, 6 * int(stability), 10 * int(local_stability))
        scan_start = max(0, first_target_frame - margin)
        scan_end = min(
            len(self.frames) - 1,
            consume_frame,
            first_target_frame + margin,
        )
        candidates = self._scan_local_elementary_events(
            scan_start,
            scan_end,
            focus_indices,
            max(1, int(stability)),
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
            local_stability=self._recovery_scan_local_stability,
        )
        out = []
        for candidate in candidates:
            if not _truthy_value(
                candidate.get("Clean_Eligible", True)
            ):
                continue
            if _truthy_value(
                candidate.get("Composite_Candidate", False)
            ):
                continue

            filtered, _rejected = (
                self._filter_conservative_endpoint_records(
                    [candidate],
                    max_changed_bonds=3,
                )
            )
            if not filtered:
                continue
            candidate = filtered[0]

            product_keys = {
                entry["key"]
                for entry in self._lineage_side_entries(
                    candidate,
                    "products",
                )
            }
            product_frame = int(
                candidate.get(
                    "Product_Frame",
                    candidate.get("Frame", 0),
                )
                or 0
            )
            if (
                target_key in product_keys
                and product_frame <= consume_frame
            ):
                out.append(candidate)
        return out

    def _exact_mapped_molecule(
        self,
        frame_idx,
        atom_indices,
    ):
        """Return the unique molecule owning exactly ``atom_indices``."""
        focus = tuple(sorted({int(index) for index in atom_indices}))
        if (
            not focus
            or frame_idx < 0
            or frame_idx >= len(self.frames)
        ):
            return None
        matches = [
            molecule
            for molecule in self._local_mols(frame_idx, focus)
            if self._molecule_atom_group(molecule) == focus
        ]
        return matches[0] if len(matches) == 1 else None

    def _unique_mapped_focus_container(
        self,
        frame_idx,
        atom_indices,
    ):
        """Return the unique molecule containing every mapped focus atom."""
        focus = {int(index) for index in atom_indices}
        if (
            not focus
            or frame_idx < 0
            or frame_idx >= len(self.frames)
        ):
            return None
        containers = []
        for molecule in self._local_mols(frame_idx, focus):
            members = set(self._molecule_atom_group(molecule))
            if focus.issubset(members):
                containers.append(molecule)
        return containers[0] if len(containers) == 1 else None

    def _mapped_endpoint_transfer_centre(
        self,
        reactant_frame,
        product_frame,
        atom_indices,
        max_raw_changed_bonds=8,
    ):
        """Identify one strictly atom-mapped intramolecular group transfer."""
        if self.traj is None:
            return None
        try:
            reactant_frame = int(reactant_frame)
            product_frame = int(product_frame)
        except (TypeError, ValueError):
            return None
        if (
            reactant_frame < 0
            or product_frame < 0
            or reactant_frame >= len(self.traj)
            or product_frame >= len(self.traj)
            or reactant_frame >= product_frame
        ):
            return None

        focus = sorted({int(index) for index in atom_indices})
        reactant_bonds = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _typ in self._get_local_bonds(
                self.traj[reactant_frame],
                focus,
            )
        }
        product_bonds = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _typ in self._get_local_bonds(
                self.traj[product_frame],
                focus,
            )
        }
        raw_changed = reactant_bonds ^ product_bonds
        if not (
            2 <= len(raw_changed) <= int(max_raw_changed_bonds)
        ):
            return None

        reactant_neighbors = defaultdict(set)
        product_neighbors = defaultdict(set)
        for left, right in reactant_bonds:
            reactant_neighbors[left].add(right)
            reactant_neighbors[right].add(left)
        for left, right in product_bonds:
            product_neighbors[left].add(right)
            product_neighbors[right].add(left)

        candidates = []
        for atom_index in focus:
            before = set(reactant_neighbors.get(atom_index, set()))
            after = set(product_neighbors.get(atom_index, set()))
            lost = before - after
            gained = after - before
            if (
                len(before) == 1
                and len(after) == 1
                and len(lost) == 1
                and len(gained) == 1
            ):
                old_neighbor = next(iter(lost))
                new_neighbor = next(iter(gained))
                transfer_edges = {
                    tuple(sorted((atom_index, old_neighbor))),
                    tuple(sorted((atom_index, new_neighbor))),
                }
                if transfer_edges.issubset(raw_changed):
                    candidates.append(
                        (int(atom_index), transfer_edges)
                    )

        if len(candidates) != 1:
            return None
        transfer_atom, transfer_edges = candidates[0]
        return {
            "transfer_atom": int(transfer_atom),
            "validated_changed": set(transfer_edges),
            "raw_changed": set(raw_changed),
            "spectator_changed": set(raw_changed) - set(transfer_edges),
        }

    def _recover_required_valence_corridor_lineage(
        self,
        gap,
        stability,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Recover a required one-frame state after an invalid-valence corridor."""
        if (
            self.frames is None
            or len(self.frames) == 0
            or self.traj is None
        ):
            return None

        record = gap["record"]
        target_entry = gap["entry"]
        target_key = target_entry["key"]
        focus = tuple(sorted({
            int(index)
            for index in target_entry.get("atom_indices", []) or []
        }))
        if not focus:
            return None

        target_frame = max(
            0,
            min(
                int(gap["consume_frame"]),
                len(self.frames) - 1,
            ),
        )
        target_molecule = self._exact_mapped_molecule(
            target_frame,
            focus,
        )
        if (
            target_molecule is None
            or not self._molecule_clean_eligible(target_molecule)
        ):
            return None
        observed_target = self._lineage_mol_entry(
            target_molecule,
            target_frame,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        if observed_target["key"] != target_key:
            return None

        # This exceptional recovery is only for a necessary one-frame state
        # directly consumed by the already detected downstream event.
        target_run_start = target_frame
        target_run_end = target_frame
        while target_run_start > 0:
            previous = self._exact_mapped_molecule(
                target_run_start - 1,
                focus,
            )
            if previous is None:
                break
            previous_entry = self._lineage_mol_entry(
                previous,
                target_run_start - 1,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if (
                not self._molecule_clean_eligible(previous)
                or previous_entry["key"] != target_key
            ):
                break
            target_run_start -= 1
        while target_run_end + 1 < len(self.frames):
            following = self._exact_mapped_molecule(
                target_run_end + 1,
                focus,
            )
            if following is None:
                break
            following_entry = self._lineage_mol_entry(
                following,
                target_run_end + 1,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if (
                not self._molecule_clean_eligible(following)
                or following_entry["key"] != target_key
            ):
                break
            target_run_end += 1
        if (
            target_run_start != target_frame
            or target_run_end != target_frame
            or int(
                record.get(
                    "Reactant_Frame",
                    record.get("Frame", target_frame),
                )
                or target_frame
            ) != target_frame
        ):
            return None

        target_formula = self._strip_surface_suffix(
            observed_target.get("formula", "")
        )
        if not target_formula:
            return None

        minimum_source_run = max(2, int(local_stability))
        maximum_corridor = max(200, 100 * max(1, int(stability)))
        lower_bound = max(0, target_frame - maximum_corridor)
        cursor = target_frame - 1
        source_molecule = None
        source_entry = None
        source_run_start = None
        source_run_end = None

        while cursor >= lower_bound:
            molecule = self._exact_mapped_molecule(cursor, focus)
            if (
                molecule is None
                or not self._molecule_clean_eligible(molecule)
            ):
                cursor -= 1
                continue
            entry = self._lineage_mol_entry(
                molecule,
                cursor,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if (
                self._strip_surface_suffix(entry.get("formula", ""))
                != target_formula
                or entry["key"] == target_key
            ):
                cursor -= 1
                continue

            run_end = cursor
            run_start = cursor
            while run_start > lower_bound:
                previous = self._exact_mapped_molecule(
                    run_start - 1,
                    focus,
                )
                if (
                    previous is None
                    or not self._molecule_clean_eligible(previous)
                ):
                    break
                previous_entry = self._lineage_mol_entry(
                    previous,
                    run_start - 1,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                if previous_entry["key"] != entry["key"]:
                    break
                run_start -= 1

            if run_end - run_start + 1 >= minimum_source_run:
                source_molecule = molecule
                source_entry = entry
                source_run_start = int(run_start)
                source_run_end = int(run_end)
                break
            cursor = run_start - 1

        if source_molecule is None:
            return None
        if source_run_end + 1 >= target_frame:
            return None

        invalid_corridor_frames = 0
        for frame_idx in range(source_run_end + 1, target_frame):
            container = self._unique_mapped_focus_container(
                frame_idx,
                focus,
            )
            if (
                container is None
                or self._molecule_clean_eligible(container)
            ):
                return None
            invalid_corridor_frames += 1
        if invalid_corridor_frames < 1:
            return None

        transfer = self._mapped_endpoint_transfer_centre(
            source_run_end,
            target_frame,
            focus,
            max_raw_changed_bonds=8,
        )
        if transfer is None:
            return None

        candidate = self._event_record(
            [source_molecule],
            [target_molecule],
            transition_frame=target_frame,
            reactant_frame=source_run_end,
            product_frame=target_frame,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        if candidate is None:
            return None

        validated_changed = set(transfer["validated_changed"])
        raw_changed = set(transfer["raw_changed"])
        candidate.update({
            "Frame": int(target_frame),
            "Reactant_Frame": int(source_run_end),
            "Product_Frame": int(target_frame),
            "Endpoint_Boundary_Reactant_Frame": int(source_run_end),
            "Endpoint_Boundary_Product_Frame": int(target_frame),
            "Representative_Reactant_Frame": int(source_run_end),
            "Representative_Product_Frame": int(target_frame),
            "Lineage_Recovered": True,
            "Recovery_Reason": "phantom_consumption_mass_gap",
            "Recovery_Mode": "valence_corridor_state_transfer",
            "Recovery_Target_Species": (
                target_entry.get("formula")
                or target_entry.get("smiles", "")
            ),
            "Recovery_Target_Family": target_entry.get("family", ""),
            "Recovery_Evidence": (
                "stable_clean_mapped_source;"
                "single_molecular_invalid_valence_corridor;"
                "one_frame_clean_target_immediately_consumed;"
                "unique_atom_mapped_endpoint_transfer"
            ),
            "Recovery_Boundary_Frame": int(target_frame),
            "Recovery_Original_Boundary_Frame": (
                self._candidate_original_frame(target_frame)
            ),

            "Recovery_Confidence": "inferred",
            "Valence_Corridor_Recovered": True,
            "Valence_Corridor_Start_Frame": int(source_run_end + 1),
            "Valence_Corridor_End_Frame": int(target_frame - 1),
            "Valence_Corridor_Length": int(invalid_corridor_frames),
            "Endpoint_Recovered": True,
            "Endpoint_Recovery_Attempted": True,
            "Endpoint_Recovery_Mode": (
                "required_one_frame_state_after_invalid_valence_corridor"
            ),
            "Endpoint_Recovery_Evidence": (
                "strict_atom_mapped_endpoint_transfer"
            ),
            "Endpoint_Reactant_Run_Length": int(
                source_run_end - source_run_start + 1
            ),
            "Endpoint_Product_Run_Length": 1,
            "Endpoint_Segmentation_Threshold": int(
                self._segmentation_local_stability
                or local_stability
            ),
            "Endpoint_Requested_Local_Stability": int(
                self._requested_local_stability
                or local_stability
            ),
            "Endpoint_Raw_Changed_Bond_Count": len(raw_changed),
            "Endpoint_Raw_Changed_Bonds": (
                self._format_changed_bond_edges(raw_changed)
            ),
            "Endpoint_Raw_Changed_Bond_Components": len(
                self._changed_bond_components_from_edges(raw_changed)
            ),
            "Endpoint_Validated_Changed_Bond_Count": len(
                validated_changed
            ),
            "Endpoint_Validated_Changed_Bonds": (
                self._format_changed_bond_edges(validated_changed)
            ),
            "Endpoint_Validated_Changed_Bond_Components": 1,
            "Endpoint_Spectator_Changed_Bonds": (
                self._format_changed_bond_edges(
                    transfer["spectator_changed"]
                )
            ),
            "Endpoint_Transfer_Atom_Count": 1,
            "Endpoint_Transfer_Atoms": str(
                transfer["transfer_atom"]
            ),
            "Endpoint_Component_Localization_Used": True,
            "Reaction_Centre_Connectivity": "changed_bond_connected",
            "Reaction_Centre_Bridge_Bond_Count": 0,
            "Reaction_Centre_Bridge_Bonds": "",
            "Changed_Bond_Count": len(validated_changed),
            "Changed_Bond_Components": 1,
            "Composite_Candidate": False,
            "Whole_Event_Fallback": False,
            "Concerted_Multi_Bond": False,
            "Elementary_Mode": "valence_corridor_transfer",
            "Elementary_Evidence": (
                "stable_clean_endpoints;atom_mapped;mass_balanced;"
                "single_connected_transfer_centre;"
                "no_clean_intermediate_in_invalid_valence_corridor"
            ),
            "Type": "Inferred Intramolecular Rearrangement",
            "Isomerization_Class": (
                "inferred_atom_mapped_intramolecular_transfer"
            ),
        })
        self._single_frame_recovery_stats[
            "valence_corridor_lineage_recoveries"
        ] += 1
        return candidate

    def _recover_one_lineage_gap(
        self,
        gap,
        stability,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
        require_family_formation=False,
    ):
        """Recover a missing mapped parent across adsorption-state oscillations."""
        if self.frames is None or len(self.frames) == 0:
            return None

        rec = gap["record"]
        target_entry = gap["entry"]
        target_key = target_entry["key"]
        target_family = target_entry.get("family") or self._lineage_family_key(
            target_entry.get("smiles", ""),
            target_entry.get("formula", ""),
        )
        consume_frame = max(
            0,
            min(int(gap["consume_frame"]), len(self.frames) - 1),
        )

        allowed_indices = set(int(i) for i in rec.get("Atom_Indices", []) or [])
        local_mols = self._local_mols(consume_frame, allowed_indices)
        target_mols = []

        for mol in local_mols:
            entry = self._lineage_mol_entry(
                mol,
                consume_frame,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if entry["key"] == target_key:
                target_mols.append(mol)

        # State classification near a surface can differ by one frame.  If the
        # exact state is absent, retain only mapped molecules of the same family
        # that intersect the event atoms.
        if not target_mols:
            for mol in local_mols:
                entry = self._lineage_mol_entry(
                    mol,
                    consume_frame,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                if entry.get("family") == target_family:
                    target_mols.append(mol)

        if not target_mols and not allowed_indices:
            for mol in self.frames[consume_frame]:
                entry = self._lineage_mol_entry(
                    mol,
                    consume_frame,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                if entry["key"] == target_key:
                    target_mols.append(mol)

        family_candidates = []
        exact_candidates = []
        family_evidence = {}

        for target_mol in target_mols:
            focus_indices = sorted(
                set(int(i) for i in target_mol.get("indices", []))
            )
            if not focus_indices:
                continue

            episode = self._find_persistent_family_start(
                focus_indices,
                target_family,
                consume_frame,
                stable_frames=max(1, int(stability)),
            )
            if episode and not episode.get("present_from_initial", False):
                boundary = int(episode["start_frame"])
                local_family_candidates = self._family_candidates_near_boundary(
                    focus_indices,
                    target_family,
                    boundary,
                    consume_frame,
                    stability,
                    local_stability,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                )
                for candidate in local_family_candidates:
                    family_candidates.append(candidate)
                    family_evidence[id(candidate)] = {
                        **episode,
                        "boundary_frame": boundary,
                    }

            if not require_family_formation:
                exact_candidates.extend(
                    self._exact_state_candidates_near_first_run(
                        focus_indices,
                        target_key,
                        consume_frame,
                        stability,
                        local_stability,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                    )
                )

        if require_family_formation:
            candidate_pool = family_candidates
            recovery_mode = "connectivity_family_formation"
        elif exact_candidates:
            candidate_pool = exact_candidates
            recovery_mode = "state_transition"
        else:
            candidate_pool = family_candidates
            recovery_mode = "connectivity_family_formation"

        if not candidate_pool:
            return None

        if recovery_mode == "connectivity_family_formation":
            candidate = min(
                candidate_pool,
                key=lambda r: (
                    abs(
                        int(r.get("Frame", 0) or 0)
                        - int(
                            family_evidence.get(id(r), {}).get(
                                "boundary_frame",
                                r.get("Frame", 0),
                            )
                        )
                    ),
                    int(r.get("Product_Frame", r.get("Frame", 0)) or 0),
                    len(r.get("Atom_Indices", []) or []),
                ),
            )
        else:
            candidate = max(
                candidate_pool,
                key=lambda r: (
                    int(r.get("Product_Frame", r.get("Frame", 0)) or 0),
                    int(r.get("Frame", 0) or 0),
                    -len(r.get("Atom_Indices", []) or []),
                ),
            )

        evidence = family_evidence.get(id(candidate), {})
        candidate = dict(candidate)
        candidate["Lineage_Recovered"] = True
        candidate["Recovery_Reason"] = "phantom_consumption_mass_gap"
        candidate["Recovery_Mode"] = recovery_mode
        candidate["Recovery_Target_Species"] = (
            target_entry["formula"] or target_entry["smiles"]
        )
        candidate["Recovery_Target_Family"] = target_family
        candidate["Recovery_Evidence"] = (
            "atom_mapped_stable_connectivity_boundary"
            if recovery_mode == "connectivity_family_formation"
            else "atom_mapped_state_transition"
        )

        endpoint_indices = candidate.get("Atom_Indices", [])
        reactant_run = self._local_signature_run_bounds(
            int(
                candidate.get(
                    "Reactant_Frame",
                    candidate.get("Frame", 0),
                )
                or 0
            ),
            endpoint_indices,
        )
        product_run = self._local_signature_run_bounds(
            int(
                candidate.get(
                    "Product_Frame",
                    candidate.get("Frame", 0),
                )
                or 0
            ),
            endpoint_indices,
        )
        if (
            self._single_frame_recovery_enabled
            and reactant_run is not None
            and product_run is not None
            and min(
                reactant_run["length"],
                product_run["length"],
            ) == 1
        ):
            candidate["Endpoint_Recovered"] = True
            candidate["Endpoint_Recovery_Mode"] = (
                "targeted_single_frame_lineage_recovery"
            )
            candidate["Endpoint_Recovery_Evidence"] = (
                "valid_atom_mapped_lineage_target_with_"
                "strict_prior_source_check"
            )
            candidate["Endpoint_Reactant_Run_Length"] = int(
                reactant_run["length"]
            )
            candidate["Endpoint_Product_Run_Length"] = int(
                product_run["length"]
            )
            candidate["Endpoint_Segmentation_Threshold"] = int(
                self._segmentation_local_stability
            )
            candidate["Endpoint_Requested_Local_Stability"] = int(
                self._requested_local_stability
            )
            self._single_frame_recovery_stats[
                "targeted_lineage_recoveries"
            ] += 1

        if recovery_mode == "connectivity_family_formation":
            boundary = int(evidence.get("boundary_frame", candidate.get("Frame", 0)))
            candidate["Recovery_Boundary_Frame"] = boundary
            candidate["Recovery_Original_Boundary_Frame"] = (
                self._candidate_original_frame(boundary)
            )
            candidate["Recovery_Family_Gap_Tolerance"] = int(
                evidence.get("gap_tolerance", max(0, int(stability) - 1))
            )
            candidate["Recovery_Short_Absence_Frames_Bridged"] = int(
                evidence.get("short_absence_frames", 0)
            )
            frame_error = abs(int(candidate.get("Frame", 0) or 0) - boundary)
            candidate["Recovery_Confidence"] = (
                "high" if frame_error <= max(2, int(local_stability)) else "medium"
            )
        else:
            candidate["Recovery_Boundary_Frame"] = int(
                candidate.get("Frame", 0) or 0
            )
            candidate["Recovery_Original_Boundary_Frame"] = (
                self._candidate_original_frame(
                    candidate["Recovery_Boundary_Frame"]
                )
            )
            candidate["Recovery_Family_Gap_Tolerance"] = ""
            candidate["Recovery_Short_Absence_Frames_Bridged"] = ""
            candidate["Recovery_Confidence"] = "high"

        return candidate

    def _lineage_reactants_have_prior_sources(
        self,
        candidate,
        existing_records,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Require every candidate reactant to be strictly reachable."""
        initial_entries = self._initial_lineage_entries(
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        candidate_react_frame = int(
            candidate.get("Reactant_Frame", candidate.get("Frame", 0)) or 0
        )
        reachable = self._reachable_lineage_state_keys(
            existing_records,
            initial_entries,
            cutoff_frame=candidate_react_frame,
        )
        exact_missing = []
        for entry in self._lineage_side_entries(candidate, "reactants"):
            if entry["key"] in reachable:
                continue
            exact_missing.append(entry)

        if exact_missing:
            candidate["Lineage_Recovery_Rejected"] = True
            candidate["Lineage_Recovery_Rejection_Reason"] = (
                "candidate_exact_atom_instances_not_available:"
                + " + ".join(
                e["formula"] or e["smiles"] for e in exact_missing
                )
            )
            return False
        return True

    def _recover_missing_lineage(
        self,
        records,
        stability,
        local_stability,
        is_sub_mask=None,
        pre_cutoffs=None,
        max_rounds=12,
    ):
        """Iteratively recover family formation and required state transitions."""
        working = list(records)
        recovered = []
        existing = {
            self._lineage_event_key(r, include_frame=True) for r in working
        }

        for _round in range(max(1, int(max_rounds))):
            gaps, initial_entries = self._lineage_gaps(
                working,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
                annotate=False,
            )
            if not gaps:
                break

            earliest = {}
            for gap in gaps:
                key = gap["entry"]["key"]
                old = earliest.get(key)
                if old is None or gap["consume_frame"] < old["consume_frame"]:
                    earliest[key] = gap

            added_this_round = 0
            for gap in sorted(
                earliest.values(), key=lambda g: g["consume_frame"]
            ):
                target_entry = gap["entry"]
                target_family = target_entry.get("family") or self._lineage_family_key(
                    target_entry.get("smiles", ""),
                    target_entry.get("formula", ""),
                )
                require_family_formation = not self._family_has_earlier_source(
                    target_family,
                    gap["consume_frame"],
                    working,
                    initial_entries,
                )

                candidate = (
                    self._recover_required_valence_corridor_lineage(
                        gap,
                        stability=stability,
                        local_stability=local_stability,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                    )
                )
                if candidate is None:
                    candidate = self._recover_one_lineage_gap(
                        gap,
                        stability=stability,
                        local_stability=local_stability,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                        require_family_formation=(
                            require_family_formation
                        ),
                    )
                if candidate is None:
                    continue

                if not self._lineage_reactants_have_prior_sources(
                    candidate,
                    working,
                    is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                ):
                    continue

                key = self._lineage_event_key(candidate, include_frame=True)
                if key in existing:
                    continue

                existing.add(key)
                working.append(candidate)
                recovered.append(candidate)
                added_this_round += 1

            if added_this_round == 0:
                break

        return recovered

    def _prior_composite_source_for_gap(
        self,
        entry,
        consume_frame,
    ):
        """Return the latest earlier composite audit row producing ``entry``."""
        candidates = []
        for row in self.unresolved_components:
            if str(row.get("Reason", "")) != (
                "non_elementary_after_local_split"
            ):
                continue
            if not _truthy_value(
                row.get("Composite_Candidate", False)
            ):
                continue

            product_frame = int(
                row.get(
                    "Product_Frame",
                    row.get("Frame", -1),
                )
                or -1
            )
            if product_frame > int(consume_frame):
                continue

            product_entries = self._lineage_side_entries(
                row,
                "products",
            )
            if any(
                product_entry.get("key") == entry.get("key")
                for product_entry in product_entries
            ):
                candidates.append((
                    product_frame,
                    str(row.get("Unresolved_ID", "")),
                    row,
                ))

        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    def _check_mass_continuity(
        self,
        records,
        is_sub_mask=None,
        pre_cutoffs=None,
    ):
        """Audit state-aware lineage after recovery and annotate every event."""
        if not records or self.frames is None or len(self.frames) == 0:
            return 0

        gaps, _initial = self._lineage_gaps(
            records,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
            annotate=True,
        )

        seen = set()
        for gap in gaps:
            rec = gap["record"]
            entry = gap["entry"]
            audit_key = (
                int(rec.get("Frame", 0) or 0),
                entry["key"],
                tuple(sorted(int(i) for i in rec.get("Atom_Indices", []) or [])),
            )
            if audit_key in seen:
                continue
            seen.add(audit_key)

            logger.debug(
                "State-aware lineage gap: species %s | %s is consumed at frame %s "
                "but was never produced earlier and is not an initial state.",
                entry["formula"], entry["smiles"], rec.get("Frame")
            )

            prior_composite = self._prior_composite_source_for_gap(
                entry,
                gap["consume_frame"],
            )
            explanation = ""
            if prior_composite is not None:
                explanation = (
                    " A prior stable, atom-mapped composite observation "
                    f"({prior_composite.get('Unresolved_ID', '')}) produces "
                    "this state, but it remains an audit explanation rather "
                    "than an elementary or automatically reachable source."
                )

            audit_row = self._record_unresolved_component(
                reason="phantom_consumption_mass_gap",
                transition_index=int(rec.get("Frame", -1) or -1),
                prev={"start": -1, "end": -1},
                curr={"start": -1, "end": -1},
                sub_r=[],
                sub_p=[],
                focus_indices=rec.get("Atom_Indices", []),
                reactant_frame=rec.get("Reactant_Frame"),
                product_frame=rec.get("Product_Frame"),
                note=(
                    f"State-aware species '{entry['formula']}' "
                    f"(SMILES: {entry['smiles']}) is consumed without an initial "
                    f"or earlier detected elementary source in "
                    f"{rec.get('Reactants_Formula','')} -> "
                    f"{rec.get('Products_Formula','')}. "
                    f"Automatic atom-mapped backward recovery found no safely "
                    f"accepted elementary formation step."
                    f"{explanation}"
                ),
                record_override=rec,
            )
            if prior_composite is not None:
                audit_row["Explained_By_Unresolved_ID"] = str(
                    prior_composite.get("Unresolved_ID", "")
                )
                audit_row["Source_Explanation"] = (
                    "prior_unreachable_or_non_elementary_composite_candidate"
                )

        return len(seen)

    def detect(
        self,
        stability=5,
        local_stability=None,
        single_frame_recovery=True,
        n_workers=1,
    ):

        detect_started = time.perf_counter()
        detect_phase_timings = {}
        stability = max(1, int(stability))
        policy = self._resolve_local_stability_policy(
            stability,
            local_stability,
            single_frame_recovery=single_frame_recovery,
        )
        requested_local_stability = policy["requested"]
        segmentation_local_stability = policy["segmentation"]
        scan_window = max(40, 8 * stability)
        self.recrossing_window = int(scan_window)
        self.unresolved_components = []
        self._unresolved_component_key_to_row = {}
        self._unresolved_duplicate_count = 0
        self._single_frame_recovery_enabled = bool(
            single_frame_recovery
        )
        self._requested_local_stability = int(
            requested_local_stability
        )
        self._segmentation_local_stability = int(
            segmentation_local_stability
        )
        self._surface_minimum_dwell = int(
            segmentation_local_stability
        )
        self._recovery_scan_local_stability = int(
            policy["recovery_scan"]
        )
        self._effective_scan_window = int(scan_window)
        self._persistent_topology_recovery_scope = (
            "all_dwell_validated_boundaries"
        )
        self._single_frame_recovery_stats = Counter()

        # Clear per-run caches so repeated calls on the same detector.
        self._reset_runtime_caches()

        if self.traj is not None:
            try:
                self._traj_obj_to_frame = {id(atoms): i for i, atoms in enumerate(self.traj)}
            except Exception:
                self._traj_obj_to_frame = {}

        print(
            "🔍 Detecting reaction events "
            "[contextual-boundary + terminal-state lineage repair v9] "
            f"(global_stability={stability}, "
            f"local_dwell={segmentation_local_stability}, "
            f"single_frame_recovery="
            f"{'on' if self._single_frame_recovery_enabled else 'off'})..."
        )
        if policy["normalized_single_frame_request"]:
            print(
                "   ℹ️ local_stability=1 selects strict one-frame "
                "recovery; ordinary local dwell remains "
                f"{segmentation_local_stability} frames."
            )

        states = []
        is_sub_mask = None
        if (self.sub_elements or self.sub_indices) and self.traj is not None and len(self.traj) > 0:
            symbols = self.traj[0].get_chemical_symbols()
            is_sub_mask = [
                (idx in self.sub_indices) or (sym in self.sub_elements)
                for idx, sym in enumerate(symbols)
            ]

        pre_cutoffs = None

        if self.traj is not None and len(self.traj) > 0:
            try:
                pre_cutoffs = natural_cutoffs(self.traj[0], mult=self.adsorption_scale)
            except Exception:
                pre_cutoffs = None

            try:
                self._bond_cutoffs = natural_cutoffs(self.traj[0], mult=self.bond_scale)
            except Exception:
                self._bond_cutoffs = None

            if is_sub_mask is not None and pre_cutoffs is not None:
                try:
                    self._ads_pair_cutoffs = self._make_adsorption_pair_cutoffs(
                        self.traj[0],
                        is_sub_mask,
                        pre_cutoffs
                    )
                except Exception:
                    self._ads_pair_cutoffs = None
            else:
                self._ads_pair_cutoffs = None

        topology_boundaries = []
        surface_boundaries = []
        previous_frame_molecules = None
        previous_memberships = None
        previous_surface_states = None
        surface_state_runs = {}

        phase_started = time.perf_counter()
        if (
            is_sub_mask is not None
            and pre_cutoffs is not None
            and self.traj is not None
        ):
            self._precompute_adsorption_indices(
                is_sub_mask,
                pre_cutoffs,
                n_workers=n_workers,
            )
        detect_phase_timings["adsorption_precompute"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        print("   - Pre-computing trajectory states for all frames...")
        for i, frame in tqdm(enumerate(self.frames), total=len(self.frames), desc="   Mapping States"):
            current_memberships = self._atom_instance_memberships(frame)
            if (
                previous_frame_molecules is not None
                and previous_memberships is not None
            ):
                for component in self._topology_boundary_components(
                    previous_frame_molecules,
                    frame,
                    reactant_memberships=previous_memberships,
                    product_memberships=current_memberships,
                ):
                    topology_boundaries.append({
                        "frame": int(i),
                        "focus_indices": component,
                    })
            previous_frame_molecules = frame
            previous_memberships = current_memberships

            if is_sub_mask is not None and pre_cutoffs is not None and self.traj is not None:
                atoms = self.traj[i]
                ads_idx = self._adsorbed_indices_for_atoms(
                    atoms, frame_idx=i, is_sub_mask=is_sub_mask, pre_cutoffs=pre_cutoffs
                )
                self._index_frame_presence(
                    i,
                    frame,
                    adsorbed_indices=ads_idx,
                    surface_system=True,
                )

                state_list = []
                current_surface_states = {}
                for m in frame:
                    is_ads = any(int(idx) in ads_idx for idx in m.get('indices', []))
                    instance_key = self._molecule_instance_key(m)
                    current_surface_states[instance_key] = bool(is_ads)
                    state_list.append((
                        self._molecule_identity(m),
                        self._molecule_atom_group(m),
                        '*' if is_ads else '(g)',
                    ))
                states.append(tuple(sorted(state_list)))
                disappeared = (
                    set(surface_state_runs)
                    - set(current_surface_states)
                )
                for instance_key in disappeared:
                    surface_state_runs.pop(instance_key, None)

                for instance_key, current_state in (
                    current_surface_states.items()
                ):
                    active_run = surface_state_runs.get(instance_key)
                    if (
                        active_run is not None
                        and active_run["state"] == current_state
                    ):
                        active_run["length"] += 1
                        if active_run.get("candidate") is not None:
                            active_run["candidate"][
                                "product_run_length"
                            ] = int(active_run["length"])
                        continue

                    new_candidate = None
                    if active_run is not None:
                        _identity, atom_group = instance_key
                        if atom_group:
                            new_candidate = {
                                "frame": int(i),
                                "focus_indices": tuple(atom_group),
                                "reactant_run_length": int(
                                    active_run["length"]
                                ),
                                "product_run_length": 1,
                                "reactant_adsorbed": bool(
                                    active_run["state"]
                                ),
                                "product_adsorbed": bool(
                                    current_state
                                ),
                            }
                            surface_boundaries.append(new_candidate)

                    surface_state_runs[instance_key] = {
                        "state": bool(current_state),
                        "length": 1,
                        "candidate": new_candidate,
                    }
                previous_surface_states = current_surface_states
            else:
                self._index_frame_presence(i, frame)
                states.append(tuple(sorted(
                    self._molecule_instance_key(m)
                    for m in frame
                )))
                previous_surface_states = None

        self._presence_index_complete = True
        lineage_interval_count = sum(
            len(bounds) // 2
            for bounds in self._lineage_family_intervals.values()
        )
        species_interval_count = sum(
            len(entry.get("intervals", ())) // 2
            for entry in self._species_presence_index.values()
        )
        detect_phase_timings["state_mapping"] = (
            time.perf_counter() - phase_started
        )

        n_frames = len(states)
        if n_frames == 0:
            return pd.DataFrame()

        # Build true stable runs; frame 0 is not accepted unless it is stable.
        stable_timeline = []
        i = 0
        while i < n_frames:
            j = i + 1
            while j < n_frames and states[j] == states[i]:
                j += 1
            if (j - i) >= stability:
                if not stable_timeline or states[i] != stable_timeline[-1]['state']:
                    stable_timeline.append({
                        'start': i,
                        'end': j - 1,
                        'state': states[i],
                        'mols': self.frames[i],
                    })
                else:
                    stable_timeline[-1]['end'] = j - 1
            i = j

        print(f"   Identified {len(stable_timeline)} stable plateaus.")        
        del states
        self._clear_heavy_scan_caches()
        all_records = []
        local_used = 0
        fallback_used = 0
        fallback_rejected = 0
        dropped_complex = 0

        transition_count = max(0, len(stable_timeline) - 1)
        transition_iter = tqdm(
            range(transition_count),
            total=transition_count,
            desc="   Scanning Transitions",
            unit="trans",
            disable=(transition_count == 0),
        )

        cache_clear_interval = 200

        phase_started = time.perf_counter()
        for k in transition_iter:
            if k > 0 and k % cache_clear_interval == 0:
                self._clear_heavy_scan_caches()

            prev = stable_timeline[k]
            curr = stable_timeline[k + 1]
            curr_start = int(curr['start'])
            prev_start = int(prev['start'])
            next_start = stable_timeline[k + 2]['start'] if k + 2 < len(stable_timeline) else n_frames - 1

            for sub_r, sub_p in self._disentangle_events(prev['mols'], curr['mols'], prev_start, curr_start):
                focus_indices = sorted(set(idx for m in list(sub_r) + list(sub_p) for idx in m.get('indices', [])))
                if not focus_indices:
                    continue

                # Scan only near the global state transition instead of re-scanning the entire old/product plateaus.
                scan_start = max(0, prev_start, curr_start - scan_window)
                scan_end = min(n_frames - 1, int(next_start) - 1, curr_start + scan_window)
                scan_end = max(scan_end, min(n_frames - 1, curr_start + stability))
                local_records = self._scan_local_elementary_events(
                    scan_start, scan_end, focus_indices, stability,
                is_sub_mask=is_sub_mask,
                    pre_cutoffs=pre_cutoffs,
                local_stability=segmentation_local_stability
                )

                endpoint_records = (
                    self._strict_single_frame_endpoint_records(
                        sub_r,
                        sub_p,
                        scan_start,
                        scan_end,
                        focus_indices,
                        stability,
                        segmentation_local_stability,
                        curr_start,
                        primary_records=local_records,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                    )
                )
                if endpoint_records:
                    all_records.extend(endpoint_records)
                    continue

                if local_records:
                    local_records, rejected = (
                        self._filter_conservative_endpoint_records(
                            local_records,
                            max_changed_bonds=3,
                            max_raw_changed_bonds=8,
                        )
                    )
                    fallback_rejected += rejected

                if local_records:
                    local_used += len(local_records)
                    all_records.extend(local_records)
                else:
                    # Strict endpoint fallback: both local endpoints must be stable
                    # near the transition, and only simple one-center changes pass.
                    react_sig = self._local_signature_at(prev_start, focus_indices)
                    prod_sig = self._local_signature_at(curr_start, focus_indices)

                    f_react = self._find_stable_local_endpoint(
                        focus_indices, react_sig,
                        max(prev_start, curr_start - scan_window),
                        curr_start - 1, stability, prefer='last'
                    )
                    f_prod = self._find_stable_local_endpoint(
                        focus_indices, prod_sig,
                        curr_start,
                        min(int(next_start) - 1, curr_start + scan_window),
                        stability, prefer='first'
                    )

                    if f_react is None or f_prod is None or f_react >= f_prod:
                        dropped_complex += 1
                        self._record_unresolved_component(
                            reason="no_stable_endpoint_for_conservative_fallback",
                            transition_index=k + 1,
                            prev=prev,
                            curr=curr,
                            sub_r=sub_r,
                            sub_p=sub_p,
                            focus_indices=focus_indices,
                            scan_start=scan_start,
                            scan_end=scan_end,
                            reactant_frame=f_react if f_react is not None else prev_start,
                            product_frame=f_prod if f_prod is not None else curr_start,
                            is_sub_mask=is_sub_mask,
                            pre_cutoffs=pre_cutoffs,
                            note="Local scan produced no accepted event, and stable local endpoint frames could not be selected safely."
                        )
                        continue

                    fallback_records = self._split_by_changed_bond_centers(
                        sub_r, sub_p, f_react, f_prod,
                        is_sub_mask=is_sub_mask,
                        pre_cutoffs=pre_cutoffs,
                        transition_frame=curr_start
                    )
                    fallback_records, rejected = self._filter_conservative_endpoint_records(
                        fallback_records,
                        max_changed_bonds=3,
                        max_raw_changed_bonds=8,
                    )
                    fallback_rejected += rejected

                    if fallback_records:
                        fallback_used += len(fallback_records)
                        all_records.extend(fallback_records)
                    else:
                        dropped_complex += 1
                        self._record_unresolved_component(
                            reason="endpoint_fallback_rejected_or_empty",
                            transition_index=k + 1,
                            prev=prev,
                            curr=curr,
                            sub_r=sub_r,
                            sub_p=sub_p,
                            focus_indices=focus_indices,
                            scan_start=scan_start,
                            scan_end=scan_end,
                            reactant_frame=f_react,
                            product_frame=f_prod,
                            is_sub_mask=is_sub_mask,
                            pre_cutoffs=pre_cutoffs,
                            endpoint_rejected_count=rejected,
                            note="Local scan produced no accepted event, and conservative endpoint fallback produced no accepted mass-balanced elementary-event candidate."
                        )

            if k % 10 == 0 or k == transition_count - 1:
                transition_iter.set_postfix(raw=len(all_records), refresh=False)

        detect_phase_timings["transition_scan"] = (
            time.perf_counter() - phase_started
        )

        # Completeness safety net: every atom-mapped topology boundary that
        # satisfies ordinary local dwell is eligible for persistent recovery.
        persistent_boundary_seeds = (
            self._persistent_topology_boundary_seeds(
                topology_boundaries,
                segmentation_local_stability,
            )
        )

        closed_topology_boundaries = (
            self._expand_scoped_topology_boundaries(
                topology_boundaries,
                persistent_boundary_seeds,
                recrossing_window=scan_window,
            )
        )
        topology_companion_count = sum(
            bool(candidate.get("closure_companion", False))
            for candidate in closed_topology_boundaries.values()
        )

        phase_started = time.perf_counter()
        atom_local_records = self._recover_persistent_atom_local_events(
            list(closed_topology_boundaries.values()),
            stable_frames=stability,
            local_stability=segmentation_local_stability,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        detect_phase_timings["topology_recovery"] = (
            time.perf_counter() - phase_started
        )
        if atom_local_records:
            all_records.extend(atom_local_records)
            print(
                "   🧬 Atom-local topology recoveries retained: "
                f"{len(atom_local_records)}."
            )

        invalid_raw_exit_seeds = self._invalid_raw_exit_boundary_seeds(
            all_records
        )
        stable_exit_boundaries = list(closed_topology_boundaries.values())
        stable_exit_boundaries.extend(invalid_raw_exit_seeds)
        stable_exit_records = (
            self._recover_stable_exits_from_ineligible_topology(
                stable_exit_boundaries,
                segmentation_local_stability,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
        )
        if stable_exit_records:
            all_records.extend(stable_exit_records)
            print(
                "   🧭 Validated corridor-net bridges retained: "
                f"{len(stable_exit_records)}."
            )
        if invalid_raw_exit_seeds:
            print(
                "   🧩 Invalid Raw endpoint fallback seeds considered: "
                f"{len(invalid_raw_exit_seeds)}."
            )

        phase_started = time.perf_counter()
        atom_local_surface_records = (
            self._recover_persistent_atom_local_surface_events(
                surface_boundaries,
                local_stability=segmentation_local_stability,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
        )
        detect_phase_timings["surface_recovery"] = (
            time.perf_counter() - phase_started
        )
        if atom_local_surface_records:
            all_records.extend(atom_local_surface_records)
            print(
                "   ⚓ Atom-local surface-state recoveries retained: "
                f"{len(atom_local_surface_records)}."
            )

        self._clear_heavy_scan_caches()

        phase_started = time.perf_counter()
        all_records = self._dedupe_records(all_records)
        all_records = self._refine_topology_ineligible_event_corridors(
            all_records,
            segmentation_local_stability,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        detect_phase_timings["corridor_refinement"] = (
            time.perf_counter() - phase_started
        )

        elementary_seed = [
            r for r in all_records
            if self._is_elementary_stoichiometry(r)
            and not _truthy_value(r.get("Composite_Candidate", False))
            and bool(r.get("Clean_Eligible", True))
        ]
        print("   ⏳ Recovering atom-mapped lineage...")
        phase_started = time.perf_counter()
        recovered_records = self._recover_missing_lineage(
            elementary_seed,
            stability=stability,
            local_stability=segmentation_local_stability,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        detect_phase_timings["lineage_recovery"] = (
            time.perf_counter() - phase_started
        )
        if recovered_records:
            all_records = self._dedupe_records(all_records + recovered_records)
            recovery_modes = Counter(
                str(r.get("Recovery_Mode", "") or "unspecified")
                for r in recovered_records
            )
            print(
                "   🔗 Atom-mapped lineage recoveries retained: "
                f"{len(recovered_records)} "
                f"({detect_phase_timings['lineage_recovery']:.1f}s)."
            )
        else:
            print(
                "   🔗 Lineage recovery completed: no additional safe step "
                f"({detect_phase_timings['lineage_recovery']:.1f}s)."
            )

        late_detected_boundary_seeds = (
            self._detected_event_topology_boundary_seeds(
                topology_boundaries,
                all_records,
            )
        )
        late_detected_boundary_seeds = {
            key: candidate
            for key, candidate in late_detected_boundary_seeds.items()
            if key not in closed_topology_boundaries
        }
        late_closed_topology_boundaries = (
            self._expand_scoped_topology_boundaries(
                topology_boundaries,
                late_detected_boundary_seeds,
                recrossing_window=scan_window,
            )
            if late_detected_boundary_seeds
            else {}
        )
        late_closed_topology_boundaries = {
            key: candidate
            for key, candidate in late_closed_topology_boundaries.items()
            if key not in closed_topology_boundaries
        }
        phase_started = time.perf_counter()
        late_atom_local_records = (
            self._recover_persistent_atom_local_events(
                list(late_closed_topology_boundaries.values()),
                stable_frames=stability,
                local_stability=segmentation_local_stability,
                is_sub_mask=is_sub_mask,
                pre_cutoffs=pre_cutoffs,
            )
            if late_closed_topology_boundaries
            else []
        )
        detect_phase_timings["late_topology_closure"] = (
            time.perf_counter() - phase_started
        )
        if late_atom_local_records:
            all_records = self._dedupe_records(
                all_records + late_atom_local_records
            )
            late_companion_count = sum(
                bool(candidate.get("closure_companion", False))
                for candidate in late_closed_topology_boundaries.values()
            )
            print(
                "   🧬 Post-refinement topology recoveries retained: "
                f"{len(late_atom_local_records)}."
            )

        phase_started = time.perf_counter()
        corridor_endpoint_recoveries = int(
            self._single_frame_recovery_stats.get(
                "corridor_endpoint_refinements",
                0,
            )
        )
        direct_endpoint_recoveries = int(
            self._single_frame_recovery_stats.get(
                "direct_endpoint_refinements",
                0,
            )
        )
        targeted_endpoint_recoveries = int(
            self._single_frame_recovery_stats.get(
                "targeted_lineage_recoveries",
                0,
            )
        )
        valence_corridor_lineage_recoveries = int(
            self._single_frame_recovery_stats.get(
                "valence_corridor_lineage_recoveries",
                0,
            )
        )
        surface_state_insufficient_dwell_rejections = int(
            self._single_frame_recovery_stats.get(
                "surface_state_insufficient_dwell_rejections",
                0,
            )
        )
        if (
            corridor_endpoint_recoveries
            or direct_endpoint_recoveries
            or targeted_endpoint_recoveries
            or valence_corridor_lineage_recoveries
        ):
            print(
                "   🪡 Strict one-frame recoveries retained: "
                f"{corridor_endpoint_recoveries + direct_endpoint_recoveries + targeted_endpoint_recoveries + valence_corridor_lineage_recoveries}."
            )
        if surface_state_insufficient_dwell_rejections:
            print(
                "   🌊 Rejected surface-state transitions with insufficient "
                "endpoint dwell: "
                f"{surface_state_insufficient_dwell_rejections}."
            )

        whole_event_fallback_count = sum(
            _truthy_value(record.get("Whole_Event_Fallback", False))
            for record in all_records
        )
        if whole_event_fallback_count:
            print(
                "   🧩 Whole-event composite fallbacks retained: "
                f"{whole_event_fallback_count}."
            )

        # Mark non-elementary events and record topology ambiguity once. Raw
        # retains these observations; strict Clean excludes them by default.
        for rec in all_records:
            # Recovery paths do not all attach provenance at the same stage.
            # Normalize surface dwell evidence before DataFrame construction so
            # Clean never has to infer recovery from a non-empty/NaN mode.
            self._ensure_surface_endpoint_evidence(
                rec,
                minimum_dwell=self._resolved_surface_minimum_dwell(),
            )
            rec.setdefault("Topology_Validated", True)
            rec.setdefault("Topology_Evidence", "legacy_or_validated")
            rec.setdefault("Valence_Warning", "")
            rec.setdefault("Isomerization_Class", "")
            rec.setdefault(
                "Reactants_Identity",
                rec.get("Reactants_SMILES", ""),
            )
            rec.setdefault(
                "Products_Identity",
                rec.get("Products_SMILES", ""),
            )
            rec.setdefault("Clean_Eligible", bool(rec["Topology_Validated"]))

            recovery_mode = str(rec.get("Recovery_Mode", "") or "").strip()
            try:
                product_dwell = int(
                    rec.get("Endpoint_Product_Run_Length", 0) or 0
                )
            except (TypeError, ValueError):
                product_dwell = 0
            if (
                recovery_mode == "valence_corridor_state_transfer"
                and product_dwell <= 1
            ):
                rec["Clean_Eligible"] = False
                rec["Is_Elementary"] = False
                rec["Recovery_Promotion_Status"] = "audit_only"
                rec["Recovery_Promotion_Reason"] = (
                    "single_frame_valence_corridor_endpoint"
                )
                self._record_unresolved_component(
                    reason="single_frame_valence_corridor_recovery",
                    transition_index=int(rec.get("Frame", -1)),
                    prev={
                        "start": rec.get("Reactant_Frame", -1),
                        "end": rec.get("Reactant_Frame", -1),
                    },
                    curr={
                        "start": rec.get("Product_Frame", -1),
                        "end": rec.get("Product_Frame", -1),
                    },
                    sub_r=[],
                    sub_p=[],
                    focus_indices=rec.get("Atom_Indices", []),
                    reactant_frame=rec.get("Reactant_Frame"),
                    product_frame=rec.get("Product_Frame"),
                    note=(
                        "Atom-mapped valence-corridor transfer has only a "
                        "one-frame clean product endpoint. Retained for Raw "
                        "lineage audit, but excluded from Clean/Pathway "
                        "chemical events until a stable endpoint is observed."
                    ),
                    record_override=rec,
                )
                continue

            if not bool(rec.get("Clean_Eligible", True)):
                rec["Clean_Eligible"] = False
                self._record_unresolved_component(
                    reason="valence_ambiguous_topology",
                    transition_index=int(rec.get("Frame", -1)),
                    prev={
                        "start": rec.get("Reactant_Frame", -1),
                        "end": rec.get("Reactant_Frame", -1),
                    },
                    curr={
                        "start": rec.get("Product_Frame", -1),
                        "end": rec.get("Product_Frame", -1),
                    },
                    sub_r=[],
                    sub_p=[],
                    focus_indices=rec.get("Atom_Indices", []),
                    reactant_frame=rec.get("Reactant_Frame"),
                    product_frame=rec.get("Product_Frame"),
                    note=(
                        "Observed element-labelled connectivity could not be "
                        "validated as a chemically sanitized species. The event "
                        "is retained in Raw and Unresolved, but excluded from "
                        "strict Clean/Pathway unless explicitly allowed. "
                        f"{rec.get('Valence_Warning', '')}"
                    ),
                    record_override=rec,
                )
            rec.setdefault("Lineage_Recovered", False)
            rec.setdefault("Recovery_Reason", "")
            rec.setdefault("Recovery_Confidence", "")
            rec.setdefault("Recovery_Mode", "")
            rec.setdefault("Recovery_Target_Species", "")
            rec.setdefault("Recovery_Target_Family", "")
            rec.setdefault("Recovery_Evidence", "")
            rec.setdefault("Recovery_Boundary_Frame", "")
            rec.setdefault("Recovery_Original_Boundary_Frame", "")
            rec.setdefault("Recovery_Family_Gap_Tolerance", "")
            rec.setdefault("Recovery_Short_Absence_Frames_Bridged", "")
            rec.setdefault("Recovery_Deferred_State_Parents", "")

            r_count, p_count = self._elementary_side_counts(rec)
            if (
                _truthy_value(rec.get("Composite_Candidate", False))
                or not (1 <= r_count <= 3 and 1 <= p_count <= 3)
            ):
                rec["Is_Elementary"] = False
                self._record_unresolved_component(
                    reason="non_elementary_after_local_split",
                    transition_index=int(rec.get("Frame", -1)),
                    prev={
                        "start": rec.get("Reactant_Frame", -1),
                        "end": rec.get("Reactant_Frame", -1),
                    },
                    curr={
                        "start": rec.get("Product_Frame", -1),
                        "end": rec.get("Product_Frame", -1),
                    },
                    sub_r=[],
                    sub_p=[],
                    focus_indices=rec.get("Atom_Indices", []),
                    reactant_frame=rec.get("Reactant_Frame"),
                    product_frame=rec.get("Product_Frame"),
                    note=(
                        "Composite event not resolved into independently "
                        "mass-balanced elementary steps after local time-splitting: "
                        f"{rec.get('Reactants_SMILES','')} -> "
                        f"{rec.get('Products_SMILES','')}. "
                        f"whole_event_fallback={rec.get('Whole_Event_Fallback', False)}; "
                        f"changed_bonds={rec.get('Changed_Bond_Count', '')}; "
                        f"changed_components={rec.get('Changed_Bond_Components', '')}."
                    ),
                    record_override=rec,
                )
            else:
                rec["Is_Elementary"] = True
                if _truthy_value(
                    rec.get("Atom_Mapping_Exchange", False)
                ):
                    rec["Type"] = "Atom-Mapping Exchange"
                    rec["Elementary_Mode"] = "atom_mapping_exchange"
                    rec.setdefault(
                        "Elementary_Evidence",
                        "chemically_identical_endpoints;"
                        "atom_instance_transfer;atom_union_conserved;"
                        "stable_endpoints;single_connected_reaction_centre",
                    )
                    rec.setdefault("Concerted_Multi_Bond", False)
                elif r_count == 3 or p_count == 3:
                    rec.setdefault("Concerted_Multi_Bond", True)
                    rec.setdefault(
                        "Elementary_Mode",
                        "concerted_multi_bond",
                    )
                    rec.setdefault(
                        "Elementary_Evidence",
                        "stable_endpoints;atom_mapped;mass_balanced;"
                        "single_connected_reaction_centre;"
                        "no_resolved_stable_intermediate",
                    )
                else:
                    rec.setdefault("Concerted_Multi_Bond", False)
                    rec.setdefault("Elementary_Mode", "standard")

            # Corridor candidates are generated before this final elementary audit.
            if str(rec.get("Elementary_Mode", "") or "").strip() == (
                "observed_corridor_net"
            ):
                rec["Observed_Corridor_Net_Elementary"] = bool(
                    _truthy_value(rec.get("Is_Elementary", False))
                    and _truthy_value(rec.get("Topology_Validated", False))
                    and _is_missing_scalar(rec.get("Valence_Warning", ""))
                    and self._is_mass_balanced_formula_text(
                        rec.get("Reactants_Formula", ""),
                        rec.get("Products_Formula", ""),
                    )
                    and self._is_elementary_stoichiometry(rec)
                )

        # Final state-aware audit.  This also writes lineage annotations into
        # raw/clean/pathway tables for downstream integrity checks.
        elementary_only = [
            r for r in all_records
            if r.get("Is_Elementary", True)
            and bool(r.get("Clean_Eligible", True))
        ]
        phantom_count = self._check_mass_continuity(
            elementary_only,
            is_sub_mask=is_sub_mask,
            pre_cutoffs=pre_cutoffs,
        )
        if phantom_count:
            print(
                f"   ⚠️ State-aware lineage: {phantom_count} unresolved source "
                f"gap(s) remain after automatic recovery. See the unresolved "
                f"audit table."
            )
        else:
            print("   ✅ State-aware lineage check passed.")

        for idx, rec in enumerate(all_records, 1):
            raw_id = f"RXN-{idx:04d}"
            rec['Reaction_ID'] = raw_id
            rec['Raw_Reaction_ID'] = raw_id
            if _truthy_value(
                rec.get("Atom_Mapping_Exchange", False)
            ):
                rec["Raw_Candidate_Status"] = (
                    "validated_atom_mapping_exchange"
                )
            elif str(rec.get("Recovery_Promotion_Status", "") or "") == (
                "audit_only"
            ):
                rec["Raw_Candidate_Status"] = (
                    "inferred_valence_corridor_recovery"
                )
            elif not bool(rec.get("Clean_Eligible", True)):
                rec["Raw_Candidate_Status"] = "audit_only_topology_ambiguous"
            elif not bool(rec.get("Is_Elementary", True)):
                rec["Raw_Candidate_Status"] = "validated_composite_candidate"
            elif bool(rec.get("Lineage_Complete", True)):
                rec["Raw_Candidate_Status"] = "validated_elementary_candidate"
            else:
                rec["Raw_Candidate_Status"] = (
                    "elementary_candidate_with_lineage_gap"
                )

        # Link unresolved audit rows back to their exact Raw event whenever a
        # one-to-one state-aware event match exists.
        def _provenance_indices(value):
            import ast
            if isinstance(value, (list, tuple, set)):
                return tuple(sorted(int(x) for x in value))
            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple, set)):
                        return tuple(sorted(int(x) for x in parsed))
                except Exception:
                    return tuple()
            return tuple()

        def _provenance_key(row):
            return (
                int(row.get("Frame", row.get("Product_Frame", 0)) or 0),
                tuple(sorted(self._parts_text(
                    row.get(
                        "Reactants_Identity",
                        row.get("Reactants_SMILES", ""),
                    )
                ))),
                tuple(sorted(self._parts_text(
                    row.get(
                        "Products_Identity",
                        row.get("Products_SMILES", ""),
                    )
                ))),
                tuple(sorted(self._parts_text(
                    row.get("Reactants_Formula", "")
                ))),
                tuple(sorted(self._parts_text(
                    row.get("Products_Formula", "")
                ))),
                _provenance_indices(row.get("Atom_Indices", [])),
            )

        raw_ids_by_key = {
            _provenance_key(rec): rec["Raw_Reaction_ID"]
            for rec in all_records
        }
        for audit_row in self.unresolved_components:
            raw_id = raw_ids_by_key.get(_provenance_key(audit_row), "")
            if raw_id:
                audit_row["Source_Raw_Reaction_ID"] = raw_id

        n_elementary = sum(1 for r in all_records if r.get('Is_Elementary', True))
        n_nonelem = len(all_records) - n_elementary
        print(f"   Raw events (all detected): {len(all_records)} "
              f"({n_elementary} elementary candidates, {n_nonelem} non-elementary)")
        detect_phase_timings["final_audit"] = (
            time.perf_counter() - phase_started
        )
        detect_phase_timings["total"] = (
            time.perf_counter() - detect_started
        )
        self.detect_timings = dict(detect_phase_timings)
        print(
            "   ⏱️ Detection time: "
            f"total={detect_phase_timings['total']:.1f}s; slowest phases: "
            + (
                ", ".join(
                    f"{name}={seconds:.1f}s"
                    for name, seconds in sorted(
                        (
                            (name, seconds)
                            for name, seconds in detect_phase_timings.items()
                            if name != "total" and seconds >= 0.05
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                )
                or "none above 0.05s"
            )
            + "."
        )
        return _reaction_id_first(pd.DataFrame(all_records))



class ReactionCleaner:
    def __init__(
        self,
        df,
        stability=5,
        allow_ambiguous_topology=False,
    ):
        self.df = df
        self.stability = max(1, int(stability))
        self.recrossing_window = max(40, 8 * self.stability)
        self.allow_ambiguous_topology = bool(allow_ambiguous_topology)

        # Runtime provenance retained for Pathway reconstruction. 
        self.lineage_dropped_records = []
        self.clean_build_stats = {}
        self.pathway_build_stats = {}
        self.pathway_rejected_candidates = pd.DataFrame()
        self.raw_clean_status = {}

    def _parse_indices(self, value):
        import ast
        import re

        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            return []
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, (list, tuple, set)):
            try:
                return sorted({int(x) for x in value})
            except (TypeError, ValueError):
                return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return sorted({int(x) for x in parsed})
                if isinstance(parsed, (int, np.integer)):
                    return [int(parsed)]
            except Exception:
                pass

            indices = set()
            for token in re.split(r"[\s,|;]+", text):
                token = token.strip().strip("[](){}")
                if not token:
                    continue
                range_match = re.fullmatch(r"(\d+)-(\d+)", token)
                if range_match:
                    start, end = map(int, range_match.groups())
                    if start <= end:
                        indices.update(range(start, end + 1))
                    else:
                        indices.update(range(start, end - 1, -1))
                    continue
                if re.fullmatch(r"[+-]?\d+", token):
                    indices.add(int(token))
            return sorted(indices)
        return []

    def _share_atoms(self, r1, r2):
        set1 = set(self._parse_indices(r1.get('Atom_Indices', [])))
        set2 = set(self._parse_indices(r2.get('Atom_Indices', [])))
        return bool(set1) and bool(set2) and not set1.isdisjoint(set2)

    def _parts(self, text):
        if not isinstance(text, str):
            return []
        return [x.strip() for x in text.split(' + ') if x.strip()]

    def _multiset_equal(self, a, b):
        return sorted(self._parts(a)) == sorted(self._parts(b))

    def _identity_text(self, row, side):
        prefix = "Reactants" if str(side).lower().startswith("r") else "Products"
        return row.get(
            f"{prefix}_Identity",
            row.get(f"{prefix}_SMILES", ""),
        )

    def _topology_valid(self, row):
        value = row.get("Topology_Validated", True)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _strip_surface_suffix(self, formula):
        if formula is None:
            return ''
        s = str(formula).strip()
        if s.endswith('*'):
            s = s[:-1]
        if s.endswith('(g)'):
            s = s[:-3]
        return s.strip()

    def _is_state_only_surface_transition(self, row):
        """Return True when molecular identity is unchanged except for * /(g)."""
        side_states = {}
        for side in ("Reactants", "Products"):
            identities = self._parts(
                row.get(
                    f"{side}_Identity",
                    row.get(f"{side}_SMILES", ""),
                )
            )
            formulas = self._parts(row.get(f"{side}_Formula", ""))
            size = max(len(identities), len(formulas))
            side_states[side] = Counter(
                (
                    identities[index] if index < len(identities) else "",
                    formulas[index] if index < len(formulas) else "",
                )
                for index in range(size)
            )

        reactant_states = side_states["Reactants"]
        product_states = side_states["Products"]
        if not reactant_states or reactant_states == product_states:
            return False

        def base_states(states):
            result = Counter()
            for (identity, formula), count in states.items():
                result[
                    (identity, self._strip_surface_suffix(formula))
                ] += count
            return result

        return base_states(reactant_states) == base_states(product_states)

    def _surface_state_evidence_complete(
        self,
        row,
        minimum_dwell=2,
    ):
        """Whether both surface endpoints have explicit, non-missing evidence."""
        if not self._is_state_only_surface_transition(row):
            return False
        minimum_dwell = max(1, int(minimum_dwell))
        for side in ("Reactant", "Product"):
            state_fields = (
                f"Endpoint_Surface_{side}_Expected_Adsorbed",
                f"Endpoint_Surface_{side}_Observed_Adsorbed",
            )
            if (
                not all(field in row for field in state_fields)
                or any(
                    _is_missing_scalar(row.get(field))
                    for field in state_fields
                )
            ):
                return False
            if self._surface_dwell_value(
                row,
                side,
                minimum_dwell=minimum_dwell,
            ) is None:
                return False
        return True

    @staticmethod
    def _surface_minimum_dwell_for_row(row, default=2):
        """Resolve the dwell threshold recorded for one surface event."""
        default = max(1, int(default))
        value = row.get("Endpoint_Surface_Minimum_Dwell", default)
        if _is_missing_scalar(value):
            return default
        try:
            return max(1, int(value))
        except (TypeError, ValueError, OverflowError):
            return default

    def _surface_dwell_value(
        self,
        row,
        side,
        minimum_dwell=2,
    ):
        """Return explicit dwell validation, supporting old and new records."""
        minimum_dwell = max(1, int(minimum_dwell))
        row_minimum = self._surface_minimum_dwell_for_row(
            row,
            default=minimum_dwell,
        )
        generic_field = (
            f"Endpoint_Surface_{side}_Dwell_Validated"
        )
        if (
            row_minimum == minimum_dwell
            and generic_field in row
            and not _is_missing_scalar(row.get(generic_field))
        ):
            return self._truthy(row.get(generic_field))

        threshold_field = (
            f"Endpoint_Surface_{side}_Dwell_At_Least_"
            f"{minimum_dwell}"
        )
        if (
            threshold_field in row
            and not _is_missing_scalar(row.get(threshold_field))
        ):
            return self._truthy(row.get(threshold_field))
        return None

    def _unstable_surface_state_transition(
        self,
        row,
        minimum_dwell=2,
    ):
        """Defensive Clean-stage guard for insufficient surface-state dwell."""
        if not self._surface_state_evidence_complete(
            row,
            minimum_dwell=minimum_dwell,
        ):
            return False

        for side in ("Reactant", "Product"):
            expected_field = (
                f"Endpoint_Surface_{side}_Expected_Adsorbed"
            )
            observed_field = (
                f"Endpoint_Surface_{side}_Observed_Adsorbed"
            )
            expected = row.get(expected_field)
            observed = row.get(observed_field)
            dwell = self._surface_dwell_value(
                row,
                side,
                minimum_dwell=minimum_dwell,
            )
            if self._truthy(expected) != self._truthy(observed):
                return True
            if dwell is not True:
                return True
        return False

    def _validated_surface_state_transition(
        self,
        row,
        minimum_dwell=2,
    ):
        """Whether a pure state transition passed the dedicated dwell rule."""
        return (
            self._surface_state_evidence_complete(
                row,
                minimum_dwell=minimum_dwell,
            )
            and not self._unstable_surface_state_transition(
                row,
                minimum_dwell=minimum_dwell,
            )
        )

    def _unstable_recovered_surface_transition(
        self,
        row,
        minimum_dwell=2,
    ):
        """Backward-compatible alias for the generalized surface-state guard."""
        return self._unstable_surface_state_transition(
            row,
            minimum_dwell=minimum_dwell,
        )

    def _formula_counter_one(self, formula):
        import re
        formula = self._strip_surface_suffix(formula)
        c = Counter()
        if not formula:
            return c
        formula = formula.replace('+', '').replace('-', '').replace('.', '')
        for sym, num in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            c[sym] += int(num) if num else 1
        return c

    def _formula_counter_many(self, text):
        total = Counter()
        for part in self._parts(text):
            total.update(self._formula_counter_one(part))
        return total

    def _mass_balanced(self, row):
        return self._formula_counter_many(row.get('Reactants_Formula', '')) == self._formula_counter_many(row.get('Products_Formula', ''))

    def _species_count(self, row):
        # Helper retained for diagnostics.
        return max(
            len(self._parts(row.get('Reactants_Formula', ''))),
            len(self._parts(row.get('Products_Formula', ''))),
            len(self._parts(row.get('Reactants_SMILES', ''))),
            len(self._parts(row.get('Products_SMILES', ''))),
        )

    def _elementary_side_counts(self, row):
        r_count = max(
            len(self._parts(row.get('Reactants_Formula', ''))),
            len(self._parts(row.get('Reactants_SMILES', ''))),
        )
        p_count = max(
            len(self._parts(row.get('Products_Formula', ''))),
            len(self._parts(row.get('Products_SMILES', ''))),
        )
        return r_count, p_count

    def _is_elementary_stoichiometry(self, row):
        r_count, p_count = self._elementary_side_counts(row)
        return 1 <= r_count <= 3 and 1 <= p_count <= 3

    def _is_identity(self, row):
        chemical_identity = self._multiset_equal(
            self._identity_text(row, "reactants"),
            self._identity_text(row, "products"),
        ) and self._multiset_equal(
            row.get('Reactants_Formula'),
            row.get('Products_Formula'),
        )
        if not chemical_identity:
            return False

        return not self._truthy(row.get("Atom_Mapping_Exchange", False))

    def _is_reverse(self, a, b):
        return self._multiset_equal(
            self._identity_text(a, "reactants"),
            self._identity_text(b, "products"),
        ) and self._multiset_equal(
            self._identity_text(a, "products"),
            self._identity_text(b, "reactants"),
        ) and self._multiset_equal(
            a.get('Reactants_Formula'),
            b.get('Products_Formula'),
        ) and self._multiset_equal(
            a.get('Products_Formula'),
            b.get('Reactants_Formula'),
        )

    def _renumber(self, records):
        for i, r in enumerate(records, 1):
            original_raw_id = str(
                r.get("Raw_Reaction_ID", r.get("Reaction_ID", "")) or ""
            )
            if original_raw_id:
                r["Raw_Reaction_ID"] = original_raw_id
            r['Reaction_ID'] = f"RXN-{i:04d}"
        return records


    def _coerce_int(self, value, fallback=0):
        try:
            if value is None:
                return int(fallback)
            if isinstance(value, float) and pd.isna(value):
                return int(fallback)
            return int(float(value))
        except Exception:
            return int(fallback)

    def finalize_clean_event_ids(self, clean_df, pathway_df):
        """Number the complete Clean table chronologically before structure export.

        Raw and Pathway identities remain stable; only references to Clean IDs
        follow the simultaneous old-to-new mapping. Call after pathway promotion.
        """
        clean = clean_df.copy()
        pathway = pathway_df.copy()
        if clean.empty:
            return clean, pathway

        for column in ("Reaction_ID", "Raw_Reaction_ID"):
            if column not in clean:
                raise ValueError(f"Final Clean table is missing {column}")
            values = clean[column].fillna("").astype(str).str.strip()
            if values.eq("").any() or values.duplicated().any():
                raise ValueError(f"Final Clean table has empty or duplicate {column}")

        sort_columns = [
            column for column in
            ("Frame", "Reactant_Frame", "Product_Frame", "Reaction_ID")
            if column in clean.columns
        ]
        clean = clean.sort_values(
            sort_columns, kind="mergesort", na_position="last"
        ).reset_index(drop=True)
        old_ids = clean["Reaction_ID"].astype(str).tolist()
        new_ids = [f"RXN-{number:04d}" for number in range(1, len(clean) + 1)]
        id_map = dict(zip(old_ids, new_ids))
        clean["Reaction_ID"] = new_ids

        def remap(value):
            if _is_missing_scalar(value):
                return value
            return id_map.get(str(value), value)

        # Map whole columns once: sequential replacement would corrupt ID swaps.
        for table in (clean, pathway):
            for column in ("Clean_Reaction_ID", "Structure_Representative_ID"):
                if column in table.columns:
                    table[column] = table[column].map(remap)

        clean_by_raw = dict(zip(clean["Raw_Reaction_ID"].astype(str), new_ids))
        for raw_id, status in self.raw_clean_status.items():
            if str(raw_id) in clean_by_raw:
                status["Clean_Reaction_ID"] = clean_by_raw[str(raw_id)]
                status["Exclusion_Reason"] = ""
            elif status.get("Clean_Reaction_ID"):
                status["Clean_Reaction_ID"] = remap(status["Clean_Reaction_ID"])
        if "raw_clean_status" in clean.attrs:
            clean.attrs["raw_clean_status"] = {
                raw_id: dict(status)
                for raw_id, status in self.raw_clean_status.items()
            }
        return clean, pathway

    def _pathway_type(self, row):
        r_count, p_count = self._elementary_side_counts(row)
        if r_count == 1 and p_count > 1:
            return "Pathway Net Fragmentation" if p_count <= 3 else "Pathway Complex Fragmentation"
        if r_count > 1 and p_count == 1:
            return "Pathway Net Association" if r_count <= 3 else "Pathway Complex Association"
        if r_count > 1 and p_count > 1:
            if r_count < p_count:
                return "Pathway Complex Fragmentation"
            if r_count > p_count:
                return "Pathway Complex Association"
            return "Pathway Complex Exchange"
        return "Pathway Net Transition"

    def _canonical_event_key(self, row, include_frame=False):
        r_s = tuple(sorted(self._parts(
            self._identity_text(row, "reactants")
        )))
        p_s = tuple(sorted(self._parts(
            self._identity_text(row, "products")
        )))
        r_f = tuple(sorted(self._parts(row.get('Reactants_Formula', ''))))
        p_f = tuple(sorted(self._parts(row.get('Products_Formula', ''))))
        idx = tuple(self._parse_indices(row.get('Atom_Indices', [])))
        r_groups = tuple(_normalize_atom_groups(
            row.get("Reactant_Atom_Groups", [])
        ))
        p_groups = tuple(_normalize_atom_groups(
            row.get("Product_Atom_Groups", [])
        ))
        if include_frame:
            return (
                self._coerce_int(
                    row.get('Frame', row.get('Product_Frame', 0))
                ),
                r_s, p_s, r_f, p_f, idx, r_groups, p_groups,
            )
        return (r_s, p_s, r_f, p_f, idx, r_groups, p_groups)

    def _pathway_record_from_unresolved(self, row):
        # Project only the detected but non-elementary connection steps.
        reason = str(row.get('Reason', '') or '')
        PROJECTABLE_REASONS = {'non_elementary_after_local_split'}
        if reason not in PROJECTABLE_REASONS:
            return None

        if (
            "Clean_Eligible" in row
            and not self._truthy(row.get("Clean_Eligible", False))
        ):
            return None

        r_formula = str(row.get('Reactants_Formula', '') or '').strip()
        p_formula = str(row.get('Products_Formula', '') or '').strip()
        r_smiles = str(row.get('Reactants_SMILES', '') or '').strip()
        p_smiles = str(row.get('Products_SMILES', '') or '').strip()
        r_identity = str(
            row.get('Reactants_Identity', r_smiles) or r_smiles
        ).strip()
        p_identity = str(
            row.get('Products_Identity', p_smiles) or p_smiles
        ).strip()
        if (
            not r_formula
            or not p_formula
            or not r_smiles
            or not p_smiles
            or not r_identity
            or not p_identity
        ):
            return None

        rec = {
            'Reaction_ID': None,
            'Frame': self._coerce_int(
                row.get('Frame', row.get('Product_Frame', None)),
                fallback=0
            ),
            'Reactant_Frame': self._coerce_int(row.get('Reactant_Frame', None), fallback=0),
            'Product_Frame': self._coerce_int(row.get('Product_Frame', None), fallback=0),
            'Reactants_Formula': r_formula,
            'Products_Formula': p_formula,
            'Reactants_SMILES': r_smiles,
            'Products_SMILES': p_smiles,
            'Reactants_Identity': r_identity,
            'Products_Identity': p_identity,
            'Topology_Validated': row.get('Topology_Validated', True),
            'Topology_Evidence': str(
                row.get('Topology_Evidence', '') or ''
            ),
            'Clean_Eligible': True,
            'Whole_Event_Fallback': self._truthy(
                row.get('Whole_Event_Fallback', False)
            ),
            'Composite_Candidate': True,
            'Composite_Evidence': str(
                row.get('Composite_Evidence', '') or ''
            ),
            'Changed_Bond_Count': row.get('Changed_Bond_Count', ''),
            'Changed_Bond_Components': row.get(
                'Changed_Bond_Components',
                '',
            ),
            'Changed_Bond_Component_Sizes': row.get(
                'Changed_Bond_Component_Sizes',
                '',
            ),
            'Split_Elementary_Records': row.get(
                'Split_Elementary_Records',
                '',
            ),
            'Atom_Indices': self._parse_indices(row.get('Atom_Indices', [])),
            'Reactant_Atom_Groups': _normalize_atom_groups(
                row.get('Reactant_Atom_Groups', [])
            ),
            'Product_Atom_Groups': _normalize_atom_groups(
                row.get('Product_Atom_Groups', [])
            ),
            'Elementary_Mode': str(
                row.get('Elementary_Mode', 'composite_bridge') or
                'composite_bridge'
            ),
            'Type': None,
            'Snapshot': 'Pathway only (not an elementary clean event)',
            'NEB_Input': 'Pathway only (not an elementary clean event)',
            'Pathway_Only': True,
            'Pathway_Source': f"unresolved:{row.get('Reason', '')}",
            'Clean_Reaction_ID': '',
            'Raw_Reaction_ID': str(
                row.get("Source_Raw_Reaction_ID", "") or ""
            ),
            'Source_Raw_Reaction_ID': str(
                row.get("Source_Raw_Reaction_ID", "") or ""
            ),
            'Source_Unresolved_ID': str(row.get('Unresolved_ID', '') or ''),
            'Lineage_Note': str(row.get('Note', '') or ''),
        }
        if self._is_identity(rec):
            return None
        if not self._mass_balanced(rec):
            return None
        rec['Type'] = self._pathway_type(rec)
        return rec


    @staticmethod
    def _truthy(value):
        return _truthy_value(value)

    def _cleaner_side_entries(self, row, side):
        if str(side).lower().startswith("r"):
            smis = self._parts(row.get("Reactants_SMILES", ""))
            identities = self._parts(
                row.get("Reactants_Identity", row.get("Reactants_SMILES", ""))
            )
            formulas = self._parts(row.get("Reactants_Formula", ""))
            atom_groups = _normalize_atom_groups(
                row.get("Reactant_Atom_Groups", [])
            )
        else:
            smis = self._parts(row.get("Products_SMILES", ""))
            identities = self._parts(
                row.get("Products_Identity", row.get("Products_SMILES", ""))
            )
            formulas = self._parts(row.get("Products_Formula", ""))
            atom_groups = _normalize_atom_groups(
                row.get("Product_Atom_Groups", [])
            )

        n = max(len(smis), len(identities), len(formulas))
        entries = []
        for i in range(n):
            smi = smis[i] if i < len(smis) else ""
            identity = identities[i] if i < len(identities) else smi
            formula = formulas[i] if i < len(formulas) else smi
            atoms = atom_groups[i] if i < len(atom_groups) else tuple()
            if not atoms and n == 1:
                atoms = tuple(self._parse_indices(row.get("Atom_Indices", [])))
            entries.append({
                "smiles": smi,
                "identity": identity,
                "formula": formula,
                "atom_indices": list(atoms),
                "key": (
                    str(identity).strip(),
                    str(formula).strip(),
                    tuple(atoms),
                ),
            })
        return entries

    def _atom_mapped_side_signature(self, row, side):
        """Return an exact per-molecule atom-mapped side signature."""
        entries = self._cleaner_side_entries(row, side)
        if not entries or any(not entry["atom_indices"] for entry in entries):
            return tuple()
        return tuple(sorted(
            (
                str(entry["identity"]).strip(),
                str(entry["formula"]).strip(),
                tuple(entry["atom_indices"]),
            )
            for entry in entries
        ))

    def _surface_state_lineage_base_key(self, key):
        """Connectivity/atom key with only the */(g) label removed."""
        if not isinstance(key, tuple) or len(key) != 3:
            return None
        identity, formula, atoms = key
        formula = str(formula or "").strip()
        if not (formula.endswith("*") or formula.endswith("(g)")):
            return None
        try:
            atom_key = tuple(int(i) for i in atoms)
        except (TypeError, ValueError):
            return None
        if not atom_key:
            return None
        return (
            str(identity or "").strip(),
            self._strip_surface_suffix(formula),
            atom_key,
        )

    def _surface_state_lineage_equivalent(self, source_key, target_key):
        """Whether two exact atom inventories differ only by */(g)."""
        source_base = self._surface_state_lineage_base_key(source_key)
        target_base = self._surface_state_lineage_base_key(target_key)
        return (
            source_key != target_key
            and source_base is not None
            and source_base == target_base
        )

    @staticmethod
    def _species_state_key_from_lineage_key(key):
        """Return the atom-agnostic state node represented by a lineage key."""
        if not isinstance(key, tuple) or len(key) != 3:
            return None
        identity, formula, _atoms = key
        identity = str(identity or "").strip()
        formula = str(formula or "").strip()
        if not identity and not formula:
            return None
        return identity, formula

    def _initial_keys_from_records(self, records):
        import json
        keys = set()

        for field in ("Initial_Species_Keys", "Initial_Reactant_Keys"):
            for row in records:
                raw = row.get(field, "")
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    continue
                try:
                    payload = json.loads(str(raw))
                except Exception:
                    payload = []
                for item in payload if isinstance(payload, list) else []:
                    if isinstance(item, dict):
                        keys.add((
                            str(item.get(
                                "identity",
                                item.get("smiles", ""),
                            )).strip(),
                            str(item.get("formula", "")).strip(),
                            tuple(sorted(set(
                                int(i)
                                for i in item.get("atom_indices", []) or []
                            ))),
                        ))
            if keys:
                break
        return keys

    @staticmethod
    def _serialize_cleaner_entries(entries):
        import json
        return json.dumps(
            [
                {
                    "smiles": str(e.get("smiles", "")),
                    "identity": str(e.get("identity", e.get("smiles", ""))),
                    "formula": str(e.get("formula", "")),
                    "atom_indices": [
                        int(i) for i in e.get("atom_indices", []) or []
                    ],
                }
                for e in entries
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _annotate_clean_lineage(self, records):
        initial_keys = self._initial_keys_from_records(records)

        if not initial_keys:
            for rec in records:
                rec.setdefault("Lineage_Complete", True)
                rec.setdefault("Missing_Reactant_Keys", "[]")
                rec.setdefault("Missing_Reactant_Species", "")
                rec.setdefault("Initial_Species_Keys", "[]")
                rec.setdefault("Initial_Reactant_Keys", "[]")
            return []

        available = {
            key: {"source": "INITIAL", "depth": 0}
            for key in initial_keys
        }
        # Atom-specific inventory is consumed chronologically, while species
        # state reachability is monotonic.
        reachable_species_states = {
            state_key
            for state_key in (
                self._species_state_key_from_lineage_key(key)
                for key in initial_keys
            )
            if state_key is not None
        }
        gaps = []

        ordered = sorted(
            records,
            key=lambda r: (
                self._coerce_int(r.get("Frame", r.get("Product_Frame", 0)), 0),
                self._coerce_int(r.get("Reactant_Frame", r.get("Frame", 0)), 0),
            ),
        )

        for rec in ordered:
            reactants = self._cleaner_side_entries(rec, "reactants")
            products = self._cleaner_side_entries(rec, "products")
            initial_reactants = [e for e in reactants if e["key"] in initial_keys]

            # Resolve exact state-aware inventory first. 
            resolved_sources = []
            missing = []
            reserved_source_keys = set()
            surface_bridges = []
            blocked_unreachable_surface_bridges = []
            allow_surface_bridge = not self._is_state_only_surface_transition(
                rec
            )
            for entry in reactants:
                target_key = entry["key"]
                source_key = None
                if (
                    target_key in available
                    and target_key not in reserved_source_keys
                ):
                    source_key = target_key
                else:
                    surface_alternatives = [
                        key
                        for key in available
                        if key not in reserved_source_keys
                        and self._surface_state_lineage_equivalent(
                            key,
                            target_key,
                        )
                    ]
                    target_state = self._species_state_key_from_lineage_key(
                        target_key
                    )
                    target_state_reachable = (
                        target_state in reachable_species_states
                    )
                    if (
                        allow_surface_bridge
                        and target_state_reachable
                        and len(surface_alternatives) == 1
                    ):
                        source_key = surface_alternatives[0]
                        surface_bridges.append({
                            "source": source_key,
                            "target": target_key,
                        })
                    elif (
                        allow_surface_bridge
                        and not target_state_reachable
                        and surface_alternatives
                    ):
                        blocked_unreachable_surface_bridges.append({
                            "sources": surface_alternatives,
                            "target": target_key,
                        })

                if source_key is None:
                    missing.append(entry)
                    resolved_sources.append(None)
                else:
                    reserved_source_keys.add(source_key)
                    resolved_sources.append(source_key)

            rec["Initial_Reactant_Keys"] = self._serialize_cleaner_entries(
                initial_reactants
            )
            rec["Lineage_Complete"] = not bool(missing)
            rec["Reachable_From_Initial"] = not bool(missing)
            rec["Missing_Reactant_Keys"] = self._serialize_cleaner_entries(missing)
            rec["Missing_Reactant_Species"] = " + ".join(
                e["formula"] or e["smiles"] for e in missing
            )
            rec["Surface_State_Lineage_Bridged"] = bool(surface_bridges)
            rec["Surface_State_Lineage_Bridge_Count"] = len(surface_bridges)
            rec["Surface_State_Lineage_Bridges"] = json.dumps(
                [
                    {
                        "identity": str(source[0]),
                        "atom_indices": [int(i) for i in source[2]],
                        "source_formula": str(source[1]),
                        "target_formula": str(target[1]),
                    }
                    for source, target in (
                        (bridge["source"], bridge["target"])
                        for bridge in surface_bridges
                    )
                ],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            rec[
                "Surface_State_Lineage_Bridge_Blocked_Unreachable_Count"
            ] = len(blocked_unreachable_surface_bridges)
            rec[
                "Surface_State_Lineage_Bridge_Blocked_Unreachable"
            ] = json.dumps(
                [
                    {
                        "identity": str(item["target"][0]),
                        "atom_indices": [
                            int(i) for i in item["target"][2]
                        ],
                        "target_formula": str(item["target"][1]),
                        "available_source_formulas": sorted({
                            str(source[1])
                            for source in item["sources"]
                        }),
                    }
                    for item in blocked_unreachable_surface_bridges
                ],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            source_ids = []
            source_depths = []
            for source_key in resolved_sources:
                source = available.get(source_key)
                if source is None:
                    continue
                source_ids.append(str(source.get("source", "")))
                source_depths.append(int(source.get("depth", 0)))
            rec["Reactant_Source_Raw_IDs"] = "|".join(source_ids)
            rec["Lineage_Depth"] = (
                max(source_depths, default=-1) + 1
                if not missing
                else ""
            )

            for entry in missing:
                gaps.append((rec, entry))
            if not missing:
                for source_key in resolved_sources:
                    available.pop(source_key, None)
                raw_id = str(
                    rec.get(
                        "Raw_Reaction_ID",
                        rec.get("Reaction_ID", ""),
                    )
                    or ""
                )
                product_depth = rec["Lineage_Depth"]
                for entry in products:
                    available[entry["key"]] = {
                        "source": raw_id,
                        "depth": product_depth,
                    }
                    state_key = self._species_state_key_from_lineage_key(
                        entry["key"]
                    )
                    if state_key is not None:
                        reachable_species_states.add(state_key)

        return gaps

    def _prune_incomplete_lineage_events(self, records):
        """Remove retained events whose reactants still have no source."""
        kept = list(records)
        removed = []

        for _round in range(50):
            gaps = self._annotate_clean_lineage(kept)
            if not gaps:
                break

            bad_ids = {id(rec) for rec, _entry in gaps}
            if not bad_ids:
                break

            next_kept = []
            changed = False
            for rec in kept:
                if id(rec) in bad_ids:
                    rec["Dropped_By_Clean_Lineage_Filter"] = True
                    rec["Lineage_Filter_Reason"] = (
                        "reactant_source_unresolved_after_recovery"
                    )
                    removed.append(rec)
                    changed = True
                else:
                    next_kept.append(rec)

            kept = next_kept
            if not changed:
                break

        self._annotate_clean_lineage(kept)
        return kept, removed

    def _restore_clean_lineage_sources(self, kept_records, candidate_pool):
        """Restore the minimum removed event set needed for source reachability."""
        kept = list(kept_records)
        kept_ids = {id(r) for r in kept}
        restored = []
        blocked_recrossing_ids = set()
        blocked_surface_recrossing_ids = set()

        producer_index = defaultdict(list)
        for candidate_position, candidate in enumerate(candidate_pool):
            candidate_product_frame = self._coerce_int(
                candidate.get(
                    "Product_Frame", candidate.get("Frame", 0)
                ),
                0,
            )
            candidate_event_frame = self._coerce_int(
                candidate.get("Frame", 0), 0
            )
            product_keys = {
                entry["key"]
                for entry in self._cleaner_side_entries(
                    candidate, "products"
                )
            }
            indexed = (
                candidate_product_frame,
                candidate_event_frame,
                int(candidate_position),
                candidate,
            )
            for product_key in product_keys:
                producer_index[product_key].append(indexed)

        for entries in producer_index.values():
            entries.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    item[2],
                )
            )

        for _round in range(20):
            gaps = self._annotate_clean_lineage(kept)
            if not gaps:
                break

            changed = False
            for gap_rec, missing in gaps:
                consume_frame = self._coerce_int(
                    gap_rec.get("Reactant_Frame", gap_rec.get("Frame", 0)), 0
                )
                best = None

                for (
                    cand_frame,
                    _event_frame,
                    _candidate_position,
                    cand,
                ) in producer_index.get(missing["key"], ()):
                    if id(cand) in kept_ids:
                        continue
                    if cand_frame > consume_frame:
                        continue

                    if self._truthy(
                        cand.get(
                            "Removed_As_Short_Time_Recrossing",
                            False,
                        )
                    ):
                        blocked_recrossing_ids.add(id(cand))
                        if self._truthy(
                            cand.get(
                                "Short_Time_Recrossing_Is_Surface_State",
                                False,
                            )
                        ):
                            blocked_surface_recrossing_ids.add(id(cand))
                        continue

                    if best is None:
                        best = cand

                if best is None:
                    continue
                best["Lineage_Restored_By_Cleaner"] = True
                kept.append(best)
                kept_ids.add(id(best))
                restored.append(best)
                changed = True

            if not changed:
                break

        kept.sort(
            key=lambda r: (
                self._coerce_int(r.get("Frame", 0), 0),
                self._coerce_int(r.get("Reactant_Frame", 0), 0),
                self._coerce_int(r.get("Product_Frame", 0), 0),
            )
        )
        self._annotate_clean_lineage(kept)
        self._lineage_recrossing_restore_blocked = len(
            blocked_recrossing_ids
        )
        self._lineage_surface_recrossing_restore_blocked = len(
            blocked_surface_recrossing_ids
        )
        return kept, restored

    def _pathway_side_keys(self, row, side):
        """State-aware species keys on one side of a pathway event."""
        return {
            entry["key"]
            for entry in self._cleaner_side_entries(row, side)
            if entry["key"][0] or entry["key"][1]
        }

    def _pathway_all_state_keys(self, records):
        keys = set()
        for rec in records:
            keys.update(self._pathway_side_keys(rec, "reactants"))
            keys.update(self._pathway_side_keys(rec, "products"))
        return keys

    def _pathway_reachable_keys(self, records, initial_keys, cutoff_frame=None):
        """Return exact atom-mapped states available at a chronological cutoff."""
        available = set(initial_keys)
        ordered = sorted(
            records,
            key=lambda r: (
                self._coerce_int(
                    r.get("Frame", r.get("Product_Frame", 0)), 0
                ),
                self._coerce_int(
                    r.get("Reactant_Frame", r.get("Frame", 0)), 0
                ),
                self._coerce_int(
                    r.get("Product_Frame", r.get("Frame", 0)), 0
                ),
            ),
        )

        for rec in ordered:
            event_frame = self._coerce_int(
                rec.get("Frame", rec.get("Product_Frame", 0)), 0
            )
            if cutoff_frame is not None and event_frame > int(cutoff_frame):
                continue

            reactants = self._pathway_side_keys(rec, "reactants")
            products = self._pathway_side_keys(rec, "products")
            # A trajectory-observed stable state is provenance metadata, not a chemical edge.
            is_observed_source = self._truthy(
                rec.get("Observed_Stable_State_Source", False)
            )
            if not products or (not reactants and not is_observed_source):
                continue
            if is_observed_source:
                available.update(products)
                continue
            if not reactants.issubset(available):
                continue
            available.difference_update(reactants)
            available.update(products)

        return available

    def _format_pathway_species_keys(self, keys):
        return " + ".join(
            (
                f"{formula or smi}@"
                f"{','.join(str(i) for i in atoms)}"
            )
            for smi, formula, atoms in sorted(
                keys, key=lambda x: (x[1], x[0], x[2])
            )
        )

    def _pathway_lineage_dropped_pool(
        self,
        lineage_dropped_df=None,
        raw_df=None,
    ):
        """Return atom-mapped candidates removed only for lineage repair."""
        records = []
        sources = []
        seen_raw_ids = set()

        def add_rows(rows, source):
            added = 0
            for row in rows or []:
                rec = dict(row)
                raw_id = str(
                    rec.get("Raw_Reaction_ID", rec.get("Reaction_ID", ""))
                    or ""
                )
                # Empty IDs cannot be made auditable or safely de-duplicated.
                if not raw_id or raw_id in seen_raw_ids:
                    continue
                seen_raw_ids.add(raw_id)
                rec.setdefault("Lineage_Repair_Pool_Source", source)
                records.append(rec)
                added += 1
            if added:
                sources.append(source)

        if lineage_dropped_df is not None and not lineage_dropped_df.empty:
            add_rows(
                lineage_dropped_df.to_dict("records"),
                "explicit_lineage_dropped",
            )

        if getattr(self, "lineage_dropped_records", None):
            add_rows(
                self.lineage_dropped_records,
                "clean_runtime",
            )

        if raw_df is not None and not raw_df.empty:
            raw_records = raw_df.to_dict("records")
            add_rows(
                [
                    row for row in raw_records
                    if self._truthy(
                        row.get("Dropped_By_Clean_Lineage_Filter", False)
                    )
                ],
                "flagged_raw_rows",
            )
            # Persisted Raw tables commonly have only Exclusion_Reason.  This
            # is the essential reproducible fallback for standalone rebuilds.
            add_rows(
                [
                    row for row in raw_records
                    if str(row.get("Exclusion_Reason", "") or "").strip()
                    == "lineage_incomplete"
                ],
                "persisted_raw_lineage_incomplete",
            )

        return records, "+".join(sources) if sources else "unavailable"

    def _raw_exclusion_reason(self, row):
        """Return the final cleaner disposition for an in-memory Raw row."""
        direct = str(row.get("Exclusion_Reason", "") or "").strip()
        if direct and direct.lower() not in {"nan", "none"}:
            return direct
        raw_id = str(
            row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or ""
        )
        status = getattr(self, "raw_clean_status", {}).get(raw_id, {})
        return str(status.get("Exclusion_Reason", "") or "").strip()

    def _observed_stable_state_source_candidates(
        self,
        raw_df,
        needed_consumers,
        minimum_dwell=None,
    ):
        """Build conservative provenance sources from settled observations."""
        if raw_df is None or raw_df.empty or not needed_consumers:
            return []
        if minimum_dwell is None:
            minimum_dwell = max(1, int(self.recrossing_window))

        raw_records = raw_df.to_dict("records")

        candidates = []
        seen = set()
        for row in raw_records:
            exclusion_reason = self._raw_exclusion_reason(row)
            is_recrossing_source = (
                exclusion_reason == "short_time_recrossing"
            )
            product_endpoint_validated = self._truthy(
                row.get(
                    "Observed_Stable_Exit_Product_Validated",
                    False,
                )
            )
            corridor_validated = self._truthy(
                row.get("Observed_Corridor_Net_Validated", False)
            )
            is_observed_corridor_net = (
                str(row.get("Elementary_Mode", "") or "").strip()
                == "observed_corridor_net"
            )
            explicit_stable_exit_source = self._truthy(
                row.get("Observed_Stable_Exit_Source", False)
            )
            
            is_stable_exit_source = product_endpoint_validated and (
                explicit_stable_exit_source
                or (corridor_validated and is_observed_corridor_net)
            )
            if not (is_recrossing_source or is_stable_exit_source):
                continue
            if is_recrossing_source:
                if not self._truthy(row.get("Topology_Validated", True)):
                    continue
                if not _is_missing_scalar(row.get("Valence_Warning", "")):
                    continue

            source_minimum_dwell = (
                max(
                    1,
                    self._coerce_int(
                        row.get(
                            "Observed_State_Source_Min_Dwell",
                            minimum_dwell,
                        ),
                        minimum_dwell,
                    ),
                )
                if is_stable_exit_source
                else int(minimum_dwell)
            )
            source_kind = (
                "topology_ineligible_stable_exit"
                if explicit_stable_exit_source
                else (
                    "topology_ineligible_stable_product"
                    if is_stable_exit_source
                    else "short_time_recrossing"
                )
            )

            source_id = str(
                row.get("Raw_Reaction_ID", row.get("Reaction_ID", ""))
                or ""
            )
            product_frame = self._coerce_int(
                row.get("Product_Frame", row.get("Frame", 0)), 0
            )
            for entry in self._cleaner_side_entries(row, "products"):
                key = entry["key"]
                if key not in needed_consumers or not entry["atom_indices"]:
                    continue

                consumer = next(
                    (
                        (frame, raw_id)
                        for frame, raw_id in needed_consumers.get(key, [])
                        if raw_id != source_id
                        and frame >= product_frame + source_minimum_dwell
                    ),
                    None,
                )
                if consumer is None:
                    continue
                consumer_frame, consumer_id = consumer
                source_key = (source_id, key)
                if source_key in seen:
                    continue
                seen.add(source_key)

                upper_reactants = str(
                    row.get("Reactants_Formula", "") or ""
                ).strip()
                upper_products = str(
                    row.get("Products_Formula", "") or ""
                ).strip()
                if upper_reactants or upper_products:
                    if is_recrossing_source:
                        upper_status = "short-time recrossing"
                    elif (
                        is_observed_corridor_net
                        or exclusion_reason.startswith("valence_ambiguous")
                    ):
                        upper_status = "invalid topology"
                    else:
                        upper_status = (
                            exclusion_reason.replace("_", " ").strip()
                            or "filtered source"
                        )
                    upper_source = (
                        f"{upper_reactants} -> {upper_products} "
                        f"{upper_status}"
                    ).strip()
                else:
                    upper_source = ""

                candidates.append({
                    "Raw_Reaction_ID": source_id,
                    "Source_Raw_Reaction_ID": source_id,
                    "Observed_State_Consumer_Raw_ID": consumer_id,
                    "Observed_State_Dwell_Frames": (
                        int(consumer_frame) - int(product_frame)
                    ),
                    "Frame": product_frame,
                    "Reactant_Frame": product_frame,
                    "Product_Frame": product_frame,
                    "Reactants_Formula": "",
                    "Reactants_SMILES": "",
                    "Reactants_Identity": "",
                    "Reactant_Atom_Groups": "[]",
                    "Products_Formula": str(entry["formula"] or ""),
                    "Products_SMILES": str(entry["smiles"] or ""),
                    "Products_Identity": str(entry["identity"] or ""),
                    "Product_Atom_Groups": _serialize_atom_groups(
                        [entry["atom_indices"]]
                    ),
                    "Atom_Indices": list(entry["atom_indices"]),
                    "Reaction_Type": "Observed Stable State",
                    "Type": "Observed Stable State",
                    "Elementary_Mode": "observed_state_source",
                    "Pathway_Only": True,
                    "Observed_Stable_State_Source": True,
                    "Pathway_Source": (
                        f"raw:{source_kind}_stable_state"
                    ),
                    "Pathway_Role": "observed_stable_state_source",
                    "Evidence_Level": "trajectory_observed_stable_state",
                    "Pathway_Eligible": False,
                    "Lineage_Gap": False,
                    "Reachable_From_Initial": False,
                    "Bridge_Required": False,
                    "Reactivated_By_Pathway": False,
                    "Lineage_Repair_Cost": 30,
                    "Lineage_Repair_Evidence": (
                        "exact_atom_group;observed_post_source_dwell>="
                        f"{source_minimum_dwell}"
                    ),
                    "Lineage_Repair_Source": (
                        f"{source_kind}_persistent_state"
                    ),
                    "Lineage_Note": (
                        "Non-chemical provenance source: an exact "
                        "atom-mapped state persisted after a filtered "
                        f"{source_kind} until a later Raw consumer."
                    ),
                    "Upper_Source": upper_source,
                    "Upper_Source_Reaction_ID": source_id,
                    "Upper_Source_Status": upper_status if upper_source else "",
                    "_candidate_kind": "observed_stable_state_source",
                    "_candidate_exact_key": (
                        "observed_stable_state_source", source_id, key
                    ),
                })
        return candidates

    def _observed_corridor_net_candidates(self, raw_df, seen_exact):
        """Project validated invalid-corridor observations as Pathway bridges."""
        if raw_df is None or raw_df.empty:
            return []
        candidates = []
        for row in raw_df.to_dict("records"):
            if (
                str(row.get("Elementary_Mode", "") or "").strip()
                != "observed_corridor_net"
            ):
                continue
            if not self._truthy(
                row.get("Observed_Corridor_Net_Validated", True)
            ):
                continue
            if not self._mass_balanced(row):
                continue
            if (
                not self._pathway_side_keys(row, "reactants")
                or not self._pathway_side_keys(row, "products")
            ):
                continue
            exact_key = self._canonical_event_key(row, include_frame=True)
            if exact_key in seen_exact:
                continue
            rec = dict(row)
            raw_id = str(
                rec.get("Raw_Reaction_ID", rec.get("Reaction_ID", "")) or ""
            )
            if not raw_id:
                continue
            rec.update({
                "Raw_Reaction_ID": raw_id,
                "Clean_Reaction_ID": "",
                "Pathway_Only": True,
                "Pathway_Source": "raw:observed_corridor_net",
                "Pathway_Role": "observed_corridor_net",
                "Evidence_Level": "validated_observed_corridor_net",
                "Pathway_Eligible": False,
                "Lineage_Gap": True,
                "Reachable_From_Initial": False,
                "Bridge_Required": True,
                "Reactivated_By_Pathway": False,
                "Lineage_Repair_Cost": 15,
                "Lineage_Repair_Evidence": (
                    "exact_atom_mapping;settled_valid_endpoints;"
                    "intervening_invalid_topology"
                ),
                "Lineage_Repair_Source": "detector_corridor_net",
                "Lineage_Note": (
                    "Observed net transformation across an invalid-topology "
                    "corridor; both endpoints are settled and atom-mapped."
                ),
                "_candidate_kind": "observed_corridor_net",
                "_candidate_exact_key": exact_key,
            })
            candidates.append(rec)
        return candidates

    def _prune_redundant_observed_state_sources(self, records):
        """Remove provenance anchors already explained by retained events."""
        retained = list(records)
        retired = []
        for rec in sorted(
            [
                item for item in retained
                if item.get("Pathway_Role")
                == "observed_stable_state_source"
            ],
            key=lambda item: (
                self._coerce_int(
                    item.get("Frame", item.get("Product_Frame", 0)), 0
                ),
                str(item.get("Raw_Reaction_ID", "")),
            ),
        ):
            # Do not publish an anchor merely because some Raw row once mentioned the state.
            anchor_keys = self._pathway_side_keys(rec, "products")
            anchor_frame = self._coerce_int(
                rec.get("Frame", rec.get("Product_Frame", 0)), 0
            )
            has_retained_consumer = False
            if anchor_keys:
                for other in retained:
                    if other is rec or other.get("Pathway_Role") == (
                        "observed_stable_state_source"
                    ):
                        continue
                    other_frame = self._coerce_int(
                        other.get(
                            "Reactant_Frame",
                            other.get("Frame", 0),
                        ),
                        0,
                    )
                    if other_frame < anchor_frame:
                        continue
                    if anchor_keys.intersection(
                        self._pathway_side_keys(other, "reactants")
                    ):
                        has_retained_consumer = True
                        break
            if not has_retained_consumer:
                rec["Pathway_Eligible"] = False
                rec["Lineage_Gap"] = False
                rec["Reachable_From_Initial"] = True
                rec["Pathway_Rejection_Reason"] = (
                    "orphan_observed_state_source_without_retained_consumer"
                )
                rec["Pathway_Selection_Reason"] = (
                    "retired_orphan_observed_state_source"
                )
                retired.append(rec)
                retained = [item for item in retained if item is not rec]
                continue

            trial = [item for item in retained if item is not rec]
            if self._annotate_clean_lineage(trial):
                continue
            rec["Pathway_Eligible"] = False
            rec["Lineage_Gap"] = False
            rec["Reachable_From_Initial"] = True
            rec["Pathway_Rejection_Reason"] = (
                "redundant_observed_state_has_retained_exact_producer"
            )
            rec["Pathway_Selection_Reason"] = (
                "retired_redundant_observed_state_source"
            )
            retired.append(rec)
            retained = trial

        # Restore correct lineage annotations after the trial checks above.
        self._annotate_clean_lineage(retained)
        return retained, retired

    def _lineage_repair_cost(self, rec):
        """Return a transparent evidence cost for a repair candidate."""
        cost = 10
        notes = ["topology_validated_elementary"]
        recovery = str(rec.get("Recovery_Result", "") or "").strip()
        if recovery == "recovered":
            cost -= 2
            notes.append("endpoint_recovered")
        mode = str(rec.get("Elementary_Mode", "standard") or "standard")
        if mode == "standard":
            cost -= 1
            notes.append("standard_elementary_mode")
        elif mode == "atom_mapping_exchange":
            cost += 1
            notes.append("atom_mapping_exchange")
        return max(1, cost), ";".join(notes)

    def _prepare_reactivated_raw_candidate(self, row, seen_exact):
        rec = dict(row)
        original_id = str(
            rec.get("Raw_Reaction_ID", rec.get("Reaction_ID", "")) or ""
        )

        if str(rec.get("Elementary_Mode", "") or "").strip() == (
            "atom_mapping_exchange"
        ):
            rec["Atom_Mapping_Exchange"] = True

        if str(rec.get("Is_Elementary", True)).strip().lower() == "false":
            return None, "raw_candidate_non_elementary"
        if not self._truthy(rec.get("Topology_Validated", True)):
            return None, "raw_candidate_topology_unvalidated"
        if not _is_missing_scalar(rec.get("Valence_Warning", "")):
            return None, "raw_candidate_valence_warning"
        if (
            not self.allow_ambiguous_topology
            and str(rec.get("Clean_Eligible", True)).strip().lower()
            in {"0", "false", "no", "n"}
        ):
            return None, "raw_candidate_valence_ambiguous"
        if self._is_identity(rec):
            return None, "raw_candidate_identity"
        if not self._mass_balanced(rec):
            return None, "raw_candidate_mass_unbalanced"
        if not self._is_elementary_stoichiometry(rec):
            return None, "raw_candidate_non_elementary_shape"

        exact_key = self._canonical_event_key(rec, include_frame=True)
        if exact_key in seen_exact:
            return None, "raw_candidate_duplicate"

        rec["Raw_Reaction_ID"] = original_id
        rec["Clean_Reaction_ID"] = ""
        rec["Pathway_Only"] = True
        rec["Pathway_Source"] = "raw:lineage_reactivated"
        rec["Pathway_Role"] = "reactivated_raw_elementary"
        rec["Evidence_Level"] = "validated_reactivated_elementary"
        rec["Pathway_Eligible"] = False
        rec["Lineage_Gap"] = True
        rec["Reachable_From_Initial"] = False
        rec["Bridge_Required"] = False
        rec["Reactivated_By_Pathway"] = False
        repair_cost, repair_evidence = self._lineage_repair_cost(rec)
        rec["Lineage_Repair_Cost"] = int(repair_cost)
        rec["Lineage_Repair_Evidence"] = repair_evidence
        rec["Lineage_Repair_Source"] = str(
            rec.get("Lineage_Repair_Pool_Source", "") or
            "clean_lineage_filter"
        )
        rec["_candidate_kind"] = "reactivated_raw_elementary"
        rec["_candidate_exact_key"] = exact_key
        return rec, ""

    def _terminal_state_backward_closure_candidates(
        self,
        clean_records,
        candidates,
        initial_keys,
        terminal_state_entries,
    ):
        """Mark chemically valid candidates needed by stable final states."""
        if not terminal_state_entries or not initial_keys:
            return set(), {
                "stable_terminal_targets": 0,
                "terminal_targets_with_candidate_producer": 0,
                "terminal_targets_closed": 0,
            }

        candidate_kinds = {
            "reactivated_raw_elementary",
            "observed_corridor_net",
        }
        usable_candidates = [
            rec for rec in candidates
            if isinstance(rec, dict)
            and rec.get("_candidate_kind", "") in candidate_kinds
        ]
        if not usable_candidates:
            return set(), {
                "stable_terminal_targets": 0,
                "terminal_targets_with_candidate_producer": 0,
                "terminal_targets_closed": 0,
            }

        def _product_frame(record):
            return self._coerce_int(
                record.get("Product_Frame", record.get("Frame", 0)),
                0,
            )

        def _reactant_frame(record):
            return self._coerce_int(
                record.get("Reactant_Frame", record.get("Frame", 0)),
                0,
            )

        terminal_targets = {}
        for entry in terminal_state_entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key")
            if not key:
                key = self._lineage_species_key(
                    entry.get("identity", entry.get("smiles", "")),
                    entry.get("formula", ""),
                    entry.get("atom_indices", []),
                )
            if not key or not (key[0] or key[1]) or not key[2]:
                continue
            terminal_targets[key] = max(
                terminal_targets.get(key, -1),
                self._coerce_int(entry.get("frame", 0), 0),
            )

        producer_index = defaultdict(list)
        for record in clean_records:
            for product_key in self._pathway_side_keys(record, "products"):
                producer_index[product_key].append(("clean", record))
        for record in usable_candidates:
            for product_key in self._pathway_side_keys(record, "products"):
                producer_index[product_key].append(("candidate", record))

        candidate_product_keys = {
            key
            for record in usable_candidates
            for key in self._pathway_side_keys(record, "products")
        }
        relevant_targets = {
            key: frame
            for key, frame in terminal_targets.items()
            if key in candidate_product_keys and key not in initial_keys
        }
        for entries in producer_index.values():
            entries.sort(
                key=lambda item: (
                    -_product_frame(item[1]),
                    0 if item[0] == "candidate" else 1,
                    -_reactant_frame(item[1]),
                )
            )

        def _resolve_state(key, deadline, stack):
            if key in initial_keys:
                return set()
            for source_kind, producer in producer_index.get(key, []):
                if _product_frame(producer) > deadline:
                    continue
                if source_kind == "clean":
                    # Clean rows have already passed the authoritative
                    # chronological source check performed by the cleaner.
                    return set()
                producer_id = id(producer)
                if producer_id in stack:
                    continue
                reactants = self._pathway_side_keys(producer, "reactants")
                if not reactants:
                    continue
                branch = {producer_id}
                complete = True
                for reactant_key in reactants:
                    parent = _resolve_state(
                        reactant_key,
                        _reactant_frame(producer),
                        stack | {producer_id},
                    )
                    if parent is None:
                        complete = False
                        break
                    branch.update(parent)
                if complete:
                    return branch
            return None

        required_candidate_ids = set()
        closed_targets = []
        for target_key, target_frame in sorted(
            relevant_targets.items(),
            key=lambda item: (item[1], item[0]),
        ):
            branch = _resolve_state(target_key, target_frame, set())
            if branch is None:
                continue
            required_candidate_ids.update(branch)
            closed_targets.append(target_key)

        targets_by_candidate = defaultdict(list)
        for target_key in closed_targets:
            branch = _resolve_state(
                target_key,
                relevant_targets[target_key],
                set(),
            )
            for candidate_id in branch or ():
                targets_by_candidate[candidate_id].append(target_key)

        for candidate in usable_candidates:
            candidate_id = id(candidate)
            if candidate_id not in required_candidate_ids:
                continue
            target_keys = targets_by_candidate.get(candidate_id, [])
            candidate["Terminal_State_Required"] = True
            candidate["Terminal_State_Target_Count"] = len(target_keys)
            candidate["Terminal_State_Targets"] = self._format_pathway_species_keys(
                set(target_keys)
            )
            candidate["Lineage_Repair_Source"] = (
                "stable_terminal_state_backward_closure"
            )
            prior_evidence = str(
                candidate.get("Lineage_Repair_Evidence", "") or ""
            ).strip(";")
            candidate["Lineage_Repair_Evidence"] = ";".join(
                part for part in (
                    prior_evidence,
                    "stable_final_state;exact_atom_instance;"
                    "backward_closed_to_initial_inventory",
                ) if part
            )

        return required_candidate_ids, {
            "stable_terminal_targets": len(relevant_targets),
            "terminal_targets_with_candidate_producer": len(
                relevant_targets
            ),
            "terminal_targets_closed": len(closed_targets),
        }

    def _promote_repaired_elementary_to_clean_output(
        self,
        clean_df,
        pathway_records,
    ):
        """Append pathway-repaired *elementary* events to the Clean output."""
        if not isinstance(clean_df, pd.DataFrame):
            return 0

        accepted_pathway_ids = set()
        for record in pathway_records or ():
            if not isinstance(record, dict):
                continue
            if not self._truthy(
                record.get("Pathway_Eligible", True)
            ):
                continue
            source_id = str(
                record.get(
                    "Raw_Reaction_ID",
                    record.get("Source_Raw_Reaction_ID", ""),
                )
                or ""
            ).strip()
            if source_id:
                accepted_pathway_ids.add(source_id)

        def source_closure_status(record):
            raw_sources = str(
                record.get("Reactant_Source_Raw_IDs", "") or ""
            ).strip()
            if not raw_sources or raw_sources.lower() in {"nan", "none"}:
                return True, ""

            source_ids = [
                item.strip()
                for item in raw_sources.split("|")
                if item.strip()
            ]
            rejected = []
            for source_id in source_ids:
                # Initial atom-mapped inventory is a legitimate provenance source.
                if source_id.upper() in {"INITIAL", "START", "INITIAL_STATE"}:
                    continue
                status = getattr(self, "raw_clean_status", {}).get(
                    source_id, {}
                )
                exclusion = str(
                    status.get("Exclusion_Reason", "") or ""
                ).strip().lower()
                clean_id = str(
                    status.get("Clean_Reaction_ID", "") or ""
                ).strip()
                if source_id in accepted_pathway_ids:
                    continue
                if clean_id and not exclusion:
                    continue
                rejected.append(
                    f"{source_id}:{exclusion or 'not_pathway_reachable'}"
                )
            if rejected:
                return False, "invalid_reactant_source:" + ";".join(rejected)
            return True, ""

        existing_raw_ids = {
            str(row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or "")
            for row in clean_df.to_dict("records")
        }
        next_clean_number = len(clean_df) + 1
        promoted = []
        source_closure_rejected = 0

        for pathway_rec in pathway_records:
            role = str(pathway_rec.get("Pathway_Role", "") or "")
            is_reactivated_raw = role == "reactivated_raw_elementary"
            is_corridor_net = (
                role == "observed_corridor_net"
                and self._truthy(
                    pathway_rec.get("Observed_Corridor_Net_Validated", False)
                )
                and self._truthy(
                    pathway_rec.get("Observed_Corridor_Net_Elementary", False)
                )
            )
            if not (is_reactivated_raw or is_corridor_net):
                continue
            mode = str(pathway_rec.get("Elementary_Mode", "standard") or "standard")
            if not is_corridor_net and mode != "standard":
                continue
            if (
                not self._truthy(pathway_rec.get("Topology_Validated", True))
                or not _is_missing_scalar(pathway_rec.get("Valence_Warning", ""))
                or not self._truthy(pathway_rec.get("Is_Elementary", True))
                or not self._mass_balanced(pathway_rec)
                or not self._is_elementary_stoichiometry(pathway_rec)
                or self._is_identity(pathway_rec)
            ):
                continue

            source_ok, source_reason = source_closure_status(pathway_rec)
            if not source_ok:
                source_closure_rejected += 1
                pathway_rec["Clean_Promotion_Rejected"] = True
                pathway_rec["Clean_Promotion_Rejection_Reason"] = source_reason
                continue

            raw_id = str(
                pathway_rec.get(
                    "Raw_Reaction_ID", pathway_rec.get("Reaction_ID", "")
                ) or ""
            )
            if not raw_id or raw_id in existing_raw_ids:
                continue

            clean_id = f"RXN-{next_clean_number:04d}"
            next_clean_number += 1
            promoted_rec = dict(pathway_rec)
            promoted_rec.update({
                "Reaction_ID": clean_id,
                "Clean_Reaction_ID": clean_id,
                "Raw_Reaction_ID": raw_id,
                "Clean_Eligible": True,
                "Clean_Tier": (
                    "trajectory_recovered_corridor_net"
                    if is_corridor_net
                    else "lineage_repaired_elementary"
                ),
                "Clean_Repair_Required": True,
                "Clean_Repair_Depends_On": (
                    "validated_settled_endpoint_corridor"
                    if is_corridor_net
                    else "validated_pathway_bridge"
                ),
                "Clean_Repair_Evidence": (
                    "settled_valid_endpoints;intervening_invalid_topology;"
                    "exact_atom_mapping;mass_balanced;elementary;"
                    "source_reachable_after_pathway_repair"
                    if is_corridor_net
                    else "topology_validated;valence_valid;mass_balanced;"
                    "elementary;source_reachable_after_pathway_repair"
                ),
                "Lineage_Complete": True,
                "Lineage_Validation_Context": "pathway_repaired",
                "Pathway_Only": False,
                "Pathway_Source": (
                    "clean:trajectory_recovered_corridor_net"
                    if is_corridor_net else "clean:lineage_repaired"
                ),
                "Pathway_Role": (
                    "repaired_clean_corridor_net"
                    if is_corridor_net else "repaired_clean_elementary"
                ),
                "Pathway_Selection_Reason": (
                    "promoted_to_clean_after_validated_corridor_recovery"
                    if is_corridor_net
                    else "promoted_to_clean_after_validated_lineage_repair"
                ),
            })
            promoted.append(promoted_rec)
            existing_raw_ids.add(raw_id)

            pathway_rec["Clean_Reaction_ID"] = clean_id
            pathway_rec["Clean_Promoted_From_Pathway"] = True
            pathway_rec["Pathway_Only"] = False
            pathway_rec["Pathway_Source"] = (
                "clean:trajectory_recovered_corridor_net"
                if is_corridor_net else "clean:lineage_repaired"
            )
            pathway_rec["Pathway_Role"] = (
                "repaired_clean_corridor_net"
                if is_corridor_net else "repaired_clean_elementary"
            )
            pathway_rec["Evidence_Level"] = (
                "validated_recovered_corridor_net"
                if is_corridor_net else "validated_repaired_elementary"
            )
            pathway_rec["Clean_Tier"] = (
                "trajectory_recovered_corridor_net"
                if is_corridor_net else "lineage_repaired_elementary"
            )
            pathway_rec["Clean_Repair_Required"] = True
            pathway_rec["Clean_Repair_Depends_On"] = (
                "validated_settled_endpoint_corridor"
                if is_corridor_net else "validated_pathway_bridge"
            )
            pathway_rec["Lineage_Validation_Context"] = "pathway_repaired"
            self.raw_clean_status.setdefault(
                raw_id, {"Clean_Reaction_ID": "", "Exclusion_Reason": ""}
            )
            self.raw_clean_status[raw_id]["Clean_Reaction_ID"] = clean_id
            self.raw_clean_status[raw_id]["Exclusion_Reason"] = ""

        if not promoted:
            if source_closure_rejected:
                print(
                    "   ⛔ Clean promotion source-closure rejections: "
                    f"{source_closure_rejected}."
                )
            return 0

        combined = pd.concat(
            [clean_df.copy(), pd.DataFrame(promoted)],
            ignore_index=True,
            sort=False,
        )
        combined = combined.sort_values(
            by=["Frame", "Reaction_ID"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
        existing_attrs = dict(clean_df.attrs)
        combined.attrs.update(existing_attrs)
        combined.attrs["clean_repaired_promotions"] = len(promoted)
        clean_df._update_inplace(combined)
        clean_df.attrs.update(combined.attrs)
        if source_closure_rejected:
            print(
                "   ⛔ Clean promotion source-closure rejections: "
                f"{source_closure_rejected}."
            )
        return len(promoted)

    def build_pathway_table(
        self,
        clean_df=None,
        unresolved_df=None,
        raw_df=None,
        lineage_dropped_df=None,
        strict_reachability=True,
        terminal_state_entries=None,
    ):
        """Build Clean + reachable bridges + terminal-state-closed Raw events."""
        if clean_df is None:
            clean_df = pd.DataFrame()
        if unresolved_df is None:
            unresolved_df = pd.DataFrame()
        if raw_df is None:
            raw_df = (
                self.df
                if isinstance(self.df, pd.DataFrame)
                else pd.DataFrame()
            )

        print(
            "🧭 Building validated lineage pathway "
            "[terminal-state lineage repair v12 + upper-source hover provenance + orphan cleanup]..."
        )

        records = []
        rejected = []
        stats = Counter()
        seen_exact = set()
        clean_loose_keys = set()

        # 1. Clean core
        if not clean_df.empty:
            for row in clean_df.to_dict("records"):
                rec = dict(row)
                if (
                    "Lineage_Complete" in rec
                    and not self._truthy(rec.get("Lineage_Complete", False))
                ):
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "clean_event_has_lineage_gap"
                    )
                    rejected.append(rec)
                    stats["clean_lineage_gap_rejected"] += 1
                    continue

                rec["Clean_Reaction_ID"] = str(
                    rec.get("Reaction_ID", "") or ""
                )
                rec["Raw_Reaction_ID"] = str(
                    rec.get(
                        "Raw_Reaction_ID",
                        rec.get("Reaction_ID", ""),
                    ) or ""
                )
                rec["Pathway_Only"] = False
                rec["Pathway_Eligible"] = True
                rec["Lineage_Gap"] = False
                rec["Reachable_From_Initial"] = True
                rec["Bridge_Required"] = False
                rec["Reactivated_By_Pathway"] = False
                rec["Pathway_Role"] = "clean_elementary"
                rec["Pathway_Selection_Reason"] = (
                    "validated_clean_elementary_event"
                )

                is_recovered = self._truthy(
                    rec.get("Lineage_Recovered", False)
                )
                is_restored = self._truthy(
                    rec.get("Lineage_Restored_By_Cleaner", False)
                )
                if is_recovered:
                    mode = str(rec.get("Recovery_Mode", "") or "")
                    rec["Pathway_Source"] = "clean:lineage_recovered"
                    rec["Evidence_Level"] = (
                        "validated_recovered_elementary"
                    )
                    rec["Pathway_Role"] = (
                        "recovered_connectivity_elementary"
                        if mode == "connectivity_family_formation"
                        else "recovered_state_elementary"
                    )
                    rec["Pathway_Selection_Reason"] = (
                        "validated_atom_mapped_lineage_recovery"
                    )
                    rec["Lineage_Note"] = (
                        "Recovered from atom-mapped trajectory; "
                        f"mode={mode}; "
                        f"boundary={rec.get('Recovery_Original_Boundary_Frame', rec.get('Recovery_Boundary_Frame', ''))}; "
                        f"confidence={rec.get('Recovery_Confidence', '')}"
                    )
                elif is_restored:
                    rec["Pathway_Source"] = "clean:lineage_restored"
                    rec["Evidence_Level"] = (
                        "validated_restored_elementary"
                    )
                    rec["Pathway_Role"] = "restored_lineage_elementary"
                    rec["Pathway_Selection_Reason"] = (
                        "validated_clean_lineage_restoration"
                    )
                    rec["Lineage_Note"] = (
                        "Restored after recrossing cleanup because it is "
                        "required as the nearest source of a retained state."
                    )
                else:
                    rec["Pathway_Source"] = "clean"
                    rec["Evidence_Level"] = "validated_elementary"
                    rec["Lineage_Note"] = ""

                exact_key = self._canonical_event_key(
                    rec, include_frame=True
                )
                if exact_key in seen_exact:
                    stats["clean_duplicate_rejected"] += 1
                    continue

                seen_exact.add(exact_key)
                clean_loose_keys.add(
                    self._canonical_event_key(rec, include_frame=False)
                )
                records.append(rec)
                stats["clean_accepted"] += 1

        clean_core_records = list(records)
        initial_keys = self._initial_keys_from_records(clean_core_records)

        clean_lineage_gaps = (
            self._annotate_clean_lineage(clean_core_records)
            if initial_keys
            else []
        )
        if clean_lineage_gaps:
            clean_missing = {
                entry["key"] for _rec, entry in clean_lineage_gaps
            }
            stats["clean_unreachable_states"] = len(clean_missing)
            print(
                "   ⚠️ Clean input contains unreachable state(s): "
                + self._format_pathway_species_keys(clean_missing)
            )

        raw_ids_by_exact_key = {}
        if raw_df is not None and not raw_df.empty:
            for raw_row in raw_df.to_dict("records"):
                raw_id = str(
                    raw_row.get(
                        "Raw_Reaction_ID",
                        raw_row.get("Reaction_ID", ""),
                    ) or ""
                )
                if raw_id:
                    raw_ids_by_exact_key[
                        self._canonical_event_key(
                            raw_row,
                            include_frame=True,
                        )
                    ] = raw_id

        # 2. Composite candidates
        candidates = []
        if not unresolved_df.empty:
            for row in unresolved_df.to_dict("records"):
                rec = self._pathway_record_from_unresolved(row)
                if rec is None:
                    continue

                stats["projectable_composite_candidates"] += 1
                exact_key = self._canonical_event_key(
                    rec, include_frame=True
                )
                if not str(rec.get("Raw_Reaction_ID", "") or ""):
                    raw_id = raw_ids_by_exact_key.get(exact_key, "")
                    if raw_id:
                        rec["Raw_Reaction_ID"] = raw_id
                        rec["Source_Raw_Reaction_ID"] = raw_id
                if not str(rec.get("Raw_Reaction_ID", "") or ""):
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "candidate_has_no_raw_ledger_provenance"
                    )
                    rejected.append(rec)
                    stats["bridge_missing_raw_provenance_rejected"] += 1
                    continue
                loose_key = self._canonical_event_key(
                    rec, include_frame=False
                )

                if exact_key in seen_exact or loose_key in clean_loose_keys:
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "duplicate_of_validated_clean_event"
                    )
                    rejected.append(rec)
                    stats["bridge_duplicate_rejected"] += 1
                    continue

                rec["Pathway_Role"] = "composite_bridge"
                rec["Evidence_Level"] = "validated_composite_bridge"
                rec["Pathway_Eligible"] = False
                rec["Lineage_Gap"] = True
                rec["Reachable_From_Initial"] = False
                rec["Bridge_Required"] = True
                rec["Reactivated_By_Pathway"] = False
                # A composite bridge is useful, but is less direct evidence
                # than a validated elementary Raw event at the same time.
                rec["Lineage_Repair_Cost"] = 20
                rec["Lineage_Repair_Evidence"] = (
                    "validated_composite_bridge"
                )
                rec["Lineage_Repair_Source"] = "unresolved_bridge"
                rec["_candidate_kind"] = "composite_bridge"
                rec["_candidate_exact_key"] = exact_key
                candidates.append(rec)

        # 3. Lineage-dropped elementary candidates
        if (
            lineage_dropped_df is None
            and not getattr(self, "lineage_dropped_records", None)
            and hasattr(clean_df, "attrs")
        ):
            attr_records = clean_df.attrs.get(
                "lineage_dropped_records", []
            )
            if attr_records:
                lineage_dropped_df = pd.DataFrame(attr_records)

        dropped_pool, dropped_pool_source = (
            self._pathway_lineage_dropped_pool(
                lineage_dropped_df=lineage_dropped_df,
                raw_df=raw_df,
            )
        )
        stats["lineage_dropped_pool_size"] = len(dropped_pool)

        corridor_nets = self._observed_corridor_net_candidates(
            raw_df, seen_exact
        )
        candidates.extend(corridor_nets)
        stats["observed_corridor_net_candidates"] = len(corridor_nets)

        needed_lineage_consumers = {}
        for row in dropped_pool:
            raw_id = str(
                row.get("Raw_Reaction_ID", row.get("Reaction_ID", ""))
                or ""
            )
            reactant_frame = self._coerce_int(
                row.get("Reactant_Frame", row.get("Frame", 0)), 0
            )
            for key in self._pathway_side_keys(row, "reactants"):
                needed_lineage_consumers.setdefault(key, []).append(
                    (reactant_frame, raw_id)
                )
        for matches in needed_lineage_consumers.values():
            matches.sort(key=lambda item: (item[0], item[1]))
        observed_sources = self._observed_stable_state_source_candidates(
            raw_df,
            needed_lineage_consumers,
        )
        candidates.extend(observed_sources)
        stats["observed_stable_state_candidates"] = len(observed_sources)

        for row in dropped_pool:
            rec, reason = self._prepare_reactivated_raw_candidate(
                row, seen_exact
            )
            if rec is None:
                stats[reason] += 1
                continue
            candidates.append(rec)
            stats["reactivatable_raw_candidates"] += 1

        terminal_required_ids, terminal_stats = (
            self._terminal_state_backward_closure_candidates(
                clean_core_records,
                candidates,
                initial_keys,
                terminal_state_entries,
            )
        )
        stats.update(terminal_stats)
        stats["terminal_state_required_candidates"] = len(
            terminal_required_ids
        )
        if terminal_stats["stable_terminal_targets"]:
            print(
                "   🔭 Stable final-state audit: "
                f"targets={terminal_stats['stable_terminal_targets']}; "
                f"backward-closed="
                f"{terminal_stats['terminal_targets_closed']}; "
                f"required candidates={len(terminal_required_ids)}."
            )

        if candidates and strict_reachability and not initial_keys:
            for rec in candidates:
                rec["Pathway_Eligible"] = False
                rec["Pathway_Rejection_Reason"] = (
                    "missing_initial_species_metadata"
                )
                rejected.append(rec)
            stats["candidate_missing_initial_metadata_rejected"] += len(
                candidates
            )
            candidates = []

        kind_priority = {
            "observed_corridor_net": 0,
            "observed_stable_state_source": 1,
            "composite_bridge": 2,
            "reactivated_raw_elementary": 3,
        }
        candidates.sort(
            key=lambda r: (
                self._coerce_int(
                    r.get("Reactant_Frame", r.get("Frame", 0)), 0
                ),
                self._coerce_int(r.get("Lineage_Repair_Cost", 99), 99),
                kind_priority.get(r.get("_candidate_kind", ""), 9),
                self._coerce_int(
                    r.get("Product_Frame", r.get("Frame", 0)), 0
                ),
            )
        )

        # 4. Joint fixed-point admission
        pending = list(candidates)
        for _round in range(max(1, len(pending) + 1)):
            if not pending:
                break

            changed = False
            next_pending = []

            for rec in pending:
                kind = rec.get("_candidate_kind", "")
                reactants = self._pathway_side_keys(rec, "reactants")
                products = self._pathway_side_keys(rec, "products")
                is_observed_source = (
                    kind == "observed_stable_state_source"
                )
                reactant_frame = self._coerce_int(
                    rec.get("Reactant_Frame", rec.get("Frame", 0)), 0
                )

                if not products or (not reactants and not is_observed_source):
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "empty_reactant_or_product_state_set"
                    )
                    rejected.append(rec)
                    stats[f"{kind}_invalid_rejected"] += 1
                    continue

                if initial_keys:
                    reachable_before = self._pathway_reachable_keys(
                        records,
                        initial_keys,
                        cutoff_frame=reactant_frame,
                    )
                    clean_reachable_before = (
                        self._pathway_reachable_keys(
                            clean_core_records,
                            initial_keys,
                            cutoff_frame=reactant_frame,
                        )
                    )
                    missing_reactants = reactants - reachable_before
                else:
                    clean_reachable_before = set()
                    missing_reactants = set()

                if is_observed_source:
                    missing_reactants = set()

                if strict_reachability and missing_reactants:
                    rec["_missing_reactants"] = missing_reactants
                    next_pending.append(rec)
                    continue

                new_products = products - reachable_before

                if not new_products:
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "candidate_adds_no_new_current_inventory_state"
                    )
                    rejected.append(rec)
                    stats[f"{kind}_redundant_rejected"] += 1
                    continue

                activated_reactants = reactants - clean_reachable_before
                if (
                    kind == "reactivated_raw_elementary"
                    and strict_reachability
                    and not activated_reactants
                    and not self._truthy(
                        rec.get("Terminal_State_Required", False)
                    )
                ):
                    rec["Pathway_Eligible"] = False
                    rec["Pathway_Rejection_Reason"] = (
                        "raw_event_not_activated_by_pathway_extension"
                    )
                    rejected.append(rec)
                    stats["reactivated_raw_not_activated_rejected"] += 1
                    continue

                rec.pop("_missing_reactants", None)
                rec.pop("_candidate_exact_key", None)
                rec.pop("_candidate_kind", None)
                rec["Pathway_Eligible"] = True
                rec["Lineage_Gap"] = False
                rec["Reachable_From_Initial"] = True
                rec["Bridge_New_Species"] = (
                    self._format_pathway_species_keys(new_products)
                )
                rec["Missing_Reactant_Keys"] = "[]"
                rec["Missing_Reactant_Species"] = ""
                rec["Lineage_Complete"] = True

                if is_observed_source:
                    rec["Bridge_Required"] = False
                    rec["Reactivated_By_Pathway"] = False
                    rec["Pathway_Selection_Reason"] = (
                        "observed_post_recrossing_state_source"
                    )
                    stats["observed_stable_state_accepted"] += 1
                elif kind == "observed_corridor_net":
                    rec["Bridge_Required"] = True
                    rec["Reactivated_By_Pathway"] = False
                    rec["Pathway_Selection_Reason"] = (
                        "reachable_observed_corridor_net_bridge"
                    )
                    stats["observed_corridor_net_accepted"] += 1
                elif kind == "composite_bridge":
                    rec["Bridge_Required"] = True
                    rec["Reactivated_By_Pathway"] = False
                    rec["Pathway_Selection_Reason"] = (
                        "reachable_composite_bridge_adds_new_state"
                    )
                    stats["composite_bridge_accepted"] += 1
                else:
                    rec["Bridge_Required"] = False
                    rec["Reactivated_By_Pathway"] = True
                    rec["Reactivated_Through_Species"] = (
                        self._format_pathway_species_keys(
                            activated_reactants
                        )
                    )
                    rec["Pathway_Selection_Reason"] = (
                        "elementary_raw_event_reactivated_after_pathway_bridge"
                    )
                    if self._truthy(
                        rec.get("Terminal_State_Required", False)
                    ):
                        rec["Pathway_Selection_Reason"] = (
                            "elementary_raw_event_reactivated_by_stable_"
                            "terminal_state_backward_closure"
                        )
                    rec["Lineage_Note"] = (
                        "Validated elementary Raw event previously removed "
                        "only for a missing upstream source; reactivated after "
                        "Pathway expansion. "
                        f"repair_cost={rec.get('Lineage_Repair_Cost', '')}; "
                        f"repair_evidence={rec.get('Lineage_Repair_Evidence', '')}; "
                        f"repair_source={rec.get('Lineage_Repair_Source', '')}."
                    )
                    stats["reactivated_raw_accepted"] += 1

                records.append(rec)
                seen_exact.add(
                    self._canonical_event_key(rec, include_frame=True)
                )
                changed = True

            pending = next_pending
            if not changed:
                break

        for rec in pending:
            kind = rec.pop("_candidate_kind", "")
            missing = rec.pop("_missing_reactants", set())
            rec.pop("_candidate_exact_key", None)
            rec["Pathway_Eligible"] = False
            rec["Lineage_Gap"] = True
            rec["Reachable_From_Initial"] = False
            rec["Pathway_Rejection_Reason"] = (
                "reactants_not_reachable_from_initial_state"
            )
            rec["Missing_Reactant_Species"] = (
                self._format_pathway_species_keys(missing)
            )
            rejected.append(rec)
            stats[f"{kind}_unreachable_rejected"] += 1

        if not records:
            self.pathway_build_stats = dict(stats)
            self.pathway_rejected_candidates = pd.DataFrame(rejected)
            print("   No validated pathway event was available.")
            return _reaction_id_first(pd.DataFrame())

        # 5. A source anchor is only a fallback. 
        if initial_keys:
            records, retired_observed_sources = (
                self._prune_redundant_observed_state_sources(records)
            )
        else:
            retired_observed_sources = []
        if retired_observed_sources:
            rejected.extend(retired_observed_sources)
            stats["redundant_observed_state_sources_retired"] += len(
                retired_observed_sources
            )
            print(
                "   ♻️ Retired "
                f"{len(retired_observed_sources)} redundant observed "
                "state source(s) with retained exact producers."
            )

        # 6. Final authoritative chronological validation. 
        final_lineage_gaps = (
            self._annotate_clean_lineage(records)
            if initial_keys
            else []
        )
        final_withdrawn_ids = set()
        for _round in range(max(1, len(records) + 1)):
            if not final_lineage_gaps:
                break

            withdraw_ids = {
                id(rec)
                for rec, _entry in final_lineage_gaps
                if self._truthy(rec.get("Pathway_Only", False))
            }

            for gap_record, missing_entry in final_lineage_gaps:
                if self._truthy(
                    gap_record.get("Pathway_Only", False)
                ):
                    continue
                missing_key = missing_entry["key"]
                gap_reactant_frame = self._coerce_int(
                    gap_record.get(
                        "Reactant_Frame",
                        gap_record.get("Frame", 0),
                    ),
                    0,
                )
                consumers = []
                for candidate in records:
                    if not self._truthy(
                        candidate.get("Pathway_Only", False)
                    ):
                        continue
                    candidate_frame = self._coerce_int(
                        candidate.get(
                            "Frame",
                            candidate.get("Product_Frame", 0),
                        ),
                        0,
                    )
                    if candidate_frame > gap_reactant_frame:
                        continue
                    reactant_keys = self._pathway_side_keys(
                        candidate,
                        "reactants",
                    )
                    product_keys = self._pathway_side_keys(
                        candidate,
                        "products",
                    )
                    if (
                        missing_key in reactant_keys
                        and missing_key not in product_keys
                    ):
                        consumers.append(candidate)
                if consumers:
                    culprit = max(
                        consumers,
                        key=lambda rec: (
                            self._coerce_int(
                                rec.get(
                                    "Frame",
                                    rec.get("Product_Frame", 0),
                                ),
                                0,
                            ),
                            self._coerce_int(
                                rec.get("Product_Frame", 0),
                                0,
                            ),
                        ),
                    )
                    withdraw_ids.add(id(culprit))

            withdraw_ids.difference_update(final_withdrawn_ids)
            if not withdraw_ids:
                break

            retained = []
            for rec in records:
                if id(rec) not in withdraw_ids:
                    retained.append(rec)
                    continue
                rec["Pathway_Eligible"] = False
                rec["Lineage_Gap"] = True
                rec["Reachable_From_Initial"] = False
                rec["Lineage_Complete"] = False
                rec["Pathway_Rejection_Reason"] = (
                    "final_atom_inventory_conflict"
                )
                rec["Pathway_Selection_Reason"] = (
                    "withdrawn_after_final_chronological_lineage_check"
                )
                rejected.append(rec)
                stats["final_lineage_conflict_rejected"] += 1
                final_withdrawn_ids.add(id(rec))
            records = retained
            final_lineage_gaps = self._annotate_clean_lineage(records)

        # Absolute safety fallback: Pathway extensions are optional, whereas
        # every Clean row is contractually mandatory. If a conflict cannot be
        # localized uniquely, roll back the remaining extensions rather than
        # publishing an incomplete or Clean-subset-violating Pathway table.
        if final_lineage_gaps:
            clean_fallback = [
                rec
                for rec in records
                if not self._truthy(rec.get("Pathway_Only", False))
            ]
            fallback_gaps = self._annotate_clean_lineage(clean_fallback)
            if not fallback_gaps:
                for rec in records:
                    if not self._truthy(
                        rec.get("Pathway_Only", False)
                    ):
                        continue
                    if id(rec) not in final_withdrawn_ids:
                        rec["Pathway_Eligible"] = False
                        rec["Lineage_Gap"] = True
                        rec["Reachable_From_Initial"] = False
                        rec["Lineage_Complete"] = False
                        rec["Pathway_Rejection_Reason"] = (
                            "final_lineage_safety_rollback"
                        )
                        rejected.append(rec)
                        stats["final_lineage_safety_rollback"] += 1
                records = clean_fallback
                final_lineage_gaps = []

        if final_lineage_gaps:
            final_missing = {
                entry["key"] for _rec, entry in final_lineage_gaps
            }
            stats["final_unreachable_states"] = len(final_missing)
            print(
                "   ⚠️ Final Pathway contains unreachable state(s): "
                + self._format_pathway_species_keys(final_missing)
            )
        else:
            print("   ✅ Final Pathway is source-reachable from initial states.")

        promoted_clean = self._promote_repaired_elementary_to_clean_output(
            clean_df,
            records,
        )
        stats["clean_repaired_elementary_promoted"] = promoted_clean
        if promoted_clean:
            print(
                "   ✅ Promoted "
                f"{promoted_clean} recovered elementary/net event(s) "
                "into the Clean output."
            )

        records.sort(
            key=lambda r: (
                self._coerce_int(
                    r.get("Frame", r.get("Product_Frame", 0)), 0
                ),
                str(r.get("Reactants_Formula", "")),
                str(r.get("Products_Formula", "")),
            )
        )
        for i, rec in enumerate(records, 1):
            rec["Reaction_ID"] = f"PTH-{i:04d}"
            for private_key in (
                "_candidate_exact_key",
                "_missing_reactants",
                "_candidate_kind",
            ):
                rec.pop(private_key, None)
            if "Atom_Indices" in rec:
                rec["Atom_Indices"] = self._parse_indices(
                    rec.get("Atom_Indices", [])
                )

        stats["pathway_total"] = len(records)
        stats["pathway_clean_events"] = sum(
            not self._truthy(rec.get("Pathway_Only", False))
            for rec in records
        )
        stats["pathway_recovered_connectivity_events"] = sum(
            rec.get("Pathway_Role") == "recovered_connectivity_elementary"
            for rec in records
        )
        stats["pathway_recovered_state_events"] = sum(
            rec.get("Pathway_Role") == "recovered_state_elementary"
            for rec in records
        )
        stats["pathway_restored_lineage_events"] = sum(
            rec.get("Pathway_Role") == "restored_lineage_elementary"
            for rec in records
        )
        stats["pathway_composite_bridges"] = sum(
            rec.get("Pathway_Role") == "composite_bridge"
            for rec in records
        )
        stats["pathway_reactivated_raw_events"] = sum(
            rec.get("Pathway_Role") == "reactivated_raw_elementary"
            for rec in records
        )
        stats["pathway_recovered_corridor_clean_events"] = sum(
            rec.get("Pathway_Role") == "repaired_clean_corridor_net"
            for rec in records
        )
        stats["pathway_observed_stable_state_sources"] = sum(
            rec.get("Pathway_Role") == "observed_stable_state_source"
            for rec in records
        )
        stats["pathway_observed_corridor_nets"] = sum(
            rec.get("Pathway_Role") == "observed_corridor_net"
            for rec in records
        )
        stats["pathway_extensions"] = (
            stats["pathway_composite_bridges"]
            + stats["pathway_reactivated_raw_events"]
            + stats["pathway_observed_stable_state_sources"]
            + stats["pathway_observed_corridor_nets"]
            + stats["pathway_recovered_corridor_clean_events"]
        )
        stats["lineage_dropped_pool_source"] = dropped_pool_source

        self.pathway_build_stats = dict(stats)
        self.pathway_rejected_candidates = pd.DataFrame(rejected)

        print(
            "   Pathway summary: "
            f"total={stats['pathway_total']} "
            f"(Clean={stats['pathway_clean_events']}, "
            f"extensions={stats['pathway_extensions']})"
            + "."
        )
        if any(
            (
                stats['pathway_recovered_connectivity_events'],
                stats['pathway_recovered_state_events'],
                stats['pathway_restored_lineage_events'],
                stats['clean_repaired_elementary_promoted'],
                stats['projectable_composite_candidates'],
                stats['reactivatable_raw_candidates'],
                stats['pathway_observed_corridor_nets'],
                stats['pathway_recovered_corridor_clean_events'],
                stats['terminal_state_required_candidates'],
                stats['redundant_observed_state_sources_retired'],
                stats['pathway_composite_bridges'],
                stats['pathway_reactivated_raw_events'],
                stats['composite_bridge_unreachable_rejected']
                + stats['reactivated_raw_elementary_unreachable_rejected'],
            )
        ):
            print(
                "   Pathway audit: "
                + "; ".join(
                    f"{label}={value}"
                    for label, value in (
                        (
                            "recovered_connectivity",
                            stats['pathway_recovered_connectivity_events'],
                        ),
                        (
                            "recovered_state",
                            stats['pathway_recovered_state_events'],
                        ),
                        (
                            "cleaner_restoration",
                            stats['pathway_restored_lineage_events'],
                        ),
                        (
                            "clean_repaired",
                            stats['clean_repaired_elementary_promoted'],
                        ),
                        (
                            "composite_pool",
                            stats['projectable_composite_candidates'],
                        ),
                        (
                            "lineage_dropped",
                            stats['reactivatable_raw_candidates'],
                        ),
                        (
                            "observed_state_sources",
                            stats['pathway_observed_stable_state_sources'],
                        ),
                        (
                            "corridor_nets",
                            stats['pathway_observed_corridor_nets'],
                        ),
                        (
                            "clean_corridor_nets",
                            stats['pathway_recovered_corridor_clean_events'],
                        ),
                        (
                            "terminal_required",
                            stats['terminal_state_required_candidates'],
                        ),
                        (
                            "retired_state_sources",
                            stats['redundant_observed_state_sources_retired'],
                        ),
                        (
                            "bridges",
                            stats['pathway_composite_bridges'],
                        ),
                        (
                            "reactivated_raw",
                            stats['pathway_reactivated_raw_events'],
                        ),
                        (
                            "unreachable",
                            stats['composite_bridge_unreachable_rejected']
                            + stats['reactivated_raw_elementary_unreachable_rejected'],
                        ),
                    )
                    if value
                )
                + "."
            )
        final_withdrawn = (
            stats["final_lineage_conflict_rejected"]
            + stats["final_lineage_safety_rollback"]
        )
        if final_withdrawn:
            print(
                "   Final atom-inventory conflicts withdrawn: "
                f"{final_withdrawn}."
            )

        if stats["pathway_extensions"] == 0:
            if stats["reactivatable_raw_candidates"] > 0:
                print(
                    "   ℹ️ Pathway is identical to Clean; no safe "
                    "Pathway-only extension was found."
                )
        elif stats["pathway_reactivated_raw_events"] > 0:
            print(
                "   🔗 Accepted bridge(s) reactivated a previously excluded "
                "elementary downstream branch."
            )

        return _reaction_id_first(pd.DataFrame(records))

    def _remove_short_time_recrossings(
        self,
        records,
        stats,
        stage,
    ):
        """Recursively remove exact short-window reverse pairs."""
        filtered = list(records)
        removed_this_pass = []
        has_nested = True

        # Cache their normalized forms
        frame_cache = {
            id(record): int(record.get("Frame", 0))
            for record in filtered
        }
        atom_cache = {
            id(record): frozenset(record.get("Atom_Indices", []))
            for record in filtered
        }

        def side_parts(record, side, field):
            if field == "identity":
                text = self._identity_text(record, side)
            else:
                prefix = (
                    "Reactants"
                    if str(side).lower().startswith("r")
                    else "Products"
                )
                text = record.get(f"{prefix}_Formula", "")
            return tuple(sorted(self._parts(text)))

        reverse_signature = {
            id(record): (
                side_parts(record, "reactants", "identity"),
                side_parts(record, "products", "identity"),
                side_parts(record, "reactants", "formula"),
                side_parts(record, "products", "formula"),
            )
            for record in filtered
        }
        surface_state_cache = {
            id(record): self._is_state_only_surface_transition(record)
            for record in filtered
        }
        lineage_recovered_cache = {
            id(record): self._truthy(
                record.get("Lineage_Recovered", False)
            )
            for record in filtered
        }
        atom_mapped_signature = {
            id(record): (
                self._atom_mapped_side_signature(record, "reactants"),
                self._atom_mapped_side_signature(record, "products"),
            )
            for record in filtered
        }

        while has_nested:
            has_nested = False
            keep = [True] * len(filtered)
            for i, curr in enumerate(filtered):
                if not keep[i]:
                    continue
                curr_id = id(curr)
                fi = frame_cache[curr_id]
                atoms_i = atom_cache[curr_id]
                curr_signature = reverse_signature[curr_id]

                for j in range(i + 1, len(filtered)):
                    if not keep[j]:
                        continue
                    nxt = filtered[j]
                    nxt_id = id(nxt)
                    fj = frame_cache[nxt_id]
                    if fj - fi > self.recrossing_window:
                        break

                    atoms_j = atom_cache[nxt_id]
                    nxt_signature = reverse_signature[nxt_id]
                    is_reverse = (
                        curr_signature[0] == nxt_signature[1]
                        and curr_signature[1] == nxt_signature[0]
                        and curr_signature[2] == nxt_signature[3]
                        and curr_signature[3] == nxt_signature[2]
                    )
                    if not (
                        atoms_i
                        and atoms_i == atoms_j
                        and is_reverse
                    ):
                        continue

                    is_surface_recrossing = (
                        surface_state_cache[curr_id]
                        and surface_state_cache[nxt_id]
                    )

                    # Preserve the historical lineage exception only for
                    # connectivity-changing recovered chemistry.  A recovered
                    # pure */(g) transition is still a physical recrossing.
                    lineage_recovered = (
                        lineage_recovered_cache[curr_id]
                        or lineage_recovered_cache[nxt_id]
                    )
                    if lineage_recovered and not is_surface_recrossing:
                        continue

                    intervening = False
                    for k in range(i + 1, j):
                        if not keep[k]:
                            continue
                        atoms_k = atom_cache[id(filtered[k])]
                        if not atoms_i.isdisjoint(atoms_k):
                            intervening = True
                            break

                    if not intervening:
                        curr_mapped = atom_mapped_signature[curr_id]
                        nxt_mapped = atom_mapped_signature[nxt_id]
                        mapped_metadata_complete = all(
                            signature
                            for signature in (
                                curr_mapped[0],
                                curr_mapped[1],
                                nxt_mapped[0],
                                nxt_mapped[1],
                            )
                        )
                        atom_mapped_reverse = (
                            mapped_metadata_complete
                            and curr_mapped[0] == nxt_mapped[1]
                            and curr_mapped[1] == nxt_mapped[0]
                        )

                        if (
                            mapped_metadata_complete
                            and not atom_mapped_reverse
                        ):
                            first_preservation = not any(
                                self._truthy(record.get(
                                    "Atom_Resolved_Recrossing_Mismatch",
                                    False,
                                ))
                                for record in (curr, nxt)
                            )
                            curr[
                                "Atom_Resolved_Recrossing_Mismatch"
                            ] = True
                            nxt[
                                "Atom_Resolved_Recrossing_Mismatch"
                            ] = True
                            if first_preservation:
                                stats[
                                    "atom_exchange_recrossing_preserved"
                                ] += 1
                            continue

                        exclusion_reason = (
                            "short_time_surface_recrossing"
                            if is_surface_recrossing
                            else "short_time_recrossing"
                        )
                        keep[i] = False
                        keep[j] = False

                        for removed in (curr, nxt):
                            was_already_marked = self._truthy(
                                removed.get(
                                    "Removed_As_Short_Time_Recrossing",
                                    False,
                                )
                            )
                            removed[
                                "Removed_As_Short_Time_Recrossing"
                            ] = True
                            removed[
                                "Short_Time_Recrossing_Is_Surface_State"
                            ] = bool(is_surface_recrossing)
                            removed[
                                "Short_Time_Recrossing_Removal_Stage"
                            ] = str(stage)
                            removed_this_pass.append(removed)

                            removed_raw_id = str(
                                removed.get(
                                    "Raw_Reaction_ID",
                                    removed.get("Reaction_ID", ""),
                                ) or ""
                            )
                            if removed_raw_id in self.raw_clean_status:
                                self.raw_clean_status[removed_raw_id][
                                    "Exclusion_Reason"
                                ] = exclusion_reason

                            if not was_already_marked:
                                stats[
                                    "short_time_exact_recrossing"
                                ] += 1
                                if is_surface_recrossing:
                                    stats[
                                        "short_time_surface_recrossing"
                                    ] += 1
                        has_nested = True
                    break

            if has_nested:
                filtered = [
                    record
                    for record, retain in zip(filtered, keep)
                    if retain
                ]

        return filtered, removed_this_pass

    def clean(self):
        if self.df.empty:
            return self.df

        clean_started = time.perf_counter()
        clean_phase_timings = {}
        phase_started = clean_started
        records = self.df.sort_values(['Frame', 'Reactant_Frame', 'Product_Frame'], kind='mergesort').to_dict('records')
        for r in records:
            r['Atom_Indices'] = self._parse_indices(r.get('Atom_Indices', []))

        print("🧹 Cleaning reaction events...")

        filtered = []
        stats = Counter()
        self.raw_clean_status = {}
        for r in records:
            raw_id = str(
                r.get("Raw_Reaction_ID", r.get("Reaction_ID", "")) or ""
            )
            self.raw_clean_status[raw_id] = {
                "Clean_Reaction_ID": "",
                "Exclusion_Reason": "",
            }

            if str(r.get('Is_Elementary', True)).strip().lower() == 'false':
                stats['non_elementary_marked'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = "non_elementary"
                continue
            if (
                not self.allow_ambiguous_topology
                and str(r.get('Clean_Eligible', True)).strip().lower()
                in {'0', 'false', 'no', 'n'}
            ):
                stats['valence_ambiguous_topology'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = "valence_ambiguous"
                continue
            surface_minimum_dwell = (
                self._surface_minimum_dwell_for_row(r)
            )
            if self._unstable_surface_state_transition(
                r,
                minimum_dwell=surface_minimum_dwell,
            ):
                stats['insufficient_surface_state_dwell'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = (
                    "insufficient_surface_state_dwell"
                )
                continue
            if self._is_identity(r):
                stats['identity'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = "identity_reaction"
                continue
            if not self._mass_balanced(r):
                stats['mass_unbalanced'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = "mass_unbalanced"
                continue
            if not self._is_elementary_stoichiometry(r):
                stats['non_elementary_shape'] += 1
                self.raw_clean_status[raw_id]["Exclusion_Reason"] = "non_elementary_stoichiometry"
                continue
            filtered.append(r)

        clean_phase_timings["prefilter"] = (
            time.perf_counter() - phase_started
        )

        # Keep the accepted pre-recrossing pool.  If later filtering makes a
        # retained downstream species source-less, the nearest physically
        # detected formation event can be restored conservatively.
        pre_recrossing_pool = list(filtered)

        phase_started = time.perf_counter()
        filtered, _initial_recrossing_removed = (
            self._remove_short_time_recrossings(
                filtered,
                stats,
                stage="pre_lineage_restore",
            )
        )
        clean_phase_timings["initial_recrossing"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        filtered, restored_events = self._restore_clean_lineage_sources(
            filtered, pre_recrossing_pool
        )
        clean_phase_timings["lineage_restore"] = (
            time.perf_counter() - phase_started
        )
        stats["lineage_source_restore_attempted"] = len(
            restored_events
        )
        stats[
            "lineage_recrossing_restore_blocked"
        ] = getattr(
            self,
            "_lineage_recrossing_restore_blocked",
            0,
        )
        stats[
            "lineage_surface_recrossing_restore_blocked"
        ] = getattr(
            self,
            "_lineage_surface_recrossing_restore_blocked",
            0,
        )

        phase_started = time.perf_counter()
        filtered, post_restore_removed = (
            self._remove_short_time_recrossings(
                filtered,
                stats,
                stage="post_lineage_restore",
            )
        )
        clean_phase_timings["post_restore_recrossing"] = (
            time.perf_counter() - phase_started
        )
        stats["post_restore_recrossing_removed"] = len(
            post_restore_removed
        )

        phase_started = time.perf_counter()
        filtered, lineage_dropped = self._prune_incomplete_lineage_events(
            filtered
        )
        clean_phase_timings["initial_lineage_prune"] = (
            time.perf_counter() - phase_started
        )

        all_lineage_dropped = list(lineage_dropped)
        post_lineage_recross_removed = []
        phase_started = time.perf_counter()
        for round_index in range(max(1, len(filtered) + 1)):
            filtered, newly_recrossed = (
                self._remove_short_time_recrossings(
                    filtered,
                    stats,
                    stage=f"post_lineage_prune_{round_index + 1}",
                )
            )
            post_lineage_recross_removed.extend(newly_recrossed)
            if not newly_recrossed:
                break

            filtered, newly_dropped = (
                self._prune_incomplete_lineage_events(filtered)
            )
            all_lineage_dropped.extend(newly_dropped)
        clean_phase_timings["fixed_point_cleanup"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        lineage_dropped = []
        seen_lineage_dropped = set()
        for dropped in all_lineage_dropped:
            key = id(dropped)
            if key in seen_lineage_dropped:
                continue
            seen_lineage_dropped.add(key)
            lineage_dropped.append(dropped)

        stats["post_lineage_prune_recrossing_removed"] = len(
            post_lineage_recross_removed
        )
        stats['lineage_source_unresolved_dropped'] = len(
            lineage_dropped
        )
        retained_ids = {id(record) for record in filtered}
        stats["lineage_source_restored"] = sum(
            id(record) in retained_ids
            for record in restored_events
        )
        stats["surface_state_lineage_bridges"] = sum(
            self._coerce_int(
                record.get("Surface_State_Lineage_Bridge_Count", 0),
                0,
            )
            for record in filtered
        )
        stats["surface_state_unreachable_bridges_blocked"] = sum(
            self._coerce_int(
                record.get(
                    "Surface_State_Lineage_Bridge_"
                    "Blocked_Unreachable_Count",
                    0,
                ),
                0,
            )
            for record in lineage_dropped
        )

        self.lineage_dropped_records = [dict(r) for r in lineage_dropped]
        self.clean_build_stats = dict(stats)

        for dropped in lineage_dropped:
            dropped_raw_id = str(
                dropped.get(
                    "Raw_Reaction_ID",
                    dropped.get("Reaction_ID", ""),
                ) or ""
            )
            if dropped_raw_id in self.raw_clean_status:
                self.raw_clean_status[dropped_raw_id][
                    "Exclusion_Reason"
                ] = "lineage_incomplete"

        cleaned = self._renumber(filtered)
        for accepted in cleaned:
            accepted_raw_id = str(
                accepted.get(
                    "Raw_Reaction_ID",
                    accepted.get("Reaction_ID", ""),
                ) or ""
            )
            self.raw_clean_status.setdefault(
                accepted_raw_id,
                {"Clean_Reaction_ID": "", "Exclusion_Reason": ""},
            )
            self.raw_clean_status[accepted_raw_id]["Clean_Reaction_ID"] = str(
                accepted.get("Reaction_ID", "") or ""
            )
            self.raw_clean_status[accepted_raw_id]["Exclusion_Reason"] = ""

        for status in self.raw_clean_status.values():
            if (
                not status.get("Clean_Reaction_ID")
                and not status.get("Exclusion_Reason")
            ):
                status["Exclusion_Reason"] = "not_selected_by_cleaner"

        print(
            "   Filter summary: "
            + (
                "; ".join(
                    f"{label}={value}"
                    for label, value in (
                        ("non_elementary", stats['non_elementary_marked']),
                        (
                            "valence_ambiguous",
                            stats['valence_ambiguous_topology'],
                        ),
                        (
                            "surface_dwell",
                            stats['insufficient_surface_state_dwell'],
                        ),
                        ("identity", stats['identity']),
                        ("mass_unbalanced", stats['mass_unbalanced']),
                        (
                            "non_elementary_stoichiometry",
                            stats['non_elementary_shape'],
                        ),
                        (
                            "exact_recrossing",
                            stats['short_time_exact_recrossing'],
                        ),
                    )
                    if value
                )
                or "no events removed"
            )
            + "."
        )
        if (
            stats['short_time_surface_recrossing']
            or stats['lineage_recrossing_restore_blocked']
            or stats['post_restore_recrossing_removed']
            or stats['post_lineage_prune_recrossing_removed']
        ):
            print(
                "   Recrossing audit: "
                + "; ".join(
                    (
                        f"restore_blocked="
                        f"{stats['lineage_recrossing_restore_blocked']} "
                        "(surface="
                        f"{stats['lineage_surface_recrossing_restore_blocked']})"
                        if label == "restore_blocked"
                        else f"{label}={value}"
                    )
                    for label, value in (
                        ("surface_subset", stats['short_time_surface_recrossing']),
                        ("restore_blocked", stats['lineage_recrossing_restore_blocked']),
                        ("post_restore", stats['post_restore_recrossing_removed']),
                        ("post_lineage", stats['post_lineage_prune_recrossing_removed']),
                    )
                    if value and not (
                        label == "surface_subset"
                        and not stats['short_time_exact_recrossing']
                    )
                )
                + "."
            )
        if (
            stats['lineage_source_restored']
            or stats['lineage_source_unresolved_dropped']
            or stats['surface_state_lineage_bridges']
            or stats['atom_exchange_recrossing_preserved']
            or stats['surface_state_unreachable_bridges_blocked']
        ):
            print(
                "   Lineage audit: "
                + "; ".join(
                    f"{label}={value}"
                    for label, value in (
                        (
                            "sources_restored",
                            stats['lineage_source_restored'],
                        ),
                        (
                            "unresolved_islands",
                            stats['lineage_source_unresolved_dropped'],
                        ),
                        (
                            "surface_state_continuity",
                            stats['surface_state_lineage_bridges'],
                        ),
                        (
                            "atom_exchange_preserved",
                            stats['atom_exchange_recrossing_preserved'],
                        ),
                        (
                            "unreachable_state_bridges_blocked",
                            stats[
                                'surface_state_unreachable_bridges_blocked'
                            ],
                        ),
                    )
                    if value
                )
                + "."
            )
        remaining_gaps = sum(
            1 for r in cleaned
            if not self._truthy(r.get('Lineage_Complete', True))
        )
        if remaining_gaps:
            print(
                f"   ⚠️ Clean-table lineage still has {remaining_gaps} event(s) "
                f"with unresolved reactant sources."
            )
        else:
            print("   ✅ Clean-table lineage is source-complete.")
        clean_result = _reaction_id_first(pd.DataFrame(cleaned))
        clean_result.attrs["lineage_dropped_records"] = [
            dict(r) for r in self.lineage_dropped_records
        ]
        clean_result.attrs["clean_build_stats"] = dict(
            self.clean_build_stats
        )
        clean_result.attrs["raw_clean_status"] = dict(
            self.raw_clean_status
        )
        clean_phase_timings["finalization"] = (
            time.perf_counter() - phase_started
        )
        clean_phase_timings["total"] = (
            time.perf_counter() - clean_started
        )
        self.clean_timings = dict(clean_phase_timings)
        print(f"   ⏱️ Cleaning time: {clean_phase_timings['total']:.1f}s.")
        return clean_result

class ReactionClusterer:
    """Count every published Clean record by its assigned event type.

    Atom-mapping exchanges are reported as lineage events, rather than silently
    omitted. Missing labels form an explicit Unclassified category.
    """

    def __init__(self, df):
        self.df = df

    def cluster(self):
        df = self.df if self.df is not None else pd.DataFrame()

        def labels_for(column):
            if column not in df:
                return pd.Series("", index=df.index, dtype=object)
            return df[column].fillna("").astype(str).str.strip()

        primary = labels_for("Type")
        fallback = labels_for("Reaction_Type")
        conflict = primary.ne("") & fallback.ne("") & primary.ne(fallback)
        if conflict.any():
            raise ValueError(
                "Conflicting Type and Reaction_Type labels in "
                f"{int(conflict.sum())} Clean record(s)"
            )
        labels = primary.where(primary.ne(""), fallback)
        mapping = (
            labels_for("Elementary_Mode").eq("atom_mapping_exchange")
            | labels.eq("Atom-Mapping Exchange")
        )
        labels = labels.mask(mapping, "Atom-Mapping Exchange")
        unclassified = labels.eq("") | labels.eq("Unclassified")
        labels = labels.mask(unclassified, "Unclassified")
        counts = labels.value_counts().rename_axis("Reaction Type").reset_index(name="Count")
        counts = counts.sort_values(
            ["Count", "Reaction Type"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        total = int(counts["Count"].sum())
        if total != len(df):
            raise ValueError("Event-type counts do not match the final Clean table")
        counts.attrs["event_type_summary"] = {
            "source_table": "Clean",
            "clean_records": len(df),
            "classified_event_records": int((~mapping & ~unclassified).sum()),
            "atom_mapping_exchange_records": int(mapping.sum()),
            "unclassified_records": int(unclassified.sum()),
            "plotted_records": total,
            "event_types": len(counts),
            "all_categories_shown": True,
        }
        return counts


class ReactionStats:
    OUTPUT_COLUMNS = [
        'Reactants_Formula',
        'Products_Formula',
        'Reactants_SMILES',
        'Products_SMILES',
        'Reaction_Type',
        'Total_Count',
        'Representative_ID',
        'First_Frame',
        'Last_Frame',
    ]

    def __init__(self, df):
        self.df = df

    @staticmethod
    def _split_species_text(value):
        if not isinstance(value, str):
            return []
        return [
            part.strip()
            for part in value.split(' + ')
            if part.strip()
        ]

    @classmethod
    def _state_side_key(cls, row, side):
        """Canonical state-aware key for one reaction side."""
        prefix = (
            'Reactants'
            if str(side).lower().startswith('r')
            else 'Products'
        )
        identities = cls._split_species_text(
            row.get(
                f'{prefix}_Identity',
                row.get(f'{prefix}_SMILES', ''),
            )
        )
        formulas = cls._split_species_text(
            row.get(f'{prefix}_Formula', '')
        )
        count = max(len(identities), len(formulas))
        pairs = []
        for index in range(count):
            identity = (
                identities[index]
                if index < len(identities)
                else ''
            )
            formula = (
                formulas[index]
                if index < len(formulas)
                else ''
            )
            pairs.append((str(identity), str(formula)))
        return tuple(sorted(pairs))

    def generate_stats(self):
        if self.df is None or self.df.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        df = self.df.copy()
        # Atom-mapping exchanges are retained in Clean/Pathway to make exact atom lineage closed.
        if "Elementary_Mode" in df.columns:
            df = df.loc[
                ~df["Elementary_Mode"].astype(str).eq(
                    "atom_mapping_exchange"
                )
            ].copy()
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)
        if 'Type' not in df.columns and 'Reaction_Type' in df.columns:
            df['Type'] = df['Reaction_Type']
        if 'Type' not in df.columns:
            df['Type'] = ''
        if 'Reaction_ID' not in df.columns:
            df['Reaction_ID'] = ''
        if 'Structure_Representative_ID' not in df.columns:
            df['Structure_Representative_ID'] = df['Reaction_ID']
        else:
            representative_ids = df['Structure_Representative_ID'].fillna('').astype(str)
            missing_representative = representative_ids.str.strip().eq('')
            representative_ids.loc[missing_representative] = (
                df.loc[missing_representative, 'Reaction_ID'].astype(str)
            )
            df['Structure_Representative_ID'] = representative_ids
        if 'Reactants_Identity' not in df.columns:
            df['Reactants_Identity'] = df.get('Reactants_SMILES', '')
        if 'Products_Identity' not in df.columns:
            df['Products_Identity'] = df.get('Products_SMILES', '')

        for column in [
            'Reactants_SMILES',
            'Products_SMILES',
            'Reactants_Formula',
            'Products_Formula',
        ]:
            if column not in df.columns:
                df[column] = ''

        df['Frame'] = pd.to_numeric(
            df.get('Frame', pd.Series(pd.NA, index=df.index)),
            errors='coerce',
        ).astype('Int64')
        # Sorting first guarantees that Representative_ID and the displayed
        # Formula/SMILES come from the same earliest event in each group.
        df = df.sort_values(
            ['Frame', 'Reaction_ID'],
            kind='mergesort',
            na_position='last',
        )
        df['_Reactants_State_Key'] = [
            self._state_side_key(row, 'reactants')
            for _, row in df.iterrows()
        ]
        df['_Products_State_Key'] = [
            self._state_side_key(row, 'products')
            for _, row in df.iterrows()
        ]

        stats = df.groupby(
            [
                '_Reactants_State_Key',
                '_Products_State_Key',
                'Type',
            ],
            dropna=False,
            sort=False,
        ).agg(
            Reactants_SMILES=('Reactants_SMILES', 'first'),
            Products_SMILES=('Products_SMILES', 'first'),
            Reactants_Formula=('Reactants_Formula', 'first'),
            Products_Formula=('Products_Formula', 'first'),
            Total_Count=('Reaction_ID', 'size'),
            Representative_ID=('Structure_Representative_ID', 'first'),
            First_Frame=('Frame', 'min'),
            Last_Frame=('Frame', 'max'),
        ).reset_index()

        stats = stats.rename(columns={'Type': 'Reaction_Type'})
        stats = stats.loc[:, self.OUTPUT_COLUMNS]
        stats = stats.sort_values(
            [
                'Total_Count',
                'First_Frame',
                'Reactants_Formula',
                'Products_Formula',
            ],
            ascending=[False, True, True, True],
            kind='mergesort',
            na_position='last',
        ).reset_index(drop=True)
        return stats

class IntermediateAnalyzer:
    REPORT_COLUMNS = [
        'Formula',
        'SMILES',
        'Species_Role',
        'Event_Count',
        'N_Appearances',
        'First_Appearance_Frame',
        'Last_Appearance_Frame',
        'Max_Lifetime_Frames',
        'Structure_File',
    ]

    def __init__(
        self,
        df,
        frames,
        traj,
        output_dir='./intermediate_structures',
        detector=None,
    ):
        self.df = df
        self.frames = frames
        self.traj = traj
        self.output_dir = output_dir
        self.detector = detector
        self._write_failures = 0
        self._reconstruct_failures = 0

    @staticmethod
    def _split_species_text(value):
        if not isinstance(value, str):
            return []
        return [part.strip() for part in value.split(' + ') if part.strip()]

    @classmethod
    def _side_species(cls, row, side):
        smiles_values = cls._split_species_text(
            row.get(f'{side}_SMILES', '')
        )
        formula_values = cls._split_species_text(
            row.get(f'{side}_Formula', '')
        )
        pairs = []
        for index, smiles in enumerate(smiles_values):
            formula = (
                formula_values[index]
                if index < len(formula_values)
                else ''
            )
            pairs.append((smiles, formula))
        return pairs

    @staticmethod
    def _species_state_key(smiles, formula):
        return (
            str(smiles or '').strip(),
            str(formula or '').strip(),
        )

    def _frame_state_formula(self, molecule, frame_idx):
        """Return a trajectory molecule formula with its adsorption state."""
        formula = str(molecule.get('formula', '') or '').strip()
        if not formula:
            return ''
        if (
            self.detector is None
            or self.traj is None
            or frame_idx < 0
            or frame_idx >= len(self.traj)
            or not (
                getattr(self.detector, 'sub_elements', set())
                or getattr(self.detector, 'sub_indices', set())
            )
        ):
            return formula

        base_formula = formula
        if hasattr(self.detector, '_strip_surface_suffix'):
            base_formula = self.detector._strip_surface_suffix(formula)
        else:
            if base_formula.endswith('*'):
                base_formula = base_formula[:-1]
            if base_formula.endswith('(g)'):
                base_formula = base_formula[:-3]
            base_formula = base_formula.strip()

        try:
            adsorbed = self.detector._is_adsorbed(
                self.traj[frame_idx],
                molecule.get('indices', []),
            )
        except Exception as exc:
            logger.debug(
                "Could not resolve adsorption state at frame %s: %s",
                frame_idx,
                exc,
            )
            return formula
        return base_formula + ('*' if adsorbed else '(g)')

    @staticmethod
    def _species_role(
        species_key,
        data,
        n_appearances,
        initial_species,
        terminal_species,
    ):
        produced = sorted(set(int(x) for x in data['production_frames']))
        consumed = sorted(set(int(x) for x in data['consumption_frames']))

        if (
            produced
            and consumed
            and (
                (len(produced) >= 2 and len(consumed) >= 2)
                or (
                    n_appearances >= 2
                    and len(produced) + len(consumed) >= 3
                )
            )
        ):
            return 'recurrent_species'

        first_produced = produced[0] if produced else None
        first_consumed = consumed[0] if consumed else None

        if (
            species_key in initial_species
            and consumed
            and (
                first_produced is None
                or first_consumed <= first_produced
            )
        ):
            return 'initial_reactant'

        if produced and consumed and any(
            production_frame < consumption_frame
            for production_frame in produced
            for consumption_frame in consumed
        ):
            return 'intermediate'

        if produced:
            last_produced = produced[-1]
            consumed_after_production = any(
                frame > last_produced for frame in consumed
            )
            if (
                not consumed_after_production
                or species_key in terminal_species
            ):
                return 'final_product'

        return 'transient'

    def find_intermediates(self):
        if self.df is None or self.df.empty:
            return pd.DataFrame(columns=self.REPORT_COLUMNS)
        if self.traj and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print("🕵️  Building species role and lifetime report...")

        species_counter = {}
        for event_position, (row_index, row) in enumerate(
            self.df.iterrows()
        ):
            if str(
                row.get("Elementary_Mode", "") or ""
            ).strip() == "atom_mapping_exchange":
                continue
            event_id = str(
                row.get('Reaction_ID', f'event-{row_index}') or
                f'event-{row_index}'
            )
            try:
                event_frame = int(float(row.get('Frame', row_index)))
            except (TypeError, ValueError):
                event_frame = int(event_position)

            event_species = set()
            for side, frame_key in (
                ('Products', 'production_frames'),
                ('Reactants', 'consumption_frames'),
            ):
                for smiles, formula in self._side_species(row, side):
                    species_key = self._species_state_key(
                        smiles,
                        formula,
                    )
                    if species_key not in species_counter:
                        species_counter[species_key] = {
                            'smiles': smiles,
                            'formula': formula,
                            'event_ids': set(),
                            'event_frames': set(),
                            'production_frames': [],
                            'consumption_frames': [],
                        }
                    data = species_counter[species_key]
                    if formula and not data.get('formula'):
                        data['formula'] = formula
                    data[frame_key].append(event_frame)
                    data['event_frames'].add(event_frame)
                    event_species.add(species_key)

            for species_key in event_species:
                species_counter[species_key]['event_ids'].add(event_id)

        # Reuse the exact compact presence index built during Mapping States.
        presence_intervals = {}
        presence_frames = defaultdict(set)
        representative = {}
        initial_species = set()
        terminal_species = set()

        compact_index = getattr(
            self.detector,
            '_species_presence_index',
            None,
        )
        compact_index_complete = bool(
            getattr(self.detector, '_presence_index_complete', False)
        )
        if compact_index_complete and compact_index is not None:
            last_trajectory_frame = len(self.frames) - 1
            for species_key, entry in compact_index.items():
                bounds = entry.get('intervals', ())
                if not bounds:
                    continue
                presence_intervals[species_key] = bounds
                representative[species_key] = entry.get(
                    'representative',
                    (int(bounds[0]), tuple()),
                )
                if int(bounds[0]) == 0:
                    initial_species.add(species_key)
                if int(bounds[-1]) == last_trajectory_frame:
                    terminal_species.add(species_key)
        elif self.frames:
            for frame_idx, mol_list in enumerate(self.frames):
                if not mol_list:
                    continue
                for mol in mol_list:
                    smiles = str(
                        mol.get('smiles')
                        or mol.get('canonical_smiles')
                        or ''
                    ).strip()
                    if not smiles:
                        continue
                    formula = self._frame_state_formula(
                        mol,
                        frame_idx,
                    )
                    species_key = self._species_state_key(
                        smiles,
                        formula,
                    )
                    presence_frames[species_key].add(frame_idx)
                    if species_key not in representative:
                        representative[species_key] = (
                            frame_idx,
                            list(mol.get('indices', [])),
                        )
                    if frame_idx == 0:
                        initial_species.add(species_key)
                    if frame_idx == len(self.frames) - 1:
                        terminal_species.add(species_key)

        sorted_species = sorted(
            species_counter.items(),
            key=lambda item: (
                -len(item[1]['event_ids']),
                str(item[1].get('formula', '')),
                str(item[1].get('smiles', '')),
            ),
        )
        results = []

        for internal_species_id, (species_key, data) in enumerate(
            sorted_species,
            1,
        ):
            smiles = data.get('smiles', species_key[0])
            formula = data.get('formula', '')
            bounds = presence_intervals.get(species_key, ())
            if bounds:
                n_appearances = len(bounds) // 2
                first_frame = int(bounds[0])
                last_frame = int(bounds[-1])
                max_lifetime = max(
                    int(bounds[i + 1]) - int(bounds[i]) + 1
                    for i in range(0, len(bounds), 2)
                )
            else:
                frame_set = set(
                    presence_frames.get(species_key, set())
                )
                # A topology-cache mismatch should not erase all timing context.
                if not frame_set:
                    frame_set.update(data.get('event_frames', set()))
                segments = self._segments_from_frames(frame_set)
                n_appearances = len(segments)
                first_frame = min(frame_set) if frame_set else pd.NA
                last_frame = max(frame_set) if frame_set else pd.NA
                max_lifetime = int(max(segments)) if segments else 0
            role = self._species_role(
                species_key,
                data,
                n_appearances,
                initial_species,
                terminal_species,
            )

            xyz_path = ''
            if self.traj and species_key in representative:
                target_frame, indices = representative[species_key]
                xyz_path = self._extract_and_write_structure(
                    internal_species_id,
                    formula,
                    smiles,
                    role,
                    target_frame,
                    indices,
                )
                if xyz_path == 'N/A':
                    xyz_path = ''

            results.append({
                'Formula': formula,
                'SMILES': smiles,
                'Species_Role': role,
                'Event_Count': len(data['event_ids']),
                'N_Appearances': n_appearances,
                'First_Appearance_Frame': first_frame,
                'Last_Appearance_Frame': last_frame,
                'Max_Lifetime_Frames': max_lifetime,
                'Structure_File': xyz_path,
            })

        if self._reconstruct_failures or self._write_failures:
            logger.warning(
                "IntermediateAnalyzer: %d reconstruction failure(s), "
                "%d write failure(s).",
                self._reconstruct_failures,
                self._write_failures,
            )

        report = pd.DataFrame(results, columns=self.REPORT_COLUMNS)
        for column in [
            'Event_Count',
            'N_Appearances',
            'First_Appearance_Frame',
            'Last_Appearance_Frame',
            'Max_Lifetime_Frames',
        ]:
            report[column] = pd.to_numeric(
                report[column], errors='coerce'
            ).astype('Int64')
        return report

    @staticmethod
    def _segments_from_frames(frame_set):
        """
        Split frame indices into continuous segments and return their lengths.

        Example: {2,3,4, 7, 9,10} -> [3, 1, 2].
        """
        if not frame_set:
            return []
        sorted_frames = sorted(int(frame) for frame in frame_set)
        segments = []
        segment_length = 1
        for previous, current in zip(
            sorted_frames,
            sorted_frames[1:],
        ):
            if current == previous + 1:
                segment_length += 1
            else:
                segments.append(segment_length)
                segment_length = 1
        segments.append(segment_length)
        return segments

    def _extract_and_write_structure(
        self,
        species_id,
        formula,
        smiles,
        role,
        frame_idx,
        indices,
    ):
        """Extract a representative structure and return its written path."""
        atoms = self._extract_structure(frame_idx, indices)
        if atoms is None:
            return "N/A"

        atoms.set_cell([0, 0, 0])
        atoms.set_pbc([False, False, False])
        atoms.info['Source_Frame'] = int(frame_idx)
        atoms.info['Formula'] = str(formula)
        atoms.info['Species_ID'] = int(species_id)
        atoms.info['SMILES'] = str(smiles)
        atoms.info['Species_Role'] = str(role)

        safe_formula = (
            formula.replace('/', '_')
            .replace('\\', '_')
            .replace(' ', '')
        )
        fname = f"species_{species_id:02d}_{safe_formula}.xyz"
        fpath = os.path.join(self.output_dir, fname)

        try:
            write(fpath, atoms)
            return fpath
        except (OSError, ValueError, RuntimeError) as exc:
            self._write_failures += 1
            logger.warning(
                "Failed to write species structure to %s: %s",
                fpath,
                exc,
            )
            return "N/A"

    def _extract_structure(self, frame_idx, indices):
        """
        Extract one species structure. Prefer detector-assisted PBC
        reconstruction and fall back to a direct trajectory slice.
        """
        if (
            not indices
            or self.traj is None
            or frame_idx < 0
            or frame_idx >= len(self.traj)
        ):
            return None

        if (
            self.detector is not None
            and hasattr(self.detector, '_reconstruct_local_atoms')
        ):
            try:
                atoms, _, _status = (
                    self.detector._reconstruct_local_atoms(
                        frame_idx,
                        list(indices),
                    )
                )
                if atoms is not None and len(atoms) > 0:
                    return atoms
            except (
                ValueError,
                IndexError,
                KeyError,
                RuntimeError,
            ) as exc:
                self._reconstruct_failures += 1
                logger.warning(
                    "PBC reconstruction failed for species at frame %d "
                    "(%d atoms): %s. Falling back to naive slice.",
                    frame_idx,
                    len(indices),
                    exc,
                )

        try:
            full_atoms = self.traj[frame_idx]
            sub = full_atoms[list(indices)]
            if (
                hasattr(full_atoms, 'pbc')
                and np.any(full_atoms.get_pbc())
            ):
                logger.debug(
                    "Species at frame %d extracted without PBC "
                    "reconstruction. Pass detector= to "
                    "IntermediateAnalyzer for proper unwrapping.",
                    frame_idx,
                )
            return sub
        except (IndexError, ValueError) as exc:
            self._reconstruct_failures += 1
            logger.warning(
                "Failed to extract structure at frame %d: %s",
                frame_idx,
                exc,
            )
            return None
