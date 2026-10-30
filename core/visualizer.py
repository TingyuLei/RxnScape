"""file: visualizer.py"""
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd 
import re
from collections import defaultdict, deque, OrderedDict
import networkx as nx  

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError: 
    HAS_PLOTLY = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError: 
    HAS_PYVIS = False


ELEMENT_FLOW_ELEMENTS = tuple("""
H He Li Be B C N O F Ne
Na Mg Al Si P S Cl Ar K Ca
Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr
Nb Mo Tc Ru Rh Pd Ag Cd In Sn
Sb Te I Xe Cs Ba La Ce Pr Nd
Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th
Pa U Np Pu Am Cm Bk Cf Es Fm
Md No Lr Rf Db Sg Bh Hs Mt Ds
Rg Cn Nh Fl Mc Lv Ts Og
""".split())

# Bump this whenever Sankey lineage semantics change so a terminal log can
# prove which visualizer implementation produced an HTML output.
SANKEY_LINEAGE_VISUALIZER_VERSION = (
    "initial-state closure + bounded-root selection v2"
)

# Combined role-color, font, drag, and no-audit-CSV revision.
SANKEY_PRESENTATION_PATCH_VERSION = "roles-font16-transition-drag-no-element-audit-csv-v2"


def _normalize_element_flow_selection(value):
    """Return a canonical, ordered list of requested element-flow views."""
    if value is None:
        return []
    if value is False:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "off", "false", "[]"}:
            return []
        raw_elements = [part.strip() for part in text.split(",")]
    elif isinstance(value, set):
        raw_elements = sorted(
            str(part).strip() for part in value if part is not None
        )
    elif isinstance(value, (list, tuple)):
        raw_elements = [str(part).strip() for part in value if part is not None]
    else:
        raw_elements = [str(value).strip()]

    canonical_by_lower = {
        element.lower(): element
        for element in ELEMENT_FLOW_ELEMENTS
    }
    selected = []
    seen = set()
    invalid = []
    for raw_element in raw_elements:
        if not raw_element:
            continue
        element = canonical_by_lower.get(raw_element.lower())
        if element is None:
            invalid.append(raw_element)
        elif element not in seen:
            selected.append(element)
            seen.add(element)

    if invalid:
        raise ValueError(
            "Unsupported element-flow selection(s): "
            f"{invalid}. Every entry must be a valid IUPAC element symbol "
            "from H through Og (for example C, N, Si, Cl, or Fe)."
        )
    return selected

class Visualizer:
    def __init__(self, output_dir='./', smiles_name_dict=None): 
        self.output_dir = output_dir
        self.smiles_name_dict = smiles_name_dict or {}
        
        # --- Publication-Quality (ACS/JCIM) Matplotlib Settings ---
        # font.family 必须写成「实际安装的字体列表」，不能用 'sans-serif' 别名：
        # 只有列表形式才会启用 matplotlib 的逐字形回退 —— Arial 缺少下标字符
        # （₀₁₂…）时自动用 DejaVu Sans 补位。用别名时整串化学式会退回方框。
        from matplotlib import font_manager

        installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
        font_stack = [
            name for name in ('Arial', 'Helvetica', 'DejaVu Sans')
            if name in installed_fonts
        ] or ['DejaVu Sans']

        plt.rcParams.update({
            'font.family': font_stack,
            'font.sans-serif': font_stack + ['DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.linewidth': 1.5,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.fontsize': 11,
            'legend.frameon': False,
            'figure.dpi': 300,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': False,
            'lines.linewidth': 2.0
        })
            
    def _to_subscript(self, formula):
        """Convert numbers in a chemical formula string to Unicode subscripts."""
        if not isinstance(formula, str): return formula
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return formula.translate(subscript_map)
        
    def plot_molecule_evolution(self, df_mols):
        if df_mols.empty: return
        print("📊 Plotting Molecule Evolution...")
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.gca()
        
        from cycler import cycler
        nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                         '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
        ax.set_prop_cycle(cycler(color=nature_colors))

        initial_cols = [
            col
            for col in df_mols.columns
            if pd.notna(df_mols[col].iloc[0])
            and float(df_mols[col].iloc[0]) > 0.0
        ]
        initial_col_set = set(initial_cols)
        variance_order = list(
            df_mols.var().sort_values(ascending=False).index
        )
        dynamic_cols = [
            col for col in variance_order
            if col not in initial_col_set
        ]
        display_count = max(10, len(initial_cols))
        selected_cols = initial_cols + dynamic_cols[
            :max(0, display_count - len(initial_cols))
        ]
        data_to_plot = df_mols[selected_cols]
        window = max(1, len(df_mols) // 50)

        smoothed = data_to_plot.rolling(
            window=window,
            min_periods=1,
            center=False,
        ).mean()

        for col in smoothed.columns:
            display_label = self._to_subscript(col)
            plt.plot(smoothed.index, smoothed[col], label=display_label,
                     linewidth=2.5, alpha=0.9, solid_capstyle='round')
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#222222')

        ax.tick_params(direction='in', length=6, width=1.5, colors='#222222', labelsize=12)
        ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.15, color='#999999')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                   frameon=False, fontsize=11, handlelength=2.5)

        plt.title('Species Population Evolution (Causal Moving Average)', fontsize=14, fontweight='bold', pad=12, color='#111111')
        plt.xlabel('Frame', fontsize=13, fontweight='bold', color='#111111')
        plt.ylabel('Counts', fontsize=13, fontweight='bold', color='#111111')
        plt.tight_layout()
        out_base = os.path.join(self.output_dir, 'molecule_evolution')
        plt.savefig(f"{out_base}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{out_base}.pdf", dpi=600, bbox_inches='tight')
        plt.close()

    def plot_bonds(self, df_bonds):
        if df_bonds.empty: return
        print("🔗 Plotting Bond Evolution...")
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.gca()

        from cycler import cycler
        nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
                         '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
        ax.set_prop_cycle(cycler(color=nature_colors))

        top_bonds = df_bonds.var().sort_values(ascending=False).head(8).index
        window = max(1, len(df_bonds) // 40)
        # Keep the reported initial bond inventory exact while smoothing only
        # over the current and preceding frames.
        smoothed = df_bonds[top_bonds].rolling(
            window=window,
            min_periods=1,
            center=False,
        ).mean()
        
        for b in top_bonds:
            display_label = self._to_subscript(b)
            plt.plot(smoothed.index, smoothed[b], label=display_label, 
                     linewidth=2.5, alpha=0.9, solid_capstyle='round')

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#222222')

        ax.tick_params(direction='in', length=6, width=1.5, colors='#222222', labelsize=12)
        ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.15, color='#999999')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                   frameon=False, fontsize=11, handlelength=2.5)

        plt.title('Chemical Bond Evolution (Causal Moving Average)', fontsize=14, fontweight='bold', pad=12, color='#111111')
        plt.xlabel('Frame', fontsize=13, fontweight='bold', color='#111111')
        plt.ylabel('Bond Counts', fontsize=13, fontweight='bold', color='#111111')

        plt.tight_layout()
        out_base = os.path.join(self.output_dir, 'bond_evolution')
        plt.savefig(f"{out_base}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{out_base}.pdf", dpi=600, bbox_inches='tight')
        plt.close()


    def plot_reaction_network(
        self,
        df_events,
        max_nodes=None,
        max_edges=None,
        initial_species=None,
        strict_lineage=False,
        write_integrity_csv=False,
        show_pathway_provenance=True,
    ):
        """Plot a state-aware, lineage-audited comprehensive reaction network."""

        if df_events.empty or not HAS_PYVIS:
            return

        print("🕸️ Plotting Reaction Network...")

        import html
        import json
        import re

        def _parts(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text = str(value).strip()
            if not text:
                return []
            return [x.strip() for x in re.split(r"\s+\+\s+", text) if x.strip()]

        def _species_key(smi, formula):
            return (str(smi).strip(), str(formula).strip())

        def _decode_key_payload(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            try:
                payload = json.loads(str(value))
            except Exception:
                return []
            out = []
            for item in payload if isinstance(payload, list) else []:
                if isinstance(item, dict):
                    out.append(_species_key(
                        item.get("smiles", ""),
                        item.get("formula", ""),
                    ))
            return out

        species_counts = defaultdict(int)
        species_formula = {}
        species_smiles = {}
        edge_weights = defaultdict(int)
        edge_types = defaultdict(set)
        edge_reactions = defaultdict(set)
        edge_sources = defaultdict(set)
        edge_roles = defaultdict(set)
        edge_evidence_classes = defaultdict(set)

        def _event_edge_class(row):
            """Map both legacy and compact Pathway schemas to display classes."""
            role = str(row.get("Pathway_Role", "") or "").strip().lower()
            legacy_source = str(
                row.get("Pathway_Source", "") or ""
            ).strip().lower()
            source_table = str(
                row.get("Source_Table", "") or ""
            ).strip().lower()

            if (
                role in {"composite_bridge", "unresolved_bridge"}
                or source_table == "unresolved"
                or legacy_source.startswith("unresolved:")
            ):
                return "composite_bridge"

            if (
                role in {
                    "reactivated_raw_elementary",
                    "reactivated_raw",
                    "validated_lineage_gap",
                }
                or (
                    source_table == "raw"
                    and role not in {
                        "clean_elementary",
                        "recovered_elementary",
                    }
                )
                or legacy_source.startswith("raw:lineage_reactivated")
            ):
                return "reactivated_raw"

            if (
                role in {
                    "recovered_elementary",
                    "recovered_connectivity_elementary",
                    "recovered_state_elementary",
                    "restored_lineage_elementary",
                }
                or legacy_source.startswith("clean:lineage_recovered")
                or legacy_source.startswith("clean:lineage_restored")
            ):
                return "recovered_lineage"

            return "clean"

        initial_nodes = set()
        # A Pathway can contain an atom-mapped state anchor recovered from a
        # settled trajectory observation after an invalid topology corridor.
        observed_state_source_nodes = set()
        # Provenance text for non-chemical observed-state anchors.  
        upper_sources_by_node = defaultdict(set)
        direct_missing_nodes = set()

        for _, row in df_events.iterrows():
            r_smi = _parts(row.get("Reactants_SMILES", ""))
            p_smi = _parts(row.get("Products_SMILES", ""))
            r_form = _parts(row.get("Reactants_Formula", ""))
            p_form = _parts(row.get("Products_Formula", ""))
            rxn_type = str(
                row.get("Type", row.get("Reaction_Type", "")) or ""
            )
            rxn_id = str(
                row.get("Reaction_ID", row.get("Pathway_ID", "")) or ""
            )
            pathway_source = str(
                row.get(
                    "Pathway_Source",
                    row.get("Source_Table", "clean"),
                ) or "clean"
            )
            source_reaction_id = str(
                row.get("Source_Reaction_ID", "") or ""
            )
            if source_reaction_id:
                pathway_source = (
                    f"{pathway_source}:{source_reaction_id}"
                )
            pathway_role = str(row.get("Pathway_Role", "") or "")
            edge_class = (
                _event_edge_class(row)
                if show_pathway_provenance
                else "clean"
            )

            r_entries = [
                (_species_key(s, f), s, f)
                for s, f in zip(r_smi, r_form)
            ]
            p_entries = [
                (_species_key(s, f), s, f)
                for s, f in zip(p_smi, p_form)
            ]

            pathway_role_normalized = pathway_role.strip().lower()
            elementary_mode_normalized = str(
                row.get("Elementary_Mode", "") or ""
            ).strip().lower()
            is_observed_state_source = (
                pathway_role_normalized == "observed_stable_state_source"
                or elementary_mode_normalized == "observed_state_source"
            )
            if is_observed_state_source and not r_entries:
                observed_state_source_nodes.update(
                    key for key, _smi, _formula in p_entries
                )
                upper_source = str(
                    row.get("Upper_Source", "") or ""
                ).strip()
                if upper_source:
                    for key, _smi, _formula in p_entries:
                        upper_sources_by_node[key].add(upper_source)

            for key, smi, formula in r_entries + p_entries:
                species_smiles[key] = smi
                species_formula[key] = formula

            # Count participation once per pathway event, even when a species
            # appears on both sides of the same event.
            event_species = {
                key for key, _smi, _formula in r_entries + p_entries
            }
            for key in event_species:
                species_counts[key] += 1

            row_initial_keys = _decode_key_payload(
                row.get("Initial_Species_Keys", "")
            )
            if row_initial_keys:
                initial_nodes.update(row_initial_keys)
            else:
                initial_nodes.update(
                    _decode_key_payload(row.get("Initial_Reactant_Keys", ""))
                )
            direct_missing_nodes.update(
                _decode_key_payload(row.get("Missing_Reactant_Keys", ""))
            )

            equation = (
                f"{row.get('Reactants_Formula', '')} → "
                f"{row.get('Products_Formula', '')}"
            )
            if rxn_id:
                equation = f"{rxn_id}: {equation}"

            def _elements(formula):
                return set(re.findall(r"[A-Z][a-z]*", str(formula)))

            for r_key, _r_smi, r_formula in r_entries:
                r_elements = _elements(r_formula)
                for p_key, _p_smi, p_formula in p_entries:
                    if r_key == p_key:
                        continue
                    p_elements = _elements(p_formula)
                    if (
                        r_elements
                        and p_elements
                        and not r_elements.intersection(p_elements)
                    ):
                        continue

                    # Present all accepted evidence as one directed edge.
                    # Provenance remains available in the hover text.
                    edge_key = (r_key, p_key)
                    edge_weights[edge_key] += 1
                    if rxn_type:
                        edge_types[edge_key].add(rxn_type)
                    edge_reactions[edge_key].add(equation)
                    if pathway_source:
                        edge_sources[edge_key].add(pathway_source)
                    if pathway_role:
                        edge_roles[edge_key].add(pathway_role)
                    if show_pathway_provenance:
                        edge_evidence_classes[edge_key].add(edge_class)

        # Allow explicit initial species from a caller/config.
        for requested in initial_species or []:
            req = str(requested).strip()
            for key in species_counts:
                smi, formula = key
                if req in {
                    smi,
                    formula,
                    str(self._to_subscript(formula)),
                }:
                    initial_nodes.add(key)

        # An observed-state anchor is useful only when a retained event uses that exact state.  
        edge_nodes = {
            node
            for edge in edge_weights
            for node in edge
        }
        orphan_observed_sources = (
            observed_state_source_nodes - edge_nodes
        )
        if orphan_observed_sources:
            for node in orphan_observed_sources:
                species_counts.pop(node, None)
                species_formula.pop(node, None)
                species_smiles.pop(node, None)
            observed_state_source_nodes.difference_update(
                orphan_observed_sources
            )
            print(
                "   Removed "
                f"{len(orphan_observed_sources)} orphan observed-state "
                "source node(s) with no retained consumer."
            )

        # Build the complete graph before any optional display cap.  
        full_graph = nx.DiGraph()
        full_graph.add_nodes_from(species_counts.keys())
        for (u, v), weight in edge_weights.items():
            full_graph.add_edge(u, v, weight=weight)

        components = sorted(
            nx.weakly_connected_components(full_graph),
            key=len,
            reverse=True,
        )
        component_of = {}
        for component_id, members in enumerate(components, start=1):
            for node in members:
                component_of[node] = component_id

        zero_indegree = {
            n for n in full_graph.nodes
            if full_graph.in_degree(n) == 0
        }

        source_seed_nodes = initial_nodes | observed_state_source_nodes

        reachable = set()
        if source_seed_nodes:
            for source in source_seed_nodes:
                if source not in full_graph:
                    continue
                reachable.add(source)
                reachable.update(nx.descendants(full_graph, source))

        unreachable_nodes = (
            set(full_graph.nodes) - reachable
            if source_seed_nodes
            else set()
        )
        unexpected_sources = (
            zero_indegree - source_seed_nodes
            if source_seed_nodes
            else set(direct_missing_nodes)
        )

        detached_component_ids = set()
        if source_seed_nodes:
            for component_id, members in enumerate(components, start=1):
                if not set(members).intersection(source_seed_nodes):
                    detached_component_ids.add(component_id)

        audit_rows = []
        seen_audit = set()

        def _add_audit(issue, severity, node=None, details=""):
            formula = species_formula.get(node, "") if node is not None else ""
            smi = species_smiles.get(node, "") if node is not None else ""
            component_id = component_of.get(node, "") if node is not None else ""
            key = (issue, formula, smi, component_id, details)
            if key in seen_audit:
                return
            seen_audit.add(key)
            audit_rows.append({
                "Issue_Type": issue,
                "Severity": severity,
                "Component_ID": component_id,
                "Species_Formula": formula,
                "Species_SMILES": smi,
                "Details": details,
            })

        for node in sorted(
            direct_missing_nodes,
            key=lambda n: (species_formula.get(n, ""), species_smiles.get(n, "")),
        ):
            if node in full_graph:
                _add_audit(
                    "Missing_Source",
                    "error",
                    node,
                    "Reaction detector/cleaner reports no initial or earlier "
                    "formation event for this reactant state.",
                )

        for node in sorted(
            unexpected_sources,
            key=lambda n: (species_formula.get(n, ""), species_smiles.get(n, "")),
        ):
            _add_audit(
                "Unexpected_Zero_Indegree",
                "error",
                node,
                "Non-initial state has no incoming edge in the complete event network.",
            )

        for node in sorted(
            unreachable_nodes - direct_missing_nodes,
            key=lambda n: (species_formula.get(n, ""), species_smiles.get(n, "")),
        ):
            _add_audit(
                "Unreachable_From_Initial_State",
                "error",
                node,
                "No directed event path from any recorded initial state.",
            )

        if len(components) > 1:
            for component_id, members in enumerate(components, start=1):
                formulas = sorted(
                    {species_formula.get(n, str(n)) for n in members}
                )
                if component_id in detached_component_ids:
                    severity = "error"
                    issue = "Detached_Component"
                    detail = (
                        "Component contains no recorded initial state: "
                        + ", ".join(formulas[:20])
                    )
                else:
                    severity = "info"
                    issue = "Additional_Component"
                    detail = (
                        "Separate component containing an initial state: "
                        + ", ".join(formulas[:20])
                    )
                _add_audit(issue, severity, None, f"Component {component_id}; {detail}")
                audit_rows[-1]["Component_ID"] = component_id

        audit_columns = [
            "Issue_Type",
            "Severity",
            "Component_ID",
            "Species_Formula",
            "Species_SMILES",
            "Details",
        ]
        audit_df = pd.DataFrame(audit_rows, columns=audit_columns)

        audit_path = os.path.join(
            self.output_dir, "reaction_network_integrity.csv"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        if write_integrity_csv:
            audit_df.to_csv(audit_path, index=False)
        else:
            try:
                if os.path.exists(audit_path):
                    os.remove(audit_path)
            except Exception:
                pass

        error_rows = (
            audit_df[audit_df["Severity"] == "error"]
            if not audit_df.empty
            else audit_df
        )

        print(
            "   Reaction-network integrity check "
            + ("failed." if not error_rows.empty else "passed.")
        )
        if not error_rows.empty:
            if write_integrity_csv:
                print(
                    f"   ⚠️ {len(error_rows)} lineage/connectivity issue(s) "
                    f"written to {audit_path}"
                )
            else:
                print(
                    f"   ⚠️ {len(error_rows)} lineage/connectivity issue(s) "
                    f"detected."
                )
            for _, issue in error_rows.head(20).iterrows():
                label = issue["Species_Formula"] or "(component)"
                print(
                    f"      {issue['Issue_Type']}: {label} "
                    f"[component {issue['Component_ID']}]"
                )
        if strict_lineage and not error_rows.empty:
            print(
                "   Reaction network was not generated because "
                "strict_lineage=True."
            )
            return

        sorted_species = sorted(
            species_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if max_nodes is not None and int(max_nodes) > 0:
            if len(sorted_species) > int(max_nodes):
                print(
                    "   ⚠️ max_nodes truncation is enabled; the displayed graph "
                    "may omit bridge species. The integrity audit still uses the "
                    "complete graph."
                )
            sorted_species = sorted_species[:int(max_nodes)]

        valid_nodes = {key for key, _count in sorted_species}

        sorted_edges = sorted(
            edge_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_edges = [
            ((u, v), w)
            for (u, v), w in sorted_edges
            if u in valid_nodes and v in valid_nodes
        ]
        if max_edges is not None and int(max_edges) > 0:
            if len(top_edges) > int(max_edges):
                print(
                    "   ⚠️ max_edges truncation is enabled; the displayed graph "
                    "may omit bridge reactions."
                )
            top_edges = top_edges[:int(max_edges)]

        connected_nodes = set()
        for (u, v), _weight in top_edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        connected_nodes.update(
            key for key in observed_state_source_nodes
            if key in valid_nodes
        )

        if not connected_nodes:
            print("   ⚠️ No displayable reaction edge was found.")
            return

        formula_to_keys = defaultdict(list)
        smiles_to_keys = defaultdict(list)
        for key, _count in sorted_species:
            formula_label = self._to_subscript(species_formula[key])
            formula_to_keys[formula_label].append(key)
            smiles_to_keys[species_smiles[key]].append(key)

        species_key_to_label = {}
        for formula_label, key_list in formula_to_keys.items():
            if len(key_list) == 1:
                key = key_list[0]
                smi = species_smiles[key]
                if smi in self.smiles_name_dict and len(smiles_to_keys[smi]) == 1:
                    species_key_to_label[key] = self.smiles_name_dict[smi]
                else:
                    species_key_to_label[key] = formula_label
            else:
                for i, key in enumerate(key_list, start=1):
                    smi = species_smiles[key]
                    if (
                        smi in self.smiles_name_dict
                        and len(smiles_to_keys[smi]) == 1
                    ):
                        species_key_to_label[key] = self.smiles_name_dict[smi]
                    else:
                        species_key_to_label[key] = f"{formula_label}-{i}"

        if top_edges:
            max_log_weight = max(
                np.log1p(weight) for _edge, weight in top_edges
            )
            max_log_weight = max(float(max_log_weight), 1.0)
        else:
            max_log_weight = 1.0

        num_nodes = len(connected_nodes)
        is_small_network = num_nodes < 35
        if is_small_network:
            spring_len = 280
            central_gravity = 0.01
            avoid_overlap = 0.8
            font_size = 22
            min_base_size = 35
            char_width_multiplier = 5.0
            log_multiplier = 6.0
        else:
            spring_len = 180
            central_gravity = 0.025
            avoid_overlap = 0.4
            font_size = 16
            min_base_size = 20
            char_width_multiplier = 3.8
            log_multiplier = 4.0

        net = Network(
            height="100vh",
            width="100vw",
            bgcolor="#ffffff",
            directed=True,
            cdn_resources="in_line",
        )

        cmap = plt.get_cmap("YlGnBu_r")
        visible_species_keys = [
            key
            for key, _count in sorted_species
            if key in connected_nodes
        ]
        visible_rank = {
            key: rank
            for rank, key in enumerate(visible_species_keys)
        }
        max_visible_rank = max(1, len(visible_species_keys) - 1)
        node_id_map = {}
        node_fill_colors = {}

        direct_problem = set(direct_missing_nodes) | set(unexpected_sources)
        downstream_problem = set(unreachable_nodes) - direct_problem

        for idx, (key, count) in enumerate(sorted_species):
            if key not in connected_nodes:
                continue

            species_id = idx + 1
            smi = species_smiles[key]
            formula = species_formula[key]
            pyvis_node_id = f"species_{species_id}"
            node_id_map[key] = pyvis_node_id

            rank_fraction = visible_rank[key] / max_visible_rank
            color_position = 0.05 + 0.60 * rank_fraction
            rgb = cmap(color_position)[:3]
            color_hex = mcolors.to_hex(rgb)
            luminance = (
                0.299 * rgb[0]
                + 0.587 * rgb[1]
                + 0.114 * rgb[2]
            )
            text_color = "#ffffff" if luminance < 0.55 else "#111111"
            node_fill_colors[key] = color_hex

            raw_id = str(species_id).strip()
            raw_name = str(
                species_key_to_label.get(key, formula)
            ).strip()
            if len(raw_id) < 3:
                raw_id = raw_id.center(3, " ")
            if len(raw_name) < 3:
                raw_name = raw_name.center(3, " ")

            label_text = f"{raw_id}\n{raw_name}"

            if key in initial_nodes:
                lineage_label = "INITIAL SOURCE"
                border_color = "#2CA02C"
                border_width = 4
            elif key in observed_state_source_nodes:
                lineage_label = "REACHABLE (OBSERVED TRAJECTORY EVIDENCE)"
                border_color = color_hex
                border_width = 0
            elif key in direct_problem:
                lineage_label = "MISSING DIRECT SOURCE"
                border_color = color_hex
                border_width = 0
            elif key in downstream_problem:
                lineage_label = "UNREACHABLE FROM INITIAL STATE"
                border_color = color_hex
                border_width = 0
            else:
                lineage_label = "REACHABLE"
                border_color = color_hex
                border_width = 0

            title_text = "\n".join([
                f"ID: {species_id}",
                f"Formula: {formula}",
                f"SMILES: {smi}",
                f"Event participation: {count}",
            ])
            upper_sources = sorted(upper_sources_by_node.get(key, set()))
            if upper_sources:
                title_text += "<br>" + "<br>".join(
                    f"<b>Upper source:</b> {html.escape(source)}"
                    for source in upper_sources
                )

            max_line_len = max(
                len(str(species_id)),
                len(raw_name.strip()),
            )
            dynamic_base_size = max(
                min_base_size,
                max_line_len * char_width_multiplier,
            )
            final_size = (
                dynamic_base_size
                + np.log1p(count) * log_multiplier
            )

            net.add_node(
                pyvis_node_id,
                label=label_text,
                title=title_text,
                color={
                    "background": color_hex,
                    "border": border_color,
                    "highlight": {
                        "background": color_hex,
                        "border": "#222222",
                    },
                    "hover": {
                        "background": color_hex,
                        "border": border_color,
                    },
                },
                shape="circle",
                size=final_size,
                borderWidth=border_width,
                borderWidthSelected=max(4, border_width),
                font={
                    "face": "arial",
                    "size": font_size,
                    "color": text_color,
                    "bold": True,
                    "align": "center",
                },
            )

        for edge_index, ((u, v), weight) in enumerate(
            top_edges, start=1
        ):
            if u not in node_id_map or v not in node_id_map:
                continue

            normalized_width = min(
                4.0,
                max(
                    1.0,
                    1.0
                    + (np.log1p(weight) / max_log_weight) * 3.0,
                ),
            )
            edge_key = (u, v)
            types = sorted(edge_types.get(edge_key, []))
            reactions = sorted(edge_reactions.get(edge_key, []))
            sources = sorted(edge_sources.get(edge_key, []))
            roles = sorted(edge_roles.get(edge_key, []))
            tooltip_lines = [f"Occurrences: {weight}"]

            if types:
                type_label = "Type" if len(types) == 1 else "Types"
                tooltip_lines.append(
                    f"{type_label}: {', '.join(types)}"
                )

            source_ids = []
            for source in sources:
                source_text = str(source).strip()
                source_tail = source_text.rsplit(":", 1)[-1].strip()
                if re.match(
                    r"^(?:RXN|PTH|RAW|UNR)-",
                    source_tail,
                    flags=re.IGNORECASE,
                ):
                    source_ids.append(source_tail)
            source_ids = sorted(set(source_ids))
            if source_ids:
                source_label = (
                    "Source" if len(source_ids) == 1 else "Sources"
                )
                tooltip_lines.append(
                    f"{source_label}: {', '.join(source_ids)}"
                )

            if reactions:
                shown = reactions[:6]
                reaction_label = (
                    "Reaction" if len(reactions) == 1 else "Reactions"
                )
                tooltip_lines.append(f"{reaction_label}:")
                tooltip_lines.extend(f"• {item}" for item in shown)
                if len(reactions) > len(shown):
                    tooltip_lines.append(
                        f"… and {len(reactions) - len(shown)} more"
                    )

            # Geometry is determined only by the unordered node pair, never by row/edge insertion order.  
            if u == v:
                smooth = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.24,
                }
            elif (v, u) in edge_weights:
                smooth = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.12,
                }
            else:
                smooth = {"enabled": False}

            net.add_edge(
                node_id_map[u],
                node_id_map[v],
                id=f"edge_{edge_index}",
                width=normalized_width,
                title="\n".join(tooltip_lines),
                color={
                    "color": node_fill_colors[u],
                    "highlight": node_fill_colors[u],
                    "hover": node_fill_colors[u],
                    "inherit": False,
                    "opacity": 0.60,
                },
                dashes=False,
                smooth=smooth,
                arrows={"to": {"enabled": True, "scaleFactor": 0.50}},
                arrowStrikethrough=False,
            )

        options_str = f"""
        var options = {{
          "layout": {{
            "randomSeed": 42,
            "improvedLayout": true
          }},
          "physics": {{
            "forceAtlas2Based": {{
              "gravitationalConstant": -150,
              "centralGravity": {central_gravity},
              "springLength": {spring_len},
              "springConstant": 0.05,
              "damping": 0.09,
              "avoidOverlap": {avoid_overlap}
            }},
            "solver": "forceAtlas2Based",
            "stabilization": {{
              "enabled": true,
              "iterations": 400,
              "updateInterval": 25,
              "fit": true
            }}
          }},
          "interaction": {{
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true,
            "multiselect": true,
            "navigationButtons": false
          }},
          "nodes": {{
            "shadow": {{"enabled": false}},
            "font": {{
              "align": "center"
            }}
          }},
          "edges": {{
            "smooth": false,
            "selectionWidth": 2,
            "hoverWidth": 1.5
          }}
        }}
        """
        net.set_options(options_str)

        out_path = os.path.join(
            self.output_dir, "reaction_network.html"
        )

        try:
            net.save_graph(out_path)

            with open(out_path, "r", encoding="utf-8") as handle:
                html_content = handle.read()

            # vis-network renders the interactive graph on a canvas.  
            publication_export_js = r"""
            <script>
            (function () {
              "use strict";
              var SVG_NS = "http://www.w3.org/2000/svg";

              function esc(value) {
                return String(value == null ? "" : value)
                  .replace(/&/g, "&amp;")
                  .replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;")
                  .replace(/"/g, "&quot;")
                  .replace(/'/g, "&apos;");
              }

              function asColor(value, fallback) {
                if (typeof value === "string") return value;
                if (value && typeof value.color === "string") {
                  return value.color;
                }
                if (value && typeof value.background === "string") {
                  return value.background;
                }
                return fallback;
              }

              // `edges.get()` retains the data originally supplied to pyvis,
              // whereas vis-network resolves that data into the actual style
              // it draws.  The latter is essential for a faithful export.
              function renderedEdgeOptions(edge) {
                var runtimeEdge = network.body && network.body.edges &&
                  network.body.edges[String(edge.id)];
                return runtimeEdge && runtimeEdge.options ? runtimeEdge.options : edge;
              }

              function edgeStyle(edge) {
                var opts = renderedEdgeOptions(edge);
                var colorValue = opts.color != null ? opts.color : edge.color;
                var color = asColor(colorValue, asColor(edge.color, "#555555"));
                var opacity = colorValue && typeof colorValue.opacity === "number"
                  ? colorValue.opacity : 0.60;
                return {
                  color: color,
                  width: Math.max(1, Number(opts.width || edge.width || 1)),
                  opacity: opacity,
                  smooth: opts.smooth != null ? opts.smooth : edge.smooth,
                  dashes: opts.dashes != null ? opts.dashes : edge.dashes,
                  arrows: opts.arrows != null ? opts.arrows : edge.arrows
                };
              }

              function toArrowEnabled(arrows) {
                if (arrows === "to" || arrows === "to, from" || arrows === "from, to") return true;
                return !!(arrows && arrows.to && arrows.to.enabled !== false);
              }

              function dashAttribute(dashes) {
                if (!dashes) return "";
                if (Array.isArray(dashes)) return ' stroke-dasharray="' + dashes.join(" ") + '"';
                return ' stroke-dasharray="6 5"';
              }

              function download(url, filename) {
                var link = document.createElement("a");
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                link.remove();
              }

              function buildSvg() {
                if (!window.network || !window.nodes || !window.edges) {
                  throw new Error("The network has not finished loading.");
                }

                var positions = network.getPositions();
                var nodeList = nodes.get().filter(function (node) {
                  return !node.hidden && positions[node.id];
                });
                if (!nodeList.length) {
                  throw new Error("There are no visible network nodes to export.");
                }

                var margin = 110;
                var minX = Infinity, minY = Infinity;
                var maxX = -Infinity, maxY = -Infinity;
                nodeList.forEach(function (node) {
                  var p = positions[node.id];
                  var radius = Math.max(8, Number(node.size || 20));
                  minX = Math.min(minX, p.x - radius);
                  maxX = Math.max(maxX, p.x + radius);
                  minY = Math.min(minY, p.y - radius);
                  maxY = Math.max(maxY, p.y + radius);
                });
                minX -= margin;
                minY -= margin;
                maxX += margin;
                maxY += margin;

                var width = Math.ceil(maxX - minX);
                var height = Math.ceil(maxY - minY);
                var nodeById = {};
                nodeList.forEach(function (node) { nodeById[node.id] = node; });
                var svg = [];
                svg.push('<?xml version="1.0" encoding="UTF-8"?>');
                svg.push('<svg xmlns="' + SVG_NS + '" width="' + width +
                  '" height="' + height + '" viewBox="' + minX + ' ' +
                  minY + ' ' + width + ' ' + height + '">');
                svg.push('<title>Complete reaction network</title>');
                svg.push('<rect x="' + minX + '" y="' + minY + '" width="' +
                  width + '" height="' + height + '" fill="#ffffff"/>');
                svg.push('<defs>');

                var edgeList = edges.get().filter(function (edge) {
                  return !edge.hidden && nodeById[edge.from] && nodeById[edge.to];
                });
                edgeList.forEach(function (edge, index) {
                  var style = edgeStyle(edge);
                  if (!toArrowEnabled(style.arrows)) return;
                  svg.push('<marker id="arrow-' + index + '" markerWidth="9" '
                    + 'markerHeight="9" refX="8" refY="4.5" orient="auto" '
                    + 'markerUnits="strokeWidth"><path d="M 0 0 L 9 4.5 L 0 9 z" '
                    + 'fill="' + esc(style.color) + '"/></marker>');
                });
                svg.push('</defs>');
                svg.push('<g id="reaction-edges" fill="none" stroke-linecap="round">');
                edgeList.forEach(function (edge, index) {
                  var source = positions[edge.from];
                  var target = positions[edge.to];
                  var style = edgeStyle(edge);
                  var color = style.color;
                  var widthPx = style.width;
                  var d;
                  if (edge.from === edge.to) {
                    var loop = Math.max(28, Number(nodeById[edge.from].size || 20) * 1.5);
                    d = 'M ' + source.x + ' ' + (source.y - 3) + ' C ' +
                      (source.x + loop) + ' ' + (source.y - loop) + ' ' +
                      (source.x - loop) + ' ' + (source.y - loop) + ' ' +
                      (source.x - 3) + ' ' + source.y;
                  } else {
                    var dx = target.x - source.x, dy = target.y - source.y;
                    var distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
                    // End on the visible node rims.  This prevents the marker
                    // from being hidden beneath the target circle.
                    var sourceRadius = Math.max(8, Number(nodeById[edge.from].size || 20));
                    var targetRadius = Math.max(8, Number(nodeById[edge.to].size || 20));
                    var startX = source.x + dx / distance * (sourceRadius + 1);
                    var startY = source.y + dy / distance * (sourceRadius + 1);
                    var endX = target.x - dx / distance * (targetRadius + 3);
                    var endY = target.y - dy / distance * (targetRadius + 3);
                    if (style.smooth && style.smooth.enabled) {
                      var bend = (style.smooth.roundness || 0.12) * distance;
                      if (style.smooth.type === "curvedCCW") bend *= -1;
                      var cx = (source.x + target.x) / 2 - dy / distance * bend;
                      var cy = (source.y + target.y) / 2 + dx / distance * bend;
                      d = 'M ' + startX + ' ' + startY + ' Q ' + cx + ' ' + cy +
                        ' ' + endX + ' ' + endY;
                    } else {
                      d = 'M ' + startX + ' ' + startY + ' L ' + endX + ' ' + endY;
                    }
                  }
                  svg.push('<path d="' + d + '" stroke="' + esc(color) +
                    '" stroke-width="' + widthPx + '" stroke-opacity="' + style.opacity + '"'
                    + dashAttribute(style.dashes)
                    + (toArrowEnabled(style.arrows) ? ' marker-end="url(#arrow-' + index + ')"' : '')
                    + '/>');
                });
                svg.push('</g><g id="reaction-nodes">');
                nodeList.forEach(function (node) {
                  var p = positions[node.id];
                  var radius = Math.max(8, Number(node.size || 20));
                  var fill = asColor(node.color, "#d9e8f5");
                  var border = node.color && node.color.border || fill;
                  var borderWidth = Math.max(0, Number(node.borderWidth || 0));
                  var font = node.font || {};
                  var fontSize = Number(font.size || 16);
                  var fontColor = font.color || "#111111";
                  var lines = String(node.label || node.id).split(/\n/);
                  var firstY = p.y - (lines.length - 1) * fontSize * 0.56;
                  svg.push('<g><title>' + esc(node.title || node.label || node.id) + '</title>');
                  svg.push('<circle cx="' + p.x + '" cy="' + p.y + '" r="' + radius +
                    '" fill="' + esc(fill) + '" stroke="' + esc(border) +
                    '" stroke-width="' + borderWidth + '"/>');
                  lines.forEach(function (line, lineIndex) {
                    svg.push('<text x="' + p.x + '" y="' +
                      (firstY + lineIndex * fontSize * 1.12) +
                      '" text-anchor="middle" dominant-baseline="middle" '
                      + 'font-family="Arial, Helvetica, sans-serif" font-size="' + fontSize +
                      '" font-weight="700" fill="' + esc(fontColor) + '">' +
                      esc(line) + '</text>');
                  });
                  svg.push('</g>');
                });
                svg.push('</g></svg>');
                return svg.join('');
              }

              window.exportReactionNetworkSvg = function (openForPdf) {
                try {
                  var svg = buildSvg();
                  var blob = new Blob([svg], {type: "image/svg+xml;charset=utf-8"});
                  var url = URL.createObjectURL(blob);
                  if (openForPdf) {
                    window.open(url, "_blank", "noopener");
                    window.setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
                  } else {
                    download(url, "reaction_network_full.svg");
                    window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
                  }
                } catch (err) {
                  window.alert("Publication export unavailable: " + err.message);
                }
              };

              window.setTimeout(function () {
                var panel = document.createElement("div");
                panel.id = "reaction-network-publication-export";
                panel.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:10000;"
                  + "padding:10px 11px;border:1px solid #B8B8B8;border-radius:7px;"
                  + "background:rgba(255,255,255,0.95);color:#222;font:13px Arial,sans-serif;"
                  + "box-shadow:0 2px 8px rgba(0,0,0,0.12);";
                panel.innerHTML = '<div style="font-weight:700;margin-bottom:6px">Publication export</div>'
                  + '<button type="button" style="margin-right:5px;padding:6px 9px;border:1px solid #888;border-radius:4px;background:#F4F4F4;cursor:pointer">Download SVG</button>'
                  + '<button type="button" style="padding:6px 9px;border:1px solid #888;border-radius:4px;background:#F4F4F4;cursor:pointer">Open SVG for PDF</button>'
                  + '<div style="margin-top:6px;color:#555;font-size:11px">Vector labels and arrows; wait for layout to settle.</div>';
                var buttons = panel.getElementsByTagName("button");
                buttons[0].onclick = function () { window.exportReactionNetworkSvg(false); };
                buttons[1].onclick = function () { window.exportReactionNetworkSvg(true); };
                document.body.appendChild(panel);
              }, 0);
            }());

            /*
             * Native Canvas publication export.  This deliberately does not
             * translate vis-network primitives to SVG: a second renderer
             * cannot reproduce Canvas arrowheads, curves, font metrics, or
             * anti-aliasing exactly.  Instead, vis-network redraws its own
             * Canvas at a larger viewport and this routine saves those pixels.
             */
            (function () {
              "use strict";

              function later() {
                return new Promise(function (resolve) {
                  window.requestAnimationFrame(function () {
                    window.requestAnimationFrame(resolve);
                  });
                });
              }

              function saveBlob(blob, filename) {
                var url = URL.createObjectURL(blob);
                var link = document.createElement("a");
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                link.remove();
                window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
              }

              function canvasForNetwork() {
                if (window.network && network.canvas && network.canvas.frame &&
                    network.canvas.frame.canvas) {
                  return network.canvas.frame.canvas;
                }
                var graph = document.getElementById("mynetwork");
                return graph ? graph.querySelector("canvas") : null;
              }

              async function exportNativeCanvas(multiplier, fitFirst) {
                if (!window.network) throw new Error("The network has not finished loading.");
                multiplier = Math.max(1, Math.min(6, Number(multiplier) || 1));

                var canvas = canvasForNetwork();
                if (!canvas) throw new Error("The vis-network canvas is unavailable.");
                var frame = network.canvas && network.canvas.frame;
                var view = network.getViewPosition();
                var scale = network.getScale();
                var cssWidth = canvas.clientWidth || canvas.width;
                var cssHeight = canvas.clientHeight || canvas.height;
                var oldFrameWidth = frame && frame.style ? frame.style.width : "";
                var oldFrameHeight = frame && frame.style ? frame.style.height : "";

                try {
                  if (fitFirst) {
                    network.fit({animation: false});
                    await later();
                    view = network.getViewPosition();
                    scale = network.getScale();
                  }

                  // Enlarging both the viewport and scale by the same factor
                  // yields the exact same visible world, with more pixels.
                  network.setSize(Math.round(cssWidth * multiplier) + "px",
                                  Math.round(cssHeight * multiplier) + "px");
                  network.moveTo({position: view, scale: scale * multiplier,
                                  animation: false});
                  network.redraw();
                  await later();

                  canvas = canvasForNetwork();
                  var bitmap = document.createElement("canvas");
                  bitmap.width = canvas.width;
                  bitmap.height = canvas.height;
                  var context = bitmap.getContext("2d");
                  // Canvas output is transparent outside the graph.  Fill it
                  // explicitly so PNG viewers never substitute a black matte.
                  context.fillStyle = "#ffffff";
                  context.fillRect(0, 0, bitmap.width, bitmap.height);
                  context.drawImage(canvas, 0, 0);

                  var filename = fitFirst
                    ? "reaction_network_full_native_" + multiplier + "x.png"
                    : "reaction_network_view_native_" + multiplier + "x.png";
                  await new Promise(function (resolve, reject) {
                    bitmap.toBlob(function (blob) {
                      if (!blob) {
                        reject(new Error("The browser could not encode the PNG."));
                        return;
                      }
                      saveBlob(blob, filename);
                      resolve();
                    }, "image/png");
                  });
                } finally {
                  network.setSize(cssWidth + "px", cssHeight + "px");
                  if (frame && frame.style) {
                    frame.style.width = oldFrameWidth;
                    frame.style.height = oldFrameHeight;
                  }
                  network.moveTo({position: view, scale: scale, animation: false});
                  network.redraw();
                }
              }

              window.exportReactionNetworkNativePng = function (fitFirst) {
                exportNativeCanvas(6, !!fitFirst).catch(function (err) {
                  window.alert("Native PNG export unavailable: " + err.message);
                });
              };

              // Replace the old SVG controls, including for HTML files made
              // with an earlier exporter.  No SVG geometry is used here.
              window.setTimeout(function () {
                var old = document.getElementById("reaction-network-publication-export");
                if (old) old.remove();
                var panel = document.createElement("div");
                panel.id = "reaction-network-native-export";
                panel.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:10000;"
                  + "padding:10px 11px;border:1px solid #B8B8B8;border-radius:7px;"
                  + "background:rgba(255,255,255,0.95);color:#222;font:13px Arial,sans-serif;"
                  + "box-shadow:0 2px 8px rgba(0,0,0,0.12);";
                panel.innerHTML = '<div style="font-weight:700;margin-bottom:6px">Native publication export</div>'
                  + '<button type="button" style="margin-right:5px;padding:6px 9px;border:1px solid #888;border-radius:4px;background:#F4F4F4;cursor:pointer">Current view · 6× PNG</button>'
                  + '<button type="button" style="padding:6px 9px;border:1px solid #888;border-radius:4px;background:#F4F4F4;cursor:pointer">Full network · 6× PNG</button>'
                  + '<div style="margin-top:6px;color:#555;font-size:11px">Same vis.js Canvas renderer; no SVG redraw.</div>';
                var buttons = panel.getElementsByTagName("button");
                buttons[0].onclick = function () { window.exportReactionNetworkNativePng(false); };
                buttons[1].onclick = function () { window.exportReactionNetworkNativePng(true); };
                document.body.appendChild(panel);
              }, 20);
            }());
            </script>
            """

            custom_js = """
            var originalNodes = {};

            function captureOriginalNodes() {
                if (Object.keys(originalNodes).length !== 0) return;
                nodes.get().forEach(function(node) {
                    originalNodes[node.id] = {
                        color: node.color,
                        font: node.font,
                        borderWidth: node.borderWidth || 0
                    };
                });
            }

            network.on("afterDrawing", captureOriginalNodes);

            window.updateNetworkHighlighting = function() {
                captureOriginalNodes();
                var selectedNodes = network.getSelectedNodes();

                if (selectedNodes.length > 0) {
                    var connectedNodesSet = new Set();
                    var connectedEdgesSet = new Set();

                    selectedNodes.forEach(function(nodeId) {
                        network.getConnectedNodes(nodeId).forEach(
                            function(id) { connectedNodesSet.add(id); }
                        );
                        network.getConnectedEdges(nodeId).forEach(
                            function(id) { connectedEdgesSet.add(id); }
                        );
                    });

                    var nodeUpdates = [];
                    nodes.getIds().forEach(function(id) {
                        var original = originalNodes[id];
                        if (!original) return;

                        if (selectedNodes.includes(id)) {
                            nodeUpdates.push({
                                id: id,
                                color: original.color,
                                borderWidth: Math.max(
                                    5, original.borderWidth
                                ),
                                font: original.font
                            });
                        } else if (connectedNodesSet.has(id)) {
                            nodeUpdates.push({
                                id: id,
                                color: original.color,
                                borderWidth: Math.max(
                                    2, original.borderWidth
                                ),
                                font: original.font
                            });
                        } else {
                            nodeUpdates.push({
                                id: id,
                                color: {
                                    background: "rgba(230,230,230,0.5)",
                                    border: "rgba(200,200,200,0.5)"
                                },
                                borderWidth: 0,
                                font: {
                                    color: "rgba(150,150,150,0.5)",
                                    align: "center"
                                }
                            });
                        }
                    });
                    nodes.update(nodeUpdates);

                    var edgeUpdates = [];
                    edges.getIds().forEach(function(id) {
                        edgeUpdates.push({
                            id: id,
                            hidden: !connectedEdgesSet.has(id)
                        });
                    });
                    edges.update(edgeUpdates);
                } else {
                    var resetNodes = [];
                    nodes.getIds().forEach(function(id) {
                        var original = originalNodes[id];
                        if (!original) return;
                        resetNodes.push({
                            id: id,
                            color: original.color,
                            borderWidth: original.borderWidth,
                            font: original.font
                        });
                    });
                    nodes.update(resetNodes);

                    var resetEdges = [];
                    edges.getIds().forEach(function(id) {
                        resetEdges.push({id: id, hidden: false});
                    });
                    edges.update(resetEdges);
                }
            };

            function plainNodeText(node) {
                return (
                    String(node.label || "") + " " +
                    String(node.title || "")
                ).toLowerCase();
            }

            window.searchReactionSpecies = function() {
                var input = document.getElementById(
                    "reaction-network-search-input"
                );
                var status = document.getElementById(
                    "reaction-network-search-status"
                );
                if (!input) return;

                var query = String(input.value || "").trim().toLowerCase();
                if (!query) {
                    window.resetReactionNetworkView();
                    return;
                }

                var matchingIds = nodes.get()
                    .filter(function(node) {
                        return plainNodeText(node).includes(query);
                    })
                    .map(function(node) { return node.id; });

                if (matchingIds.length === 0) {
                    network.unselectAll();
                    window.updateNetworkHighlighting();
                    if (status) status.textContent = "No matching species";
                    return;
                }

                network.selectNodes(matchingIds);
                window.updateNetworkHighlighting();
                if (matchingIds.length === 1) {
                    network.focus(matchingIds[0], {
                        scale: 1.25,
                        animation: {
                            duration: 450,
                            easingFunction: "easeInOutQuad"
                        }
                    });
                } else {
                    network.fit({
                        nodes: matchingIds,
                        animation: {
                            duration: 450,
                            easingFunction: "easeInOutQuad"
                        }
                    });
                }
                if (status) {
                    status.textContent =
                        matchingIds.length + " matching species";
                }
            };

            window.resetReactionNetworkView = function() {
                var input = document.getElementById(
                    "reaction-network-search-input"
                );
                var status = document.getElementById(
                    "reaction-network-search-status"
                );
                if (input) input.value = "";
                if (status) status.textContent = "";
                network.unselectAll();
                window.updateNetworkHighlighting();
                network.fit({
                    animation: {
                        duration: 500,
                        easingFunction: "easeInOutQuad"
                    }
                });
            };

            var searchInput = document.getElementById(
                "reaction-network-search-input"
            );
            if (searchInput) {
                searchInput.addEventListener("keydown", function(event) {
                    if (event.key === "Enter") {
                        window.searchReactionSpecies();
                    }
                });
            }

            network.on("selectNode", window.updateNetworkHighlighting);
            network.on("deselectNode", window.updateNetworkHighlighting);
            network.once("stabilized", function() {
                network.setOptions({physics: {enabled: false}});
                network.fit({
                    animation: {
                        duration: 500,
                        easingFunction: "easeInOutQuad"
                    }
                });
            });
            """

            html_content = html_content.replace(
                "return network;",
                custom_js + "\n                  return network;",
            )
            html_content = html_content.replace(
                "</body>", publication_export_js + "</body>", 1
            )

            tooltip_css = """
            <style>
                div.vis-tooltip {
                    max-width: 460px !important;
                    padding: 10px 12px !important;
                    border: 1px solid #9A9A9A !important;
                    border-radius: 7px !important;
                    background: rgba(255,255,255,0.98) !important;
                    color: #222222 !important;
                    box-shadow: 0 3px 12px rgba(0,0,0,0.18) !important;
                    font: 13px/1.5 Arial, sans-serif !important;
                    white-space: pre-wrap !important;
                    overflow-wrap: anywhere !important;
                    text-align: left !important;
                }
            </style>
            """
            html_content = html_content.replace(
                "</head>", tooltip_css + "</head>", 1
            )

            search_controls = """
            <div id="reaction-network-controls" style="
                position:fixed;right:14px;top:14px;z-index:10000;
                padding:10px 11px;border:1px solid #B8B8B8;
                border-radius:7px;background:rgba(255,255,255,0.95);
                color:#222222;font:13px Arial,sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.12);
            ">
                <div style="font-weight:700;margin-bottom:6px;">
                    Find species
                </div>
                <div style="display:flex;gap:5px;align-items:center;">
                    <input id="reaction-network-search-input"
                        type="text" placeholder="Formula or SMILES"
                        aria-label="Search species by formula or SMILES"
                        style="width:190px;padding:6px 7px;
                        border:1px solid #AAAAAA;border-radius:4px;
                        font:13px Arial,sans-serif;">
                    <button type="button"
                        onclick="window.searchReactionSpecies()"
                        style="padding:6px 9px;border:1px solid #888888;
                        border-radius:4px;background:#F4F4F4;
                        cursor:pointer;font:13px Arial,sans-serif;">Search</button>
                    <button type="button"
                        onclick="window.resetReactionNetworkView()"
                        style="padding:6px 9px;border:1px solid #888888;
                        border-radius:4px;background:#F4F4F4;
                        cursor:pointer;font:13px Arial,sans-serif;">Reset</button>
                </div>
                <div id="reaction-network-search-status"
                    style="min-height:16px;margin-top:5px;color:#555555;
                    font-size:12px;"></div>
            </div>
            """
            html_content = html_content.replace(
                "<body>", "<body>" + search_controls, 1
            )

            if initial_nodes:
                initial_legend = """
                <div style="
                    position:fixed;left:14px;bottom:14px;z-index:9998;
                    padding:9px 11px;border:1px solid #BBBBBB;
                    border-radius:7px;background:rgba(255,255,255,0.94);
                    color:#222222;font:13px Arial,sans-serif;
                    box-shadow:0 2px 8px rgba(0,0,0,0.12);
                ">
                    <span style="display:inline-block;width:17px;height:17px;
                        border:4px solid #2CA02C;border-radius:50%;
                        margin-right:8px;vertical-align:middle;
                        background:#DCEAF7;"></span>
                    <span style="vertical-align:middle;">Initial source</span>
                </div>
                """
                html_content = html_content.replace(
                    "<body>", "<body>" + initial_legend, 1
                )

            if not error_rows.empty:
                direct_count = len(direct_problem)
                detached_count = len(detached_component_ids)
                audit_hint = (
                    " See reaction_network_integrity.csv."
                    if write_integrity_csv
                    else ""
                )
                message = (
                    f"Lineage warning: {direct_count} direct missing-source/"
                    f"zero-indegree state(s), {len(unreachable_nodes)} state(s) "
                    f"unreachable from recorded initial conditions, and "
                    f"{detached_count} detached component(s)."
                    f"{audit_hint}"
                )
                banner = f"""
                <div style="
                    position:fixed;
                    top:10px;
                    left:50%;
                    transform:translateX(-50%);
                    z-index:9999;
                    max-width:80vw;
                    padding:10px 16px;
                    border:2px solid #D62728;
                    border-radius:8px;
                    background:rgba(255,255,255,0.95);
                    color:#8B0000;
                    font:600 14px Arial,sans-serif;
                    box-shadow:0 2px 8px rgba(0,0,0,0.18);
                ">{html.escape(message)}</div>
                """
                html_content = html_content.replace(
                    "<body>",
                    "<body>" + banner,
                    1,
                )

            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(html_content)

            print(f"✨ Reaction Network saved: {out_path}")
            if write_integrity_csv:
                print(f"   Integrity audit saved: {audit_path}")

        except Exception as exc:
            print(f"Warning: {exc}")

    def plot_reaction_clusters(self, df_clusters, clean_event_count=None):
        total = pd.to_numeric(df_clusters["Count"], errors="raise").sum()
        if clean_event_count is not None and total != clean_event_count:
            raise ValueError(
                f"Plot counts ({total}) differ from final Clean records ({clean_event_count})"
            )
        if df_clusters.empty: return
        print("📊 Plotting Reaction Types...")
        
        top = df_clusters.copy()
        fig, ax = plt.subplots(figsize=(10, max(8, 0.48 * len(top) + 1.5)), dpi=300)

        import numpy as np
        cmap = plt.get_cmap("YlGnBu_r")
        colors = cmap(np.linspace(0.1, 0.6, len(top)))

        bars = ax.barh(top['Reaction Type'], top['Count'],
                       color=colors, edgecolor='none', height=0.65)

        ax.invert_yaxis()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.tick_params(axis='y', length=0, labelsize=12, colors='#222222')
        ax.tick_params(axis='x', direction='out', length=5, width=1.5, labelsize=11, colors='#222222')
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.6, linewidth=1.0)

        max_count = top['Count'].max()
        ax.set_xlim(0, max_count * 1.15) 

        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_count * 0.015),
                    bar.get_y() + bar.get_height()/2,
                    f'{int(width)}',
                    ha='left', va='center',
                    fontsize=11, fontweight='bold', color='#444444')

        ax.set_title('Reaction Types', fontsize=15, fontweight='bold', pad=15, color='#111111')
        ax.set_xlabel('Frequency', fontsize=13, fontweight='bold', color='#111111')

        plt.tight_layout()
        out_base = os.path.join(self.output_dir, 'reaction_clusters')
        plt.savefig(f"{out_base}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{out_base}.pdf", dpi=600, bbox_inches='tight')
        plt.close()

    def _get_formula_elements(self, formula):
        """Return all elements contained in a reported formula."""
        import re
        return set(re.findall(r'[A-Z][a-z]*', str(formula)))

    def _is_false_flux_edge(self, r_formula, p_formula):
        """Reject pairwise edges with no possible elemental continuity."""
        r_elements = self._get_formula_elements(r_formula)
        p_elements = self._get_formula_elements(p_formula)

        if not r_elements or not p_elements:
            return True
        return not bool(r_elements.intersection(p_elements))

    def _detect_roots(self, df_events, top_n_roots=15):
        """
        Universal Root Detection for arbitrary chemical systems.
        Identifies species that act primarily as 'sources' across the entire trajectory.
        """
        if df_events.empty: return []

        def _parts(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text = str(value).strip()
            if not text:
                return []
            # The reporting separator is whitespace + '+' + whitespace.
            # A bare split('+') corrupts charged SMILES such as [O+].
            return [
                item.strip()
                for item in re.split(r"\s+\+\s+", text)
                if item.strip()
            ]
        
        # Count the true "source" flux in the net flux map.
        raw_weights = defaultdict(float)
        for _, row in df_events.iterrows():
            r_list = _parts(row.get('Reactants_SMILES', ''))
            p_list = _parts(row.get('Products_SMILES', ''))
            r_form = _parts(row.get('Reactants_Formula', ''))
            p_form = _parts(row.get('Products_Formula', ''))
            w = 1.0 
            
            import re
            def get_elements(formula):
                return set(re.findall(r'[A-Z][a-z]*', formula))
                
            for r, r_f in zip(r_list, r_form):
                for p, p_f in zip(p_list, p_form):
                    if r == p: continue
                    if self._is_false_flux_edge(r_f, p_f):
                        continue
                    raw_weights[(r, p)] += w
                    
        # Compute the total net outflow of each node
        net_outflux = defaultdict(float)
        processed_pairs = set()
        threshold = (max(raw_weights.values()) if raw_weights else 1) * 0.01
        
        for (u, v), w in raw_weights.items():
            pair = tuple(sorted((u, v)))
            if pair in processed_pairs: continue
            
            fwd = raw_weights.get((u, v), 0)
            bwd = raw_weights.get((v, u), 0)
            
            net = fwd - bwd
            if abs(net) < threshold: continue
            
            # net > 0 means flow u -> v
            if net > 0:
                net_outflux[u] += net
                net_outflux[v] -= net
            else:
                net_outflux[v] += abs(net)
                net_outflux[u] -= abs(net)
            
            processed_pairs.add(pair)
            
        sorted_roots = sorted(net_outflux.items(), key=lambda x: x[1], reverse=True)
        
        # 3. select nodes with positive net output as physical sources (maximum 15).
        roots = [n[0] for n in sorted_roots if n[1] > 0][:top_n_roots]
        
        # Fallback
        if not roots:
            freq_counts = defaultdict(int)
            for _, row in df_events.iterrows():
                for r in _parts(row.get('Reactants_SMILES', '')):
                    freq_counts[r] += 1
            roots = [n[0] for n in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_roots]]
            
        return roots

    def plot_element_flow_sankey(
        self,
        df_events,
        element="C",
        initial_species=None,
        atom_symbols=None,
        final_state_inventory=None,
    ):
        """Plot a conserved, atom-mapped element-flow Sankey.

        Parameters
        ----------
        element : str
            Valid IUPAC element symbol displayed in the flow diagram.
        atom_symbols : mapping or sequence, optional
            Explicit atom-index-to-symbol data.  Supplying it resolves any
            allocation that is underdetermined by formulas alone.
        final_state_inventory : iterable of mappings, optional
            Stable exact molecular instances observed at the final trajectory
            frame. They annotate retained end-state species only; they never
            create a source or an artificial flow ribbon.
        """
        selected_elements = _normalize_element_flow_selection(element)
        if len(selected_elements) != 1:
            raise ValueError(
                "plot_element_flow_sankey requires exactly one element; "
                f"received {element!r}."
            )
        element = selected_elements[0]

        out_path = os.path.join(
            self.output_dir,
            f"reaction_element_flow_{element}.html",
        )
        audit_path = os.path.join(
            self.output_dir,
            f"reaction_element_flow_{element}_audit.csv",
        )

        def _remove_stale():
            for stale_path in (out_path, audit_path):
                try:
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
                except Exception:
                    pass

        try:
            if os.path.exists(audit_path):
                os.remove(audit_path)
        except Exception:
            pass

        if df_events.empty or not HAS_PLOTLY:
            _remove_stale()
            return

        import ast
        import html
        import json
        from collections import Counter

        def _parts(value):
            if value is None or (
                isinstance(value, float) and pd.isna(value)
            ):
                return []
            text = str(value).strip()
            if not text:
                return []
            return [
                item.strip()
                for item in re.split(r"\s+\+\s+", text)
                if item.strip()
            ]

        def _pairs(row, side):
            smiles = _parts(row.get(f"{side}_SMILES", ""))
            formulas = _parts(row.get(f"{side}_Formula", ""))
            if len(smiles) != len(formulas):
                raise ValueError(
                    f"{side} SMILES/formula counts do not match."
                )
            return list(zip(smiles, formulas))

        def _formula_counts(formula):
            counts = Counter()
            for symbol, number in re.findall(
                r"([A-Z][a-z]?)(\d*)",
                str(formula),
            ):
                counts[symbol] += int(number or 1)
            return counts

        def _normalise_group(item):
            if isinstance(item, dict):
                for key in ("indices", "atom_indices", "atoms"):
                    if key in item:
                        item = item[key]
                        break
            if isinstance(item, str):
                return [
                    int(value)
                    for value in re.findall(r"-?\d+", item)
                ]
            if isinstance(item, (list, tuple, set, np.ndarray)):
                return [int(value) for value in item]
            return []

        def _atom_groups(value, expected):
            if value is None or (
                isinstance(value, float) and pd.isna(value)
            ):
                return []
            payload = value
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                try:
                    payload = json.loads(text)
                except Exception:
                    try:
                        payload = ast.literal_eval(text)
                    except Exception:
                        bracket_groups = re.findall(
                            r"\[([^\[\]]*)\]",
                            text,
                        )
                        if bracket_groups:
                            payload = [
                                _normalise_group(group)
                                for group in bracket_groups
                            ]
                        elif expected == 1:
                            payload = [_normalise_group(text)]
                        else:
                            return []

            if not isinstance(payload, (list, tuple)):
                return []
            if expected == 1 and (
                not payload
                or not isinstance(
                    payload[0],
                    (list, tuple, set, dict, np.ndarray),
                )
            ):
                payload = [payload]
            groups = [_normalise_group(item) for item in payload]
            return groups if len(groups) == expected else []

        def _symbol_for_index(index):
            if atom_symbols is None:
                return None
            if isinstance(atom_symbols, dict):
                value = atom_symbols.get(
                    index,
                    atom_symbols.get(str(index)),
                )
            else:
                try:
                    value = atom_symbols[index]
                except (IndexError, KeyError, TypeError):
                    value = None
            return str(value) if value is not None else None

        node_formula = {}
        node_smiles = {}
        first_seen = {}
        edge_data = defaultdict(
            lambda: {"weight": 0.0, "events": set()}
        )
        skipped_malformed = 0
        skipped_ambiguous = 0
        skipped_state_anchors = 0
        accepted_events = 0

        sort_columns = [
            column
            for column in ("Frame", "Reactant_Frame", "Product_Frame")
            if column in df_events.columns
        ]
        ordered_events = (
            df_events.sort_values(sort_columns)
            if sort_columns
            else df_events
        )

        for row_index, row in ordered_events.iterrows():
            try:
                role = str(row.get("Pathway_Role", "")).strip()
                raw_reactant_formula = row.get("Reactants_Formula", "")
                reactant_formula = (
                    ""
                    if pd.isna(raw_reactant_formula)
                    else str(raw_reactant_formula).strip()
                )
                if (
                    role == "observed_stable_state_source"
                    or not reactant_formula
                ):
                    skipped_state_anchors += 1
                    continue
                reactants = _pairs(row, "Reactants")
                products = _pairs(row, "Products")
                r_groups = _atom_groups(
                    row.get("Reactant_Atom_Groups"),
                    len(reactants),
                )
                p_groups = _atom_groups(
                    row.get("Product_Atom_Groups"),
                    len(products),
                )
                if not r_groups or not p_groups:
                    raise ValueError("missing per-species atom groups")

                r_sets = [set(group) for group in r_groups]
                p_sets = [set(group) for group in p_groups]
                if (
                    set().union(*r_sets) != set().union(*p_sets)
                    or sum(map(len, r_sets))
                    != len(set().union(*r_sets))
                    or sum(map(len, p_sets))
                    != len(set().union(*p_sets))
                ):
                    raise ValueError("atom groups are not a conserved partition")
                if any(
                    sum(_formula_counts(formula).values())
                    != len(group)
                    for (_smiles, formula), group in zip(
                        reactants,
                        r_sets,
                    )
                ) or any(
                    sum(_formula_counts(formula).values())
                    != len(group)
                    for (_smiles, formula), group in zip(
                        products,
                        p_sets,
                    )
                ):
                    raise ValueError(
                        "formula atom count does not match exported atom group"
                    )

                r_amounts = [
                    _formula_counts(formula).get(element, 0)
                    for _smiles, formula in reactants
                ]
                p_amounts = [
                    _formula_counts(formula).get(element, 0)
                    for _smiles, formula in products
                ]
                if sum(r_amounts) != sum(p_amounts):
                    raise ValueError("selected element is not conserved")
                total_amount = int(sum(r_amounts))
                if total_amount == 0:
                    continue

                capacities = [
                    [
                        len(r_group.intersection(p_group))
                        for p_group in p_sets
                    ]
                    for r_group in r_sets
                ]

                # When atom symbols are supplied, the exact mapped allocation can be counted directly. 
                if atom_symbols is not None:
                    allocation = [
                        [
                            sum(
                                _symbol_for_index(index) == element
                                for index in r_group.intersection(p_group)
                            )
                            for p_group in p_sets
                        ]
                        for r_group in r_sets
                    ]
                    if (
                        [sum(row_values) for row_values in allocation]
                        != r_amounts
                        or [
                            sum(allocation[i][j] for i in range(len(r_sets)))
                            for j in range(len(p_sets))
                        ]
                        != p_amounts
                    ):
                        raise ValueError(
                            "atom_symbols disagrees with molecular formulas"
                        )
                else:
                    source = ("source", row_index)
                    sink = ("sink", row_index)
                    flow_graph = nx.DiGraph()
                    for i, amount in enumerate(r_amounts):
                        flow_graph.add_edge(
                            source,
                            ("r", i),
                            capacity=int(amount),
                        )
                    for i, row_caps in enumerate(capacities):
                        for j, capacity in enumerate(row_caps):
                            if capacity > 0:
                                flow_graph.add_edge(
                                    ("r", i),
                                    ("p", j),
                                    capacity=int(capacity),
                                )
                    for j, amount in enumerate(p_amounts):
                        flow_graph.add_edge(
                            ("p", j),
                            sink,
                            capacity=int(amount),
                        )

                    flow_value, flow = nx.maximum_flow(
                        flow_graph,
                        source,
                        sink,
                    )
                    if int(flow_value) != total_amount:
                        raise ValueError(
                            "no formula-consistent atom allocation exists"
                        )
                    allocation = [
                        [
                            int(flow.get(("r", i), {}).get(("p", j), 0))
                            for j in range(len(p_sets))
                        ]
                        for i in range(len(r_sets))
                    ]

                    residual = nx.DiGraph()
                    for i, row_caps in enumerate(capacities):
                        for j, capacity in enumerate(row_caps):
                            value = allocation[i][j]
                            if value < capacity:
                                residual.add_edge(("r", i), ("p", j))
                            if value > 0:
                                residual.add_edge(("p", j), ("r", i))
                    has_alternate_allocation = any(
                        len(cycle) >= 4
                        for cycle in nx.simple_cycles(residual)
                    )
                    if has_alternate_allocation:
                        skipped_ambiguous += 1
                        continue

                frame = int(row.get("Frame", row_index))
                event_id = str(
                    row.get(
                        "Pathway_ID",
                        row.get(
                            "Reaction_ID",
                            row.get("Raw_Reaction_ID", row_index),
                        ),
                    )
                )
                for side in (reactants, products):
                    for smiles, formula in side:
                        key = (str(smiles).strip(), str(formula).strip())
                        node_smiles[key] = str(smiles).strip()
                        node_formula[key] = str(formula).strip()
                        first_seen[key] = min(
                            first_seen.get(key, frame),
                            frame,
                        )

                for i, (r_smiles, r_formula) in enumerate(reactants):
                    u = (str(r_smiles).strip(), str(r_formula).strip())
                    for j, (p_smiles, p_formula) in enumerate(products):
                        value = int(allocation[i][j])
                        v = (str(p_smiles).strip(), str(p_formula).strip())
                        if value <= 0 or u == v:
                            continue
                        edge_data[(u, v)]["weight"] += float(value)
                        edge_data[(u, v)]["events"].add(event_id)
                accepted_events += 1
            except Exception:
                skipped_malformed += 1

        mapped_graph = nx.DiGraph()
        for (u, v), data in edge_data.items():
            mapped_graph.add_edge(
                u,
                v,
                weight=data["weight"],
                events=set(data["events"]),
            )
        if not mapped_graph.edges:
            _remove_stale()
            print(
                f"⚠️ No uniquely mapped {element}-flow edge was available."
            )
            return

        def _species_sort_key(species):
            return (
                first_seen.get(species, float("inf")),
                node_formula.get(species, species[1]),
                node_smiles.get(species, species[0]),
            )

        is_surface_system = any(
            str(node_formula.get(node, node[1])).strip().endswith("*")
            for node in mapped_graph.nodes
        )
        surface_exchange_records = []
        surface_cycle_records = []

        net_throughput_mode = True
        if net_throughput_mode:
            pairwise_net = nx.DiGraph()
            seen_pairs = set()
            for u, v in sorted(
                mapped_graph.edges(),
                key=lambda edge: (
                    _species_sort_key(edge[0]),
                    _species_sort_key(edge[1]),
                ),
            ):
                pair = frozenset((u, v))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                left, right = sorted(pair, key=_species_sort_key)
                left_data = mapped_graph.get_edge_data(left, right, {})
                right_data = mapped_graph.get_edge_data(right, left, {})
                left_to_right = float(left_data.get("weight", 0.0))
                right_to_left = float(right_data.get("weight", 0.0))
                exchange = min(left_to_right, right_to_left)
                if exchange > 0.0:
                    surface_exchange_records.append({
                        "left": left,
                        "right": right,
                        "left_to_right": left_to_right,
                        "right_to_left": right_to_left,
                        "exchange": exchange,
                    })

                difference = left_to_right - right_to_left
                if abs(difference) < 1e-12:
                    continue
                if difference > 0.0:
                    source, target = left, right
                    gross_forward = left_to_right
                    gross_reverse = right_to_left
                    net_weight = difference
                else:
                    source, target = right, left
                    gross_forward = right_to_left
                    gross_reverse = left_to_right
                    net_weight = -difference
                events = set(left_data.get("events", set()))
                events.update(right_data.get("events", set()))
                pairwise_net.add_edge(
                    source,
                    target,
                    weight=float(net_weight),
                    pairwise_net_weight=float(net_weight),
                    gross_forward=float(gross_forward),
                    gross_reverse=float(gross_reverse),
                    cycle_removed=0.0,
                    events=events,
                )

            while pairwise_net.edges:
                try:
                    cycle = nx.find_cycle(
                        pairwise_net,
                        orientation="original",
                    )
                except nx.NetworkXNoCycle:
                    break
                cycle_edges = [(edge[0], edge[1]) for edge in cycle]
                circulation = min(
                    float(pairwise_net[u][v]["weight"])
                    for u, v in cycle_edges
                )
                cycle_path = [cycle_edges[0][0]] + [
                    target for _source, target in cycle_edges
                ]
                surface_cycle_records.append({
                    "amount": float(circulation),
                    "path": cycle_path,
                })
                for u, v in cycle_edges:
                    data = pairwise_net[u][v]
                    data["weight"] = float(data["weight"]) - circulation
                    data["cycle_removed"] = (
                        float(data.get("cycle_removed", 0.0)) + circulation
                    )
                    if data["weight"] <= 1e-12:
                        pairwise_net.remove_edge(u, v)

            if not nx.is_directed_acyclic_graph(pairwise_net):
                _remove_stale()
                print(
                    f"⚠️ {element}-flow cycle decomposition failed; "
                    "no potentially misleading Sankey was written."
                )
                return

            surface_nodes = sorted(
                {
                    node
                    for edge in pairwise_net.edges
                    for node in edge
                },
                key=_species_sort_key,
            )
            components = [{node} for node in surface_nodes]
            component_of = {
                node: component_id
                for component_id, members in enumerate(components)
                for node in members
            }
            component_members = {
                component_id: set(members)
                for component_id, members in enumerate(components)
            }
            condensed = nx.DiGraph()
            condensed.add_nodes_from(component_members)
            for u, v, data in pairwise_net.edges(data=True):
                condensed.add_edge(
                    component_of[u],
                    component_of[v],
                    **dict(data),
                )
            if not condensed.edges:
                _remove_stale()
                print(
                    f"⚠️ {element}-flow has no net throughput after "
                    "reversible exchange and closed circulation are removed."
                )
                return
        else:
            # Plotly Sankey diagrams require a DAG.  
            components = list(nx.strongly_connected_components(mapped_graph))
            component_of = {
                node: component_id
                for component_id, members in enumerate(components)
                for node in members
            }
            component_members = {
                component_id: set(members)
                for component_id, members in enumerate(components)
            }
            condensed_edges = defaultdict(
                lambda: {"weight": 0.0, "events": set()}
            )
            for u, v, data in mapped_graph.edges(data=True):
                cu = component_of[u]
                cv = component_of[v]
                if cu == cv:
                    continue
                condensed_edges[(cu, cv)]["weight"] += float(data["weight"])
                condensed_edges[(cu, cv)]["events"].update(data["events"])

            condensed = nx.DiGraph()
            condensed.add_nodes_from(component_members)
            for (u, v), data in condensed_edges.items():
                condensed.add_edge(u, v, **data)
            if not condensed.edges:
                _remove_stale()
                print(
                    f"⚠️ The {element}-flow contains only internal cyclic "
                    "transfer; no acyclic Sankey ribbon can be shown."
                )
                return

        # Keep end-state observation separate from event throughput.  
        final_state_counts = Counter()
        final_state_dwell = {}
        for entry in final_state_inventory or []:
            if not isinstance(entry, dict):
                continue
            smiles = str(
                entry.get("smiles", entry.get("identity", "")) or ""
            ).strip()
            formula = str(entry.get("formula", "") or "").strip()
            if not (smiles or formula):
                continue
            state_key = (smiles, formula)
            final_state_counts[state_key] += 1
            final_state_dwell[state_key] = max(
                int(final_state_dwell.get(state_key, 0)),
                int(entry.get("stable_dwell_frames", 0) or 0),
            )

        requested_initial = {
            str(value).strip()
            for value in (initial_species or [])
            if str(value).strip()
        }

        def _matches_initial(node):
            for species in component_members[node]:
                smiles, formula = species
                if requested_initial.intersection({
                    str(smiles),
                    str(formula),
                    str(self._to_subscript(formula)),
                }):
                    return True
            return False

        initial_components = {
            node for node in condensed.nodes if _matches_initial(node)
        }
        if requested_initial and not initial_components:
            _remove_stale()
            print(
                f"⚠️ None of the requested initial species could be mapped "
                f"into the {element}-flow graph: "
                + ", ".join(sorted(requested_initial))
            )
            return
        if not initial_components:
            initial_components = {
                node
                for node in condensed.nodes
                if condensed.in_degree(node) == 0
            }

        boundary_return_edges = [
            (u, v)
            for u, v in condensed.edges()
            if v in initial_components and u not in initial_components
        ]
        condensed.remove_edges_from(boundary_return_edges)

        # A plotted element-flow branch must be traceable from a supplied initial state.  
        reachable_components = set(initial_components)
        for source in initial_components:
            reachable_components.update(nx.descendants(condensed, source))
        excluded_unreachable_components = set(condensed.nodes) - reachable_components
        if excluded_unreachable_components:
            condensed = condensed.subgraph(reachable_components).copy()
            print(
                "   Excluded "
                f"{len(excluded_unreachable_components)} non-initial "
                f"{element}-flow state(s) not traceable from the recorded "
                "initial inventory."
            )

        visible_components = {
            node
            for edge in condensed.edges
            for node in edge
        }
        visible_components.update(
            node for node in initial_components if node in condensed
        )
        if not visible_components or not condensed.edges:
            _remove_stale()
            print(
                f"⚠️ No {element}-flow branch remains after initial-state "
                "lineage validation."
            )
            return
        draw_nodes = list(nx.topological_sort(
            condensed.subgraph(visible_components)
        ))
        node_index = {
            node: index for index, node in enumerate(draw_nodes)
        }
        initial_components = initial_components.intersection(draw_nodes)
        terminal_components = {
            node
            for node in draw_nodes
            if condensed.out_degree(node) == 0
        }
        final_components = {
            node
            for node in draw_nodes
            if any(
                final_state_counts.get(species, 0) > 0
                for species in component_members[node]
            )
        }

        visible_species = sorted(
            {
                species
                for node in draw_nodes
                for species in component_members[node]
            },
            key=lambda species: (
                0
                if component_of[species] in initial_components
                else 1,
                first_seen.get(species, float("inf")),
                node_formula.get(species, species[1]),
                node_smiles.get(species, species[0]),
            ),
        )
        species_ids = {
            species: f"S{index:02d}"
            for index, species in enumerate(visible_species, start=1)
        }
        cycle_components = sorted(
            [
                node
                for node in draw_nodes
                if len(component_members[node]) > 1
            ],
            key=lambda node: tuple(
                sorted(
                    species_ids[species]
                    for species in component_members[node]
                )
            ),
        )
        module_ids = {
            node: f"M{index:02d}"
            for index, node in enumerate(cycle_components, start=1)
        }

        def _safe(value):
            return html.escape(str(value), quote=True)

        def _internal_transition_details(node):
            """Return concise accepted-event counts inside a collapsed SCC."""
            members = component_members[node]
            formula_frequency = Counter(
                node_formula.get(species, species[1])
                for species in members
            )

            def _transition_caption(species):
                formula = node_formula.get(species, species[1])
                rendered = self._to_subscript(formula)
                if formula_frequency[formula] > 1:
                    return f"{species_ids[species]} ({rendered})"
                return rendered

            directed = []
            for source, target, data in mapped_graph.edges(data=True):
                if source not in members or target not in members:
                    continue
                directed.append((
                    source,
                    target,
                    len(data.get("events", set())),
                ))
            transition_counts = {}
            for source, target, event_count in directed:
                transition_counts[(source, target)] = event_count

            direction_rows = []
            net_rows = []
            pairs = sorted(
                {
                    frozenset((source, target))
                    for source, target, _event_count in directed
                },
                key=lambda pair: tuple(sorted(
                    species_ids[species] for species in pair
                )),
            )
            for pair in pairs:
                left, right = sorted(
                    pair,
                    key=lambda species: species_ids[species],
                )
                left_to_right = transition_counts.get((left, right), 0)
                right_to_left = transition_counts.get((right, left), 0)
                if right_to_left > left_to_right:
                    source, target = right, left
                    forward, reverse = right_to_left, left_to_right
                else:
                    source, target = left, right
                    forward, reverse = left_to_right, right_to_left

                if forward:
                    direction_rows.append(
                        f"{_safe(_transition_caption(source))} → "
                        f"{_safe(_transition_caption(target))}: {forward}"
                    )
                if reverse:
                    direction_rows.append(
                        f"{_safe(_transition_caption(target))} → "
                        f"{_safe(_transition_caption(source))}: {reverse}"
                    )
                difference = forward - reverse
                if difference == 0:
                    net_rows.append(
                        f"Net {_safe(_transition_caption(left))} ↔ "
                        f"{_safe(_transition_caption(right))}: 0"
                    )
                else:
                    net_rows.append(
                        f"Net {_safe(_transition_caption(source))} → "
                        f"{_safe(_transition_caption(target))}: {difference}"
                    )
            return sum(transition_counts.values()), direction_rows, net_rows

        def _component_role(node):
            is_initial = node in initial_components
            is_terminal = node in terminal_components
            is_final_retained = node in final_components
            if is_initial and is_terminal and is_final_retained:
                return "Initial/source and terminal; final-frame retained"
            if is_initial and is_terminal:
                return "Initial/source and terminal"
            if is_initial and is_final_retained:
                return "Initial/source; final-frame retained"
            if is_initial:
                return "Initial/source"
            if is_terminal and is_final_retained:
                return "Terminal; final-frame retained"
            if is_terminal:
                return "Terminal"
            if is_final_retained:
                return "Intermediate; final-frame retained"
            return "Intermediate"

        def _component_label(node):
            members = component_members[node]
            if len(members) > 1:
                return module_ids[node]
            species = next(iter(members))
            formula = self._to_subscript(
                node_formula.get(species, species[1])
            )
            return f"{species_ids[species]} {formula}"

        sankey_palette = [
            "#39B54A",  # green
            "#2CB7A9",  # teal
            "#3E91D8",  # blue
            "#7758C7",  # violet
            "#D24FA8",  # magenta
            "#F06449",  # orange-red
            "#F2BE14",  # golden yellow
        ]

        def _hex_rgb(color):
            color = str(color).lstrip("#")
            return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

        def _rgb_hex(rgb):
            return "#{:02X}{:02X}{:02X}".format(
                *[int(round(max(0, min(255, value)))) for value in rgb]
            )

        def _palette_color(position):
            position = max(0.0, min(1.0, float(position)))
            scaled = position * (len(sankey_palette) - 1)
            left = int(np.floor(scaled))
            right = min(left + 1, len(sankey_palette) - 1)
            fraction = scaled - left
            rgb_left = _hex_rgb(sankey_palette[left])
            rgb_right = _hex_rgb(sankey_palette[right])
            return _rgb_hex(tuple(
                (1.0 - fraction) * a + fraction * b
                for a, b in zip(rgb_left, rgb_right)
            ))

        def _blend_hex(color_a, color_b, fraction=0.5):
            fraction = max(0.0, min(1.0, float(fraction)))
            rgb_a = _hex_rgb(color_a)
            rgb_b = _hex_rgb(color_b)
            return _rgb_hex(tuple(
                (1.0 - fraction) * a + fraction * b
                for a, b in zip(rgb_a, rgb_b)
            ))

        def _hex_rgba(color, alpha):
            r, g, b = _hex_rgb(color)
            return f"rgba({r}, {g}, {b}, {float(alpha):.3f})"

        node_color_map = {}
        for node in draw_nodes:
            if node in initial_components:
                node_color_map[node] = sankey_palette[0]
            elif node in terminal_components:
                # Terminal states are orange, including retained terminals.
                node_color_map[node] = sankey_palette[5]
            elif node in final_components:
                # Final-frame-retained intermediate states are blue.
                node_color_map[node] = sankey_palette[2]
            else:
                # Ordinary intermediates are neutral gray.
                node_color_map[node] = "#9AA3AD"

        labels = [_component_label(node) for node in draw_nodes]
        colors = [node_color_map[node] for node in draw_nodes]

        def _role_color(node):
            if node in initial_components:
                return sankey_palette[0]
            if node in terminal_components:
                return sankey_palette[5]
            if node in final_components:
                return sankey_palette[2]
            return "#9AA3AD"

        component_labels = {
            node: labels[node_index[node]]
            for node in draw_nodes
        }

        def _component_hover_label(node):
            members = component_members[node]
            if len(members) > 1:
                return component_labels[node]
            species = next(iter(members))
            formula = node_formula.get(species, species[1])
            rendered_formula = self._to_subscript(formula)
            return f"{species_ids[species]} ({rendered_formula})"

        component_hover_labels = {
            node: _component_hover_label(node)
            for node in draw_nodes
        }

        component_flow_metrics = {}
        for node in draw_nodes:
            incoming = sum(
                float(data.get("weight", 0.0))
                for _u, _v, data in condensed.in_edges(
                    node,
                    data=True,
                )
            )
            outgoing = sum(
                float(data.get("weight", 0.0))
                for _u, _v, data in condensed.out_edges(
                    node,
                    data=True,
                )
            )
            component_flow_metrics[node] = {
                "incoming": float(incoming),
                "outgoing": float(outgoing),
                # Positive values indicate net accumulation in the displayed
                # mapped-event ledger; negative values indicate depletion.
                "change": float(incoming - outgoing),
            }

        def _state_change_text(change):
            if change > 1e-12:
                return f"+{change:.3g} (net accumulation)"
            if change < -1e-12:
                return f"{change:.3g} (net depletion)"
            return "0 (balanced)"

        hover_rows = []
        for node in draw_nodes:
            members = sorted(
                component_members[node],
                key=lambda item: species_ids[item],
            )
            incoming = component_flow_metrics[node]["incoming"]
            outgoing = component_flow_metrics[node]["outgoing"]
            state_change = component_flow_metrics[node]["change"]
            incoming_rows = [
                (
                    f"{_safe(component_hover_labels[source])}: "
                    f"{float(data.get('weight', 0.0)):.3g}"
                )
                for source, _target, data in condensed.in_edges(
                    node,
                    data=True,
                )
            ]
            outgoing_rows = [
                (
                    f"{_safe(component_hover_labels[target])}: "
                    f"{float(data.get('weight', 0.0)):.3g}"
                )
                for _source, target, data in condensed.out_edges(
                    node,
                    data=True,
                )
            ]
            final_count = sum(
                final_state_counts.get(species, 0)
                for species in members
            )
            final_dwell = max(
                (final_state_dwell.get(species, 0) for species in members),
                default=0,
            )
            final_inventory_text = (
                f"Stable final-frame instances: {final_count}<br>"
                f"Final-state dwell: ≥{final_dwell} frame(s)<br>"
                if final_count else ""
            )
            if len(members) > 1:
                total, direction_rows, net_rows = (
                    _internal_transition_details(node)
                )
                hover_rows.append(
                    f"<b>{_safe(component_hover_labels[node])}</b><br>"
                    f"Internal state transitions: {total}<br>"
                    f"{'<br>'.join(direction_rows) if direction_rows else 'None'}"
                    f"<br>{'<br>'.join(net_rows) if net_rows else 'None'}<br>"
                    f"Mapped {element}-atom input: {incoming:.3g}<br>"
                    f"Mapped {element}-atom output: {outgoing:.3g}<br>"
                    f"Implied mapped {element}-state change: "
                    f"{_state_change_text(state_change)}<br>"
                    f"{final_inventory_text}"
                )
                continue
            else:
                species = members[0]
                smiles, formula = species
                member_text = (
                    f"Formula: {_safe(self._to_subscript(formula))}<br>"
                    f"SMILES: {_safe(smiles)}"
                )
            hover_rows.append(
                f"<b>{_safe(component_hover_labels[node])}</b><br>"
                f"Role: {_safe(_component_role(node))}<br>"
                f"{member_text}<br>"
                f"Displayed net "
                f"{element}-atom input: {incoming:.3g}<br>"
                f"Displayed net "
                f"{element}-atom output: {outgoing:.3g}<br>"
                "Implied net "
                f"{element}-state change: "
                f"{_state_change_text(state_change)}<br>"
                f"{final_inventory_text}"
                f"Incoming ribbons:<br>"
                f"{'<br>'.join(incoming_rows) if incoming_rows else 'None'}"
                f"<br>Outgoing ribbons:<br>"
                f"{'<br>'.join(outgoing_rows) if outgoing_rows else 'None'}"
            )

        sources = []
        targets = []
        values = []
        edge_pairs = []
        link_hover = []
        for u, v, data in condensed.edges(data=True):
            if u not in node_index or v not in node_index:
                continue
            sources.append(node_index[u])
            targets.append(node_index[v])
            values.append(float(data["weight"]))
            edge_pairs.append((u, v))
            event_ids = sorted(data.get("events", set()))
            shown = ", ".join(event_ids[:10])
            if len(event_ids) > 10:
                shown += f", … (+{len(event_ids) - 10})"
            if net_throughput_mode:
                link_hover.append(
                    f"<b>{_safe(component_hover_labels[u])} → "
                    f"{_safe(component_hover_labels[v])}</b><br>"
                    f"Forward mapped {element} atoms: "
                    f"{float(data.get('gross_forward', 0.0)):.3g}<br>"
                    f"Reverse mapped {element} atoms: "
                    f"{float(data.get('gross_reverse', 0.0)):.3g}<br>"
                    f"Pairwise net: "
                    f"{float(data.get('pairwise_net_weight', 0.0)):.3g}<br>"
                    f"Closed-cycle circulation removed: "
                    f"{float(data.get('cycle_removed', 0.0)):.3g}<br>"
                    f"Displayed net throughput: "
                    f"{float(data['weight']):.3g}<br>"
                    f"Supporting events: {shown or '(not reported)'}"
                )

            else:
                link_hover.append(
                    f"<b>{_safe(component_hover_labels[u])} → "
                    f"{_safe(component_hover_labels[v])}</b><br>"
                    f"Mapped {element}-atom transitions: "
                    f"{float(data['weight']):.3g}<br>"
                    f"Supporting events: {shown or '(not reported)'}"
                )

        element_link_colors = [
            _hex_rgba(
                _blend_hex(node_color_map[source], node_color_map[target], 0.48),
                0.24,
            )
            for source, target in edge_pairs
        ]

        element_names = {
            "C": "Carbon",
            "H": "Hydrogen",
            "O": "Oxygen",
            "Cl": "Chlorine",
        }
        excluded = skipped_malformed + skipped_ambiguous
        if net_throughput_mode:
            scope_note = (
                f"Ribbon width is net {element}-atom throughput after direct "
                "reverse exchange and closed circulation are removed without "
                "changing any node's net element balance. Removed exchange "
                "and circulation are reported below. Node height is flow "
                "support, not species abundance. Sxx identifies one exact "
                "species."
            )
        else:
            scope_note = (
                f"Ribbon width is mapped {element}-atom transfer through "
                "accepted events; node height is flow support, not species "
                "abundance. Sxx identifies one exact species. Mxx identifies "
                "a strongly connected state group collapsed only to make the "
                "Sankey layout acyclic; it is not a new species or a confirmed "
                "reaction cycle."
            )
        if excluded:
            scope_note += (
                f" {excluded} event(s) were excluded because their exported "
                "mapping was missing, inconsistent, or pairwise-ambiguous."
            )
        if skipped_state_anchors:
            scope_note += (
                f" {skipped_state_anchors} non-reactive lineage state "
                "anchor(s) were excluded before atom-flow mapping."
            )
        if boundary_return_edges:
            scope_note += (
                f" {len(boundary_return_edges)} return edge(s) into the "
                "recorded initial inventory were treated as recirculation "
                "and omitted from the source-to-product layout."
            )
        if excluded_unreachable_components:
            scope_note += (
                f" {len(excluded_unreachable_components)} state(s) not "
                "traceable from the recorded initial inventory were excluded."
            )
        if final_components:
            scope_note += (
                " Final-frame-retained nodes are identified from stable exact "
                "molecular instances in the terminal trajectory window; this "
                "annotation does not add artificial flow ribbons."
            )
        scope_note += (
            " Node colors encode role: green=Initial/source, blue="
            "final-frame retained, gray=Intermediate, orange=Terminal."
        )

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=24,
                thickness=18,
                label=labels,
                color=colors,
                line=dict(color="white", width=0.8),
                customdata=hover_rows,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=element_link_colors,
                customdata=link_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )])
        fig.update_layout(
            title=dict(
                text=(
                    f"Net {element_names[element]} Throughput Sankey "
                    "(Initial-state lineage constrained)"
                    if net_throughput_mode
                    else (
                        f"{element_names[element]} Atom-Flow Sankey "
                        "(Mapped Pathway Events)"
                    )
                ),
                font=dict(size=22, color="#111111", family="Arial"),
            ),
            # Element-flow node-label font size (Plotly Sankey labels).
            font=dict(size=20, color="#111111", family="Arial"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=75, l=55, r=55, b=45),
            height=min(1800, max(620, 38 * len(draw_nodes))),
        )
        config = {
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"reaction_element_flow_{element}_HighRes",
                "scale": 5,
            },
        }

        module_rows = []
        for node in cycle_components:
            members = sorted(
                component_members[node],
                key=lambda species: species_ids[species],
            )
            member_key = "<br>".join(
                "<b>{}</b> · {}".format(
                    species_ids[species],
                    _safe(self._to_subscript(
                        node_formula.get(species, species[1])
                    )),
                )
                for species in members
            )
            total, direction_rows, net_rows = _internal_transition_details(node)
            direction_key = "<br>".join(direction_rows) or "None"
            net_key = "<br>".join(net_rows) or "None"
            module_rows.append(
                "<tr>"
                f"<td><b>{module_ids[node]}</b></td>"
                f"<td>{member_key}</td>"
                f"<td>{total}</td>"
                f"<td>{direction_key}</td>"
                f"<td>{net_key}</td>"
                "</tr>"
            )

        species_rows = []
        for species in visible_species:
            node = component_of[species]
            smiles, formula = species
            module = module_ids.get(node, "—")
            role_color = _role_color(node)
            species_rows.append(
                "<tr>"
                f"<td><b>{species_ids[species]}</b></td>"
                f"<td>{_safe(self._to_subscript(formula))}</td>"
                f"<td class=\"ef-smiles\"><code>{_safe(smiles)}</code></td>"
                f"<td><span class=\"ef-role-swatch\" style=\"background:{role_color}\"></span>"
                f"{_safe(_component_role(node))}</td>"
                + ("" if net_throughput_mode else f"<td>{module}</td>")
                + "</tr>"
            )

        balance_audit_records = []
        balance_audit_rows = []
        for node in draw_nodes:
            members = sorted(
                component_members[node],
                key=lambda species: species_ids[species],
            )
            formulas = [
                str(node_formula.get(species, species[1]))
                for species in members
            ]
            smiles_values = [
                str(node_smiles.get(species, species[0]))
                for species in members
            ]
            incoming = component_flow_metrics[node]["incoming"]
            outgoing = component_flow_metrics[node]["outgoing"]
            state_change = component_flow_metrics[node]["change"]
            if state_change > 1e-12:
                interpretation = "net accumulation"
            elif state_change < -1e-12:
                interpretation = "net depletion"
            else:
                interpretation = "balanced"
            node_id = component_labels[node]
            species_id_text = "; ".join(
                species_ids[species]
                for species in members
            )
            formula_text = "; ".join(formulas)
            smiles_text = "; ".join(smiles_values)
            role = _component_role(node)
            tolerance = 1e-12
            if role == "Intermediate":
                reconciled = abs(state_change) <= tolerance
            elif role == "Initial/source":
                reconciled = incoming <= tolerance
            elif role == "Terminal":
                reconciled = outgoing <= tolerance
            else:
                reconciled = incoming <= tolerance and outgoing <= tolerance
            balance_audit_records.append({
                "Node_ID": node_id,
                "Species_IDs": species_id_text,
                "Formula_State": formula_text,
                "SMILES": smiles_text,
                "Diagram_Role": role,
                "Element": element,
                "Flow_Basis": (
                    "displayed_net_mapped_atom_throughput"
                    if net_throughput_mode
                    else "mapped_atom_transfer"
                ),
                "Displayed_Element_Input": incoming,
                "Displayed_Element_Output": outgoing,
                "Implied_Net_State_Change": state_change,
                "Interpretation": interpretation,
                "Endpoint_Inventory_Reconciled": bool(reconciled),
            })
            rendered_formula_text = _safe("; ".join(
                self._to_subscript(value)
                for value in formulas
            ))
            balance_audit_rows.append(
                "<tr>"
                f"<td><b>{_safe(node_id)}</b></td>"
                f"<td>{rendered_formula_text}</td>"
                f"<td>{incoming:.3g}</td>"
                f"<td>{outgoing:.3g}</td>"
                f"<td>{_safe(_state_change_text(state_change))}</td>"
                "</tr>"
            )

        balance_section = (
            "<details open>"
            "<summary>Displayed element-balance audit</summary>"
            f"<p class=\"ef-note\">Implied net {element}-state change is "
            "displayed input minus displayed output. It is derived from the "
            "mapped-event ledger and is not an independently reconciled "
            "absolute initial/final inventory.</p>"
            "<div class=\"ef-table-wrap\"><table>"
            "<thead><tr><th>Node</th><th>Formula / state</th>"
            f"<th>{element} input</th><th>{element} output</th>"
            "<th>Implied net state change</th>"
            "</tr></thead><tbody>"
            + "".join(balance_audit_rows)
            + "</tbody></table></div></details>"
        )

        module_section = ""
        if module_rows:
            module_section = (
                "<details open>"
                "<summary>Strongly connected state-group key</summary>"
                "<p class=\"ef-note\">Each Mxx node is a group of species "
                "linked by directed paths in both directions. It is collapsed "
                "only to make the Sankey layout acyclic; it is not a new "
                "species or proof of a chemical cycle. Values below are "
                "accepted directional state-transition counts.</p>"
                "<div class=\"ef-table-wrap\"><table>"
                "<thead><tr><th>State group</th><th>Members</th>"
                "<th>Internal state transitions</th>"
                "<th>Directional internal transitions</th>"
                "<th>Net internal transitions</th>"
                "</tr></thead><tbody>"
                + "".join(module_rows)
                + "</tbody></table></div></details>"
            )

        surface_section = ""
        if net_throughput_mode:
            surface_formula_frequency = Counter(
                node_formula.get(species, species[1])
                for species in mapped_graph.nodes
            )

            def _surface_species_label(species):
                formula = node_formula.get(species, species[1])
                rendered = self._to_subscript(formula)
                if surface_formula_frequency[formula] > 1:
                    return (
                        f"{rendered} "
                        f"[{node_smiles.get(species, species[0])}]"
                    )
                return rendered

            exchange_rows = []
            for record in surface_exchange_records:
                left = record["left"]
                right = record["right"]
                left_to_right = float(record["left_to_right"])
                right_to_left = float(record["right_to_left"])
                difference = left_to_right - right_to_left
                if abs(difference) < 1e-12:
                    net_text = "0"
                elif difference > 0.0:
                    net_text = (
                        f"{_safe(_surface_species_label(left))} → "
                        f"{_safe(_surface_species_label(right))}: "
                        f"{difference:.3g}"
                    )
                else:
                    net_text = (
                        f"{_safe(_surface_species_label(right))} → "
                        f"{_safe(_surface_species_label(left))}: "
                        f"{-difference:.3g}"
                    )
                exchange_rows.append(
                    "<tr>"
                    f"<td>{_safe(_surface_species_label(left))} ↔ "
                    f"{_safe(_surface_species_label(right))}</td>"
                    f"<td>{left_to_right:.3g}</td>"
                    f"<td>{right_to_left:.3g}</td>"
                    f"<td>{float(record['exchange']):.3g}</td>"
                    f"<td>{net_text}</td>"
                    "</tr>"
                )

            cycle_rows = []
            for index, record in enumerate(
                surface_cycle_records,
                start=1,
            ):
                path_text = " → ".join(
                    _safe(_surface_species_label(species))
                    for species in record["path"]
                )
                cycle_rows.append(
                    "<tr>"
                    f"<td><b>C{index:02d}</b></td>"
                    f"<td>{float(record['amount']):.3g}</td>"
                    f"<td>{path_text}</td>"
                    "</tr>"
                )

            exchange_section = ""
            if exchange_rows:
                exchange_section = (
                    "<details open>"
            "<summary>Reversible pairwise exchange removed from net flow</summary>"
                    f"<p class=\"ef-note\">Values are mapped {element}-atom "
                    "units. “Canceled each way” is removed equally from both "
                    "directions and therefore does not change net element "
                    "balance.</p>"
                    "<div class=\"ef-table-wrap\"><table>"
                    "<thead><tr><th>Species pair</th>"
                    "<th>Left → right</th><th>Right → left</th>"
                    "<th>Canceled each way</th><th>Pairwise net</th>"
                    "</tr></thead><tbody>"
                    + "".join(exchange_rows)
                    + "</tbody></table></div></details>"
                )

            cycle_section = ""
            if cycle_rows:
                cycle_section = (
                    "<details open>"
            "<summary>Closed element circulation removed from net flow</summary>"
                    f"<p class=\"ef-note\">Each listed circulation removes "
                    f"the same mapped {element}-atom amount from every edge "
                    "in one closed directed path. Node net balances are "
                    "unchanged.</p>"
                    "<div class=\"ef-table-wrap\"><table>"
                    "<thead><tr><th>Cycle</th><th>Circulation removed</th>"
                    "<th>Closed path</th></tr></thead><tbody>"
                    + "".join(cycle_rows)
                    + "</tbody></table></div></details>"
                )

            surface_section = exchange_section + cycle_section

        # Keep the species key (including Diagram role) visible for large element plots as well.
        species_open = " open"
        role_legend = (
            "<p class=\"ef-role-legend\">"
            "<span><i class=\"ef-role-swatch\" style=\"background:#39B54A\"></i>"
            "Initial/source</span>"
            "<span><i class=\"ef-role-swatch\" style=\"background:#3E91D8\"></i>"
            "final-frame retained</span>"
            "<span><i class=\"ef-role-swatch\" style=\"background:#9AA3AD\"></i>"
            "Intermediate</span>"
            "<span><i class=\"ef-role-swatch\" style=\"background:#F06449\"></i>"
            "Terminal</span></p>"
        )
        legend_panel = (
            "<section class=\"ef-key\" aria-label=\"Element-flow legend\">"
            "<h2>Interactive element-flow key</h2>"
            + role_legend
            + f"<p class=\"ef-note\">{_safe(scope_note)} Formula state suffixes, "
            "when present, remain part of species identity. Colors describe "
            "diagram role, not chemical class.</p>"
            + balance_section
            + surface_section
            + module_section
            + f"<details{species_open}>"
            "<summary>Species key</summary>"
            "<div class=\"ef-table-wrap ef-species-wrap\"><table>"
            "<thead><tr><th>ID</th><th>Formula / state</th>"
            "<th>Full SMILES</th><th>Diagram role</th>"
            + ("" if net_throughput_mode else "<th>State group</th>")
            + "</tr></thead><tbody>"
            + "".join(species_rows)
            + "</tbody></table></div></details></section>"
            + "<style>"
            ".ef-key{font-family:Arial,sans-serif;max-width:1500px;"
            "margin:8px auto 36px;padding:0 24px;color:#1f2933}"
            ".ef-key h2{font-size:20px;margin:12px 0 6px}"
            ".ef-key details{border:1px solid #d7dde3;border-radius:8px;"
            "margin:12px 0;background:#fff}"
            ".ef-key summary{cursor:pointer;font-weight:700;padding:12px 14px;"
            "background:#f5f7f9;border-radius:8px}"
            ".ef-note{font-size:13px;line-height:1.45;color:#52606d;"
            "margin:8px 14px 12px}"
            ".ef-role-legend{display:flex;flex-wrap:wrap;gap:14px;"
            "margin:8px 14px 12px;font-size:13px;color:#334e68}"
            ".ef-role-legend span{white-space:nowrap}"
            ".ef-role-swatch{display:inline-block;width:11px;height:11px;"
            "border-radius:2px;margin-right:5px;vertical-align:-1px}"
            ".ef-table-wrap{overflow:auto;max-height:560px}"
            ".ef-key table{border-collapse:collapse;width:100%;font-size:12px}"
            ".ef-key th,.ef-key td{padding:8px 10px;border-bottom:"
            "1px solid #e6e9ed;text-align:left;vertical-align:top}"
            ".ef-key th{position:sticky;top:0;background:#eef2f5;z-index:1}"
            ".ef-smiles{min-width:320px;max-width:760px;"
            "overflow-wrap:anywhere;word-break:break-word}"
            ".ef-key code{font-size:11px;white-space:normal}"
            "</style>"
        )
        page = fig.to_html(
            full_html=True,
            include_plotlyjs=True,
            config=config,
        )
        element_label_css = (
            "<style>g.sankey-node text{"
            "text-shadow:none !important;"
            "stroke:none !important;"
            "paint-order:normal !important;"
            "font-weight:600 !important;"
            "fill:#111111 !important;"
            "}</style>"
        )
        page = page.replace("</head>", element_label_css + "</head>", 1)
        page = page.replace("</body>", legend_panel + "</body>", 1)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(page)
        print(f"✨ {element}-atom Flow Sankey saved: {out_path}")
        if skipped_ambiguous or skipped_malformed:
            print(
                "   ⚠️ Sankey mapping exclusions: "
                f"ambiguous={skipped_ambiguous}, "
                f"malformed={skipped_malformed}."
            )

    def plot_sankey(self, df_events, top_n=30, initial_species=None,
                strict_initial_species=True, element="C",
                atom_symbols=None, final_state_inventory=None):
        """
        Net event-support transition map plus configurable element-flow Sankeys.
        """
        selected_elements = _normalize_element_flow_selection(element)
        out_path = os.path.join(
            self.output_dir,
            'reaction_transition_sankey.html',
        )
       
        def _remove_stale_sankey():
            """Remove a previous Sankey HTML when the current diagram is not generated."""
            for stale_path in (out_path, ):
                try:
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
                        print(
                            "   Removed stale transition-map file: "
                            f"{stale_path}"
                        )
                except Exception as exc:
                    print(
                        "⚠️ Could not remove stale transition-map file "
                        f"{stale_path}: {exc}"
                    )

        def _remove_unselected_element_outputs():
            """Prevent outputs from an earlier selection looking current."""
            selected_set = set(selected_elements)
            for omitted_element in ELEMENT_FLOW_ELEMENTS:
                if omitted_element in selected_set:
                    continue
                stale_paths = (
                    os.path.join(
                        self.output_dir,
                        f"reaction_element_flow_{omitted_element}.html",
                    ),
                    os.path.join(
                        self.output_dir,
                        f"reaction_element_flow_{omitted_element}_audit.csv",
                    ),
                )
                for stale_path in stale_paths:
                    try:
                        if os.path.exists(stale_path):
                            os.remove(stale_path)
                            print(
                                "   Removed unselected element-flow output: "
                                f"{stale_path}"
                            )
                    except Exception as exc:
                        print(
                            "⚠️ Could not remove unselected element-flow "
                            f"output {stale_path}: {exc}"
                        )

        _remove_unselected_element_outputs()

        if df_events.empty or not HAS_PLOTLY:
            _remove_stale_sankey()
            for selected_element in selected_elements:
                for suffix in (".html", "_audit.csv"):
                    stale_path = os.path.join(
                        self.output_dir,
                        "reaction_element_flow_"
                        f"{selected_element}{suffix}",
                    )
                    try:
                        if os.path.exists(stale_path):
                            os.remove(stale_path)
                    except Exception:
                        pass
            return

        # Generate the element-conserving view independently. 
        print(
            "   Sankey lineage mode: "
            f"{SANKEY_LINEAGE_VISUALIZER_VERSION}"
        )
        for selected_element in selected_elements:
            self.plot_element_flow_sankey(
                df_events,
                element=selected_element,
                initial_species=initial_species,
                atom_symbols=atom_symbols,
                final_state_inventory=final_state_inventory,
            )

        try:
            print("🌊 Plotting Net Event-Support Hybrid Reaction Map...")
        except UnicodeEncodeError:
            print("Plotting Net Event-Support Hybrid Reaction Map...")

        from collections import Counter

        def _parts(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text = str(value).strip()
            if not text:
                return []
            return [
                item.strip()
                for item in re.split(r"\s+\+\s+", text)
                if item.strip()
            ]

        def _paired_species(row, smiles_column, formula_column):
            smiles_parts = _parts(row.get(smiles_column, ""))
            formula_parts = _parts(row.get(formula_column, ""))
            if len(smiles_parts) != len(formula_parts):
                event_id = str(
                    row.get(
                        "Pathway_ID",
                        row.get("Reaction_ID", "(unknown event)"),
                    )
                )
                raise ValueError(
                    "Sankey species parsing mismatch for "
                    f"{event_id}: {smiles_column} contains "
                    f"{len(smiles_parts)} species but {formula_column} "
                    f"contains {len(formula_parts)}."
                )
            return list(zip(smiles_parts, formula_parts))

        def _node_key(smi, formula):
            # Use the same state identity as the interactive network.  
            return (str(smi).strip(), str(formula).strip())

        # 1. First-occurrence frame + initial reactant determination.
        first_rxn = {}
        first_as_reactant = {}
        first_as_product = {}
        node_formula = {}   # key -> formula with suffix (for labeling/element judgment)
        node_smiles = {}    # key -> bare SMILES (for coloring, hovering, and isomer names)
        for _, row in df_events.sort_values('Frame').iterrows():
            f = int(row['Frame'])
            rf = int(row['Reactant_Frame']) if 'Reactant_Frame' in row and pd.notna(row['Reactant_Frame']) else f
            pf = int(row['Product_Frame']) if 'Product_Frame' in row and pd.notna(row['Product_Frame']) else f

            reactant_pairs = _paired_species(
                row,
                'Reactants_SMILES',
                'Reactants_Formula',
            )
            product_pairs = _paired_species(
                row,
                'Products_SMILES',
                'Products_Formula',
            )

            for smi, form in reactant_pairs:
                key = _node_key(smi, form)
                node_formula[key] = form
                node_smiles[key] = smi
                if key not in first_as_reactant or rf < first_as_reactant[key]:
                    first_as_reactant[key] = rf
                if key not in first_rxn or rf < first_rxn[key]:
                    first_rxn[key] = rf
            for smi, form in product_pairs:
                key = _node_key(smi, form)
                if key not in node_formula:
                    node_formula[key] = form
                node_smiles[key] = smi
                if key not in first_as_product or pf < first_as_product[key]:
                    first_as_product[key] = pf
                if key not in first_rxn or pf < first_rxn[key]:
                    first_rxn[key] = pf

        if not first_rxn:
            _remove_stale_sankey()
            return
        
        is_surface_system = any(
            str(f).strip().endswith('*')
            for f in node_formula.values()
        )

        def _is_gas_like(n):
            """Whether a node is explicitly labeled as gas-phase."""
            formula = str(node_formula.get(n, n)).strip()
            return formula.endswith('(g)')

        def _is_source_like(n):
            """
            Generic source-like rule based only on event timing.
            """
            r = first_as_reactant.get(n)
            p = first_as_product.get(n)

            if r is None:
                return False

            if p is None:
                return True

            return r <= p

        def _side_signature(counts):
            """Canonical, hashable species multiset for one reaction side."""
            return tuple(sorted(
                (
                    (species, int(count))
                    for species, count in counts.items()
                    if int(count) > 0
                ),
                key=lambda item: (
                    str(item[0][1]),
                    str(item[0][0]),
                    item[1],
                ),
            ))

        def _signature_counts(signature):
            return OrderedDict(
                (species, int(count))
                for species, count in signature
            )

        def _format_side(signature):
            labels = []
            for species, count in signature:
                formula = self._to_subscript(
                    str(node_formula.get(species, species[1]))
                )
                labels.append(f"{count if count > 1 else ''}{formula}")
            return " + ".join(labels) if labels else "∅"

        def _is_reaction_node(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "__reaction__"
            )

        # 2. Aggregate complete reaction events after cancelling exact spectators.
        directed_reactions = OrderedDict()
        for _, row in df_events.iterrows():
            event_id = str(
                row.get(
                    "Pathway_ID",
                    row.get(
                        "Reaction_ID",
                        row.get("Raw_Reaction_ID", ""),
                    ),
                )
                or ""
            ).strip()
            reactant_pairs = _paired_species(
                row,
                'Reactants_SMILES',
                'Reactants_Formula',
            )
            product_pairs = _paired_species(
                row,
                'Products_SMILES',
                'Products_Formula',
            )

            r_keys = [_node_key(s, f) for s, f in reactant_pairs]
            p_keys = [_node_key(s, f) for s, f in product_pairs]
            for k, (s, f) in (
                list(zip(r_keys, reactant_pairs))
                + list(zip(p_keys, product_pairs))
            ):
                node_formula.setdefault(k, f)
                node_smiles[k] = s

            r_counts = Counter(r_keys)
            p_counts = Counter(p_keys)

            for sp in list(r_counts.keys()):
                if sp in p_counts:
                    common = min(r_counts[sp], p_counts[sp])
                    r_counts[sp] -= common
                    p_counts[sp] -= common

            rs_active = {k: c for k, c in r_counts.items() if c > 0}
            ps_active = {k: c for k, c in p_counts.items() if c > 0}

            reactant_signature = _side_signature(rs_active)
            product_signature = _side_signature(ps_active)
            if not reactant_signature or not product_signature:
                continue
            if reactant_signature == product_signature:
                continue

            reaction_key = (reactant_signature, product_signature)
            frame = int(row.get("Frame", 0) or 0)
            reaction_type = str(row.get("Reaction_Type", "") or "").strip()
            record = directed_reactions.setdefault(
                reaction_key,
                {
                    "count": 0.0,
                    "events": set(),
                    "types": set(),
                    "first_frame": frame,
                    "last_frame": frame,
                },
            )
            record["count"] += 1.0
            if event_id:
                record["events"].add(event_id)
            if reaction_type:
                record["types"].add(reaction_type)
            record["first_frame"] = min(record["first_frame"], frame)
            record["last_frame"] = max(record["last_frame"], frame)

        # 3. Net only exact reverse reaction signatures. 
        reaction_data = OrderedDict()
        processed_reaction_keys = set()
        reaction_index = 0
        for reaction_key, forward_record in directed_reactions.items():
            if reaction_key in processed_reaction_keys:
                continue
            reverse_key = (reaction_key[1], reaction_key[0])
            reverse_record = directed_reactions.get(reverse_key)
            forward_weight = float(forward_record["count"])
            reverse_weight = float(
                reverse_record["count"] if reverse_record else 0.0
            )
            processed_reaction_keys.add(reaction_key)
            processed_reaction_keys.add(reverse_key)
            if forward_weight == reverse_weight:
                continue

            if forward_weight > reverse_weight:
                kept_key = reaction_key
                kept_record = forward_record
                opposite_record = reverse_record
                kept_forward = forward_weight
                kept_reverse = reverse_weight
            else:
                kept_key = reverse_key
                kept_record = reverse_record
                opposite_record = forward_record
                kept_forward = reverse_weight
                kept_reverse = forward_weight

            reaction_index += 1
            reaction_node = ("__reaction__", reaction_index)
            reactant_signature, product_signature = kept_key
            reaction_events = set(kept_record.get("events", set()))
            if opposite_record:
                reaction_events.update(opposite_record.get("events", set()))
            reaction_types = set(kept_record.get("types", set()))
            reaction_equation = (
                f"{_format_side(reactant_signature)} → "
                f"{_format_side(product_signature)}"
            )
            reaction_data[reaction_node] = {
                "reactants": _signature_counts(reactant_signature),
                "products": _signature_counts(product_signature),
                "weight": float(kept_forward - kept_reverse),
                "forward_weight": kept_forward,
                "reverse_weight": kept_reverse,
                "events": reaction_events,
                "types": reaction_types,
                "equation": reaction_equation,
                "first_frame": int(kept_record.get("first_frame", 0)),
                "last_frame": int(kept_record.get("last_frame", 0)),
            }
            first_rxn[reaction_node] = reaction_data[reaction_node]["first_frame"]
            node_formula[reaction_node] = "Reaction"
            node_smiles[reaction_node] = reaction_equation

        if not reaction_data:
            _remove_stale_sankey()
            return

        def _reaction_edge_common(reaction_node, data):
            return {
                "reaction_node": reaction_node,
                "reaction_weight": float(data["weight"]),
                "forward_weight": float(data["forward_weight"]),
                "reverse_weight": float(data["reverse_weight"]),
                "events": set(data["events"]),
                "types": set(data["types"]),
                "equation": data["equation"],
            }

        def _has_multiple_reactants(data):
            """True only for two or more distinct reactant species."""
            return len(data["reactants"]) > 1

        def _internal_reaction_edges(reaction_node, data):
            """Balanced bipartite edges used for all graph operations."""
            reactant_total = max(1, sum(data["reactants"].values()))
            product_total = max(1, sum(data["products"].values()))
            common = _reaction_edge_common(reaction_node, data)
            for species, coefficient in data["reactants"].items():
                edge_data = dict(common)
                edge_data.update({
                    "weight": (
                        float(data["weight"])
                        * float(coefficient)
                        / float(reactant_total)
                    ),
                    "coefficient": int(coefficient),
                    "role": "reactant",
                })
                yield species, reaction_node, edge_data
            for species, coefficient in data["products"].items():
                edge_data = dict(common)
                edge_data.update({
                    "weight": (
                        float(data["weight"])
                        * float(coefficient)
                        / float(product_total)
                    ),
                    "coefficient": int(coefficient),
                    "role": "product",
                })
                yield reaction_node, species, edge_data

        def _display_reaction_edges(reaction_node, data):
            """Collapse single-reactant reactions to direct display links."""
            if _has_multiple_reactants(data):
                yield from _internal_reaction_edges(reaction_node, data)
                return

            reactant, reactant_coefficient = next(
                iter(data["reactants"].items())
            )
            product_total = max(1, sum(data["products"].values()))
            common = _reaction_edge_common(reaction_node, data)
            for product, product_coefficient in data["products"].items():
                edge_data = dict(common)
                edge_data.update({
                    "weight": (
                        float(data["weight"])
                        * float(product_coefficient)
                        / float(product_total)
                    ),
                    "coefficient": int(product_coefficient),
                    "reactant_coefficient": int(reactant_coefficient),
                    "role": "single_reactant_direct",
                })
                yield reactant, product, edge_data

        def _build_reaction_graph(reaction_nodes):
            graph = nx.DiGraph()
            for reaction_node in reaction_nodes:
                data = reaction_data[reaction_node]
                for u, v, edge_data in _internal_reaction_edges(
                    reaction_node,
                    data,
                ):
                    graph.add_edge(u, v, **edge_data)
            return graph

        # 4. Keep every complete net reaction for source validation, reachability, support filtering, and pathway repair.  
        active_reaction_nodes = list(reaction_data.keys())
        G_full = _build_reaction_graph(active_reaction_nodes)

        if not active_reaction_nodes or not G_full.edges:
            _remove_stale_sankey()
            return

        active_reaction_set = set(active_reaction_nodes)
        full_nodes = {
            node for node in G_full.nodes
            if not _is_reaction_node(node)
        }
        
        initial_species_list = []
        _seen_initial = set()

        for x in (initial_species or []):
            s = str(x).strip()
            if not s:
                continue
            if s not in _seen_initial:
                initial_species_list.append(s)
                _seen_initial.add(s)

        initial_species_set = set(initial_species_list)


        def _matches_one_initial_species(n, req):
            formula = str(node_formula.get(n, n)).strip()
            formula_sub = str(self._to_subscript(formula)).strip()
            smi = str(node_smiles.get(n, "")).strip()
            req = str(req).strip()

            return req in (formula, formula_sub, smi)


        def _node_flux_in_out(G, n):
            if n not in G:
                return 0.0, 0.0

            in_w = sum(
                float(d.get('weight', 0.0))
                for _, _, d in G.in_edges(n, data=True)
            )
            out_w = sum(
                float(d.get('weight', 0.0))
                for _, _, d in G.out_edges(n, data=True)
            )
            return in_w, out_w


        if initial_species_set:
            all_known_nodes = set(node_formula.keys())

            initial_nodes = []
            missing_requested = []
            no_net_outflow = []

            for req in initial_species_list:
                matched_nodes = [
                    n for n in all_known_nodes
                    if _matches_one_initial_species(n, req)
                ]

                matched_nodes = sorted(
                    matched_nodes,
                    key=lambda n: (
                        first_rxn.get(n, 10**12),
                        str(node_formula.get(n, n)),
                        str(node_smiles.get(n, ""))
                    )
                )

                if not matched_nodes:
                    missing_requested.append(req)
                    continue

                valid_nodes = []
                invalid_nodes = []

                for n in matched_nodes:
                    in_w, out_w = _node_flux_in_out(G_full, n)

                    if n in G_full and out_w > 0.0:
                        valid_nodes.append(n)
                    else:
                        invalid_nodes.append((n, in_w, out_w))

                if not valid_nodes:
                    no_net_outflow.append((req, invalid_nodes))
                    continue

                for n in valid_nodes:
                    if n not in initial_nodes:
                        initial_nodes.append(n)

            if missing_requested or no_net_outflow:
                print(
                    "⚠️ User-defined sankey_initial_species validation failed."
                )
                print("   Requested initial species:", initial_species_list)

                if missing_requested:
                    print("   Species not found in reaction events:")
                    for req in missing_requested:
                        print("      ", req)

                if no_net_outflow:
                    print(
                        "   Species found, but not valid net-transition sources "
                        "(no positive net event support in the transition graph):"
                    )
                    for req, records in no_net_outflow:
                        print("      Requested:", req)
                        for n, in_w, out_w in records:
                            print(
                                "         ",
                                node_formula.get(n, n),
                                "|",
                                node_smiles.get(n, ""),
                                "| net-in:",
                                in_w,
                                "| net-out:",
                                out_w
                            )

                if strict_initial_species:
                    print(
                        "   Net reaction transition map was not generated. "
                        "Please check bond_scale / adsorption_scale, "
                        "sankey_initial_species labels, or reaction_events_pathway.csv."
                    )
                    _remove_stale_sankey()
                    return

                print(
                    "   Proceeding with only valid user-defined source species "
                    "because strict_initial_species=False."
                )

            if not initial_nodes:
                print(
                    "⚠️ User-defined sankey_initial_species was provided, "
                    "but none are valid net-transition sources."
                )
                print("   Net reaction transition map was not generated.")
                _remove_stale_sankey()
                return

        else:
            initial_nodes = []

        def _flux_in_out(G, n):
            in_w = sum(
                float(d.get('weight', 0.0))
                for _, _, d in G.in_edges(n, data=True)
            )
            out_w = sum(
                float(d.get('weight', 0.0))
                for _, _, d in G.out_edges(n, data=True)
            )
            return in_w, out_w

        if not initial_nodes and not initial_species_set:
            if is_surface_system:
                gas_candidates = [
                    n for n in full_nodes
                    if _is_gas_like(n) and G_full.out_degree(n) > 0
                ]

                gas_sources = []

                for n in gas_candidates:
                    in_w, out_w = _flux_in_out(G_full, n)

                    if out_w > in_w:
                        gas_sources.append(n)

                if not gas_sources and gas_candidates:
                    gas_sources = sorted(
                        gas_candidates,
                        key=lambda n: (
                            _flux_in_out(G_full, n)[1] - _flux_in_out(G_full, n)[0],
                            _flux_in_out(G_full, n)[1]
                        ),
                        reverse=True
                    )[:max(1, min(5, len(gas_candidates)))]

                initial_nodes = sorted(
                    gas_sources,
                    key=lambda n: (
                        first_as_reactant.get(n, 10**12),
                        -_flux_in_out(G_full, n)[1]
                    )
                )

                has_explicit_gas_sources = bool(initial_nodes)

                if not initial_nodes:
                    initial_nodes = [
                        n for n in full_nodes
                        if _is_source_like(n)
                        and G_full.in_degree(n) == 0
                        and G_full.out_degree(n) > 0
                    ]

            else:
                initial_nodes = [
                    n for n in full_nodes
                    if _is_source_like(n)
                    and G_full.in_degree(n) == 0
                    and G_full.out_degree(n) > 0
                ]

        if not initial_nodes and not initial_species_set:
            initial_nodes = sorted(
                [n for n in full_nodes if first_as_reactant.get(n) is not None],
                key=lambda n: first_as_reactant.get(n, 10**12)
            )[:max(1, min(5, len(full_nodes)))]

        initial_set = set(initial_nodes)

        # Main pathway closure requires every reactant of a reaction to be reachable.
        if initial_nodes:
            reachable_species = set(initial_nodes)
            reachable_reactions = set()
            changed = True
            ordered_active_reactions = sorted(
                active_reaction_nodes,
                key=lambda node: (
                    int(reaction_data[node]["first_frame"]),
                    int(node[1]),
                ),
            )
            while changed:
                changed = False
                for reaction_node in ordered_active_reactions:
                    if reaction_node in reachable_reactions:
                        continue
                    reactants = set(reaction_data[reaction_node]["reactants"])
                    if reactants.issubset(reachable_species):
                        reachable_reactions.add(reaction_node)
                        before = len(reachable_species)
                        reachable_species.update(
                            reaction_data[reaction_node]["products"]
                        )
                        changed = changed or len(reachable_species) > before
                        changed = True
            candidate_reactions = [
                node for node in active_reaction_nodes
                if node in reachable_reactions
            ]
            if not candidate_reactions:
                _remove_stale_sankey()
                return
        else:
            candidate_reactions = list(active_reaction_nodes)

        # 6. Dominant complete-reaction selection. 
        candidate_reactions = sorted(
            candidate_reactions,
            key=lambda node: (
                -float(reaction_data[node]["weight"]),
                int(reaction_data[node]["first_frame"]),
                int(node[1]),
            ),
        )
        max_weight = float(reaction_data[candidate_reactions[0]]["weight"])

        # Global minimum weight threshold.
        global_min_weight = max(1.0, max_weight * 0.005)

        # Each species keeps its two strongest consuming reactions, plus any
        # reaction carrying at least the configured local share.
        max_out_reactions_per_species = 2

        # An edge is retained if its fraction of the node’s total outflow meets this value.
        local_keep_ratio = 0.08

        consuming_by_species = defaultdict(list)
        for reaction_node in candidate_reactions:
            data = reaction_data[reaction_node]
            for species in data["reactants"]:
                consuming_by_species[species].append(reaction_node)

        selected_reaction_set = set()
        for species, items in consuming_by_species.items():
            items_sorted = sorted(
                items,
                key=lambda node: (
                    -float(reaction_data[node]["weight"]),
                    int(reaction_data[node]["first_frame"]),
                    int(node[1]),
                ),
            )
            total_out = sum(
                float(reaction_data[node]["weight"])
                for node in items_sorted
            )
            for rank, reaction_node in enumerate(items_sorted):
                weight = float(reaction_data[reaction_node]["weight"])
                keep_source_reaction = species in initial_set
                keep_local_top = (
                    rank < max_out_reactions_per_species
                    and weight >= global_min_weight
                )
                keep_local_share = (
                    total_out > 0.0
                    and weight / total_out >= local_keep_ratio
                    and weight >= global_min_weight
                )
                if keep_source_reaction or keep_local_top or keep_local_share:
                    selected_reaction_set.add(reaction_node)

        if not selected_reaction_set:
            selected_reaction_set.add(candidate_reactions[0])

        if (
            top_n is not None
            and int(top_n) > 0
            and len(selected_reaction_set) > int(top_n)
        ):
            # Reserve a bounded, evidence-ranked root budget
            root_reactions = sorted(
                [
                    node for node in selected_reaction_set
                    if set(reaction_data[node]["reactants"]) & initial_set
                ],
                key=lambda node: (
                    -float(reaction_data[node]["weight"]),
                    int(reaction_data[node]["first_frame"]),
                    int(node[1]),
                ),
            )
            root_budget = min(
                int(top_n),
                len(root_reactions),
                max(
                    len(initial_set),
                    int(np.ceil(float(int(top_n)) * 0.25)),
                ),
            )
            protected_reactions = set(root_reactions[:root_budget])
            other_reactions = sorted(
                selected_reaction_set - protected_reactions,
                key=lambda node: (
                    -float(reaction_data[node]["weight"]),
                    int(reaction_data[node]["first_frame"]),
                    int(node[1]),
                ),
            )
            remaining = max(0, int(top_n) - len(protected_reactions))
            selected_reaction_set = protected_reactions | set(
                other_reactions[:remaining]
            )
            print(
                "   Transition selection: bounded initial-adjacent "
                f"reservation={len(protected_reactions)}/{int(top_n)}."
            )

        producing_by_species = defaultdict(list)
        for reaction_node in candidate_reactions:
            for species in reaction_data[reaction_node]["products"]:
                producing_by_species[species].append(reaction_node)

        for species in producing_by_species:
            producing_by_species[species].sort(
                key=lambda node: (
                    int(reaction_data[node]["first_frame"]),
                    -float(reaction_data[node]["weight"]),
                    int(node[1]),
                )
            )

        max_repair_iter = max(50, len(candidate_reactions) + 1)
        unresolved_selected_species = set()
        for _ in range(max_repair_iter):
            unresolved_selected_species = set()
            changed = False
            for consumer in sorted(
                selected_reaction_set,
                key=lambda node: (
                    int(reaction_data[node]["first_frame"]),
                    int(node[1]),
                ),
            ):
                consumer_frame = int(reaction_data[consumer]["first_frame"])
                for species in reaction_data[consumer]["reactants"]:
                    if species in initial_set:
                        continue
                    selected_producer_exists = any(
                        producer in selected_reaction_set
                        and int(reaction_data[producer]["first_frame"])
                        <= consumer_frame
                        for producer in producing_by_species.get(species, [])
                    )
                    if selected_producer_exists:
                        continue
                    candidates = [
                        producer
                        for producer in producing_by_species.get(species, [])
                        if producer not in selected_reaction_set
                        and int(reaction_data[producer]["first_frame"])
                        <= consumer_frame
                    ]
                    if not candidates:
                        unresolved_selected_species.add(species)
                        continue
                    selected_reaction_set.add(candidates[0])
                    changed = True
            if not changed:
                break

        if unresolved_selected_species:
            print(
                "   Excluding "
                f"{len(unresolved_selected_species)} selected species without "
                "an earlier producer or an initial-state origin."
            )

        selected_reactions = sorted(
            selected_reaction_set,
            key=lambda node: (
                int(reaction_data[node]["first_frame"]),
                int(node[1]),
            ),
        )
        print(
            "   Selected "
            f"{len(selected_reactions)} of {len(candidate_reactions)} "
            "source-reachable net reaction(s) after support and top_n "
            "filtering."
        )

        # Convert the selected reactions into an acyclic main-transition view.
        final_edges = []
        G_display_builder = nx.DiGraph()
        hidden_cycle_products = defaultdict(Counter)
        hidden_cycle_links = 0
        hidden_cycle_only_reactions = 0

        for reaction_node in selected_reactions:
            reaction_edges = list(_display_reaction_edges(
                reaction_node,
                reaction_data[reaction_node],
            ))
            trial_graph = G_display_builder.copy()
            staged_edges = []
            staged_hidden = Counter()
            has_visible_product = False

            for u, v, edge_data in reaction_edges:
                closes_cycle = (
                    u == v
                    or (
                        u in trial_graph
                        and v in trial_graph
                        and nx.has_path(trial_graph, v, u)
                    )
                )
                is_product_link = edge_data.get("role") in {
                    "product",
                    "single_reactant_direct",
                }
                if closes_cycle:
                    if is_product_link:
                        staged_hidden[v] += float(edge_data["weight"])
                        hidden_cycle_links += 1
                    continue

                staged_edges.append((u, v, edge_data))
                trial_graph.add_edge(u, v, weight=edge_data["weight"])
                if is_product_link:
                    has_visible_product = True

            if not has_visible_product:
                hidden_cycle_only_reactions += 1
                continue

            final_edges.extend(staged_edges)
            G_display_builder = trial_graph
            if staged_hidden:
                hidden_cycle_products[reaction_node].update(staged_hidden)

        # 7. Reconstruct the plotting network and arrange nodes by graph depth starting from sources.
        G_draw = G_display_builder

        # The Sankey layout is a projection of the complete, lineage-closed selection.
        if initial_nodes:
            reachable_draw_nodes = set(initial_set).intersection(G_draw.nodes)
            for source in list(reachable_draw_nodes):
                reachable_draw_nodes.update(nx.descendants(G_draw, source))
            omitted_display_nodes = set(G_draw.nodes) - reachable_draw_nodes
            if omitted_display_nodes:
                G_draw = G_draw.subgraph(reachable_draw_nodes).copy()
                final_edges = [
                    (u, v, data)
                    for u, v, data in final_edges
                    if u in G_draw and v in G_draw
                ]
                print(
                    "   Omitted "
                    f"{len(omitted_display_nodes)} display-only node(s) whose "
                    "acyclic projection has no path from the recorded initial "
                    "inventory."
                )

            pseudo_sources = [
                n for n in G_draw.nodes
                if n not in initial_set and G_draw.in_degree(n) == 0
            ]
            if pseudo_sources:
                _remove_stale_sankey()
                print(
                    "⚠️ Transition-map lineage closure failed; non-initial "
                    "sources remain after projection pruning. No Sankey was "
                    "written."
                )
                for n in pseudo_sources:
                    print("      ", node_formula.get(n, n), "|", node_smiles.get(n, ""))
                return

        draw_nodes_set = set(G_draw.nodes)
        initial_nodes = [n for n in initial_nodes if n in draw_nodes_set]
        initial_set = set(initial_nodes)

        depth = {}
        for n in initial_nodes:
            depth[n] = 0

        try:
            topo_nodes = list(nx.topological_sort(G_draw))
        except Exception:
            topo_nodes = list(draw_nodes_set)

        for u in topo_nodes:
            if u not in depth:
                continue
            for v in G_draw.successors(u):
                nd = depth[u] + 1
                if v not in depth or nd > depth[v]:
                    depth[v] = nd

        unreached = [n for n in draw_nodes_set if n not in depth]
        for n in unreached:
            depth[n] = 1 + int(first_rxn.get(n, 0) > 0)

        max_depth = max(depth.values()) if depth else 1
        N_COLS = max(6, min(10, max_depth + 1))

        col_of_node = {}
        for n in draw_nodes_set:
            d = depth.get(n, 1)
            if max_depth <= N_COLS - 1:
                col = d
            else:
                col = int(round(d * (N_COLS - 1) / max(1, max_depth)))
            col_of_node[n] = min(max(col, 0), N_COLS - 1)

        for n in initial_nodes:
            col_of_node[n] = 0

        col_x = {
            c: (0.05 + 0.87 * c / (N_COLS - 1))
            for c in range(N_COLS)
        }

        x_of_node = {
            n: col_x[col_of_node[n]]
            for n in draw_nodes_set
        }
        
        col_members = defaultdict(list)
        for n in draw_nodes_set:
            col_members[col_of_node[n]].append(n)

        node_y_map = {}
        for col, members in col_members.items():
            members_sorted = sorted(
                members,
                key=lambda nn: (
                    -sum(
                        d['weight']
                        for u, v, d in final_edges
                        if u == nn
                    ),
                    1 if _is_reaction_node(nn) else 0,
                    first_rxn.get(nn, 10**12),
                    str(node_formula.get(nn, nn)),
                    str(node_smiles.get(nn, "")),
                ),
            )
            k = len(members_sorted)
            for j, nn in enumerate(members_sorted):
                node_y_map[nn] = (0.05 + 0.90 * j / (k - 1)) if k > 1 else 0.5

        draw_nodes = sorted(
            draw_nodes_set,
            key=lambda n: (
                col_of_node.get(n, 0),
                node_y_map.get(n, 0.5),
                1 if _is_reaction_node(n) else 0,
                first_rxn.get(n, 10**12),
                str(node_formula.get(n, n)),
                str(node_smiles.get(n, "")),
            ),
        )

        n_map = {n: i for i, n in enumerate(draw_nodes)}
        node_x = [x_of_node.get(n, 0.5) for n in draw_nodes]
        node_y = [node_y_map.get(n, 0.5) for n in draw_nodes]

        # 8. Species labels retain the existing formula/isomer convention.
        species_draw_nodes = [
            node for node in draw_nodes
            if not _is_reaction_node(node)
        ]
        reaction_draw_nodes = [
            node for node in draw_nodes if _is_reaction_node(node)
        ]
        formula_to_keys = defaultdict(list)
        for n in species_draw_nodes:
            f = self._to_subscript(node_formula.get(n, node_smiles.get(n, str(n))))
            formula_to_keys[f].append(n)

        key_to_label = {}
        for f, klist in formula_to_keys.items():
            if len(klist) == 1:
                k = klist[0]
                smi = node_smiles.get(k, '')
                key_to_label[k] = self.smiles_name_dict.get(smi, f) \
                    if hasattr(self, 'smiles_name_dict') else f
            else:
                for i, k in enumerate(klist):
                    smi = node_smiles.get(k, '')
                    if hasattr(self, 'smiles_name_dict') and smi in self.smiles_name_dict:
                        key_to_label[k] = self.smiles_name_dict[smi]
                    else:
                        key_to_label[k] = f"{f}-{i+1}"

        species_digits = max(
            2,
            len(str(max(1, len(species_draw_nodes)))),
        )
        reaction_digits = max(
            2,
            len(str(max(1, len(reaction_draw_nodes)))),
        )
        node_short_ids = {}
        for index, node in enumerate(species_draw_nodes, start=1):
            node_short_ids[node] = f"S{index:0{species_digits}d}"
        visible_reactions = [
            node for node in selected_reactions
            if node in draw_nodes_set
        ]
        reaction_label_order = {
            node: index
            for index, node in enumerate(visible_reactions, start=1)
        }
        for node in reaction_draw_nodes:
            index = reaction_label_order[node]
            short_id = f"R{index:0{reaction_digits}d}"
            node_short_ids[node] = short_id
            support = float(reaction_data[node]["weight"])
            key_to_label[node] = f"{short_id} ×{support:.3g}"

        # retain their short Rxx label so their gray geometry stays readable.
        compact_key_mode = len(species_draw_nodes) > 30
        draw_labels = []
        for node in draw_nodes:
            if _is_reaction_node(node):
                draw_labels.append(key_to_label[node])
            elif compact_key_mode:
                draw_labels.append(node_short_ids[node])
            else:
                draw_labels.append(
                    key_to_label.get(node, str(node_smiles.get(node, node)))
                )

        sankey_palette = [
            "#39B54A",  # green
            "#2CB7A9",  # teal
            "#3E91D8",  # blue
            "#7758C7",  # violet
            "#D24FA8",  # magenta
            "#F06449",  # orange-red
            "#F2BE14",  # golden yellow
        ]

        def _hex_rgb(color):
            color = str(color).lstrip("#")
            return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

        def _rgb_hex(rgb):
            return "#{:02X}{:02X}{:02X}".format(
                *[int(round(max(0, min(255, value)))) for value in rgb]
            )

        def _palette_color(position):
            """Interpolate continuously through the Sankey palette."""
            position = max(0.0, min(1.0, float(position)))
            scaled = position * (len(sankey_palette) - 1)
            left = int(np.floor(scaled))
            right = min(left + 1, len(sankey_palette) - 1)
            fraction = scaled - left
            rgb_left = _hex_rgb(sankey_palette[left])
            rgb_right = _hex_rgb(sankey_palette[right])
            return _rgb_hex(tuple(
                (1.0 - fraction) * a + fraction * b
                for a, b in zip(rgb_left, rgb_right)
            ))

        def _blend_hex(color_a, color_b, fraction=0.5):
            """Blend two node colors for a softly colored Sankey ribbon."""
            fraction = max(0.0, min(1.0, float(fraction)))
            rgb_a = _hex_rgb(color_a)
            rgb_b = _hex_rgb(color_b)
            return _rgb_hex(tuple(
                (1.0 - fraction) * a + fraction * b
                for a, b in zip(rgb_a, rgb_b)
            ))

        def _hex_rgba(color, alpha):
            r, g, b = _hex_rgb(color)
            return f"rgba({r}, {g}, {b}, {float(alpha):.3f})"

        terminal_nodes = {
            node for node in draw_nodes
            if not _is_reaction_node(node)
            and G_draw.out_degree(node) == 0
        }
        node_color_map = {}
        for node in draw_nodes:
            column_fraction = (
                col_of_node.get(node, 0) / max(1, N_COLS - 1)
            )

            if _is_reaction_node(node):
                node_color_map[node] = "#D9DDE3"
                continue
            elif node in initial_set:
                color_position = 0.0
            elif node in terminal_nodes:
                # Products occupy the warm end of the palette. Their y-position
                # adds an orange -> yellow variation from top to bottom.
                y_fraction = max(
                    0.0,
                    min(1.0, float(node_y_map.get(node, 0.5))),
                )
                color_position = max(
                    column_fraction,
                    0.82 + 0.18 * y_fraction,
                )
            else:
                color_position = column_fraction

            node_color_map[node] = _palette_color(color_position)

        node_colors = [node_color_map[node] for node in draw_nodes]

        node_in = []
        node_out = []
        for n in draw_nodes:
            in_val = sum(d['weight'] for u, v, d in final_edges if v == n)
            out_val = sum(d['weight'] for u, v, d in final_edges if u == n)
            node_in.append(in_val)
            node_out.append(out_val)

        sources = [n_map[u] for u, v, d in final_edges]
        targets = [n_map[v] for u, v, d in final_edges]
        values = [d['weight'] for u, v, d in final_edges]

        def _hidden_cycle_html(reaction_node):
            hidden = hidden_cycle_products.get(reaction_node)
            if not hidden:
                return ""
            items = []
            for species, normalized_support in hidden.items():
                label = key_to_label.get(
                    species,
                    self._to_subscript(
                        node_formula.get(species, str(species))
                    ),
                )
                items.append(
                    f"{label} (normalized support "
                    f"{float(normalized_support):.3g})"
                )
            return (
                "<br>Cycle-closing product(s) omitted from Sankey: "
                + ", ".join(items)
            )

        link_custom = []
        for u, v, data in final_edges:
            if compact_key_mode:
                heading = (
                    f"<b>{node_short_ids[u]} → "
                    f"{node_short_ids[v]}</b><br>"
                    f"{key_to_label.get(u, str(u))} → "
                    f"{key_to_label.get(v, str(v))}"
                )
            else:
                heading = (
                    f"<b>{key_to_label.get(u, str(u))} → "
                    f"{key_to_label.get(v, str(v))}</b>"
                )
            forward_weight = float(
                data.get("forward_weight", data["weight"])
            )
            reverse_weight = float(data.get("reverse_weight", 0.0))
            reaction_weight = float(
                data.get("reaction_weight", data["weight"])
            )
            supporting_events = sorted(data.get("events", set()))
            shown_events = ", ".join(supporting_events[:10])
            if len(supporting_events) > 10:
                shown_events += (
                    f", … (+{len(supporting_events) - 10})"
                )
            reaction_types = sorted(data.get("types", set()))
            role_text = {
                "reactant": "Reactant → reaction",
                "product": "Reaction → product",
                "single_reactant_direct": (
                    "Single-species reactant direct transition"
                ),
            }.get(data.get("role"), "Reaction participant")
            hidden_cycle_html = _hidden_cycle_html(
                data.get("reaction_node")
            )
            link_custom.append(
                f"{heading}<br>"
                f"<b>{data.get('equation', '')}</b><br>"
                f"Participant role: {role_text}<br>"
                f"Stoichiometric coefficient: "
                f"{int(data.get('coefficient', 1))}<br>"
                f"Reaction type: "
                f"{', '.join(reaction_types) or '(not reported)'}<br>"
                f"Forward event support: {forward_weight:.3g}<br>"
                f"Reverse event support: {reverse_weight:.3g}<br>"
                f"Net reaction-event support: {reaction_weight:.3g}<br>"
                f"Normalized ribbon share: "
                f"{float(data['weight']):.3g}<br>"
                f"Supporting pathways: "
                f"{shown_events or '(not reported)'}"
                f"{hidden_cycle_html}"
            )

        link_colors = []
        for u, v, _data in final_edges:
            ribbon_hex = _blend_hex(
                node_color_map[u],
                node_color_map[v],
                fraction=0.48,
            )
            link_colors.append(_hex_rgba(ribbon_hex, 0.24))

        node_cd = []
        for index, node in enumerate(draw_nodes):
            full_label = key_to_label.get(
                node,
                str(node_smiles.get(node, node)),
            )
            if _is_reaction_node(node):
                reaction_types = sorted(reaction_data[node]["types"])
                hover_heading = (
                    f"{node_short_ids[node]} — Aggregated reaction"
                )
                detail_html = (
                    f"Reaction: {reaction_data[node]['equation']}<br>"
                    f"Type: {', '.join(reaction_types) or '(not reported)'}<br>"
                    f"Frames: {reaction_data[node]['first_frame']}–"
                    f"{reaction_data[node]['last_frame']}<br>"
                    f"Forward / reverse / net events: "
                    f"{reaction_data[node]['forward_weight']:.3g} / "
                    f"{reaction_data[node]['reverse_weight']:.3g} / "
                    f"{reaction_data[node]['weight']:.3g}"
                    f"{_hidden_cycle_html(node)}"
                )
            else:
                hover_heading = (
                    f"{node_short_ids[node]} — {full_label}"
                    if compact_key_mode
                    else full_label
                )
                detail_html = (
                    f"Formula: "
                    f"{self._to_subscript(node_formula.get(node, ''))}<br>"
                    f"SMILES: {node_smiles.get(node, '')}"
                )
            node_cd.append((
                detail_html,
                node_in[index],
                node_out[index],
                hover_heading,
            ))

        num_nodes = len(draw_labels)
        num_species = len(species_draw_nodes)
        max_nodes_in_column = max(
            (len(members) for members in col_members.values()),
            default=1,
        )

        if num_nodes <= 15:
            dynamic_height = max(520, num_nodes * 48)
            dynamic_width = 980
            node_pad = 38
            node_thickness = 18
            label_font_size = 18
        elif num_nodes <= 40:
            dynamic_height = max(
                720,
                260 + max_nodes_in_column * 82,
            )
            dynamic_width = max(1250, N_COLS * 165)
            node_pad = 24
            node_thickness = 16
            label_font_size = 17
        else:
            dynamic_height = min(
                1700,
                max(900, 280 + max_nodes_in_column * 105),
            )
            dynamic_width = min(
                2600,
                max(1750, N_COLS * 225),
            )
            node_pad = 18
            node_thickness = 14
            label_font_size = 16

        legend_columns = 0
        legend_rows = 0
        sankey_domain_end = 1.0
        sankey_domain_y0 = 0.0
        sankey_domain_y1 = 1.0
        figure_width = dynamic_width
        node_border_width = 0.6
        if compact_key_mode:
            if num_species <= 24:
                legend_columns = 1
            elif num_species <= 48:
                legend_columns = 2
            else:
                legend_columns = 3
            legend_rows = int(np.ceil(num_species / legend_columns))
            legend_width = 55 + legend_columns * 360
            figure_width = dynamic_width + legend_width
            dynamic_height = max(
                dynamic_height,
                160 + legend_rows * 31,
            )
            sankey_domain_end = min(
                0.81,
                max(0.56, dynamic_width / figure_width - 0.012),
            )
            sankey_domain_y0 = 0.05
            sankey_domain_y1 = 0.95
            node_pad = max(node_pad, 24)
            node_thickness = 10
            # Keep compact transition maps readable as well.
            label_font_size = 16
            node_border_width = 0.35

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            domain=dict(
                x=[0.0, sankey_domain_end],
                y=[sankey_domain_y0, sankey_domain_y1],
            ),
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                label=draw_labels,
                color=node_colors,
                line=dict(color="white", width=node_border_width),
                x=node_x,
                customdata=node_cd,
                hovertemplate=(
                    "<b>%{customdata[3]}</b><br>"
                    "%{customdata[0]}<br>"
                    "Incoming normalized support: %{customdata[1]:.3g}<br>"
                    "Outgoing normalized support: %{customdata[2]:.3g}"
                    "<extra></extra>"
                )
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                customdata=link_custom,
                hovertemplate="%{customdata}<extra></extra>",
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=(
                    "Net Event-Support Hybrid Reaction Map"
                    "<br><sup> pale-gray blocks mark reactions with multipl"
                    "distinct reactant species; Ribbon width is normalized"
                    "event support, not material flow or rate.</sup>"
                ),
                font=dict(size=22, color="#111111", family="Arial"),
                pad=dict(b=18)
            ),
            font=dict(size=label_font_size, color="#000000", family="Arial"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(
                t=105,
                l=70,
                r=45 if compact_key_mode else 120,
                b=40,
            ),
            height=dynamic_height,
            width=figure_width,
        )

        if compact_key_mode:
            legend_left = sankey_domain_end + 0.012
            legend_area_width = max(0.10, 0.995 - legend_left)
            column_width = legend_area_width / legend_columns
            row_step = min(0.050, 0.86 / max(1, legend_rows))
            legend_top = 0.925

            fig.add_annotation(
                x=legend_left,
                y=0.985,
                xref='paper',
                yref='paper',
                text=(
                    '<b>Species key</b> · Gray Rxx blocks are '
                    'multi-species-reactant reactions'
                ),
                showarrow=False,
                xanchor='left',
                yanchor='top',
                font=dict(size=14, color='#111111', family='Arial'),
            )
            fig.add_annotation(
                x=legend_left,
                y=0.950,
                xref='paper',
                yref='paper',
                text=(
                    'Full formula, SMILES, and omitted cycle-closing products '
                    'remain available on reaction hover.'
                ),
                showarrow=False,
                xanchor='left',
                yanchor='top',
                font=dict(size=10, color='#666666', family='Arial'),
            )

            for index, node in enumerate(species_draw_nodes):
                column = index // legend_rows
                row = index % legend_rows
                x0 = legend_left + column * column_width
                y = legend_top - row * row_step
                swatch_width = min(0.010, column_width * 0.055)
                swatch_half_height = min(0.009, row_step * 0.30)
                fig.add_shape(
                    type='rect',
                    xref='paper',
                    yref='paper',
                    x0=x0,
                    x1=x0 + swatch_width,
                    y0=y - swatch_half_height,
                    y1=y + swatch_half_height,
                    fillcolor=node_color_map[node],
                    line=dict(color='white', width=node_border_width),
                    layer='above',
                )
                fig.add_annotation(
                    x=x0 + swatch_width + 0.006,
                    y=y,
                    xref='paper',
                    yref='paper',
                    text=(
                        f"<b>{node_short_ids[node]}</b>  "
                        f"{key_to_label.get(node, str(node))}"
                    ),
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    align='left',
                    font=dict(size=14, color='#111111', family='Arial'),
                )

        config = {
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'net_event_support_hybrid_reaction_map_HighRes',
                'height': dynamic_height,
                'width': figure_width,
                'scale': 5,
            },
            'displaylogo': False,
        }
        
        fig.write_html(out_path, config=config)
        
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            tooltip_css = f"""
            <style>
                html, body {{
                    min-width: {figure_width}px;
                    overflow-x: auto !important;
                }}
                div.plotly-notifier {{ visibility: hidden !important; }}
                .hovertext text {{
                    font-family: Arial, sans-serif !important;
                }}
                g.sankey-node text {{
                    text-shadow: none !important;
                    stroke: none !important;
                    paint-order: normal !important;
                    font-weight: 600 !important;
                    fill: #000000 !important;
                }}
            </style>
            """
            html_content = html_content.replace(
                '</head>',
                tooltip_css + '</head>',
                1,
            )

            sankey_js = """
            <script>
            function bindSankeyEvents() {
                var plotElements = document.getElementsByClassName('plotly-graph-div');
                if (plotElements.length === 0) {
                    setTimeout(bindSankeyEvents, 500);
                    return;
                }
                var plotElement = plotElements[0];
                
                var style = document.createElement('style');
                style.innerHTML = '.sankey-node, .node-rect { cursor: pointer; transition: opacity 0.2s; } .sankey-link { transition: opacity 0.2s; } ' + 
                                  'g.sankey-node text { text-shadow: none !important; stroke: none !important; paint-order: normal !important; font-weight: 600 !important; fill: #000000 !important; }';
                document.head.appendChild(style);
                
                setInterval(function() {
                    var texts = document.querySelectorAll('g.sankey-node text');
                    texts.forEach(function(t) {
                        if (t.style.textShadow !== 'none') {
                            t.style.setProperty('text-shadow', 'none', 'important');
                            t.style.setProperty('stroke', 'none', 'important');
                            t.style.setProperty('paint-order', 'normal', 'important');
                            t.style.setProperty('font-weight', '600', 'important');
                            t.style.setProperty('fill', '#000000', 'important');
                        }
                    });
                }, 500);
                
                if (!plotElement.on) {
                    setTimeout(bindSankeyEvents, 500);
                    return;
                }
                
                function showFeedback(msg, isError) {
                    var t = document.createElement('div');
                    t.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:10px 20px;border-radius:5px;z-index:10000;font-family:sans-serif;color:white;transition:opacity 0.5s;box-shadow: 0 2px 5px rgba(0,0,0,0.2);';
                    t.style.background = isError ? 'rgba(220,53,69,0.9)' : 'rgba(40,167,69,0.9)';
                    t.innerText = msg;
                    document.body.appendChild(t);
                    setTimeout(function(){ t.style.opacity = '0'; setTimeout(function(){t.remove();}, 500); }, 2000);
                }

                window.onerror = function(msg, url, lineNo, columnNo, error) {
                    console.error(msg);
                    return false;
                };
                
                var originalNodeColors = null;
                var originalLinkColors = null;
                var sources = null;
                var targets = null;
                var selectedNodes = new Set();
                
                function handleNodeClick(nodeId, isMulti) {
                    if (nodeId === undefined || nodeId === null) return;
                    
                    if (!originalNodeColors) {
                        try {
                            var data0 = plotElement.data[0];
                            originalNodeColors = [...data0.node.color];
                            sources = data0.link.source;
                            targets = data0.link.target;
                            
                            var lc = data0.link.color;
                            if (Array.isArray(lc)) {
                                originalLinkColors = [...lc];
                            } else {
                                originalLinkColors = Array(sources.length).fill(lc);
                            }
                        } catch(err) {
                            showFeedback('Init Error: ' + err.message, true);
                            console.error(err);
                            return;
                        }
                    }
                    
                    if (isMulti) {
                        if (selectedNodes.has(nodeId)) selectedNodes.delete(nodeId);
                        else selectedNodes.add(nodeId);
                    } else {
                        if (selectedNodes.has(nodeId) && selectedNodes.size === 1) {
                            selectedNodes.clear();
                        } else {
                            selectedNodes.clear();
                            selectedNodes.add(nodeId);
                        }
                    }
                    
                    try {
                        updateSankeyColors();
                    } catch(err) {
                        showFeedback('Update Error: ' + err.message, true);
                    }
                }

                plotElement.on('plotly_click', function(data) {
                    if (!data.points || data.points.length === 0) return;
                    var pt = data.points[0];
                    if (pt.hasOwnProperty('source') && pt.hasOwnProperty('target')) return; 
                    var nodeId = pt.pointNumber !== undefined ? pt.pointNumber : pt.index;
                    if (nodeId === undefined) return;
                    
                    var e = data.event || window.event || {};
                    var now = Date.now();
                    if (now - lastClickTime < 200 && lastClickedNode === nodeId) return;
                    
                    handleNodeClick(nodeId, e.ctrlKey || e.metaKey || e.shiftKey);
                });
                
                var startX = 0, startY = 0, startTime = 0;
                plotElement.addEventListener('mousedown', function(e) {
                    startX = e.clientX;
                    startY = e.clientY;
                    startTime = Date.now();
                }, true);

                plotElement.addEventListener('mouseup', function(e) {
                    var dx = e.clientX - startX;
                    var dy = e.clientY - startY;
                    var dt = Date.now() - startTime;
                    
                    if (Math.sqrt(dx*dx + dy*dy) > 5 || dt > 1000) return;

                    var target = e.target;
                    var current = target;
                    var nodeGroup = null;
                    var cls;
                    
                    while (current && current !== plotElement) {
                        cls = current.getAttribute ? current.getAttribute('class') : '';
                        if (cls && (cls.indexOf('sankey-node') !== -1 || cls.indexOf('node-rect') !== -1)) {
                            nodeGroup = current;
                            break;
                        }
                        current = current.parentNode;
                    }
                    
                    if (!nodeGroup) {
                        var isSvg = target.tagName && target.tagName.toLowerCase() === 'svg';
                        var isBg = target.getAttribute && (target.getAttribute('class')||'').indexOf('sankey') !== -1;
                        if ((isSvg || isBg) && selectedNodes.size > 0) {
                            selectedNodes.clear();
                            updateSankeyColors();
                        }
                        return;
                    }
                    
                    var d3Data = nodeGroup.__data__ || (nodeGroup.parentNode && nodeGroup.parentNode.__data__);
                    if (!d3Data && nodeGroup.querySelector) {
                         var child = nodeGroup.querySelector('path, rect');
                         if (child) d3Data = child.__data__;
                    }

                    if (!d3Data) return;
                    
                    var nodeId = d3Data.pointNumber;
                    if (nodeId === undefined) nodeId = d3Data.index;
                    if (nodeId === undefined) nodeId = d3Data.i;
                    if (nodeId === undefined && d3Data.node) nodeId = d3Data.node.index; 
                    
                    if (nodeId !== undefined) {
                        var now = Date.now();
                        if (now - lastClickTime < 200 && lastClickedNode === nodeId) return;
                        
                        lastClickTime = now;
                        lastClickedNode = nodeId;
                        
                        handleNodeClick(nodeId, e.ctrlKey || e.metaKey || e.shiftKey);
                    }
                }, true);
                
                var lastClickTime = 0;
                var lastClickedNode = -1;
                
                plotElement.on('plotly_doubleclick', function() {
                    if (selectedNodes.size > 0) {
                        selectedNodes.clear();
                        updateSankeyColors();
                    }
                });
                
                function updateSankeyColors() {
                    if (selectedNodes.size === 0) {
                        Plotly.restyle(plotElement, {
                            'node.color': [originalNodeColors],
                            'link.color': [originalLinkColors]
                        });
                        return;
                    }
                    
                    var newNodeColors = originalNodeColors.map(c => 'rgba(200,200,200,0.3)'); 
                    var newLinkColors = originalLinkColors.map(c => 'rgba(230,230,230,0.05)'); 
                    
                    var activeNodes = new Set(selectedNodes);
                    var activeLinks = new Set();
                    
                    if (selectedNodes.size > 0) {
                        var forwardQueue = Array.from(selectedNodes);
                        var visitedFwd = new Set(forwardQueue);
                        while(forwardQueue.length > 0) {
                            var curr = forwardQueue.shift();
                            for (var i = 0; i < sources.length; i++) {
                                if (sources[i] === curr) {
                                    activeLinks.add(i);
                                    if (!activeNodes.has(targets[i])) {
                                        activeNodes.add(targets[i]);
                                        if (!visitedFwd.has(targets[i])) {
                                            visitedFwd.add(targets[i]);
                                            forwardQueue.push(targets[i]);
                                        }
                                    }
                                }
                            }
                        }
                        
                        var backwardQueue = Array.from(selectedNodes);
                        var visitedBwd = new Set(backwardQueue);
                        while(backwardQueue.length > 0) {
                            var curr = backwardQueue.shift();
                            for (var i = 0; i < targets.length; i++) {
                                if (targets[i] === curr) {
                                    activeLinks.add(i);
                                    if (!activeNodes.has(sources[i])) {
                                        activeNodes.add(sources[i]);
                                        if (!visitedBwd.has(sources[i])) {
                                            visitedBwd.add(sources[i]);
                                            backwardQueue.push(sources[i]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    activeNodes.forEach(function(nId) {
                        if (selectedNodes.has(nId)) {
                            newNodeColors[nId] = '#e74c3c'; 
                        } else {
                            newNodeColors[nId] = originalNodeColors[nId]; 
                        }
                    });
                    
                    function withOpacity(color, opacity) {
                        if (typeof color !== 'string') {
                            return 'rgba(52, 152, 219,' + opacity + ')';
                        }
                        var match = color.match(/^#([0-9a-f]{6})$/i);
                        if (!match) return color;
                        var value = parseInt(match[1], 16);
                        var red = (value >> 16) & 255;
                        var green = (value >> 8) & 255;
                        var blue = value & 255;
                        return 'rgba(' + red + ',' + green + ',' + blue
                            + ',' + opacity + ')';
                    }

                    activeLinks.forEach(function(lId) {
                        newLinkColors[lId] = withOpacity(
                            originalNodeColors[sources[lId]],
                            0.82
                        );
                    });
                    
                    Plotly.restyle(plotElement, {
                        'node.color': [newNodeColors],
                        'link.color': [newLinkColors]
                    });
                }
            }
            setTimeout(bindSankeyEvents, 500);
            </script>
            """
            
            html_content = html_content.replace(
                '</body>',
                sankey_js + '\n</body>',
                1,
            )
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                            
            print(
                "✨ Net Event-Support Hybrid Reaction Map saved: "
                f"{out_path}"
            )
        except Exception as e:
            print(f"Warning: Could not finalize transition-map HTML: {e}")
