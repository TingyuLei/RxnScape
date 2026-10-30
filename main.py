#!/usr/bin/env python
import os
import sys
import builtins

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

# Prevent OpenBLAS / OMP from multithreading inside multiprocessing workers, preventing deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ASE to use minimal memory
os.environ["ASE_DFTB_COMMAND"] = ""

import argparse
import ast
import inspect
import re
import json
import hashlib
import platform
import math
import time
import pandas as pd
import pickle
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)

try:
    if __package__:
        from .loader import TrajectoryLoader
        from .analyzer import TrajectoryAnalyzer
        from .reaction import (
            ReactionDetector, ReactionCleaner, ReactionClusterer,
            ReactionStats, IntermediateAnalyzer,
        )
        from .visualizer import Visualizer
        from .config import Config
        from . import __version__ as RXNSCAPE_VERSION
    elif os.path.isfile(os.path.join(script_dir, "core", "__init__.py")):
        # Standard distribution layout:
        #   main.py / rxn.py / core/{loader,analyzer,reaction,...}.py
        from core import (
            TrajectoryLoader, TrajectoryAnalyzer, ReactionDetector,
            ReactionCleaner, ReactionClusterer, ReactionStats,
            IntermediateAnalyzer, Visualizer,
        )
        from core.config import Config
        from core import __version__ as RXNSCAPE_VERSION
    else:
        # Also support a compact, flat script directory for development or
        # single-folder deployments.
        from loader import TrajectoryLoader
        from analyzer import TrajectoryAnalyzer
        from reaction import (
            ReactionDetector, ReactionCleaner, ReactionClusterer,
            ReactionStats, IntermediateAnalyzer,
        )
        from visualizer import Visualizer
        from config import Config
        RXNSCAPE_VERSION = "1.0.0"
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)

CACHE_FILE = "topology_cache.pkl"
CONFIG_FILE = "rxn.yaml"


DEFAULTS = {
    "input_file": None,
    "remove": None,
    "bond_scale": 1.2,
    "adsorption_scale": 1.25,
    "hmm_prob": 0.99,
    "workers": None,
    "max_size": 60,
    "rings": False,
    "auto_neb": True,
    "sankey": True,
    "sankey_initial_species": None,
    "sankey_elements": list(Config.SANKEY_DEFAULT_ELEMENTS),
    "stage": "all",
    "stability": 5,
    "local_stability": None,   # None = automatically take max(2, stability // 2)
    "single_frame_recovery": True,
    "allow_ambiguous_topology": False,
    "cell": None,
    "pbc": None,
}


CLEAN_EVENT_COLUMNS = [
    "Reaction_ID",
    "Raw_Reaction_ID",
    "Frame",
    "Reactant_Frame",
    "Product_Frame",
    "Reactants_Formula",
    "Products_Formula",
    "Reactants_SMILES",
    "Products_SMILES",
    "Reaction_Type",
    "Isomerization_Class",
    "Elementary_Mode",
    "Reactant_Source_Raw_IDs",
    "Changed_Bonds",
    "Reactant_Atom_Groups",
    "Product_Atom_Groups",
    "Structure_Representative_ID",
    "Snapshot",
    "NEB_Input",
]




RAW_EVENT_COLUMNS = [
    "Raw_Reaction_ID",
    "Frame",
    "Reactant_Frame",
    "Product_Frame",
    "Reactants_Formula",
    "Products_Formula",
    "Reactants_SMILES",
    "Products_SMILES",
    "Reaction_Type",
    "Is_Elementary",
    "Isomerization_Class",
    "Topology_Validated",
    "Valence_Warning",
    "Lineage_Complete",
    "Elementary_Mode",
    "Recovery_Type",
    "Recovery_Result",
    "Observed_Stable_Exit_Source",
    "Observed_Stable_Exit_Product_Validated",
    "Observed_Corridor_Net_Validated",
    "Observed_Corridor_Net_Elementary",
    "Observed_State_Source_Min_Dwell",
    "Observed_Stable_Exit_Settled_After_Frames",
    "Exclusion_Reason",
    "Changed_Bonds",
    "Reactant_Atom_Groups",
    "Product_Atom_Groups",
    "Clean_Reaction_ID",
]

PATHWAY_EVENT_COLUMNS = [
    "Pathway_ID",
    "Source_Table",
    "Source_Reaction_ID",
    "Source_Raw_Reaction_ID",
    "Frame",
    "Reactants_Formula",
    "Products_Formula",
    "Reactants_SMILES",
    "Products_SMILES",
    "Reaction_Type",
    "Pathway_Role",
    "Elementary_Mode",
    "Inclusion_Reason",
    "Lineage_Note",
    "Upper_Source",
    "Upper_Source_Reaction_ID",
    "Upper_Source_Status",
    "Reactant_Atom_Groups",
    "Product_Atom_Groups",
]

UNRESOLVED_EVENT_COLUMNS = [
    "Unresolved_ID",
    "Source_Raw_Reaction_ID",
    "Frame",
    "Reactant_Frame",
    "Product_Frame",
    "Reactants_Formula",
    "Products_Formula",
    "Reactants_SMILES",
    "Products_SMILES",
    "Reaction_Type",
    "Reason",
    "Recovery_Type",
    "Recovery_Result",
    "Pathway_Status",
    "Pathway_IDs",
    "Changed_Bonds",
    "Reactant_Atom_Groups",
    "Product_Atom_Groups",
    "Elementary_Mode",
    "Raw_Changed_Bond_Count",
    "Validated_Changed_Bond_Count",
    "Changed_Bond_Components",
    "Note",
]


def _is_missing_export_value(value):
    """Return True for scalar values that should become an empty CSV cell."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return missing
    try:
        # NumPy scalar booleans expose ``item``; arrays must not be coerced.
        if getattr(missing, "ndim", None) == 0:
            return bool(missing.item())
    except Exception:
        pass
    return False


def _parse_export_atom_indices(value):
    """Parse legacy/list/string atom-index values into sorted unique integers."""
    if _is_missing_export_value(value):
        return []

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = re.findall(r"-?\d+", text)

    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        parsed = [parsed]
    elif not isinstance(parsed, (list, tuple, set)):
        parsed = re.findall(r"-?\d+", str(parsed))

    indices = set()
    for item in parsed:
        try:
            indices.add(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(indices)


def _format_export_atom_indices(value):
    """Write zero-based atom indices as a compact, deterministic pipe list."""
    return "|".join(str(index) for index in _parse_export_atom_indices(value))


def _format_export_atom_groups(value):
    """Write species-aligned atom groups as compact deterministic JSON."""
    if _is_missing_export_value(value):
        return "[]"
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "[]"
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return "[]"
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    if not isinstance(parsed, (list, tuple)):
        return "[]"
    if parsed and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in parsed
    ):
        parsed = [parsed]
    groups = []
    for group in parsed:
        if not isinstance(group, (list, tuple, set)):
            continue
        indices = set()
        for item in group:
            if isinstance(item, bool):
                continue
            try:
                indices.add(int(item))
            except (TypeError, ValueError):
                continue
        groups.append(sorted(indices))
    return json.dumps(groups, separators=(",", ":"))


def _parse_export_edge_pairs(value):
    """Parse edge collections such as ``0-1;2-3`` or ``[(0, 1)]``."""
    if _is_missing_export_value(value):
        return set()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        pairs = re.findall(r"(-?\d+)\s*-\s*(-?\d+)", text)
        if pairs:
            return {
                tuple(sorted((int(left), int(right))))
                for left, right in pairs
            }
        try:
            value = ast.literal_eval(text)
        except Exception:
            return set()

    pairs = set()
    if isinstance(value, dict):
        value = value.keys()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, (list, tuple, set)) and len(item) >= 2:
                try:
                    pair = tuple(sorted((int(list(item)[0]), int(list(item)[1]))))
                except (TypeError, ValueError):
                    continue
                pairs.add(pair)
    return pairs


def _row_has_explicit_validated_edges(row):
    """Distinguish an explicit zero validated-edge result from missing audit data."""
    count = row.get("Endpoint_Validated_Changed_Bond_Count", None)
    if not _is_missing_export_value(count) and str(count).strip():
        return True
    value = row.get("Endpoint_Validated_Changed_Bonds", None)
    return not _is_missing_export_value(value) and bool(str(value).strip())


def _format_clean_changed_bonds(row, detector):
    """Return ``broken:...; formed:...`` using validated endpoint changes first."""
    indices = _parse_export_atom_indices(row.get("Atom_Indices", []))
    if not indices or detector is None or detector.traj is None:
        return ""

    def endpoint_frame(boundary_field, representative_field):
        boundary_value = row.get(boundary_field, None)
        value = (
            boundary_value
            if not _is_missing_export_value(boundary_value)
            and str(boundary_value).strip()
            else row.get(representative_field)
        )
        return int(float(value))

    try:
        reactant_frame = endpoint_frame(
            "Endpoint_Boundary_Reactant_Frame",
            "Reactant_Frame",
        )
        product_frame = endpoint_frame(
            "Endpoint_Boundary_Product_Frame",
            "Product_Frame",
        )
    except (TypeError, ValueError):
        return ""

    if not (
        0 <= reactant_frame < len(detector.traj)
        and 0 <= product_frame < len(detector.traj)
    ):
        return ""

    try:
        reactant_atoms = detector.traj[reactant_frame]
        product_atoms = detector.traj[product_frame]
        reactant_bonds = {
            tuple(sorted((int(edge[0]), int(edge[1])))): bond_type
            for edge, bond_type in detector._get_local_bonds(
                reactant_atoms, indices
            )
        }
        product_bonds = {
            tuple(sorted((int(edge[0]), int(edge[1])))): bond_type
            for edge, bond_type in detector._get_local_bonds(
                product_atoms, indices
            )
        }
        symbols = reactant_atoms.get_chemical_symbols()
    except Exception:
        return ""

    broken = set(reactant_bonds) - set(product_bonds)
    formed = set(product_bonds) - set(reactant_bonds)

    # Endpoint recovery can identify a smaller, validated reaction centre.
    if _row_has_explicit_validated_edges(row):
        validated_pairs = _parse_export_edge_pairs(
            row.get("Endpoint_Validated_Changed_Bonds", "")
        )
        broken &= validated_pairs
        formed &= validated_pairs

    def atom_label(index):
        try:
            return f"{symbols[index]}{index}"
        except Exception:
            return str(index)

    def bond_label(pair):
        left, right = pair
        return f"{atom_label(left)}-{atom_label(right)}"

    parts = []
    if broken:
        parts.append(
            "broken:" + "|".join(
                bond_label(pair) for pair in sorted(broken)
            )
        )
    if formed:
        parts.append(
            "formed:" + "|".join(
                bond_label(pair) for pair in sorted(formed)
            )
        )
    return "; ".join(parts)


def _normalize_structure_output(value, kind, detector=None):
    """Normalize paths."""
    if _is_missing_export_value(value) or not str(value).strip():
        return "Skipped (Structure Unavailable)"

    text = str(value).strip()
    if text.startswith("Structure_Error:"):
        detail = text.split(":", 1)[1].strip()
        return f"Generation Failed ({detail})" if detail else "Generation Failed"

    lowered = text.lower()
    if lowered == "n/a (surface system)":
        return "Skipped (Surface System)"
    if lowered in {"n/a", "na", "none", "null"}:
        if kind == "NEB_Input" and getattr(detector, "neb_dir", None) == "DISABLED":
            return "Skipped (NEB Disabled)"
        return "Skipped (Structure Unavailable)"
    if lowered == "disabled":
        return "Skipped (NEB Disabled)" if kind == "NEB_Input" else "Skipped (Disabled)"
    return text


def build_clean_export_table(final_df, detector=None):
    """Build the fixed-schema Clean reaction table."""
    if final_df is None:
        final_df = pd.DataFrame()
    df = final_df.copy()
    export = pd.DataFrame(index=df.index)

    direct_columns = [
        "Reaction_ID",
        "Raw_Reaction_ID",
        "Reactants_Formula",
        "Products_Formula",
        "Reactants_SMILES",
        "Products_SMILES",
        "Isomerization_Class",
        "Elementary_Mode",
        "Reactant_Source_Raw_IDs",
        "Structure_Representative_ID",
    ]
    for column in direct_columns:
        export[column] = df[column] if column in df.columns else ""

    for column in ["Frame", "Reactant_Frame", "Product_Frame"]:
        values = df[column] if column in df.columns else pd.Series(
            pd.NA, index=df.index
        )
        export[column] = pd.to_numeric(values, errors="coerce").astype("Int64")

    if "Type" in df.columns:
        export["Reaction_Type"] = df["Type"]
    elif "Reaction_Type" in df.columns:
        export["Reaction_Type"] = df["Reaction_Type"]
    else:
        export["Reaction_Type"] = ""

    export["Changed_Bonds"] = df.apply(
        lambda row: _format_clean_changed_bonds(row, detector),
        axis=1,
    )
    for column in ["Reactant_Atom_Groups", "Product_Atom_Groups"]:
        export[column] = (
            df[column].map(_format_export_atom_groups)
            if column in df.columns
            else "[]"
        )
    export["Snapshot"] = (
        df["Snapshot"].map(
            lambda value: _normalize_structure_output(
                value, "Snapshot", detector
            )
        )
        if "Snapshot" in df.columns
        else "Skipped (Structure Unavailable)"
    )
    export["NEB_Input"] = (
        df["NEB_Input"].map(
            lambda value: _normalize_structure_output(
                value, "NEB_Input", detector
            )
        )
        if "NEB_Input" in df.columns
        else "Skipped (Structure Unavailable)"
    )

    # All text-like non-applicable values are written as empty cells.
    for column in CLEAN_EVENT_COLUMNS:
        if column not in export.columns:
            export[column] = ""
        if column not in {"Frame", "Reactant_Frame", "Product_Frame"}:
            export[column] = export[column].map(
                lambda value: "" if _is_missing_export_value(value) else value
            )

    return export.loc[:, CLEAN_EVENT_COLUMNS]


def _export_truthy(value):
    if _is_missing_export_value(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _export_bool(value):
    if _is_missing_export_value(value) or str(value).strip() == "":
        return ""
    return bool(_export_truthy(value))


def _export_positive_number(value):
    if _is_missing_export_value(value):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _recovery_summary(row, unresolved=False):
    endpoint_recovered = _export_truthy(
        row.get("Endpoint_Recovered", False)
    )
    endpoint_mode_value = row.get("Endpoint_Recovery_Mode", "")
    endpoint_mode = (
        ""
        if _is_missing_export_value(endpoint_mode_value)
        else str(endpoint_mode_value).strip()
    )
    endpoint_rejection_value = row.get("Endpoint_Rejection_Reason", "")
    endpoint_rejected = (
        False
        if _is_missing_export_value(endpoint_rejection_value)
        else bool(str(endpoint_rejection_value).strip())
    )

    endpoint_attempted = (
        endpoint_recovered
        or bool(endpoint_mode)
        or endpoint_rejected
    )

    lineage_recovered = (
        _export_truthy(row.get("Lineage_Recovered", False))
        or _export_truthy(row.get("Lineage_Restored_By_Cleaner", False))
    )
    composite_split = _export_positive_number(
        row.get("Split_Elementary_Records", 0)
    )

    kinds = []
    if endpoint_attempted or endpoint_recovered:
        kinds.append("endpoint")
    if lineage_recovered:
        kinds.append("lineage")
    if composite_split:
        kinds.append("composite_split")

    if "endpoint" in kinds and "lineage" in kinds:
        recovery_type = "endpoint_and_lineage"
    elif kinds:
        recovery_type = kinds[0]
    else:
        recovery_type = "none"

    successes = int(endpoint_recovered) + int(lineage_recovered) + int(composite_split)
    attempted = endpoint_attempted or lineage_recovered or composite_split

    if not attempted:
        recovery_result = "not_attempted" if unresolved else "not_required"
    elif successes and endpoint_attempted and not endpoint_recovered:
        recovery_result = "partially_recovered"
    elif successes:
        recovery_result = "recovered"
    elif endpoint_rejected:
        recovery_result = "rejected"
    else:
        recovery_result = "failed"

    return recovery_type, recovery_result


def _coerce_export_int_series(df, column):
    values = (
        df[column]
        if column in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def _finalize_export_table(export, columns, integer_columns=()):
    for column in columns:
        if column not in export.columns:
            export[column] = ""
        if column not in set(integer_columns):
            export[column] = export[column].map(
                lambda value: "" if _is_missing_export_value(value) else value
            )
    return export.loc[:, columns]


def build_raw_export_table(raw_df, clean_df=None, cleaner=None, detector=None):
    """Build the fixed-schema Raw event table after Clean decisions exist."""
    if raw_df is None:
        raw_df = pd.DataFrame()
    if clean_df is None:
        clean_df = pd.DataFrame()
    df = raw_df.copy()
    export = pd.DataFrame(index=df.index)

    raw_ids = (
        df["Raw_Reaction_ID"]
        if "Raw_Reaction_ID" in df.columns
        else df.get("Reaction_ID", pd.Series("", index=df.index))
    )
    export["Raw_Reaction_ID"] = raw_ids

    for column in ["Frame", "Reactant_Frame", "Product_Frame"]:
        export[column] = _coerce_export_int_series(df, column)

    for column in [
        "Reactants_Formula",
        "Products_Formula",
        "Reactants_SMILES",
        "Products_SMILES",
        "Isomerization_Class",
        "Valence_Warning",
        "Elementary_Mode",
    ]:
        export[column] = df[column] if column in df.columns else ""

    export["Reaction_Type"] = (
        df["Type"] if "Type" in df.columns
        else df.get("Reaction_Type", pd.Series("", index=df.index))
    )
    for column in [
        "Is_Elementary",
        "Topology_Validated",
        "Lineage_Complete",
    ]:
        export[column] = (
            df[column].map(_export_bool)
            if column in df.columns
            else ""
        )

    recovery = df.apply(
        lambda row: _recovery_summary(row, unresolved=False),
        axis=1,
    )
    export["Recovery_Type"] = recovery.map(lambda item: item[0])
    export["Recovery_Result"] = recovery.map(lambda item: item[1])

    # Persist recovery provenance so a later rebuild can distinguish a stable
    # corridor endpoint from a transient invalid intermediate.  
    for column in [
        "Observed_Stable_Exit_Source",
        "Observed_Stable_Exit_Product_Validated",
        "Observed_Corridor_Net_Validated",
        "Observed_Corridor_Net_Elementary",
    ]:
        export[column] = (
            df[column].map(_export_bool)
            if column in df.columns
            else ""
        )
    export["Observed_State_Source_Min_Dwell"] = _coerce_export_int_series(
        df,
        "Observed_State_Source_Min_Dwell",
    )
    export["Observed_Stable_Exit_Settled_After_Frames"] = (
        _coerce_export_int_series(
            df,
            "Observed_Stable_Exit_Settled_After_Frames",
        )
    )

    status_map = getattr(cleaner, "raw_clean_status", {}) if cleaner else {}
    clean_map = {}
    if clean_df is not None and not clean_df.empty:
        for row in clean_df.to_dict("records"):
            raw_id = str(
                row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or ""
            )
            if raw_id:
                clean_map[raw_id] = str(row.get("Reaction_ID", "") or "")

    def raw_status(row):
        raw_id = str(
            row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or ""
        )
        status = dict(status_map.get(raw_id, {}))
        clean_id = str(
            status.get("Clean_Reaction_ID", clean_map.get(raw_id, "")) or ""
        )
        reason = str(status.get("Exclusion_Reason", "") or "")
        if clean_id:
            reason = ""
        elif not reason:
            reason = "not_selected_by_cleaner"
        return reason, clean_id

    statuses = df.apply(raw_status, axis=1)
    export["Exclusion_Reason"] = statuses.map(lambda item: item[0])
    export["Clean_Reaction_ID"] = statuses.map(lambda item: item[1])

    # Cleaner is the authoritative final lineage decision.  
    clean_linked = (
        export["Clean_Reaction_ID"].fillna("").astype(str).str.strip().ne("")
    )
    export.loc[clean_linked, "Lineage_Complete"] = True

    export["Changed_Bonds"] = df.apply(
        lambda row: _format_clean_changed_bonds(row, detector),
        axis=1,
    )
    for column in ["Reactant_Atom_Groups", "Product_Atom_Groups"]:
        export[column] = (
            df[column].map(_format_export_atom_groups)
            if column in df.columns
            else "[]"
        )

    return _finalize_export_table(
        export,
        RAW_EVENT_COLUMNS,
        integer_columns={
            "Frame",
            "Reactant_Frame",
            "Product_Frame",
            "Observed_State_Source_Min_Dwell",
            "Observed_Stable_Exit_Settled_After_Frames",
        },
    )


def _pathway_source(row):
    source = str(row.get("Pathway_Source", "") or "")
    if source.startswith("clean"):
        return "clean", str(
            row.get("Clean_Reaction_ID", row.get("Reaction_ID", "")) or ""
        )
    if source.startswith("raw"):
        return "raw", str(
            row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or ""
        )
    if source.startswith("unresolved"):
        return "unresolved", str(
            row.get("Source_Unresolved_ID", row.get("Unresolved_ID", "")) or ""
        )
    if row.get("Clean_Reaction_ID"):
        return "clean", str(row.get("Clean_Reaction_ID") or "")
    if row.get("Source_Unresolved_ID"):
        return "unresolved", str(row.get("Source_Unresolved_ID") or "")
    return "raw", str(
        row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or ""
    )


def _pathway_role(row, source_table):
    role = str(row.get("Pathway_Role", "") or "")
    if source_table == "unresolved" and role == "composite_bridge":
        return "unresolved_bridge"
    if role in {
        "recovered_connectivity_elementary",
        "recovered_state_elementary",
        "restored_lineage_elementary",
    }:
        return "recovered_elementary"
    if role == "reactivated_raw_elementary":
        return "reactivated_raw"
    if role == "repaired_clean_corridor_net":
        return "recovered_corridor_net"
    return role or "clean_elementary"


def _pathway_inclusion_reason(value):
    mapping = {
        "validated_clean_elementary_event": "Accepted clean event",
        "validated_atom_mapped_lineage_recovery": (
            "Accepted atom-mapped lineage recovery"
        ),
        "validated_clean_lineage_restoration": (
            "Retained as a required Clean lineage source"
        ),
        "reachable_composite_bridge_adds_new_state": (
            "Required bridge that adds a reachable state"
        ),
        "elementary_raw_event_reactivated_after_pathway_bridge": (
            "Raw elementary event reactivated after pathway expansion"
        ),
        "promoted_to_clean_after_validated_corridor_recovery": (
            "Validated settled-endpoint corridor net promoted to Clean"
        ),
        "observed_post_recrossing_state_source": (
            "Observed settled state used as a lineage source"
        ),
    }
    text = str(value or "").strip()
    return mapping.get(text, text.replace("_", " ").strip().capitalize())


def _pathway_lineage_note(row):
    notes = []
    bridge_species = row.get("Bridge_New_Species", "")
    reactivated_species = row.get("Reactivated_Through_Species", "")
    for value in [
        row.get("Lineage_Note", ""),
        (
            f"New reachable species: {bridge_species}"
            if not _is_missing_export_value(bridge_species)
            and str(bridge_species).strip()
            else ""
        ),
        (
            "Reactivated through species: "
            f"{reactivated_species}"
            if not _is_missing_export_value(reactivated_species)
            and str(reactivated_species).strip()
            else ""
        ),
    ]:
        if _is_missing_export_value(value):
            continue
        text = str(value or "").strip()
        if text and text not in notes:
            notes.append(text)
    return "; ".join(notes)


def build_pathway_export_table(
    pathway_df,
    detector=None,
    include_internal_state_sources=False,
):
    """Build the Pathway export, optionally retaining provenance anchors."""
    if pathway_df is None:
        pathway_df = pd.DataFrame()
    df = pathway_df.copy()
    internal_source_count = 0
    if not include_internal_state_sources and not df.empty:
        roles = df.get(
            "Pathway_Role", pd.Series("", index=df.index)
        ).astype(str).str.strip()
        internal_source_mask = roles.eq("observed_stable_state_source")
        internal_source_count = int(internal_source_mask.sum())
        if internal_source_count:
            df = df.loc[~internal_source_mask].copy()
    export = pd.DataFrame(index=df.index)

    export["Pathway_ID"] = (
        df["Reaction_ID"] if "Reaction_ID" in df.columns
        else df.get("Pathway_ID", pd.Series("", index=df.index))
    )
    sources = df.apply(_pathway_source, axis=1)
    export["Source_Table"] = sources.map(lambda item: item[0])
    export["Source_Reaction_ID"] = sources.map(lambda item: item[1])
    export["Source_Raw_Reaction_ID"] = df.apply(
        lambda row: str(_first_present(
            row,
            ["Raw_Reaction_ID", "Source_Raw_Reaction_ID"],
            "",
        )),
        axis=1,
    )
    export["Frame"] = _coerce_export_int_series(df, "Frame")

    for column in [
        "Reactants_Formula",
        "Products_Formula",
        "Reactants_SMILES",
        "Products_SMILES",
    ]:
        export[column] = df[column] if column in df.columns else ""

    export["Reaction_Type"] = (
        df["Type"] if "Type" in df.columns
        else df.get("Reaction_Type", pd.Series("", index=df.index))
    )
    export["Pathway_Role"] = [
        _pathway_role(row, source[0])
        for row, source in zip(df.to_dict("records"), list(sources))
    ]
    export["Elementary_Mode"] = (
        df["Elementary_Mode"] if "Elementary_Mode" in df.columns else ""
    )
    export["Inclusion_Reason"] = (
        df["Pathway_Selection_Reason"].map(_pathway_inclusion_reason)
        if "Pathway_Selection_Reason" in df.columns else ""
    )
    export["Lineage_Note"] = df.apply(_pathway_lineage_note, axis=1)
    for column in [
        "Upper_Source",
        "Upper_Source_Reaction_ID",
        "Upper_Source_Status",
    ]:
        export[column] = (
            df[column] if column in df.columns else ""
        )
    for column in ["Reactant_Atom_Groups", "Product_Atom_Groups"]:
        export[column] = (
            df[column].map(_format_export_atom_groups)
            if column in df.columns
            else "[]"
        )

    export = _finalize_export_table(
        export, PATHWAY_EVENT_COLUMNS, integer_columns={"Frame"}
    )
    export.attrs["internal_state_sources_omitted"] = internal_source_count
    return export


def _first_present(row, fields, default=""):
    for field in fields:
        value = row.get(field, None)
        if not _is_missing_export_value(value) and str(value).strip() != "":
            return value
    return default


def _unresolved_note(row):
    notes = []
    for field in ["Note", "Source_Explanation", "Endpoint_Rejection_Reason"]:
        value = row.get(field, "")
        if _is_missing_export_value(value):
            continue
        text = str(value or "").strip()
        if text and text not in notes:
            notes.append(text)
    return "; ".join(notes)


def append_lineage_dropped_unresolved(
    unresolved_df,
    lineage_dropped_records,
):
    """Add Clean lineage drops to the unresolved audit table."""
    base = (
        unresolved_df.copy()
        if unresolved_df is not None
        else pd.DataFrame()
    )
    if (
        "Source_Raw_Reaction_ID" in base.columns
        and "Reason" in base.columns
    ):
        existing_values = base.loc[
            base["Reason"].astype(str).eq("lineage_incomplete"),
            "Source_Raw_Reaction_ID",
        ]
    else:
        existing_values = pd.Series(dtype=object)
    existing_sources = {
        str(value).strip()
        for value in existing_values
        if not _is_missing_export_value(value) and str(value).strip()
    }
    rows = []
    for record in list(lineage_dropped_records or []):
        row = dict(record)
        source_raw_id = str(
            row.get(
                "Raw_Reaction_ID",
                row.get("Reaction_ID", ""),
            )
            or ""
        ).strip()
        if source_raw_id and source_raw_id in existing_sources:
            continue
        row.update({
            "Reason": "lineage_incomplete",
            "Source_Raw_Reaction_ID": source_raw_id,
            "Clean_Eligible": False,
            "Note": (
                "Clean candidate was excluded because one or more "
                "reactant lineages had no validated source."
            ),
        })
        rows.append(row)
        if source_raw_id:
            existing_sources.add(source_raw_id)

    if not rows:
        return base

    combined = pd.concat(
        [base, pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    )
    combined["Unresolved_ID"] = [
        f"UNRES-{index + 1:04d}"
        for index in range(len(combined))
    ]
    return combined


def _pathway_recovery_index(pathway_df):
    """Index final Pathway rows by unresolved and Raw provenance IDs."""
    by_unresolved = {}
    by_raw = {}
    if pathway_df is None or pathway_df.empty:
        return by_unresolved, by_raw

    for row in pathway_df.to_dict("records"):
        pathway_id = str(_first_present(
            row,
            ["Pathway_ID", "Reaction_ID"],
            "",
        )).strip()
        if not pathway_id:
            continue

        source_table, source_id = _pathway_source(row)
        if source_table == "unresolved" and source_id:
            by_unresolved.setdefault(source_id, set()).add(pathway_id)

        raw_id = str(_first_present(
            row,
            ["Source_Raw_Reaction_ID", "Raw_Reaction_ID"],
            "",
        )).strip()
        if raw_id:
            by_raw.setdefault(raw_id, set()).add(pathway_id)

    return by_unresolved, by_raw


def build_unresolved_export_table(
    unresolved_df,
    detector=None,
    pathway_df=None,
):
    """Build the fixed-schema unresolved-event review table."""
    if unresolved_df is None:
        unresolved_df = pd.DataFrame()
    df = unresolved_df.copy()
    export = pd.DataFrame(index=df.index)

    for column in [
        "Unresolved_ID",
        "Source_Raw_Reaction_ID",
        "Reactants_Formula",
        "Products_Formula",
        "Reactants_SMILES",
        "Products_SMILES",
        "Reason",
    ]:
        export[column] = df[column] if column in df.columns else ""

    for column in ["Frame", "Reactant_Frame", "Product_Frame"]:
        export[column] = _coerce_export_int_series(df, column)

    export["Reaction_Type"] = (
        df["Type"] if "Type" in df.columns
        else df.get("Reaction_Type", pd.Series("", index=df.index))
    )
    recovery = df.apply(
        lambda row: _recovery_summary(row, unresolved=True),
        axis=1,
    )
    export["Recovery_Type"] = recovery.map(lambda item: item[0])
    export["Recovery_Result"] = recovery.map(lambda item: item[1])

    # Recovery_Result describes an attempted local recovery.  It does not say
    # whether the corresponding event was ultimately admitted to Pathway.
    pathway_by_unresolved, pathway_by_raw = _pathway_recovery_index(pathway_df)

    def final_pathway_ids(row):
        unresolved_id = str(row.get("Unresolved_ID", "") or "").strip()
        raw_id = str(row.get("Source_Raw_Reaction_ID", "") or "").strip()
        pathway_ids = set(pathway_by_unresolved.get(unresolved_id, set()))
        pathway_ids.update(pathway_by_raw.get(raw_id, set()))
        return sorted(pathway_ids)

    final_pathways = df.apply(final_pathway_ids, axis=1)
    export["Pathway_Status"] = final_pathways.map(
        lambda pathway_ids: (
            "recovered_in_pathway" if pathway_ids else "not_recovered"
        )
    )
    export["Pathway_IDs"] = final_pathways.map(
        lambda pathway_ids: json.dumps(
            pathway_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    export["Changed_Bonds"] = df.apply(
        lambda row: _format_clean_changed_bonds(row, detector),
        axis=1,
    )
    for column in ["Reactant_Atom_Groups", "Product_Atom_Groups"]:
        export[column] = (
            df[column].map(_format_export_atom_groups)
            if column in df.columns
            else "[]"
        )
    export["Elementary_Mode"] = (
        df["Elementary_Mode"] if "Elementary_Mode" in df.columns else ""
    )

    export["Raw_Changed_Bond_Count"] = df.apply(
        lambda row: _first_present(
            row,
            ["Endpoint_Raw_Changed_Bond_Count", "Changed_Bond_Count"],
        ),
        axis=1,
    )
    export["Validated_Changed_Bond_Count"] = df.apply(
        lambda row: _first_present(
            row, ["Endpoint_Validated_Changed_Bond_Count"]
        ),
        axis=1,
    )
    export["Changed_Bond_Components"] = df.apply(
        lambda row: _first_present(
            row,
            [
                "Endpoint_Validated_Changed_Bond_Components",
                "Endpoint_Raw_Changed_Bond_Components",
                "Changed_Bond_Components",
            ],
        ),
        axis=1,
    )
    export["Note"] = df.apply(_unresolved_note, axis=1)

    for column in [
        "Raw_Changed_Bond_Count",
        "Validated_Changed_Bond_Count",
        "Changed_Bond_Components",
    ]:
        export[column] = pd.to_numeric(
            export[column], errors="coerce"
        ).astype("Int64")

    return _finalize_export_table(
        export,
        UNRESOLVED_EVENT_COLUMNS,
        integer_columns={
            "Frame",
            "Reactant_Frame",
            "Product_Frame",
            "Raw_Changed_Bond_Count",
            "Validated_Changed_Bond_Count",
            "Changed_Bond_Components",
        },
    )


def validate_event_table_invariants(raw_df, clean_df, pathway_df):
    """Fail fast if the published three-table contract is violated.
    The contract is provenance-based rather than count-only:
    every Clean and Pathway row must point to an existing Raw ledger event,
    and every Clean Raw event must also occur in Pathway.
    """
    raw_df = raw_df if raw_df is not None else pd.DataFrame()
    clean_df = clean_df if clean_df is not None else pd.DataFrame()
    pathway_df = pathway_df if pathway_df is not None else pd.DataFrame()

    def ids(df, primary, fallback=None):
        if df.empty:
            return set()
        if primary in df.columns:
            values = df[primary]
        elif fallback and fallback in df.columns:
            values = df[fallback]
        else:
            return set()
        return {
            str(value).strip()
            for value in values
            if not _is_missing_export_value(value) and str(value).strip()
        }

    raw_ids = ids(raw_df, "Raw_Reaction_ID", "Reaction_ID")
    clean_raw_ids = ids(clean_df, "Raw_Reaction_ID", "Reaction_ID")
    pathway_raw_ids = ids(
        pathway_df, "Raw_Reaction_ID", "Source_Raw_Reaction_ID"
    )

    missing_clean_raw = clean_raw_ids - raw_ids
    missing_pathway_raw = pathway_raw_ids - raw_ids
    missing_clean_pathway = clean_raw_ids - pathway_raw_ids
    errors = []
    if len(raw_df) < len(pathway_df) or len(pathway_df) < len(clean_df):
        errors.append(
            "row-count order violated "
            f"(Raw={len(raw_df)}, Pathway={len(pathway_df)}, "
            f"Clean={len(clean_df)})"
        )
    if missing_clean_raw:
        errors.append(
            "Clean references missing Raw IDs: "
            + ", ".join(sorted(missing_clean_raw))
        )
    if missing_pathway_raw:
        errors.append(
            "Pathway references missing Raw IDs: "
            + ", ".join(sorted(missing_pathway_raw))
        )
    if missing_clean_pathway:
        errors.append(
            "Clean events absent from Pathway: "
            + ", ".join(sorted(missing_clean_pathway))
        )
    if (
        not clean_df.empty
        and "Lineage_Complete" in clean_df.columns
        and not clean_df["Lineage_Complete"].map(_export_truthy).all()
    ):
        errors.append("Clean contains event(s) with incomplete atom lineage")
    if (
        not pathway_df.empty
        and "Lineage_Complete" in pathway_df.columns
        and not pathway_df["Lineage_Complete"].map(_export_truthy).all()
    ):
        errors.append("Pathway contains event(s) with incomplete atom lineage")

    def has_raw_provenance(row):
        return bool(str(_first_present(
            row,
            ["Raw_Reaction_ID", "Source_Raw_Reaction_ID"],
            "",
        )).strip())

    if not clean_df.empty and not all(
        has_raw_provenance(row)
        for row in clean_df.to_dict("records")
    ):
        errors.append("Clean contains event(s) without Raw provenance")
    if not pathway_df.empty and not all(
        has_raw_provenance(row)
        for row in pathway_df.to_dict("records")
    ):
        errors.append("Pathway contains event(s) without Raw provenance")

    def groups_match_side(row, side):
        formula_parts = [
            part.strip()
            for part in str(row.get(f"{side}s_Formula", "") or "").split(" + ")
            if part.strip()
        ]
        field = f"{side}_Atom_Groups"
        try:
            groups = json.loads(_format_export_atom_groups(row.get(field, [])))
        except Exception:
            return False
        return (
            len(groups) == len(formula_parts)
            and all(bool(group) for group in groups)
        )

    malformed_clean_groups = [
        str(row.get("Raw_Reaction_ID", row.get("Reaction_ID", "")) or "")
        for row in clean_df.to_dict("records")
        if not groups_match_side(row, "Reactant")
        or not groups_match_side(row, "Product")
    ]
    if malformed_clean_groups:
        errors.append(
            "Clean has missing/misaligned per-species atom groups: "
            + ", ".join(malformed_clean_groups)
        )

    if errors:
        raise RuntimeError(
            "Reaction table invariant check failed: " + "; ".join(errors)
        )

    return {
        "raw": len(raw_df),
        "pathway": len(pathway_df),
        "clean": len(clean_df),
    }


def print_banner():
    width = 78
    inner = width - 6

    def line(text=""):
        text = str(text)[:inner]
        return f"│  {text:<{inner}}  │"

    print()
    print("╭" + "─" * (width - 2) + "╮")
    print(line("RxnScape"))
    print(line("From Reactive MD to Interpretable Reaction Pathways and Networks"))
    print()
    print(line("Topology  →  Events  →  Pathways  →  Lineage"))
    print(line("Atom Mapping  ·  Molecular Lineage  ·  Traceable Evidence"))
    print()
    print(line(f"Version {RXNSCAPE_VERSION}"))
    print(line("Dexin Sun · Tingyu Lei"))
    print("╰" + "─" * (width - 2) + "╯")
    print()

def load_yaml_config():
    """Safely load rxn.yaml from the current working directory."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"⚠️ Warning: failed to read {CONFIG_FILE}: {exc}")
    return {}


def _first_non_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _parse_indices_string(idx_str):
    """Parse strings like '0-500,623' into a list of integers."""
    result = []
    if not idx_str:
        return result

    if isinstance(idx_str, list):
        # Support [0, 1, "5-9"]
        parts = []
        for x in idx_str:
            parts.extend(str(x).split(','))
    else:
        parts = str(idx_str).split(',')

    for part in parts:
        part = str(part).strip()
        if not part:
            continue

        if '-' in part and not part.startswith('-'):
            bounds = part.split('-')
            if len(bounds) == 2:
                try:
                    s, e = int(bounds[0].strip()), int(bounds[1].strip())
                    if e >= s:
                        result.extend(range(s, e + 1))
                    else:
                        result.extend(range(s, e - 1, -1))
                except ValueError:
                    print(f"⚠️ Warning: Could not parse range '{part}'. Ignoring.")
            else:
                print(f"⚠️ Warning: Invalid range format '{part}'. Ignoring.")
        else:
            try:
                result.append(int(part))
            except ValueError:
                print(f"⚠️ Warning: Could not parse index '{part}'. Ignoring.")
    return result


def _as_bool(value):
    """Convert YAML/CLI-ish booleans robustly."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _parse_species_list(value):
    """Parse Sankey initial species from YAML/CLI."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        species = [str(x).strip() for x in value if str(x).strip()]
    else:
        species = [x.strip() for x in str(value).split(',') if x.strip()]

    return species or None


def _parse_sankey_elements(value):
    """Parse selected atom-mapped element-flow outputs."""
    if value is None:
        return None
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
        for element in Config.SANKEY_ALLOWED_ELEMENTS
    }
    selected = []
    seen = set()
    for raw_element in raw_elements:
        if not raw_element:
            continue
        element = canonical_by_lower.get(raw_element.lower(), raw_element)
        if element not in seen:
            selected.append(element)
            seen.add(element)
    return selected


def _require_numba_for_topology():
    """Fail early if Numba is unavailable.
    HMM bond-state filtering is part of the scientific algorithm.
    """
    try:
        import numba  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RxnScape requires Numba for HMM bond-state filtering. "
            "Please install the required dependencies before running topology analysis."
        ) from exc


def _require_rdkit_for_topology():
    """Fail early if RDKit is unavailable.
    RDKit is required for reproducible SMILES-based species identification.
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RxnScape requires RDKit for reproducible SMILES-based species "
            "identification. Please install RDKit, preferably via conda-forge."
        ) from exc


def _file_sha256(path, block_size=1024 * 1024):
    """Return a SHA256 digest for an input file, or None when unavailable."""
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _get_module_version(module_name):
    """Return an installed module version for reproducibility metadata."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def write_run_metadata(cfg):
    """Write a reproducibility record for the current RxnScape run."""
    from datetime import datetime

    rxnscape_version = RXNSCAPE_VERSION

    input_file = cfg.get('input_file')
    abs_input = os.path.abspath(input_file) if input_file else None

    metadata = {
        "software": "RxnScape",
        "version": rxnscape_version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "input_file": abs_input,
        "input_sha256": _file_sha256(input_file),
        "effective_config": cfg,
        "topology_config": _normalized_topology_config(cfg),
        "topology_config_hash": _topology_config_hash(cfg),
        "cache_file": os.path.abspath(CACHE_FILE),
        "output_files": Config.OUTPUT_FILES,
        "dependencies": {
            "numpy": _get_module_version("numpy"),
            "pandas": _get_module_version("pandas"),
            "scipy": _get_module_version("scipy"),
            "ase": _get_module_version("ase"),
            "networkx": _get_module_version("networkx"),
            "numba": _get_module_version("numba"),
            "matplotlib": _get_module_version("matplotlib"),
            "plotly": _get_module_version("plotly"),
            "pyvis": _get_module_version("pyvis"),
            "yaml": _get_module_version("yaml"),
            "rdkit": _get_module_version("rdkit"),
        },
        "species_identity_model": {
            "name": "rdkit_sanitized_connectivity_smiles",
            "bond_order_model": "all_single",
            "bond_order_inference": False,
            "stereochemistry_inference": False,
            "note": (
                "SMILES is a connectivity-derived label. It must not be "
                "interpreted as a validated bond-order, charge, electronic-"
                "state, aromaticity, or stereochemistry assignment."
            ),
        },
        "reaction_detection_policy": {
            "global_plateau_stability": int(cfg.get("stability", 1)),
            "ordinary_local_dwell": int(
                cfg.get("segmentation_local_stability", 1)
            ),
            "single_frame_recovery_scan": int(
                cfg.get("recovery_scan_local_stability", 1)
            ),
            "global_scan_window": int(
                max(40, 8 * int(cfg.get("stability", 1)))
            ),
            "cleaner_recrossing_window": int(
                max(40, 8 * int(cfg.get("stability", 1)))
            ),
            "lineage_search_margin": int(max(
                30,
                6 * int(cfg.get("stability", 1)),
                10 * int(cfg.get("segmentation_local_stability", 1)),
            )),
            "lineage_short_gap_tolerance": int(
                max(0, int(cfg.get("stability", 1)) - 1)
            ),
            "persistent_topology_recovery_scope": (
                "all_dwell_validated_boundaries"
            ),
            "note": (
                "Global stability localizes ordinary corridors. Persistent "
                "atom-local topology recovery independently considers every "
                "boundary satisfying ordinary_local_dwell; all candidates "
                "still pass the standard conservative chemistry filters."
            ),
        },
    }

    out_path = os.path.join(Config.DIR_DATA, "run_metadata.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"🧾 Run metadata saved: {out_path}")
    except Exception as exc:
        print(f"⚠️ Warning: failed to write run metadata: {exc}")


def update_run_timings(stage_timings):
    """Append lightweight wall-clock timings to the reproducibility record."""
    out_path = os.path.join(Config.DIR_DATA, "run_metadata.json")
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["timings_seconds"] = {
            str(name): round(float(seconds), 3)
            for name, seconds in stage_timings.items()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
    except Exception as exc:
        print(f"⚠️ Warning: failed to update run timings: {exc}")


def _normalized_topology_config(cfg):
    """Parameters that affect cached topology/frame_molecule results."""
    input_file = cfg.get('input_file')
    abs_input = os.path.abspath(input_file) if input_file else None
    return {
        'input_file': abs_input,
        'input_sha256': _file_sha256(input_file),
        'remove': cfg.get('remove'),
        'cell': cfg.get('cell'),
        'pbc': cfg.get('pbc'),
        'bond_scale': cfg.get('bond_scale'),
        'hmm_prob': cfg.get('hmm_prob'),
        'max_size': cfg.get('max_size'),
        'substrate': {
            'indices': sorted(int(i) for i in cfg.get('substrate', {}).get('indices', [])),
            'elements': sorted(str(e) for e in cfg.get('substrate', {}).get('elements', [])),
        },
    }


def _topology_config_hash(cfg):
    payload = json.dumps(_normalized_topology_config(cfg), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _build_cache_payload(frame_molecules, cfg):
    return {
        'schema_version': 3,
        'topology_config': _normalized_topology_config(cfg),
        'topology_config_hash': _topology_config_hash(cfg),
        'runtime': {
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'frame_molecules': frame_molecules,
    }


def _load_valid_topology_cache(cfg):
    """Load topology cache only when it matches the current topology parameters."""
    with open(CACHE_FILE, 'rb') as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or 'frame_molecules' not in payload:
        raise ValueError("Topology cache has an unsupported format.")

    schema = payload.get('schema_version')
    saved_hash = payload.get('topology_config_hash')
    current_hash = _topology_config_hash(cfg)

    if schema != 3 or not saved_hash:
        raise ValueError(
            "Topology cache was created by an older version and cannot be validated. "
            "Please rerun topology analysis."
        )

    if saved_hash != current_hash:
        raise ValueError(
            "Topology cache does not match the current input/topology parameters. "
            "Please rerun topology analysis."
        )

    return payload['frame_molecules']


def build_parser():
    parser = argparse.ArgumentParser(description="RxnScape Analysis Pipeline")
    parser.add_argument('input_file', nargs='?', default=None, help="Path to trajectory file")
    parser.add_argument('--cell', default=None, help="Cell override/source: e.g. '30 30 30', 'ams.rkf', or 'auto'")
    parser.add_argument('--pbc', default=None, help="PBC flags: e.g. 'T T T', 'T T F', or false")

    # Defaults are None here so YAML can sit between CLI and built-in defaults.
    parser.add_argument('--remove', default=None, help="Elements to remove, e.g. Ar,Ne")
    parser.add_argument('--bond_scale', type=float, default=None, help="Bond scale factor")
    parser.add_argument('--adsorption_scale', type=float, default=None, help="Adsorption cutoff scale factor")
    parser.add_argument('--hmm_prob', type=float, default=None, help="HMM persistence probability")
    parser.add_argument('--workers', type=int, default=None, help="CPU workers")
    parser.add_argument('--max_size', type=int, default=None, help="Max molecule size for RDKit/SMILES")
    parser.add_argument('--max_structure_size', dest='max_size', type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--stability', type=int, default=None,
                        help="Minimum number of frames for a stable reaction state")
    parser.add_argument('--local_stability', type=int, default=None,
                        help="Minimum frames for ordinary local segmentation "
                             "(default: max(2, stability//2)). A value of 1 is "
                             "handled by strict endpoint recovery unless that "
                             "feature is disabled.")
    parser.add_argument(
        '--single_frame_recovery',
        dest='single_frame_recovery',
        action='store_true',
        default=None,
        help=(
            "Recover valid one-frame endpoints under strict atom-mapped, "
            "mass-balanced, stable-product checks."
        ),
    )
    parser.add_argument(
        '--no_single_frame_recovery',
        '--no-single-frame-recovery',
        dest='single_frame_recovery',
        action='store_false',
        help=(
            "Disable strict one-frame endpoint recovery and restore legacy "
            "local_stability=1 segmentation behaviour."
        ),
    )
    parser.add_argument(
        '--allow_ambiguous_topology',
        dest='allow_ambiguous_topology',
        action='store_true',
        default=None,
        help=(
            "Allow valence-ambiguous connectivity observations into Clean. "
            "Default is strict audit-only handling."
        ),
    )
    parser.add_argument(
        '--strict_topology',
        dest='allow_ambiguous_topology',
        action='store_false',
        help="Exclude valence-ambiguous topology from Clean/Pathway.",
    )
                             
    # Substrate
    parser.add_argument('--sub_indices', default=None, help="Substrate indices")
    parser.add_argument('--sub_elements', default=None, help="Substrate elements")

    # Feature flags; support explicit on/off without breaking old flags.
    parser.add_argument('--rings', dest='rings', action='store_true', default=None,
                        help="Enable rings")
    parser.add_argument('--no_rings', '--no-rings', dest='rings', action='store_false',
                        help="Disable rings")

    parser.add_argument('--auto_neb', dest='auto_neb', action='store_true', default=None,
                        help="Enable Auto-NEB")
    parser.add_argument('--no_neb', dest='auto_neb', action='store_false',
                        help="Disable Auto-NEB")

    parser.add_argument('--sankey', dest='sankey', action='store_true', default=None,
                        help="Enable Sankey")
    parser.add_argument('--no_sankey', dest='sankey', action='store_false',
                        help="Disable Sankey")
    parser.add_argument('--sankey_initial_species', default=None,
                        help="Comma-separated initial species for Sankey roots, e.g. 'CH4,O2' or 'C80H122Cl40'")
    parser.add_argument(
        '--sankey_elements', '--sankey-elements',
        default=None,
        help=(
            "Atom-mapped element-flow outputs using valid element symbols, "
            "e.g. 'C', 'C,Cl', or 'Si,O'. Use 'none' to generate only the "
            "reaction transition map."
        ),
    )

    parser.add_argument('--stage', default=None, choices=['topo', 'reaction', 'viz', 'all'],
                        help="Execution stage")

    return parser


def resolve_config(args, yaml_config):
    """Resolve effective config with priority: CLI > rxn.yaml > built-in defaults."""
    params = yaml_config.get('parameters', {}) or {}
    features = yaml_config.get('features', {}) or {}
    visualization = yaml_config.get('Visualization', yaml_config.get('visualization', {})) or {}
    reaction = params.get('reaction', {}) or {}

    cfg = {}
    cfg['input_file'] = _first_non_none(args.input_file, yaml_config.get('input_file'), DEFAULTS['input_file'])
    cfg['remove'] = _first_non_none(args.remove, params.get('remove'), DEFAULTS['remove'])
    cfg['cell'] = _first_non_none(args.cell, params.get('cell'), DEFAULTS['cell'])
    cfg['pbc'] = _first_non_none(args.pbc, params.get('pbc'), DEFAULTS['pbc'])

    cfg['bond_scale'] = _first_non_none(args.bond_scale, params.get('bond_scale'), DEFAULTS['bond_scale'])
    cfg['adsorption_scale'] = _first_non_none(args.adsorption_scale, params.get('adsorption_scale'), DEFAULTS['adsorption_scale'])
    cfg['hmm_prob'] = _first_non_none(args.hmm_prob, params.get('hmm_prob'), DEFAULTS['hmm_prob'])
    cfg['workers'] = _first_non_none(args.workers, params.get('workers'), DEFAULTS['workers'])
    cfg['max_size'] = _first_non_none(args.max_size, params.get('max_size'), params.get('max_structure_size'), DEFAULTS['max_size'])

    # User-facing YAML uses one reaction-stability parameter.
    cfg['stability'] = _first_non_none(
        args.stability,
        params.get('stability'),
        DEFAULTS['stability']
    )
    
    _local_stab = _first_non_none(
        args.local_stability,
        params.get('local_stability'),
        DEFAULTS['local_stability']
    )
    cfg['local_stability'] = _local_stab
    cfg['single_frame_recovery'] = _as_bool(_first_non_none(
        args.single_frame_recovery,
        params.get('single_frame_recovery'),
        DEFAULTS['single_frame_recovery'],
    ))
    cfg['allow_ambiguous_topology'] = _as_bool(_first_non_none(
        args.allow_ambiguous_topology,
        params.get('allow_ambiguous_topology'),
        DEFAULTS['allow_ambiguous_topology'],
    ))

    # Reaction detection policy is fixed for reproducibility:
    # hybrid local scan + mass-balanced elementary candidates only.

    cfg['rings'] = _as_bool(_first_non_none(args.rings, features.get('rings'), DEFAULTS['rings']))
    cfg['auto_neb'] = _as_bool(_first_non_none(args.auto_neb, features.get('auto_neb'), DEFAULTS['auto_neb']))
    cfg['sankey'] = _as_bool(_first_non_none(args.sankey, features.get('sankey'), DEFAULTS['sankey']))
    cfg['sankey_initial_species'] = _parse_species_list(_first_non_none(
        args.sankey_initial_species,
        visualization.get('sankey_initial_species'),
        features.get('sankey_initial_species'),
        DEFAULTS['sankey_initial_species']
    ))
    cfg['sankey_elements'] = _parse_sankey_elements(_first_non_none(
        getattr(args, 'sankey_elements', None),
        visualization.get('sankey_elements'),
        features.get('sankey_elements'),
        DEFAULTS['sankey_elements'],
    ))

    cfg['stage'] = _first_non_none(args.stage, DEFAULTS['stage'])

    # Substrate: CLI > YAML > empty
    yaml_sub = params.get('substrate', {}) or {}
    sub_config = {}

    if args.sub_indices is not None:
        sub_config['indices'] = _parse_indices_string(args.sub_indices)
    elif yaml_sub.get('indices'):
        sub_config['indices'] = _parse_indices_string(yaml_sub.get('indices'))
    else:
        sub_config['indices'] = []

    if args.sub_elements is not None:
        sub_config['elements'] = [x.strip() for x in args.sub_elements.split(',') if x.strip()]
    elif yaml_sub.get('elements'):
        if isinstance(yaml_sub.get('elements'), str):
            sub_config['elements'] = [x.strip() for x in yaml_sub.get('elements').split(',') if x.strip()]
        else:
            sub_config['elements'] = list(yaml_sub.get('elements') or [])
    else:
        sub_config['elements'] = []

    cfg['substrate'] = sub_config
    return cfg


def validate_config(cfg):
    """Reject values that would otherwise fail late or change the algorithm silently."""
    positive = {
        "bond_scale": cfg.get("bond_scale"),
        "adsorption_scale": cfg.get("adsorption_scale"),
        "max_size": cfg.get("max_size"),
        "stability": cfg.get("stability"),
    }
    for name, value in positive.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, got {value!r}") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")

    cfg["bond_scale"] = float(cfg["bond_scale"])
    cfg["adsorption_scale"] = float(cfg["adsorption_scale"])
    for name in ("max_size", "stability"):
        numeric = float(cfg[name])
        if not numeric.is_integer():
            raise ValueError(f"{name} must be an integer, got {cfg[name]!r}")
        cfg[name] = int(numeric)

    local_stability = cfg.get("local_stability")
    if local_stability is not None:
        try:
            local_numeric = float(local_stability)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"local_stability must be an integer or null, got {local_stability!r}"
            ) from exc
        if local_numeric <= 0 or not local_numeric.is_integer():
            raise ValueError(
                f"local_stability must be a positive integer or null, got {local_stability!r}"
            )
        cfg["local_stability"] = int(local_numeric)
    else:
        cfg["local_stability"] = None

    workers = cfg.get("workers")
    if workers is not None:
        try:
            worker_numeric = float(workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"workers must be an integer or null, got {workers!r}"
            ) from exc
        if worker_numeric <= 0 or not worker_numeric.is_integer():
            raise ValueError(
                f"workers must be a positive integer or null, got {workers!r}"
            )
        cfg["workers"] = int(worker_numeric)
    else:
        cfg["workers"] = None

    hmm_prob = float(cfg.get("hmm_prob"))
    if not math.isfinite(hmm_prob) or not 0.0 <= hmm_prob <= 1.0:
        raise ValueError(f"hmm_prob must be between 0 and 1, got {hmm_prob!r}")
    cfg["hmm_prob"] = hmm_prob

    negative_indices = [
        index
        for index in cfg.get("substrate", {}).get("indices", [])
        if int(index) < 0
    ]
    if negative_indices:
        raise ValueError(
            "substrate indices must be zero-based non-negative integers; "
            f"got {negative_indices}"
        )
    cfg["substrate"]["indices"] = [
        int(index) for index in cfg.get("substrate", {}).get("indices", [])
    ]

    sankey_elements = cfg.get("sankey_elements")
    if sankey_elements is None:
        sankey_elements = list(Config.SANKEY_DEFAULT_ELEMENTS)
    if not isinstance(sankey_elements, list):
        raise ValueError(
            "sankey_elements must be a list or comma-separated string; "
            f"got {sankey_elements!r}"
        )
    invalid_elements = [
        element
        for element in sankey_elements
        if element not in Config.SANKEY_ALLOWED_ELEMENTS
    ]
    if invalid_elements:
        raise ValueError(
            "sankey_elements contains unsupported value(s): "
            f"{invalid_elements}. Every entry must be a valid IUPAC element "
            "symbol from H through Og (for example C, N, Si, Cl, or Fe)."
        )
    cfg["sankey_elements"] = list(sankey_elements)

    automatic_local = max(2, cfg["stability"] // 2)
    requested_local = (
        automatic_local
        if cfg["local_stability"] is None
        else cfg["local_stability"]
    )
    normalized = bool(cfg.get("single_frame_recovery", True)) and requested_local == 1
    cfg["segmentation_local_stability"] = (
        automatic_local if normalized else requested_local
    )
    cfg["recovery_scan_local_stability"] = (
        1
        if bool(cfg.get("single_frame_recovery", True))
        else cfg["segmentation_local_stability"]
    )
    cfg["local_stability_request_normalized"] = normalized


def load_effective_trajectory(cfg):
    """Load coordinates with exactly the same removal policy in every stage."""
    remove = cfg.get("remove")
    remove_species = (
        [item.strip() for item in remove.split(",") if item.strip()]
        if isinstance(remove, str)
        else remove
    )
    traj = make_trajectory_loader(cfg).load(remove_species)
    invalid_indices = [
        index
        for index in cfg.get("substrate", {}).get("indices", [])
        if index >= len(traj[0])
    ]
    if invalid_indices:
        raise ValueError(
            "substrate indices exceed the post-removal atom range "
            f"0..{len(traj[0]) - 1}: {invalid_indices}"
        )
    return traj


def _filter_kwargs_for_callable(func, kwargs, label):
    """Only pass kwargs supported by the installed core implementation."""
    sig = inspect.signature(func)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return dict(kwargs), []

    allowed = {}
    ignored = []
    for k, v in kwargs.items():
        if k in sig.parameters:
            allowed[k] = v
        else:
            ignored.append(k)
    if ignored:
        print(f"⚠️ {label} does not accept these new parameters yet: {', '.join(ignored)}")
        print("   They were ignored. Upgrade core/reaction.py for them to affect reaction detection.")
    return allowed, ignored


def _call_with_supported_kwargs(func, kwargs, label):
    allowed, _ = _filter_kwargs_for_callable(func, kwargs, label)
    return func(**allowed)


def make_trajectory_loader(cfg):
    return TrajectoryLoader(cfg['input_file'], cell=cfg.get('cell'), pbc=cfg.get('pbc'))


def print_effective_config(cfg):
    print("⚙️ Effective settings (priority: CLI > rxn.yaml > defaults):")
    requested_local_stability = cfg.get('local_stability')
    segmentation_local_stability = cfg.get(
        'segmentation_local_stability'
    )
    if requested_local_stability is None:
        local_stability_display = (
            f"auto -> {segmentation_local_stability} frames"
        )
    elif cfg.get('local_stability_request_normalized', False):
        local_stability_display = (
            f"{requested_local_stability} requested -> "
            f"{segmentation_local_stability}-frame ordinary dwell"
        )
    else:
        local_stability_display = (
            f"{segmentation_local_stability} frames"
        )

    sections = [
        ("Input", [
            ("input_file", cfg['input_file']),
            ("stage", cfg['stage']),
            ("cell", cfg['cell']),
            ("pbc", cfg['pbc']),
        ]),
        ("Topology", [
            ("bond_scale", cfg['bond_scale']),
            ("adsorption_scale", cfg['adsorption_scale']),
            ("hmm_prob", cfg['hmm_prob']),
            ("workers", cfg['workers']),
            ("max_size", cfg['max_size']),
        ]),
        ("Reaction analysis", [
            ("global_stability", f"{cfg['stability']} frames"),
            ("local_stability", local_stability_display),
            (
                "single_frame_recovery",
                (
                    "on (targeted 1-frame scan)"
                    if cfg['single_frame_recovery']
                    else "off"
                ),
            ),
            (
                "ambiguous_topology",
                (
                    "allow"
                    if cfg['allow_ambiguous_topology']
                    else "reject"
                ),
            ),
        ]),
        ("Substrate", [
            ("indices", cfg['substrate'].get('indices', [])),
            ("elements", cfg['substrate'].get('elements', [])),
        ]),
        ("Features", [
            ("auto_neb", cfg['auto_neb']),
            ("sankey", cfg['sankey']),
            ("sankey_initial_species", cfg.get('sankey_initial_species')),
            ("sankey_elements", cfg.get('sankey_elements')),
            ("rings", cfg['rings']),
        ]),
    ]
    for title, rows in sections:
        print(f"   [{title}]")
        for k, v in rows:
            print(f"     {k}: {v}")


def _initial_species_from_report(report_df):
    """Return stable initial-species identifiers from species_report.csv.

    SMILES is preferred because a formula may represent multiple isomers.
    Formula is used only when SMILES is unavailable.
    """
    if report_df is None or report_df.empty:
        return []
    if "Species_Role" not in report_df.columns:
        return []

    initial_rows = report_df[
        report_df["Species_Role"].astype(str).str.strip().str.lower()
        == "initial_reactant"
    ]
    resolved = []
    seen = set()
    for _, row in initial_rows.iterrows():
        smiles = str(row.get("SMILES", "") or "").strip()
        formula = str(row.get("Formula", "") or "").strip()
        identifier = smiles or formula
        if identifier and identifier not in seen:
            seen.add(identifier)
            resolved.append(identifier)
    return resolved


def _resolve_network_initial_species(cfg, species_report_df):
    """Resolve reaction-network initial species with explicit config priority."""
    configured = cfg.get("sankey_initial_species") or []
    if isinstance(configured, str):
        configured = [configured]
    configured = [
        str(value).strip()
        for value in configured
        if str(value).strip()
    ]
    if configured:
        return configured, "configuration"

    inferred = _initial_species_from_report(species_report_df)
    if inferred:
        return inferred, "species_report.csv"

    return [], "unavailable"

def main():
    run_started = time.perf_counter()
    stage_timings = {}
    parser = build_parser()
    args = parser.parse_args()

    print_banner()
    Config.make_dirs()

    yaml_config = load_yaml_config()
    try:
        cfg = resolve_config(args, yaml_config)
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        print(f"❌ Configuration Error: {exc}")
        return 2
    print(f"🔬 Pipeline mode: {cfg['stage'].upper()}")
    print_effective_config(cfg)
    write_run_metadata(cfg)

    sub_config = cfg['substrate']
    sub_elements = sub_config.get('elements', [])
    sub_indices = sub_config.get('indices', [])

    # === Stage 1: Topology ===
    frame_molecules, traj = None, None
    viz_needs_reaction = (
        cfg['stage'] == 'viz'
        and not os.path.exists(Config.OUTPUT_FILES["clean_events"])
    )
    need_topo = (
        cfg['stage'] in ['topo', 'all']
        or (
            (cfg['stage'] == 'reaction' or viz_needs_reaction)
            and not os.path.exists(CACHE_FILE)
        )
    )

    if need_topo:
        topology_started = time.perf_counter()
        if not cfg['input_file']:
            print("❌ Input file required for topology analysis.")
            return 1
        try:
            _require_numba_for_topology()
            _require_rdkit_for_topology()
        except ImportError as e:
            print(f"❌ Dependency Error: {e}")
            return 1
        try:
            traj = load_effective_trajectory(cfg)
        except Exception as e:
            print(f"❌ Load Error: {e}")
            return 1

        analyzer = TrajectoryAnalyzer(traj)
        analyzer.analyze(
            bond_scale=cfg['bond_scale'],
            hmm_persistence=cfg['hmm_prob'],
            n_workers=cfg['workers'],
            do_ring_stat=cfg['rings'],
            substrate_config=sub_config,
            max_size=cfg['max_size']
        )

        molecule_table = analyzer.get_molecule_table()
        bond_table = analyzer.get_bond_table()
        molecule_table.to_csv(
            Config.OUTPUT_FILES["mol_csv"],
            index=True,
            index_label="Frame",
        )
        bond_table.to_csv(
            Config.OUTPUT_FILES["bond_csv"],
            index=True,
            index_label="Frame",
        )
        if cfg['rings']:
            analyzer.get_ring_table().to_csv(Config.OUTPUT_FILES["ring_csv"])

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(_build_cache_payload(analyzer.frame_molecules, cfg), f)
        frame_molecules = analyzer.frame_molecules
        stage_timings["topology"] = (
            time.perf_counter() - topology_started
        )
        print("✅ Topology analysis completed & cached.")

    elif cfg['stage'] != 'viz' or viz_needs_reaction:
        if os.path.exists(CACHE_FILE):
            print("📂 Loading topology cache...")
            try:
                frame_molecules = _load_valid_topology_cache(cfg)
            except Exception as exc:
                print(f"❌ Cache validation failed: {exc}")
                print("👉 Please rerun with --stage topo or --stage all to rebuild topology_cache.pkl.")
                return 1
            if cfg['input_file']:
                traj = load_effective_trajectory(cfg)
        else:
            print("❌ Cache not found. Run with --stage topo first.")
            return 1

    # === Stage 2: Reaction ===
    clean_events_df = pd.DataFrame()
    # This is the dataframe that should feed user-facing Clean statistics and visualizations. 
    published_clean_df = pd.DataFrame()
    pathway_events_df = pd.DataFrame()
    species_report_df = pd.DataFrame()
    terminal_state_entries = []
    need_reaction = (
        cfg['stage'] in ['reaction', 'all']
        or viz_needs_reaction
    )

    if need_reaction:
        reaction_started = time.perf_counter()
        reaction_detail_timings = {}
        if frame_molecules is None or len(frame_molecules) == 0:
            print("❌ Reaction analysis has no topology frames to process.")
            return 1
        if traj is None and cfg['input_file']:
            traj = load_effective_trajectory(cfg)

        neb_dir = Config.OUTPUT_FILES["neb"] if cfg['auto_neb'] else "DISABLED"

        detector = ReactionDetector(
            frame_molecules,
            trajectory=traj,
            output_dir=Config.OUTPUT_FILES["snapshots"],
            neb_dir=neb_dir,
            bond_scale=cfg['bond_scale'],
            substrate_elements=sub_elements,
            substrate_indices=sub_indices,
            adsorption_scale=cfg['adsorption_scale']
        )

        detail_started = time.perf_counter()
        raw_df = detector.detect(
            stability=cfg['stability'],
            local_stability=cfg.get('local_stability'),
            single_frame_recovery=cfg.get(
                'single_frame_recovery',
                True,
            ),
            n_workers=cfg['workers'],
        )
        reaction_detail_timings["detect"] = (
            time.perf_counter() - detail_started
        )
        # Final-frame states are observational evidence, not user-specified
        # products.  Keep only exact molecular instances that persist for the
        # configured dwell interval so the pathway builder can close valid Raw
        # chains backwards from real terminal states without inventing sources.
        terminal_state_entries = detector.final_stable_lineage_entries(
            minimum_dwell=cfg['stability'],
        )
        print(
            "   Stable final molecular states available for lineage audit: "
            f"{len(terminal_state_entries)}."
        )
        # Persist the exact final-frame inventory so a later visualization-only
        # run can still classify retained states without rerunning reaction
        # detection or loading the full trajectory again.
        final_inventory_path = os.path.join(
            Config.DIR_DATA, "final_state_inventory.json"
        )
        try:
            with open(final_inventory_path, "w", encoding="utf-8") as handle:
                json.dump(
                    terminal_state_entries,
                    handle,
                    ensure_ascii=False,
                    default=str,
                )
            print(
                "   Final-state inventory saved: "
                f"{final_inventory_path}"
            )
        except Exception as exc:
            print(f"⚠️ Could not save final-state inventory: {exc}")
        for phase_name, seconds in getattr(
            detector, "detect_timings", {}
        ).items():
            reaction_detail_timings[
                f"detect.{phase_name}"
            ] = seconds

        # Unresolved/audit components: feed the Sankey/pathway projection AND are
        # exported as an audit table (mass-continuity gaps, non-elementary
        # residuals, rejected fallbacks) for transparency.
        unresolved_df = detector.get_unresolved_components_table()

        if not raw_df.empty:
            cleaner = ReactionCleaner(
                raw_df,
                stability=cfg['stability'],
                allow_ambiguous_topology=cfg['allow_ambiguous_topology'],
            )
            detail_started = time.perf_counter()
            clean_events_df = cleaner.clean()
            reaction_detail_timings["clean"] = (
                time.perf_counter() - detail_started
            )
            for phase_name, seconds in getattr(
                cleaner, "clean_timings", {}
            ).items():
                reaction_detail_timings[
                    f"clean.{phase_name}"
                ] = seconds
            unresolved_df = append_lineage_dropped_unresolved(
                unresolved_df,
                cleaner.lineage_dropped_records,
            )

            # ``ReactionCleaner.clean`` keeps a backwards-compatible in-memory handoff in DataFrame.attrs.
            clean_events_df.attrs.clear()

            # Pathway repair can still append Clean events. Finalize their IDs
            # before selecting representatives or writing any structure files.
            final_df = clean_events_df.copy()

            # Build every tier in memory first.  Nothing is published until the
            # provenance and containment contract has passed.
            detail_started = time.perf_counter()
            pathway_raw_df = raw_df.copy()
            raw_status = getattr(cleaner, "raw_clean_status", {})
            if raw_status:
                raw_ids = pathway_raw_df.get(
                    "Raw_Reaction_ID",
                    pathway_raw_df.get("Reaction_ID", pd.Series(
                        "", index=pathway_raw_df.index
                    )),
                )
                prior_reasons = pathway_raw_df.get(
                    "Exclusion_Reason",
                    pd.Series("", index=pathway_raw_df.index),
                )
                pathway_raw_df["Exclusion_Reason"] = [
                    str(
                        raw_status.get(str(raw_id), {}).get(
                            "Exclusion_Reason", prior_reason
                        )
                        or prior_reason
                        or ""
                    )
                    for raw_id, prior_reason in zip(
                        raw_ids, prior_reasons
                    )
                ]
            pathway_events_df = cleaner.build_pathway_table(
                final_df,
                unresolved_df,
                raw_df=pathway_raw_df,
                lineage_dropped_df=pd.DataFrame(
                    cleaner.lineage_dropped_records
                ),
                terminal_state_entries=terminal_state_entries,
            )

            final_df, pathway_events_df = cleaner.finalize_clean_event_ids(
                final_df, pathway_events_df,
            )
            reaction_detail_timings["pathway"] = (
                time.perf_counter() - detail_started
            )

            print(f"💾 Generating files for {len(final_df)} final valid events...")
            detail_started = time.perf_counter()
            final_df = detector.generate_output_files(
                final_df,
                snap_duration=10,
                n_workers=cfg['workers'],
            )
            reaction_detail_timings["structures"] = (
                time.perf_counter() - detail_started
            )
            clean_events_df = final_df
            published_clean_df = final_df
            invariant_counts = validate_event_table_invariants(
                raw_df,
                final_df,
                pathway_events_df,
            )
            print(
                "✅ Event tables verified: "
                f"Raw={invariant_counts['raw']} ≥ "
                f"Pathway={invariant_counts['pathway']} ≥ "
                f"Clean={invariant_counts['clean']}; "
                "Clean ⊆ Pathway ⊆ Raw."
            )

            print("🧾 Preparing event tables...")
            event_tables_started = time.perf_counter()
            raw_csv_path = Config.OUTPUT_FILES.get(
                "raw_events",
                os.path.join(Config.DIR_DATA, "reaction_events_raw.csv"),
            )
            detail_started = time.perf_counter()
            raw_csv_df = build_raw_export_table(
                raw_df,
                final_df,
                cleaner=cleaner,
                detector=detector,
            )
            reaction_detail_timings["event_tables.raw_build"] = (
                time.perf_counter() - detail_started
            )
            detail_started = time.perf_counter()
            final_csv_df = build_clean_export_table(final_df, detector)
            reaction_detail_timings["event_tables.clean_build"] = (
                time.perf_counter() - detail_started
            )
            detail_started = time.perf_counter()
            pathway_csv_df = build_pathway_export_table(
                pathway_events_df,
                detector=detector,
                # Persist exact-atom source anchors.  
                include_internal_state_sources=True,
            )
            reaction_detail_timings["event_tables.pathway_build"] = (
                time.perf_counter() - detail_started
            )
            reaction_detail_timings["event_tables.total_build"] = (
                time.perf_counter() - event_tables_started
            )
            print(
                "✅ Event tables ready "
                f"({reaction_detail_timings['event_tables.total_build']:.1f}s)."
            )

            detail_started = time.perf_counter()
            raw_csv_df.to_csv(raw_csv_path, index=False)
            final_csv_df.to_csv(Config.OUTPUT_FILES["clean_events"], index=False)
            pathway_csv_path = Config.OUTPUT_FILES.get(
                "pathway_events",
                os.path.join(Config.DIR_DATA, "reaction_events_pathway.csv")
            )
            pathway_csv_df.to_csv(pathway_csv_path, index=False)
            reaction_detail_timings["event_table_exports"] = (
                time.perf_counter() - detail_started
            )
            print(
                "💾 Saved event tables: "
                f"Raw={len(raw_csv_df)}, Clean={len(final_csv_df)}, "
                f"Pathway={len(pathway_csv_df)}"
                + (
                    " "
                    f"(internal state anchors omitted="
                    f"{pathway_csv_df.attrs.get('internal_state_sources_omitted', 0)})"
                    if pathway_csv_df.attrs.get(
                        "internal_state_sources_omitted", 0
                    )
                    else ""
                )
                + f" -> {Config.DIR_DATA}/"
            )
            
            print("📊 Generating Statistics Report...")
            detail_started = time.perf_counter()
            stats_df = ReactionStats(final_df).generate_stats()
            stats_df.to_csv(Config.OUTPUT_FILES["stats_csv"], index=False)
            reaction_detail_timings["statistics"] = (
                time.perf_counter() - detail_started
            )
            print(f"   Saved to: {Config.OUTPUT_FILES['stats_csv']}")

            detail_started = time.perf_counter()
            species_report_df = IntermediateAnalyzer(
                final_df,
                frame_molecules,
                traj,
                output_dir=Config.OUTPUT_FILES["intermediates"],
                detector=detector,
            ).find_intermediates()
            species_report_df.to_csv(
                Config.OUTPUT_FILES["inter_csv"],
                index=False,
            )
            reaction_detail_timings["species_report"] = (
                time.perf_counter() - detail_started
            )
            print(
                f"🧪 Saved {len(species_report_df)} species records to "
                f"{Config.OUTPUT_FILES['inter_csv']}"
            )

        else:
            print("⚠️ No reactions detected.")
            raw_csv_path = Config.OUTPUT_FILES.get(
                "raw_events",
                os.path.join(Config.DIR_DATA, "reaction_events_raw.csv"),
            )
            pathway_csv_path = Config.OUTPUT_FILES.get(
                "pathway_events",
                os.path.join(Config.DIR_DATA, "reaction_events_pathway.csv"),
            )
            build_raw_export_table(
                raw_df,
                clean_df=pd.DataFrame(),
                detector=detector,
            ).to_csv(raw_csv_path, index=False)
            build_clean_export_table(
                pd.DataFrame(), detector
            ).to_csv(Config.OUTPUT_FILES["clean_events"], index=False)
            build_pathway_export_table(
                pd.DataFrame(), detector=detector
            ).to_csv(pathway_csv_path, index=False)
            ReactionStats(pd.DataFrame()).generate_stats().to_csv(
                Config.OUTPUT_FILES["stats_csv"], index=False
            )
            species_report_df = IntermediateAnalyzer(
                pd.DataFrame(),
                frame_molecules,
                traj,
                output_dir=Config.OUTPUT_FILES["intermediates"],
                detector=detector,
            ).find_intermediates()
            species_report_df.to_csv(
                Config.OUTPUT_FILES["inter_csv"], index=False
            )
            print("📄 Wrote header-only reaction and species reports.")

        # Keep the four-table output contract even when there are no unresolved
        # records.  A header-only CSV has zero data rows and remains directly
        # readable by pandas and spreadsheet software.
        unresolved_csv_path = Config.OUTPUT_FILES.get(
            "unresolved_events",
            os.path.join(Config.DIR_DATA, "reaction_events_unresolved.csv"),
        )
        detail_started = time.perf_counter()
        unresolved_csv_df = build_unresolved_export_table(
            unresolved_df,
            detector=detector,
            pathway_df=pathway_events_df,
        )
        unresolved_csv_df.to_csv(unresolved_csv_path, index=False)
        reaction_detail_timings["unresolved_export"] = (
            time.perf_counter() - detail_started
        )
        print(
            f"🔎 Saved {len(unresolved_csv_df)} unresolved records to "
            f"{unresolved_csv_path}"
        )

        published_clean_df = (
            final_df if "final_df" in locals() else clean_events_df
        )
        lineage_repaired_clean_count = (
            int(
                published_clean_df.get(
                    "Clean_Tier", pd.Series("", index=published_clean_df.index)
                ).astype(str).eq("lineage_repaired_elementary").sum()
            )
            if isinstance(published_clean_df, pd.DataFrame)
            else 0
        )
        corridor_recovered_clean_count = (
            int(
                published_clean_df.get(
                    "Clean_Tier", pd.Series("", index=published_clean_df.index)
                ).astype(str).eq("trajectory_recovered_corridor_net").sum()
            )
            if isinstance(published_clean_df, pd.DataFrame)
            else 0
        )
        repaired_clean_count = (
            lineage_repaired_clean_count + corridor_recovered_clean_count
        )
        strict_clean_count = max(
            0, len(published_clean_df) - repaired_clean_count
        )
        repair_parts = []
        if lineage_repaired_clean_count:
            repair_parts.append(
                f"lineage-repaired={lineage_repaired_clean_count}"
            )
        if corridor_recovered_clean_count:
            repair_parts.append(
                f"corridor-recovered={corridor_recovered_clean_count}"
            )
        clean_summary = (
            f"{len(published_clean_df)} "
            f"(strict={strict_clean_count}, "
            + ", ".join(repair_parts)
            + ")"
            if repaired_clean_count
            else str(len(published_clean_df))
        )
        internal_state_anchor_count = (
            int(
                pathway_events_df.get(
                    "Pathway_Role",
                    pd.Series("", index=pathway_events_df.index),
                ).astype(str).eq("observed_stable_state_source").sum()
            )
            if isinstance(pathway_events_df, pd.DataFrame)
            else 0
        )
        pathway_summary = (
            f"{len(pathway_events_df) - internal_state_anchor_count} "
            f"(internal-state-anchors={internal_state_anchor_count})"
            if internal_state_anchor_count
            else str(len(pathway_events_df))
        )
        print(
            "✅ Reaction stage completed: "
            f"Raw={len(raw_df)}, Clean={clean_summary}, "
            f"Pathway={pathway_summary}, "
            f"Unresolved={len(unresolved_csv_df)}."
        )
        stage_timings["reaction"] = (
            time.perf_counter() - reaction_started
        )
        for detail_name, seconds in reaction_detail_timings.items():
            stage_timings[f"reaction.{detail_name}"] = seconds
        report_seconds = sum(
            reaction_detail_timings.get(name, 0.0)
            for name in (
                "statistics",
                "species_report",
                "unresolved_export",
            )
        )
        print(
            "⏱️ Reaction stage: "
            f"detect={reaction_detail_timings.get('detect', 0.0):.1f}s, "
            f"clean={reaction_detail_timings.get('clean', 0.0):.1f}s, "
            f"structures={reaction_detail_timings.get('structures', 0.0):.1f}s, "
            f"pathway={reaction_detail_timings.get('pathway', 0.0):.1f}s, "
            "table_build="
            f"{reaction_detail_timings.get('event_tables.total_build', 0.0):.1f}s, "
            f"reports={report_seconds:.1f}s, "
            f"total={stage_timings['reaction']:.1f}s."
        )

    elif cfg['stage'] == 'viz' and os.path.exists(Config.OUTPUT_FILES["clean_events"]):
        clean_events_df = pd.read_csv(Config.OUTPUT_FILES["clean_events"])
        if "Reaction_Type" in clean_events_df.columns and "Type" not in clean_events_df.columns:
            clean_events_df["Type"] = clean_events_df["Reaction_Type"]
        published_clean_df = clean_events_df

    pathway_csv_path = Config.OUTPUT_FILES.get(
        "pathway_events",
        os.path.join(Config.DIR_DATA, "reaction_events_pathway.csv")
    )
    if cfg['stage'] in ['viz', 'all'] and os.path.exists(pathway_csv_path):
        try:
            pathway_events_df = pd.read_csv(pathway_csv_path)
            if "Pathway_ID" in pathway_events_df.columns and "Reaction_ID" not in pathway_events_df.columns:
                pathway_events_df["Reaction_ID"] = pathway_events_df["Pathway_ID"]
            if "Reaction_Type" in pathway_events_df.columns and "Type" not in pathway_events_df.columns:
                pathway_events_df["Type"] = pathway_events_df["Reaction_Type"]
        except Exception as exc:
            print(f"⚠️ Could not read pathway events from {pathway_csv_path}: {exc}")
            pathway_events_df = pd.DataFrame()

    species_report_path = Config.OUTPUT_FILES.get(
        "inter_csv",
        os.path.join(Config.DIR_DATA, "species_report.csv"),
    )
    if not terminal_state_entries:
        final_inventory_path = os.path.join(
            Config.DIR_DATA, "final_state_inventory.json"
        )
        if os.path.exists(final_inventory_path):
            try:
                with open(final_inventory_path, "r", encoding="utf-8") as handle:
                    loaded_inventory = json.load(handle)
                if isinstance(loaded_inventory, list):
                    terminal_state_entries = loaded_inventory
                print(
                    "   Loaded final-state inventory for visualization: "
                    f"{len(terminal_state_entries)} state(s)."
                )
            except Exception as exc:
                print(f"⚠️ Could not load final-state inventory: {exc}")
    if (
        cfg['stage'] in ['viz', 'all']
        and species_report_df.empty
        and os.path.exists(species_report_path)
    ):
        try:
            species_report_df = pd.read_csv(species_report_path)
        except Exception as exc:
            print(
                f"⚠️ Could not read initial-species metadata from "
                f"{species_report_path}: {exc}"
            )
            species_report_df = pd.DataFrame()

    # === Stage 3: Viz ===
    if cfg['stage'] in ['viz', 'all']:
        visualization_started = time.perf_counter()
        print("🌊 Generating Visualizations...")
        viz = Visualizer(output_dir=Config.DIR_PLOTS)

        if os.path.exists(Config.OUTPUT_FILES["mol_csv"]):
            viz.plot_molecule_evolution(pd.read_csv(Config.OUTPUT_FILES["mol_csv"], index_col=0))
        if os.path.exists(Config.OUTPUT_FILES["bond_csv"]):
            viz.plot_bonds(pd.read_csv(Config.OUTPUT_FILES["bond_csv"], index_col=0))
        if not clean_events_df.empty:
            network_events_df = (
                pathway_events_df
                if not pathway_events_df.empty
                else clean_events_df
            )
            if "Elementary_Mode" in network_events_df.columns:
                network_events_df = network_events_df.loc[
                    ~network_events_df["Elementary_Mode"].astype(str).eq(
                        "atom_mapping_exchange"
                    )
                ].copy()

            network_initial_species, initial_species_source = (
                _resolve_network_initial_species(
                    cfg,
                    species_report_df,
                )
            )
            if network_initial_species:
                print(
                    "   Reaction-network initial species "
                    f"({initial_species_source}): "
                    + ", ".join(network_initial_species)
                )
            else:
                print(
                    "   ⚠️ Reaction-network initial species could not be "
                    "resolved from configuration or species_report.csv."
                )

            viz.plot_reaction_network(
                network_events_df,
                initial_species=network_initial_species or None,
            )
            if cfg['sankey']:
                viz.plot_sankey(
                    network_events_df,
                    initial_species=cfg.get('sankey_initial_species'),
                    strict_initial_species=True,
                    element=cfg.get(
                        'sankey_elements',
                        list(Config.SANKEY_DEFAULT_ELEMENTS),
                    ),
                    # Reuse the persisted final-frame inventory during both
                    # full runs and visualization-only reruns. 
                    final_state_inventory=terminal_state_entries or None,
                )
            # Use the final published Clean table.  
            cluster_source_df = (
                published_clean_df
                if isinstance(published_clean_df, pd.DataFrame)
                and not published_clean_df.empty
                else clean_events_df
            )
            cluster_df = ReactionClusterer(cluster_source_df).cluster()
            print(
                "   Reaction-cluster source: final Clean events="
                f"{len(cluster_source_df)}, all event records counted="
                f"{int(cluster_df['Count'].sum()) if not cluster_df.empty else 0}."
            )
            viz.plot_reaction_clusters(
                cluster_df, clean_event_count=len(cluster_source_df),
            )

        stage_timings["visualization"] = (
            time.perf_counter() - visualization_started
        )

    stage_timings["total"] = time.perf_counter() - run_started
    update_run_timings(stage_timings)
    top_level_timing_names = (
        "topology",
        "reaction",
        "visualization",
        "total",
    )
    top_level_timings = [
        (name, stage_timings[name])
        for name in top_level_timing_names
        if name in stage_timings
    ]
    print(
        "⏱️ Run summary: "
        + ", ".join(
            f"{name}={seconds:.1f}s"
            for name, seconds in top_level_timings
        )
    )
    print("\n🎉 RxnScape analysis completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
