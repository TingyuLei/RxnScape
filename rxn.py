#!/usr/bin/env python
import sys
import os
import argparse
import yaml
import subprocess
import shutil
import shlex

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
CONFIG_FILE = "rxn.yaml"


DEFAULT_CONFIG_TEXT = """input_file: input.traj

parameters:
  cell: null      # e.g. "30 30 30"
  pbc: null       # e.g. "T T T"

  bond_scale: 1.2
  adsorption_scale: 1.25
  hmm_prob: 0.99
  workers: 4
  max_structure_size: 60
  stability: 5
  local_stability: null   # null = auto; when set to 1, regular segmentation still uses the automatic threshold, and single frames only enter strict restoration scanning

  substrate:
    indices: ''      # e.g. "0-499,623"
    elements: []     # e.g. ["Au", "Pt"]

features:
  auto_neb: true
  sankey: true
  rings: false

Visualization:
  sankey_initial_species: []   # Strictly validated Sankey roots; invalid or no-outflow roots skip Sankey.
  sankey_elements: [C]         # Any valid element symbols; [] disables element-flow Sankeys.
"""


def create_default_config(force=False):
    """Generate a clean default configuration YAML file.

    Existing rxn.yaml is never overwritten unless force=True, so hand-tuned
    parameters cannot be lost by accidentally re-running ``init``.
    """
    if os.path.exists(CONFIG_FILE) and not force:
        print(f"⚠️  {CONFIG_FILE} already exists; not overwritten.")
        print(f"   Re-run 'python {sys.argv[0]} init --force' to overwrite it.")
        return False
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(DEFAULT_CONFIG_TEXT)
    print(f"✅ Configuration generated: {CONFIG_FILE}")
    print(f"👉 Please edit '{CONFIG_FILE}' and run 'python {sys.argv[0]} run'")
    return True


def load_config():
    """Load the YAML configuration file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return None


def _as_bool(value):
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


def _add_value(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def _add_bool(cmd, true_flag, false_flag, value):
    value = _as_bool(value)
    if value is True:
        cmd.append(true_flag)
    elif value is False:
        cmd.append(false_flag)


def _add_species_list(cmd, flag, value):
    """Add a comma-separated species list to the main.py CLI."""
    if value is None:
        return

    if isinstance(value, (list, tuple, set)):
        species = [str(x).strip() for x in value if str(x).strip()]
    else:
        species = [x.strip() for x in str(value).split(',') if x.strip()]

    if species:
        cmd.extend([flag, ",".join(species)])


def _add_element_list(cmd, flag, value):
    """Forward selected element-flow views to main.py.

    An explicitly empty list is meaningful: it disables element-flow Sankeys
    while retaining the reaction transition map.  Forward it as ``none`` so
    command-line precedence preserves that intent.  Element-symbol validation
    and case normalization remain centralized in main.py/config.py.
    """
    if value is None:
        return

    if value is False:
        elements = []
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "off", "false", "[]"}:
            elements = []
        else:
            elements = [
                item.strip()
                for item in text.split(',')
                if item.strip()
            ]
    elif isinstance(value, set):
        elements = sorted(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    elif isinstance(value, (list, tuple)):
        elements = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    else:
        elements = [str(value).strip()]

    cmd.extend([flag, ",".join(elements) if elements else "none"])


def build_cmd(config, action):
    """Construct command-line arguments for main.py."""
    cmd = [sys.executable, MAIN_SCRIPT]

    input_file = config.get("input_file")
    if not input_file and action in {"run", "topo", "reaction"}:
        print("❌ Config Error: 'input_file' missing in rxn.yaml.")
        sys.exit(1)

    if input_file:
        cmd.append(input_file)

    stage_map = {
        "run": "all",
        "topo": "topo",
        "reaction": "reaction",
        "viz": "viz",
    }
    if action in stage_map:
        cmd.extend(["--stage", stage_map[action]])

    p = config.get("parameters", {}) or {}
    # Cell / PBC parameters
    _add_value(cmd, "--cell", p.get("cell"))
    _add_value(cmd, "--pbc", p.get("pbc"))

    # Topology / general parameters
    _add_value(cmd, "--bond_scale", p.get("bond_scale"))
    _add_value(cmd, "--adsorption_scale", p.get("adsorption_scale"))
    _add_value(cmd, "--hmm_prob", p.get("hmm_prob"))
    _add_value(cmd, "--workers", p.get("workers"))
    _add_value(cmd, "--max_size", p.get("max_size", p.get("max_structure_size")))
    _add_value(cmd, "--remove", p.get("remove"))

    # Reaction stability. The public interface uses parameters.stability only.
    _add_value(cmd, "--stability", p.get("stability"))
    _add_value(cmd, "--local_stability", p.get("local_stability"))
    
    # Substrate
    sub = p.get("substrate", {}) or {}
    if sub.get("indices"):
        if isinstance(sub["indices"], str):
            cmd.extend(["--sub_indices", sub["indices"]])
        elif isinstance(sub["indices"], list):
            cmd.extend(["--sub_indices", ",".join(map(str, sub["indices"]))])

    if sub.get("elements"):
        if isinstance(sub["elements"], str):
            cmd.extend(["--sub_elements", sub["elements"]])
        elif isinstance(sub["elements"], list):
            cmd.extend(["--sub_elements", ",".join(map(str, sub["elements"]))])

    # Feature flags
    f = config.get("features", {}) or {}
    _add_bool(cmd, "--rings", "--no_rings", f.get("rings"))
    _add_bool(cmd, "--auto_neb", "--no_neb", f.get("auto_neb"))
    _add_bool(cmd, "--sankey", "--no_sankey", f.get("sankey"))

    # Visualization options
    v = config.get("Visualization", config.get("visualization", {})) or {}
    _add_species_list(cmd, "--sankey_initial_species", v.get("sankey_initial_species"))
    _add_element_list(
        cmd,
        "--sankey_elements",
        v.get("sankey_elements", f.get("sankey_elements")),
    )

    return cmd


def main():
    parser = argparse.ArgumentParser(description="RxnScape Smart Launcher", add_help=True)
    parser.add_argument('command', choices=['init', 'run', 'topo', 'reaction', 'viz', 'clean'],
                        help="Command to execute")
    parser.add_argument('--sub_indices', help="Override substrate indices from CLI")
    parser.add_argument('--sub_elements', help="Override substrate elements from CLI")
    parser.add_argument('--print_cmd', action='store_true',
                        help="Print the final main.py command before execution")
    parser.add_argument('--dry_run', '--dry-run', action='store_true',
                        help="Print the final main.py command and do not execute it")
    parser.add_argument('--yes', '-y', action='store_true',
                        help="Confirm deletion; required by 'clean' to remove files")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite an existing rxn.yaml when used with 'init'")

    # Unknown args are appended after YAML-derived args, so they override YAML in argparse.
    args, unknown = parser.parse_known_args()

    if args.command == 'init':
        create_default_config(force=args.force)
        return

    if args.command == 'clean':
        targets = [
            "00_Data", "01_Plots", "02_Structures",
            "topology_cache.pkl", "rxn.log",
        ]
        existing = [t for t in targets if os.path.exists(t)]
        if not existing:
            print("🧹 Nothing to clean.")
            return
        # 默认只预览，必须显式 --yes 才会真正删除
        if args.dry_run or not args.yes:
            print("🧾 Would remove (nothing deleted):")
            for t in existing:
                print(f"   - {t}")
            if not args.dry_run:
                print(f"   Re-run 'python {sys.argv[0]} clean --yes' to delete them.")
            return
        for t in existing:
            if os.path.isdir(t):
                shutil.rmtree(t)
            else:
                os.remove(t)
            print(f"🗑️  Removed: {t}")
        print("🧹 Workspace cleaned.")
        return

    config = load_config()
    if not config:
        print("⚠️ Config not found. Run 'init' first.")
        sys.exit(1)

    print(f"🚀 RxnScape Launcher: [{args.command.upper()} MODE]")
    try:
        final_cmd = build_cmd(config, args.command)

        # Explicit launcher-level substrate overrides.
        if args.sub_indices:
            final_cmd.extend(["--sub_indices", args.sub_indices])
        if args.sub_elements:
            final_cmd.extend(["--sub_elements", args.sub_elements])

        # Pass through extra CLI args, appended last so they can override YAML-derived values.
        final_cmd.extend(unknown)

        if args.print_cmd or args.dry_run:
            print("🧾 Final command:")
            print(" ".join(shlex.quote(x) for x in final_cmd))

        if args.dry_run:
            return

        subprocess.run(final_cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Execution Failed with return code {e.returncode}")
        return int(e.returncode) or 1
    except (TypeError, ValueError) as e:
        print(f"\n❌ Configuration Error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by User")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
