# RxnScape

**English** | [简体中文](README.zh-CN.md)

**From Reactive Molecular Dynamics to Interpretable Reaction Pathways and Networks**

RxnScape is a Python tool for analyzing reactive molecular dynamics (Reactive MD) trajectories. It identifies molecular connectivity changes, reconstructs reaction events, and generates reaction pathways, reaction networks, and visualizations.

## Features

- Molecular topology identification and temporal stabilization
- Reaction-event detection, atom mapping, and molecular lineage tracking
- Export of Raw, Clean, Pathway, and Unresolved event tables
- Reaction pathways, reaction networks, and element-flow visualizations
- Species and bond evolution statistics, and related structure exports

## Installation

Python 3.11 and a dedicated Python environment are recommended.

```bash
git clone https://gitee.com/TingyuLei/RxnScape.git
cd RxnScape
python -m pip install -r requirements.txt
```

Alternatively, download the source from [GitHub](https://github.com/TingyuLei/RxnScape).

If the repository includes `pyproject.toml`, you can also install it from the source directory with `python -m pip install .`. **This does not mean the package has been published to PyPI.**

## Quick Start

From a working directory containing your input trajectory, run the following command (replace the path with the actual location of the RxnScape source):

```bash
python /path/to/RxnScape/rxn.py init
```

Edit the generated `rxn.yaml` file. Set `input_file` to your trajectory path and adjust the bonding, stability, and substrate parameters for your system.

Then run:

```bash
python /path/to/RxnScape/rxn.py run
```

You can also run individual stages:

```bash
python /path/to/RxnScape/rxn.py topo
python /path/to/RxnScape/rxn.py reaction
python /path/to/RxnScape/rxn.py viz
```

## Main Outputs

| Directory | Contents |
| --- | --- |
| `00_Data/` | Species statistics, reaction-event tables, and analysis data |
| `01_Plots/` | Species/bond evolution, reaction networks, and flow visualizations |
| `02_Structures/` | Reaction snapshots, intermediates, and related structure files |

The files `reaction_events_raw.csv`, `reaction_events_clean.csv`, `reaction_events_pathway.csv`, and `reaction_events_unresolved.csv` record candidate events, filtered events, pathway-related events, and events requiring further inspection, respectively. **These tables are not four mutually exclusive datasets.**

## License

See [LICENSE](LICENSE) for conditions of use and redistribution.
