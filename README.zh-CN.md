# RxnScape

[English](README.md) | 简体中文

**从反应分子动力学轨迹到可解释的反应路径与反应网络**

RxnScape 是一个用于反应分子动力学（Reactive MD）轨迹后处理的 Python 工具，可从轨迹中识别分子结构变化、重构反应事件，并生成反应路径、反应网络和可视化结果。

## 主要功能

- 分子拓扑识别与时间序列稳定化
- 反应事件检测、原子映射与分子来源追踪
- Raw、Clean、Pathway 和 Unresolved 事件表导出
- 反应路径、反应网络及元素流向可视化
- 物种、化学键演化统计和相关结构导出

## 安装

建议使用 Python 3.11 和独立的 Python 环境。

```bash
git clone https://gitee.com/TingyuLei/RxnScape.git
cd RxnScape
python -m pip install -r requirements.txt
```

也可以从 [GitHub](https://github.com/TingyuLei/RxnScape) 下载源码。

若仓库中包含 `pyproject.toml`，还可以在源码根目录执行 `python -m pip install .`。**这不等于软件已发布到 PyPI。**

## 快速使用

在包含输入轨迹的工作目录运行以下命令（将路径替换为实际的 RxnScape 源码位置）：

```bash
python /path/to/RxnScape/rxn.py init
```

编辑自动生成的 `rxn.yaml`，将 `input_file` 设置为轨迹文件路径，并根据体系调整成键、稳定性和基底参数。

然后运行：

```bash
python /path/to/RxnScape/rxn.py run
```

还可以分阶段运行：

```bash
python /path/to/RxnScape/rxn.py topo
python /path/to/RxnScape/rxn.py reaction
python /path/to/RxnScape/rxn.py viz
```

## 主要输出

| 目录 | 内容 |
| --- | --- |
| `00_Data/` | 物种统计、反应事件表和分析数据 |
| `01_Plots/` | 物种/键演化、反应网络和流向图 |
| `02_Structures/` | 反应结构快照和中间体等结构文件 |

`reaction_events_raw.csv`、`reaction_events_clean.csv`、`reaction_events_pathway.csv` 和 `reaction_events_unresolved.csv` 分别用于记录候选事件、筛选后的事件、路径相关事件以及待核查事件。这些表并非四个互不重叠的数据集。

## 许可

使用和再分发条件见 [LICENSE](LICENSE)。
