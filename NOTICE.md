# NOTICE

## 本仓的数据全部是合成的

`network/campus_graph.json` 与所有速度历史由 `scripts/build_network.py` 与
`campus_bus/congestion.py::make_history` 确定性生成，**不含任何真实路网、
真实轨迹或第三方数据集**。README 的 Provenance 一节说明了为什么：它要替代的那份
2024 年课程实践的原始数据（长安大学校园路网与北斗轨迹）没有留档。

## 第三方依赖

| 包 | 用途 | 许可证 |
|---|---|---|
| [NumPy](https://numpy.org/) | 生成路网与历史、GPS 噪声与统计 | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org/) | 拥堵预测的岭回归基线 | BSD-3-Clause |
| [NetworkX](https://networkx.org/) | **仅测试用**：把静态快照交给第三方实现独立求一遍 | BSD-3-Clause |
| [PyTorch](https://pytorch.org/) | **可选**：`bench/bench_all.py --lstm` 的单层 LSTM，CI 不装 | BSD-3-Clause |

运行时（`python -m campus_bus.service` 与 Docker 镜像）**不依赖上述任何一个**：
`Network.load` 在 numpy 缺席时退回纯 Python 列表，CI 里专门卸掉 numpy 验证过。

配对自助（`campus_bus/stats.py`）是自己写的纯标准库实现，不用 numpy——
`Generator` 的流跨大版本不承诺稳定（NEP 19），而这个仓的结论要跨机器复现。
