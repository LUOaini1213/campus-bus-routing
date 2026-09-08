# -*- coding: utf-8 -*-
"""生成 network/campus_graph.json：23 节点 / 58 有向边 + 分时速度剖面。

**这张网是合成的。** 2024 年那份长安大学校园路网与北斗轨迹没有留档
（见 README「Provenance」），所以这里按一个常见校园的拓扑重建：
一条主环路 + 两条穿场道 + 若干支路，节点是宿舍区、教学楼、图书馆、
食堂、体育场、校门与班车站点。坐标以米为单位，原点在西南角。

速度剖面 = 自由流速度 × 拥堵因子(时刻)，拥堵因子按早晚高峰双峰形状生成，
主路比支路更受影响；再加一点确定性的伪随机扰动（固定种子，可复现）。
生成后强制 FIFO（见 network.enforce_fifo）。

    python scripts/build_network.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campus_bus.network import BUCKET_MIN, N_BUCKETS, Edge, Network  # noqa: E402

SEED = 20241001

# 23 个节点：(名字, x_m, y_m)
NODES = [
    ("west_gate",     0,    600), ("north_gate",  900,   1200), ("east_gate",  1800,  600),
    ("south_gate",    900,      0),
    ("dorm_a",      200,    900), ("dorm_b",      200,    350), ("dorm_c",    1600,   950),
    ("dorm_d",     1600,    300),
    ("teach_1",     700,    850), ("teach_2",    1100,    850), ("teach_3",    700,   400),
    ("teach_4",    1100,    400),
    ("library",     900,    650), ("canteen_n",   600,   1050), ("canteen_s",  1200,   200),
    ("gym",        1450,    700), ("stadium",     450,    150), ("lab_block", 1350,  1050),
    ("admin",       750,   1100), ("hub_w",       450,    650), ("hub_e",     1350,   650),
    ("hub_n",       900,    950), ("hub_s",       900,    300),
]

# 58 条有向边：(u, v, kind)。主环路双向，支路多为双向，两条单行。
LINKS = [
    # 主环（双向）
    ("west_gate", "hub_w", "main"), ("hub_w", "hub_n", "main"), ("hub_n", "hub_e", "main"),
    ("hub_e", "east_gate", "main"), ("east_gate", "hub_e", "main"), ("hub_e", "hub_s", "main"),
    ("hub_s", "hub_w", "main"), ("hub_w", "west_gate", "main"),
    ("hub_n", "hub_w", "main"), ("hub_e", "hub_n", "main"), ("hub_s", "hub_e", "main"),
    ("hub_w", "hub_s", "main"),
    # 穿场道
    ("hub_n", "library", "campus"), ("library", "hub_s", "campus"),
    ("hub_s", "library", "campus"), ("library", "hub_n", "campus"),
    ("hub_w", "library", "campus"), ("library", "hub_e", "campus"),
    # 校门接入
    ("north_gate", "hub_n", "main"), ("hub_n", "north_gate", "main"),
    ("south_gate", "hub_s", "main"), ("hub_s", "south_gate", "main"),
    # 宿舍区
    ("dorm_a", "hub_w", "campus"), ("hub_w", "dorm_a", "campus"),
    ("dorm_b", "hub_w", "campus"), ("hub_w", "dorm_b", "campus"),
    ("dorm_c", "hub_e", "campus"), ("hub_e", "dorm_c", "campus"),
    ("dorm_d", "hub_e", "campus"), ("hub_e", "dorm_d", "campus"),
    ("dorm_a", "canteen_n", "service"), ("canteen_n", "dorm_a", "service"),
    ("dorm_d", "canteen_s", "service"), ("canteen_s", "dorm_d", "service"),
    # 教学区
    ("teach_1", "hub_n", "campus"), ("hub_n", "teach_1", "campus"),
    ("teach_2", "hub_n", "campus"), ("hub_n", "teach_2", "campus"),
    ("teach_3", "hub_s", "campus"), ("hub_s", "teach_3", "campus"),
    ("teach_4", "hub_s", "campus"), ("hub_s", "teach_4", "campus"),
    ("teach_1", "library", "service"), ("library", "teach_1", "service"),
    ("teach_4", "library", "service"), ("library", "teach_4", "service"),
    # 其它设施
    ("canteen_n", "hub_n", "service"), ("hub_n", "canteen_n", "service"),
    ("canteen_s", "hub_s", "service"), ("hub_s", "canteen_s", "service"),
    ("gym", "hub_e", "service"), ("hub_e", "gym", "service"),
    ("stadium", "hub_s", "service"), ("hub_s", "stadium", "service"),
    ("lab_block", "hub_n", "service"), ("hub_n", "lab_block", "service"),
    ("admin", "hub_n", "service"), ("hub_n", "admin", "service"),
]
# 曾经这里是 ("admin","hub_n") 单向 + ("stadium","west_gate") 单向，为的是
# "留下不对称"。后果是 **admin 入度为 0、永远到不了**：60 组 OD 里有 3 组以它
# 为终点，30 个查询被静默跳过，README 里"60 组 OD × 10 个出发时刻 = 570"
# 这个算式因此对不上（600 - 30）。校园班车到不了行政楼也说不通。
# 现在 admin 双向、去掉体育场那条单行，边数仍是 58。
# 有向性并不依赖单行边：每个方向是独立的 Edge，各有自己的相位与扰动，
# u→v 与 v→u 的速度剖面本来就不同。

FREE_KMH = {"main": 40.0, "campus": 30.0, "service": 20.0}
# 高峰期速度掉到自由流的这个比例（主路最敏感）
PEAK_FLOOR = {"main": 0.42, "campus": 0.55, "service": 0.72}


def peak_factor(t_min: float) -> float:
    """双峰拥堵形状：0=完全自由流，1=最堵。7:40 与 17:50 为峰。"""
    def bell(center, width):
        return math.exp(-0.5 * ((t_min - center) / width) ** 2)
    return min(1.0, 1.05 * bell(460, 42) + 0.95 * bell(1070, 55) + 0.25 * bell(720, 35))


def build(seed: int = SEED) -> Network:
    """按给定种子生成整张分时路网。

    抽成函数是为了让 `bench/bench_robustness.py` 能换种子重建，而不是复制
    一份构图代码——两处一旦不同步，稳健性扫描扫的就不是同一个模型。
    """
    rng = np.random.default_rng(seed)
    nodes = {n: (float(x), float(y)) for n, x, y in NODES}
    edges = []
    for i, (u, v, kind) in enumerate(LINKS):
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        length = math.hypot(x2 - x1, y2 - y1)
        edges.append(Edge(i, u, v, round(length, 1), FREE_KMH[kind], kind))

    speed = np.zeros((len(edges), N_BUCKETS))
    for e in edges:
        floor = PEAK_FLOOR[e.kind]
        # 每条边一个固定偏移，让同类边不完全同步
        phase = float(rng.uniform(-12, 12))
        jitter = rng.normal(0.0, 0.02, N_BUCKETS)
        for b in range(N_BUCKETS):
            t = b * BUCKET_MIN
            f = peak_factor(t + phase)
            ratio = 1.0 - (1.0 - floor) * f
            speed[e.eid][b] = e.free_kmh * max(0.25, ratio + jitter[b])

    net = Network(nodes, edges, speed)
    net.enforce_fifo()
    return net


def main() -> int:
    net = build(SEED)
    nodes, edges, speed = net.nodes, net.edges, net.speed
    assert net.n_edges() == 58, "边数应为 58，实际 %d" % net.n_edges()
    net.dump(Path(__file__).resolve().parents[1] / "network" / "campus_graph.json")

    unreachable = [n for n in net.nodes if not any(e.v == n for e in net.edges)]
    longest = max(e.length_m for e in edges)
    slowest = float(speed.min())
    print("nodes=%d edges=%d buckets=%d (%d min each)" % (
        net.n_nodes(), net.n_edges(), N_BUCKETS, BUCKET_MIN))
    print("入度为 0 的节点: %s" % (unreachable or "无"))
    print("最长边 %.0f m、最低速 %.1f km/h -> 最坏单边行程 %.2f min"
          % (longest, slowest, longest / 1000.0 / slowest * 60.0))
    print("check_fifo（步长 %.1f min）看得见的违反: %d 条"
          % (BUCKET_MIN / 2, len(net.check_fifo())))
    print("但连续意义上的 FIFO 违反: **%d** 个槽边界，最大下降 %.4f min。"
          % (net.boundary_drop_count(), net.max_boundary_drop()))
    print("  -> 「违反数 0」是采样分辨率的产物，不是结构保证。分段常速剖面在槽边界上")
    print("     几乎必然违反连续 FIFO；真正让 Dijkstra 可用的是**违反幅度有上界**，")
    print("     次优性被上面那个 %.4f min 界住。详见 README 第 2 节与 network.check_fifo 的 docstring。"
          % net.max_boundary_drop())
    print("（check_fifo / enforce_fifo 由 tests/test_all.py 用一条构造的 8 km 长边验证。）")
    assert not unreachable, "存在入度为 0 的节点：%s" % unreachable
    return 0


if __name__ == "__main__":
    sys.exit(main())
