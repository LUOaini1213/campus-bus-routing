# -*- coding: utf-8 -*-
"""校园路网：23 节点 / 58 有向边的分时加权图。

分时图的一个前提常被忽略：**只有满足 FIFO（先进先出，又叫非超车）性质，
Dijkstra 才仍然最优**。FIFO 是说 t1 < t2 时必有 t1 + tt(e,t1) <= t2 + tt(e,t2)
—— 晚出发不可能早到。分段常速的速度剖面很容易违反它：如果速度在某个
时刻突然大幅回升，早一分钟出发的车反而会晚到。

所以这里做两件事：`check_fifo` 把违反点找出来，`enforce_fifo` 通过限制
速度上升的斜率把剖面修成 FIFO 的。两个函数都有测试。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_JSON = ROOT / "network" / "campus_graph.json"

MINUTES_PER_DAY = 1440
BUCKET_MIN = 5                      # 速度剖面的时间粒度
N_BUCKETS = MINUTES_PER_DAY // BUCKET_MIN


@dataclass(frozen=True)
class Edge:
    eid: int
    u: str
    v: str
    length_m: float
    free_kmh: float
    kind: str                       # main / campus / service


class Network:
    """节点坐标 + 有向边 + 每条边一天 288 个 5 分钟槽的速度剖面。"""

    def __init__(self, nodes: dict, edges: list, speed_kmh):
        self.nodes = nodes                                   # name -> (x_m, y_m)
        self.edges = edges                                   # list[Edge]
        self.speed = speed_kmh                               # (n_edges, N_BUCKETS)
        self.out: dict[str, list[Edge]] = {n: [] for n in nodes}
        for e in edges:
            self.out[e.u].append(e)
        # A* 的启发式除以这个上界，所以它必须是**剖面里真实出现过的最高速**，
        # 不是标称的 free_kmh。第一版写的是 max(free_kmh)=40，而生成器的 jitter
        # 让实际速度到过 42.73 km/h（1182 个 (边,槽) 单元超标），于是 h(v) 会高估、
        # 启发式不再可采纳。`test_heuristic_is_admissible` 就是为这个加的。
        self.max_kmh = max(float(max(e.free_kmh for e in edges)),
                           max(float(max(row)) for row in speed_kmh))

    # ---------------------------------------------------------------- 基本量
    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)

    def euclid_m(self, a: str, b: str) -> float:
        (x1, y1), (x2, y2) = self.nodes[a], self.nodes[b]
        return math.hypot(x2 - x1, y2 - y1)

    def bucket(self, t_min: float) -> int:
        return int(t_min // BUCKET_MIN) % N_BUCKETS

    def travel_time(self, e: Edge, t_min: float) -> float:
        """在 t_min（分）进入边 e 的行程时间，单位分钟。"""
        kmh = float(self.speed[e.eid][self.bucket(t_min)])
        return (e.length_m / 1000.0) / kmh * 60.0

    def arrival(self, e: Edge, t_min: float) -> float:
        return t_min + self.travel_time(e, t_min)

    # ---------------------------------------------------------------- FIFO
    def check_fifo(self, samples_per_bucket: int = 2) -> list[tuple]:
        """**按 step 分辨率采样**看得见的 FIFO 违反，返回
        (eid, t_early, t_late, arr_early, arr_late)。

        名字容易骗人，所以把话说死：**这不是"是否存在违反"的判定，是"在
        step = BUCKET_MIN/samples_per_bucket 的分辨率下是否可见"的判定。**
        分段常速的剖面在槽边界上几乎必然违反连续意义的 FIFO——只要 tt 在边界
        处下降，早一瞬进入的车就晚到。本网上 16646 个槽边界里有 8324 个
        （50%）如此，但最大下降只有 0.3145 min，远小于默认 step = 2.5 min，
        所以采样器一个也看不见（见 `max_boundary_drop()`）。

        真正让 Dijkstra 可用的不是"违反数为 0"，而是**违反的幅度有上界**：
        次优性被 `max_boundary_drop()` 界住。README 第 2 节按这个口径写。
        """
        bad = []
        step = BUCKET_MIN / samples_per_bucket
        for e in self.edges:
            t = 0.0
            prev_arr = self.arrival(e, t)
            t += step
            while t < MINUTES_PER_DAY:
                arr = self.arrival(e, t)
                if arr < prev_arr - 1e-9:
                    bad.append((e.eid, t - step, t, prev_arr, arr))
                prev_arr = arr
                t += step
        return bad

    def max_boundary_drop(self) -> float:
        """槽边界上到达时刻的最大下降量（分钟）——连续 FIFO 违反的幅度上界。

        对每条边、每个槽边界，比较刚好在边界前后进入的两辆车的到达时刻。
        这个量与采样步长无关，是剖面自身的性质，也是 Dijkstra 在本网上
        次优性的上界。
        """
        worst = 0.0
        for e in self.edges:
            for b in range(1, N_BUCKETS):
                t0 = b * BUCKET_MIN
                drop = self.arrival(e, t0 - 1e-6) - self.arrival(e, t0 + 1e-6)
                if drop > worst:
                    worst = drop
        return worst

    def boundary_drop_count(self) -> int:
        """有多少个槽边界上到达时刻是下降的（即连续 FIFO 被违反）。"""
        n = 0
        for e in self.edges:
            for b in range(1, N_BUCKETS):
                t0 = b * BUCKET_MIN
                if self.arrival(e, t0 + 1e-6) < self.arrival(e, t0 - 1e-6) - 1e-12:
                    n += 1
        return n

    def enforce_fifo(self, samples_per_bucket: int = 2) -> int:
        """把速度剖面修成 FIFO 的，返回被下调的槽数。

        判据必须与 `check_fifo` 用同一套采样，否则「修完仍报错」——第一版就是
        这么错的：它只比相邻槽**起点**的到达时刻，等于默许每槽最多提前一整个
        槽宽；而 check_fifo 是在槽内也采样的，看到的是槽内**最后一个**采样点。
        测试 `test_check_fifo_actually_catches_a_violation` 用一条 8 km 的长边
        （行程时间跨十几个槽）把这个差别暴露了出来。

        正确条件：设采样步长 step，槽 b 内最后一个采样点在 (b+1)*B - step，
        于是下一槽首个采样点要求 tt(b+1) >= tt(b) - step —— 每槽行程时间最多
        下降一个 step。前向扫一遍，超出的就把速度压回去。
        """
        step = BUCKET_MIN / samples_per_bucket
        touched = 0
        for e in self.edges:
            km = e.length_m / 1000.0
            row = self.speed[e.eid]
            prev_tt = km / float(row[0]) * 60.0
            for b in range(1, N_BUCKETS):
                tt = km / float(row[b]) * 60.0
                floor_tt = prev_tt - step
                if tt < floor_tt - 1e-12:
                    tt = floor_tt
                    row[b] = km / tt * 60.0
                    touched += 1
                prev_tt = tt
        return touched

    # ---------------------------------------------------------------- 序列化
    @staticmethod
    def load(path: Path | None = None) -> "Network":
        """numpy 有就用 numpy，没有就用 list of list。

        路由与服务只做 `speed[eid][bucket]` 这种索引，两者都支持——所以运行
        镜像可以不装 numpy（见 Dockerfile）。numpy 只有 bench 与预测器需要。
        """
        path = Path(path or NETWORK_JSON)
        raw = json.loads(path.read_text(encoding="utf-8"))
        nodes = {n["id"]: (float(n["x"]), float(n["y"])) for n in raw["nodes"]}
        edges = [Edge(i, e["u"], e["v"], float(e["length_m"]), float(e["free_kmh"]), e["kind"])
                 for i, e in enumerate(raw["edges"])]
        rows = raw["speed_kmh"]
        assert len(rows) == len(edges) and len(rows[0]) == N_BUCKETS, "速度矩阵形状不对"
        try:
            import numpy as np
            speed = np.asarray(rows, dtype=float)
        except ImportError:                      # 运行镜像里没有 numpy
            speed = [[float(x) for x in r] for r in rows]
        return Network(nodes, edges, speed)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": ("Synthetic campus network, rebuilt 2026-09-08. The 2024 original "
                     "(Chang'an University shuttle network and its Beidou traces) was not kept; "
                     "see README section 'Provenance'."),
            "bucket_minutes": BUCKET_MIN,
            "nodes": [{"id": k, "x": v[0], "y": v[1]} for k, v in self.nodes.items()],
            "edges": [{"u": e.u, "v": e.v, "length_m": round(e.length_m, 1),
                       "free_kmh": e.free_kmh, "kind": e.kind} for e in self.edges],
            "speed_kmh": [[round(float(x), 4) for x in row] for row in self.speed],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
