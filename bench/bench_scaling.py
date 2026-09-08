# -*- coding: utf-8 -*-
"""规模实验：23 节点太小，小到看不出任何算法选择的差别。

原始简历写「用二叉堆把复杂度优化到 O(|E| + |V| log |V|)」。在 23 节点上这句
话无法证伪——总共才展开十来个节点，用列表线性扫也一样快。所以这里把同一套
代码放到 10² ~ 10⁴ 节点的网格路网上，量三件事：

  1. 堆 vs 线性扫描的实际差距，看它从多大规模开始才值得说；
  2. A* 相对 Dijkstra 省下的展开随规模怎么变；
  3. 动态松弛阈值在大网上是否终于开始划算（小网上的结论是「不划算」）。

网格路网是合成的、确定性生成的，只用来做规模对照，不参与其它结论。

    python bench/bench_scaling.py --out results/scaling.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus.network import N_BUCKETS, Edge, Network        # noqa: E402
from campus_bus.routing import astar, dijkstra, label_correcting  # noqa: E402

INF = float("inf")


def grid_network(side: int, seed: int = 3) -> Network:
    """side × side 的网格，四邻接双向边，边长与限速带确定性扰动。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    nodes, name = {}, {}
    for i in range(side):
        for j in range(side):
            n = "n%d_%d" % (i, j)
            name[(i, j)] = n
            nodes[n] = (i * 120.0, j * 120.0)
    edges, eid = [], 0
    for i in range(side):
        for j in range(side):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= side or b[1] >= side:
                    continue
                L = 120.0 * float(rng.uniform(0.85, 1.3))
                v = float(rng.choice([20.0, 30.0, 40.0]))
                edges.append(Edge(eid, name[a], name[b], L, v, "grid")); eid += 1
                edges.append(Edge(eid, name[b], name[a], L, v, "grid")); eid += 1
    speed = np.empty((len(edges), N_BUCKETS))
    for e in edges:                       # 双峰拥堵，与主网同形状
        phase = float(rng.uniform(-15, 15))
        for b in range(N_BUCKETS):
            t = b * (1440 / N_BUCKETS) + phase
            f = math.exp(-0.5 * ((t - 460) / 45) ** 2) + 0.9 * math.exp(-0.5 * ((t - 1070) / 55) ** 2)
            speed[e.eid][b] = e.free_kmh * (1.0 - 0.55 * min(1.0, f))
    net = Network(nodes, edges, speed)
    net.enforce_fifo()
    assert not net.check_fifo()
    return net


def dijkstra_linear(net: Network, src: str, dst: str, depart: float):
    """同一个算法，但用线性扫描找最小标签——用来隔离「堆」这一个变量。"""
    best = {src: depart}
    done: set[str] = set()
    expansions = 0
    while True:
        u, tu = None, INF
        for k, v in best.items():                 # O(V) 找最小
            if k not in done and v < tu:
                u, tu = k, v
        if u is None:
            break
        done.add(u); expansions += 1
        if u == dst:
            break
        for e in net.out[u]:
            if e.v in done:
                continue
            arr = net.arrival(e, tu)
            if arr < best.get(e.v, INF):
                best[e.v] = arr
    return best.get(dst, INF), expansions


def _pairs(net: Network, k: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    names = sorted(net.nodes)
    return [tuple(rng.choice(names, 2, replace=False)) for _ in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", type=int, nargs="+", default=[10, 20, 40, 70, 100])
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling.json"))
    args = ap.parse_args()

    rows = []
    for side in args.sides:
        net = grid_network(side)
        V, E = net.n_nodes(), net.n_edges()
        pairs = _pairs(net, args.queries, seed=side)
        t_heap, t_lin, exp_d, exp_a, ok = [], [], [], [], 0
        thr_exp, thr_ok = [], 0
        for s, t in pairs:
            t0 = time.perf_counter(); d = dijkstra(net, s, t, 460.0); t_heap.append(time.perf_counter() - t0)
            if V <= 4900:                       # 线性扫描在 10⁴ 节点上太慢，只跑到中等规模
                t0 = time.perf_counter(); arr_l, _ = dijkstra_linear(net, s, t, 460.0)
                t_lin.append(time.perf_counter() - t0)
                assert abs(arr_l - d.arrival) < 1e-9, "堆版与线性版结果必须一致"
            a = astar(net, s, t, 460.0)
            ok += abs(a.arrival - d.arrival) < 1e-9
            exp_d.append(d.expansions); exp_a.append(a.expansions)
            r = dijkstra(net, s, t, 460.0, relax_threshold=0.25)
            thr_exp.append(r.expansions); thr_ok += abs(r.arrival - d.arrival) < 1e-9
        row = {
            "side": side, "nodes": V, "edges": E,
            "heap_ms_mean": round(statistics.mean(t_heap) * 1000, 4),
            "linear_ms_mean": round(statistics.mean(t_lin) * 1000, 4) if t_lin else None,
            "heap_speedup_x": round(statistics.mean(t_lin) / statistics.mean(t_heap), 2) if t_lin else None,
            "expansions": {"dijkstra": round(statistics.mean(exp_d), 1),
                           "astar": round(statistics.mean(exp_a), 1)},
            "astar_saving_pct": round((1 - statistics.mean(exp_a) / statistics.mean(exp_d)) * 100, 2),
            "astar_optimal": "%d/%d" % (ok, len(pairs)),
            "threshold_0p25": {"expansions": round(statistics.mean(thr_exp), 1),
                               "saving_pct": round((1 - statistics.mean(thr_exp) / statistics.mean(exp_d)) * 100, 2),
                               "optimal": "%d/%d" % (thr_ok, len(pairs))},
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out = {"rows": rows,
           "note": ("网格路网仅用于规模对照，不参与其它结论。线性扫描版只跑到 "
                    "4900 节点，再大就太慢了。")}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
