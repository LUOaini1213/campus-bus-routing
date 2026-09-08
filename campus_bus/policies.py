# -*- coding: utf-8 -*-
"""四种选路策略，回答一个本该最先回答的问题：分时权重到底值多少？

前面的 bench 都在比「算得对不对、算得快不快」，但没有比**分时这件事本身
值不值**。如果拿自由流速度选出来的路和拿分时速度选出来的路是同一条，那
整套分时机制就是白搭。

四种策略选路时看到的边权不同，但**一律放回真实的分时路网里走一遍**来结算：

  free_flow    只看限速（length / free_kmh），完全不知道拥堵
  daily_mean   每条边用全天平均速度，知道哪条路整体慢，但不知道什么时候慢
  peak_profile 用早高峰那一档的速度当常数，知道高峰有多堵，但当成全天如此
  time_dep     真正按出发时刻查分时剖面（本仓默认）

结算指标是**实现行程时间**，不是选路时以为的行程时间——后者是自欺欺人。
"""
from __future__ import annotations

from .network import BUCKET_MIN, N_BUCKETS, Network
from .routing import dijkstra, path_cost

PEAK_BUCKET = 460 // BUCKET_MIN          # 早高峰 7:40 所在槽


def _constant_speed_view(net: Network, kmh_per_edge) -> Network:
    """造一份速度恒定的同构路网，用来模拟「看不见分时」的策略。"""
    speed = [[float(kmh_per_edge[e.eid])] * N_BUCKETS for e in net.edges]
    return Network(net.nodes, net.edges, speed)


def views(net: Network) -> dict:
    """返回 {策略名: 用于选路的路网}。time_dep 就是原网。"""
    free = {e.eid: e.free_kmh for e in net.edges}
    mean = {e.eid: sum(net.speed[e.eid]) / N_BUCKETS for e in net.edges}
    peak = {e.eid: float(net.speed[e.eid][PEAK_BUCKET]) for e in net.edges}
    return {
        "free_flow": _constant_speed_view(net, free),
        "daily_mean": _constant_speed_view(net, mean),
        "peak_profile": _constant_speed_view(net, peak),
        "time_dep": net,
    }


def compare(net: Network, pairs, departures) -> dict:
    """每个策略在每个 (OD, 出发时刻) 上选路，再放回真实路网结算。"""
    truth_best = {}
    for s, t in pairs:
        for dep in departures:
            r = dijkstra(net, s, t, dep)
            if r.arrival < float("inf"):
                truth_best[(s, t, dep)] = r.arrival - dep

    out = {}
    for name, view in views(net).items():
        excess, worse, same, n = [], 0, 0, 0
        for (s, t, dep), best in truth_best.items():
            p = dijkstra(view, s, t, dep).path
            if not p:
                continue
            got = path_cost(net, p, dep) - dep          # 真实路网里走
            if got == float("inf"):
                continue
            n += 1
            e = (got - best) / best * 100.0
            excess.append(e)
            if e > 1e-9:
                worse += 1
            else:
                same += 1
        out[name] = {
            "n": n,
            "mean_excess_pct": round(sum(excess) / len(excess), 4),
            "p95_excess_pct": round(sorted(excess)[int(len(excess) * 0.95)], 4),
            "max_excess_pct": round(max(excess), 4),
            "routes_worse_than_optimal": "%d/%d" % (worse, n),
            "routes_optimal": "%d/%d" % (same, n),
        }
    return out
