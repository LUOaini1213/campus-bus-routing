# -*- coding: utf-8 -*-
"""分时最早到达路由：二叉堆 Dijkstra、A*、以及一个用来判对错的参照实现。

三个求解器解同一个问题（给定出发时刻的最早到达），但可信度不同：

  label_correcting  标签修正（Bellman-Ford 式），不依赖 Dijkstra 的最优性前提，
                    在 FIFO 网络上必然最优 —— 拿它当**参照答案**。
  dijkstra          二叉堆分时 Dijkstra，O((V+E) log V)。要判它对不对，
                    就是跟上面那个比。
  astar             同一个堆，加一个可采纳启发式 h = 欧氏距离 / 最高限速。
                    只减少展开的节点数，不改变答案。

`relax_threshold` 是「动态松弛阈值」：改进小于该阈值（分钟）的松弛直接丢弃。
它能少展开一些节点，代价是可能错过最优解——所以它默认关闭，
`bench/bench_all.py::bench_threshold` 把「省了多少展开 / 丢了多少最优」一起量出来。
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass

from .network import Network

INF = float("inf")


@dataclass
class Result:
    arrival: float                  # 到达时刻（分）
    path: list[str]                 # 节点序列
    expansions: int                 # 从堆里弹出并展开的节点数
    settled: int                    # 有限标签的节点数

    # 曾经这里有一个 `duration` 属性，返回的却是 arrival —— 名实不符，
    # 且全仓无人调用。Result 不带出发时刻，正确的实现也写不出来，故删除。
    # 需要行程时间的地方一律显式写 `arrival - depart`。


def _rebuild(prev: dict, src: str, dst: str) -> list[str]:
    if dst not in prev and dst != src:
        return []
    out, cur = [dst], dst
    while cur != src:
        cur = prev[cur]
        out.append(cur)
    out.reverse()
    return out


def dijkstra(net: Network, src: str, dst: str, depart_min: float,
             relax_threshold: float = 0.0, prune: bool = False) -> Result:
    """分时 Dijkstra。标签是「最早到达时刻」，不是「距离」。

    `relax_threshold` 是「动态松弛阈值」，**它省不下任何展开，这是它的实现
    位置决定的、不是实验结论**：阈值加在松弛条件上，而一个节点首次被发现时
    `cur = INF`，`INF - thr` 仍是 INF，所以它永远挡不住首次入堆；展开数的下界
    恒等于可达节点数。把它开到 1e9（等于让算法彻底退化）实测展开仍是 12.201，
    只比 0 少 0.4%。留着它是为了把这个教训摆在明处，默认 0。

    `prune` 才是真正能减少展开的近似：用 A* 的可采纳启发式做**上界剪枝**
    —— 当 `arr + h(v) >= 已知的 dst 最优到达` 时，这个节点不可能改进答案，
    直接不入堆。因为 h 可采纳，剪掉的都是真的没用的，**不损失最优性**。
    """
    def h(n: str) -> float:
        return net.euclid_m(n, dst) / 1000.0 / net.max_kmh * 60.0

    best = {src: depart_min}
    prev: dict[str, str] = {}
    done: set[str] = set()
    heap = [(depart_min, src)]
    expansions = 0
    ub = INF                                    # 已知的 dst 最优到达（剪枝上界）
    while heap:
        t, u = heapq.heappop(heap)
        if u in done:
            continue
        if prune and t >= ub:
            continue
        done.add(u)
        expansions += 1
        if u == dst:
            break
        for e in net.out[u]:
            if e.v in done:
                continue
            arr = net.arrival(e, t)
            cur = best.get(e.v, INF)
            if arr < cur - relax_threshold:
                if e.v == dst:
                    ub = min(ub, arr)
                elif prune and arr + h(e.v) >= ub:
                    continue                     # 不可能改进答案，不入堆
                best[e.v] = arr
                prev[e.v] = u
                heapq.heappush(heap, (arr, e.v))
    return Result(best.get(dst, INF), _rebuild(prev, src, dst), expansions, len(done))


def brute_force(net: Network, src: str, dst: str, depart_min: float,
                max_nodes: int = 8) -> Result:
    """枚举全部简单路径取最早到达。**唯一不假设 FIFO 的真值来源。**

    另外三个求解器都遵循「每个节点只保留最早标签」，在非 FIFO 网上会**一起**
    给出次优解——它们互相印证不了这一点，因为共享同一个假设。只有穷举能。
    仅供小图（默认 ≤ 8 节点）使用。
    """
    if net.n_nodes() > max_nodes:
        raise ValueError("brute_force 只用于 <= %d 节点的小图，当前 %d"
                         % (max_nodes, net.n_nodes()))
    best, best_path = INF, []
    others = [n for n in sorted(net.nodes) if n not in (src, dst)]
    for r in range(len(others) + 1):
        for mid in itertools.permutations(others, r):
            p = [src] + list(mid) + [dst]
            c = path_cost(net, p, depart_min)
            if c < best:
                best, best_path = c, p
    return Result(best, best_path, 0, 0)


def astar(net: Network, src: str, dst: str, depart_min: float,
          relax_threshold: float = 0.0) -> Result:
    """A*：h(v) = 欧氏距离(v, dst) / 最高限速，可采纳（不会高估）。"""
    def h(n: str) -> float:
        return net.euclid_m(n, dst) / 1000.0 / net.max_kmh * 60.0

    best = {src: depart_min}
    prev: dict[str, str] = {}
    done: set[str] = set()
    heap = [(depart_min + h(src), depart_min, src)]
    expansions = 0
    while heap:
        _, t, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        expansions += 1
        if u == dst:
            break
        for e in net.out[u]:
            if e.v in done:
                continue
            arr = net.arrival(e, t)
            if arr < best.get(e.v, INF) - relax_threshold:
                best[e.v] = arr
                prev[e.v] = u
                heapq.heappush(heap, (arr + h(e.v), arr, e.v))
    return Result(best.get(dst, INF), _rebuild(prev, src, dst), expansions, len(done))


def label_correcting(net: Network, src: str, dst: str, depart_min: float) -> Result:
    """参照实现：不做「弹出即定终身」的假设，反复松弛直到不再改进。"""
    best = {n: INF for n in net.nodes}
    best[src] = depart_min
    prev: dict[str, str] = {}
    queue = deque([src])                # deque 而非 list：list.pop(0) 是 O(n)，
    in_q = {src}                        # 这是参照实现，在网格网上会被调用很多次
    expansions = 0
    while queue:
        u = queue.popleft()
        in_q.discard(u)
        expansions += 1
        for e in net.out[u]:
            arr = net.arrival(e, best[u])
            if arr < best[e.v] - 1e-12:
                best[e.v] = arr
                prev[e.v] = u
                if e.v not in in_q:
                    queue.append(e.v)
                    in_q.add(e.v)
    return Result(best[dst], _rebuild(prev, src, dst), expansions,
                  sum(1 for v in best.values() if v < INF))


def path_cost(net: Network, path: list[str], depart_min: float) -> float:
    """沿给定节点序列实际走一遍，返回到达时刻。用来验证路径与标签自洽。"""
    if not path:
        return INF
    t = depart_min
    for a, b in zip(path, path[1:]):
        cand = [e for e in net.out[a] if e.v == b]
        if not cand:
            return INF
        t = net.arrival(min(cand, key=lambda e: net.travel_time(e, t)), t)
    return t


SOLVERS = {"dijkstra": dijkstra, "astar": astar, "label_correcting": label_correcting,
           "brute_force": brute_force}
