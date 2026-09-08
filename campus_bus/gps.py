# -*- coding: utf-8 -*-
"""GPS 轨迹：模拟 -> 成串丢点 + 测量噪声 -> 补偿 -> 量误差。

**第一版的结论是模拟器造出来的假象，这一版把它推翻了。**第一版让车在每条边上
匀速直线行驶，于是「按时间线性插值」在数学上就是精确的——测出来 15% 丢失率下
误差 0.00 m，看着像「补偿没用」，其实是模拟器太干净。真实公交有两件事让位置
不再是时间的线性函数：

  1. **进站停靠**：到站减速、停 D 秒、再加速，位置在这段时间里是平的；
  2. **加减速**：即便不停站，起步与到达也不是瞬间到巡航速度。

再加上第三件事：**GPS 本身有测量噪声**（几米量级）。噪声一来，保留下来的点也
不再是真值，「地图匹配」这一步才有了存在的理由——把点投到路上，横向噪声直接
被消掉。

所以这一版：
  - `simulate` 用平滑加减速（smoothstep）+ 站点停靠生成真值轨迹；
  - `observe` 在真值上叠高斯噪声，再按两态马尔可夫成串丢点；
  - `compensate` 两种估计：raw 线性插值 / 地图匹配 + 单调弧长插值；
  - 误差**分别报保留点与丢失点**：前者衡量去噪，后者衡量补点。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .network import Network

DWELL_S = 18.0                      # 站点停靠时长（秒）
NOISE_M = 6.0                       # GPS 单轴标准差（米），民用量级


@dataclass
class Trajectory:
    t: np.ndarray                   # 采样时刻（分）
    xy: np.ndarray                  # (n, 2) 真值坐标
    edge_idx: np.ndarray            # 每个采样点所在边的 eid
    s_on_edge: np.ndarray           # 该边上的弧长位置（米）
    moving: np.ndarray              # 是否在动（False = 停站中）


def _smoothstep(u: float) -> float:
    """0->1 的平滑推进，两端导数为 0，即起步与到站都有加减速。"""
    return u * u * (3.0 - 2.0 * u)


def simulate(net: Network, path: list[str], depart_min: float, hz: float = 1.0,
             stops: set[str] | None = None, dwell_s: float = DWELL_S) -> Trajectory:
    """按 path 走一遍并采样真值位置。

    stops 里的中间节点会停靠 dwell_s 秒；每条边内部按 smoothstep 推进，
    所以位置**不是**时间的线性函数——这正是补偿要面对的困难。
    """
    if stops is None:                                   # 默认每隔一个节点设一站
        stops = set(path[1:-1][::2])
    ts, xys, eids, ss, mv = [], [], [], [], []
    t = depart_min
    dt = 1.0 / hz / 60.0
    for a, b in zip(path, path[1:]):
        e = next(e for e in net.out[a] if e.v == b)
        (x1, y1), (x2, y2) = net.nodes[a], net.nodes[b]
        tt = net.travel_time(e, t)
        n = max(2, int(round(tt / dt)))
        for k in range(n):
            u = _smoothstep(k / n)
            ts.append(t + k * tt / n)
            xys.append((x1 + u * (x2 - x1), y1 + u * (y2 - y1)))
            eids.append(e.eid); ss.append(u * e.length_m); mv.append(True)
        t += tt
        if b in stops:                                  # 到站停靠
            for k in range(max(1, int(round(dwell_s / 60.0 / dt)))):
                ts.append(t + k * dt)
                xys.append((x2, y2))
                eids.append(e.eid); ss.append(e.length_m); mv.append(False)
            t += dwell_s / 60.0
    return Trajectory(np.asarray(ts), np.asarray(xys), np.asarray(eids),
                      np.asarray(ss), np.asarray(mv))


def observe(traj: Trajectory, noise_m: float = NOISE_M, seed: int = 21) -> np.ndarray:
    """真值 + 各向同性高斯噪声 = 接收机报出来的位置。"""
    rng = np.random.default_rng(seed)
    return traj.xy + rng.normal(0.0, noise_m, traj.xy.shape)


def drop(traj: Trajectory, rate: float = 0.15, mean_gap: float = 6.0,
         seed: int = 7) -> np.ndarray:
    """两态马尔可夫成串丢失，返回保留点的布尔掩码。

    坏状态平均持续 mean_gap 个采样点；由整体丢失率反解进入坏状态的概率。
    真实车载定位是成串丢的（楼间、树荫、地下通道口），按独立丢点建模会低估难度。
    """
    rng = np.random.default_rng(seed)
    n = len(traj.t)
    p_leave = 1.0 / mean_gap
    p_enter = rate * p_leave / max(1e-9, 1.0 - rate)    # 稳态丢失率 = rate
    keep = np.ones(n, dtype=bool)
    bad = False
    for i in range(n):
        bad = (rng.random() > p_leave) if bad else (rng.random() < p_enter)
        keep[i] = not bad
    keep[0] = keep[-1] = True
    return keep


def _pava(y: np.ndarray) -> np.ndarray:
    """保序回归（pool-adjacent-violators）：在"不减"约束下的最小二乘拟合。

    为什么不用 `np.maximum.accumulate`：累积最大值也能让序列不减，但它只会
    **抬高**、从不下调，于是把对称的观测噪声整流成**系统性前偏**——一次偏大的
    观测会把后面所有估计一起顶上去。实测就是这个后果：用 cummax 时地图匹配在
    保留点上赢（噪声被投影消掉），在**丢失点上反而输给朴素插值**。PAVA 会把
    违反单调的一段合并取均值，既满足约束又无偏。
    """
    vals, wts = [], []                           # 每段的均值与点数
    for x in y:
        vals.append(float(x)); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-12:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    out = []
    for v_, w_ in zip(vals, wts):
        out.extend([v_] * int(round(w_)))
    return np.asarray(out[:len(y)])


def compensate(net: Network, traj: Trajectory, obs: np.ndarray, keep: np.ndarray,
               mode: str = "arclength", path: list[str] | None = None) -> np.ndarray:
    """由观测（含噪、有缺失）估计每个采样时刻的位置。

    mode="raw"       直接对观测坐标按时间线性插值：噪声原样留着，
                     跨停站的空档会把车"插着穿过"停靠时间。
    mode="arclength" 先把观测投到路径的边上（地图匹配，消掉横向噪声），
                     再沿路径累计弧长做单调插值。
    """
    out = np.array(obs, dtype=float)
    idx = np.flatnonzero(keep)
    if mode == "raw":
        out[:, 0] = np.interp(traj.t, traj.t[idx], obs[idx, 0])
        out[:, 1] = np.interp(traj.t, traj.t[idx], obs[idx, 1])
        return out

    edge_by_id = {e.eid: e for e in net.edges}
    if path:
        order = [next(e.eid for e in net.out[a] if e.v == b) for a, b in zip(path, path[1:])]
    else:
        order = list(dict.fromkeys(int(x) for x in traj.edge_idx))
    cum, total = {}, 0.0
    for eid in order:
        cum[eid] = total
        total += edge_by_id[eid].length_m
    allowed = [edge_by_id[eid] for eid in order]

    def project(p, lo: int):
        """在 order[lo-1:] 里找最近边。

        两个坑叠在一起，都踩过：
        1. 全图找最近边会跳到反向边或平行支路，弧长一乱全乱（误差比不做还大 100 倍）；
        2. 但只在 order[lo:] 里找、且 lo 只增不减，是**不可逆贪心**——6 m 噪声在
           路口附近足以把一个点投到下一条边上，此后所有点都被锁在那条边之后，
           再也回不来。实测 15% 丢失率下有 12% 的观测点被永久锁死在错误的边上。
        所以允许回退一格（lo-1），既挡住乱跳，又给噪声一次改正机会。"""
        lo = max(0, lo - 1)
        best, best_d, best_k = (order[lo], 0.0), math.inf, lo
        for k in range(lo, len(allowed)):
            e = allowed[k]
            (x1, y1), (x2, y2) = net.nodes[e.u], net.nodes[e.v]
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            s = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0]-x1)*vx + (p[1]-y1)*vy) / L2))
            d = (p[0] - (x1 + s*vx))**2 + (p[1] - (y1 + s*vy))**2
            if d < best_d:
                best_d, best, best_k = d, (e.eid, s * e.length_m), k
        return best[0], best[1], best_k

    obs_s, lo = [], 0
    for i in idx:
        eid, s, lo = project(obs[i], lo)
        obs_s.append(cum[eid] + s)
    obs_s = _pava(np.asarray(obs_s))                     # 弧长单调不减（保序回归，非 cummax）
    est_s = np.interp(traj.t, traj.t[idx], obs_s)

    for k, s_glob in enumerate(est_s):
        eid = order[-1]
        for cand in order:
            if s_glob <= cum[cand] + edge_by_id[cand].length_m:
                eid = cand
                break
        e = edge_by_id[eid]
        frac = min(1.0, max(0.0, (s_glob - cum[eid]) / e.length_m))
        (x1, y1), (x2, y2) = net.nodes[e.u], net.nodes[e.v]
        out[k] = (x1 + frac * (x2 - x1), y1 + frac * (y2 - y1))
    return out


def errors(traj: Trajectory, est: np.ndarray, keep: np.ndarray) -> dict:
    """分别报保留点与丢失点的误差：前者衡量去噪，后者衡量补点。"""
    d = np.hypot(est[:, 0] - traj.xy[:, 0], est[:, 1] - traj.xy[:, 1])
    def stat(mask):
        if not mask.any():
            return {"n": 0}
        v = d[mask]
        return {"n": int(mask.sum()), "mean_m": round(float(v.mean()), 3),
                "p95_m": round(float(np.percentile(v, 95)), 3),
                "max_m": round(float(v.max()), 3)}
    return {"kept": stat(keep), "gap": stat(~keep), "all": stat(np.ones(len(d), bool))}
