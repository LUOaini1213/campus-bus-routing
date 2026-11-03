# -*- coding: utf-8 -*-
"""跑完四组测量，把结果写进 results/。

    python bench/bench_all.py            # 不含 LSTM（无需 torch）
    python bench/bench_all.py --lstm     # 含 LSTM
    python bench/bench_all.py --check    # 只校验 results/ 与新跑一致，不覆盖

四组：
  1. 路由正确性与代价：Dijkstra / A* 对参照实现的最优率、展开节点数、查询延迟
  2. 动态松弛阈值消融：省了多少展开，丢了多少最优
  3. GPS 成串丢点补偿：平面插值 vs 弧长插值
  4. 拥堵预测：四个模型的 MAE，以及**用预测速度选路后实际多付多少行程时间**
  5. 分时权重值多少：四种「选路时看到什么」的策略，一律放回真实路网结算
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus import congestion, fleet, gps, policies, stats   # noqa: E402
from campus_bus.network import BUCKET_MIN, Network               # noqa: E402
from campus_bus.routing import (astar, brute_force, dijkstra,     # noqa: E402
                                label_correcting, path_cost)

RESULTS = ROOT / "results"
DEPARTURES = [420, 450, 480, 510, 720, 1020, 1050, 1080, 1110, 150]   # 覆盖双峰与平峰

# --- 漂移比对：序数事实严格相等，连续量留容差 -------------------------------
# 照 eda-copilot 的分层做法。那个仓的 CI 首跑就红在这上面：同一份代码在 ubuntu
# runner 上把 MRR 从 0.7130 算成 0.7109，因为两篇文档的分数只差 4e-5、BLAS 末位
# 一变就换位。这里的连续量（多付百分比、误差米数、MAE、CI 端点）同理，全字符串
# 严比等于把 CI 押在最后一位小数上。
#
# **计数与分数式字符串（"600/600"、"46/280"）反而必须严格相等**——它们是序数
# 事实，一变就说明选路真的变了，不是浮点噪声。
REL_TOL = 1e-6
# 机器相关、不参与比对的字段：延迟是墙钟，换机器必变。
EXEMPT_SUFFIX = ("latency_ms",)


def compare(a, b, path: str = "") -> list:
    """递归逐字段比对，返回路径化的差异列表（仿 malaysia-auto-ask 的 _compare）。"""
    if any(path.endswith("/" + s) for s in EXEMPT_SUFFIX):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append("%s/%s: 新增" % (path, k))
            elif k not in b:
                out.append("%s/%s: 缺失" % (path, k))
            else:
                out += compare(a[k], b[k], "%s/%s" % (path, k))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return ["%s: 长度 %d vs %d" % (path, len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare(x, y, "%s[%d]" % (path, i))
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else ["%s: %s vs %s" % (path, a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return [] if a == b else ["%s: %d vs %d" % (path, a, b)]
        if abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b)):
            return []
        return ["%s: %r vs %r" % (path, a, b)]
    return [] if a == b else ["%s: %r vs %r" % (path, a, b)]


OD_FILE = ROOT / "bench" / "od_pairs.json"


def od_pairs(net: Network, limit: int = 60, seed: int = 5):
    """取 OD 对。**优先读提交进仓的 bench/od_pairs.json。**

    原来是现场用 `np.random.default_rng(seed).choice(..., replace=False)` 抽的。
    两个问题：① `Generator` 的流跨 numpy 大版本不承诺稳定（NEP 19），CI 上装的
    是最新 numpy，一旦 `choice` 的实现变了，**所有 OD 全变**，routing / policies /
    threshold 三组结论连同 `--check` 一起红，而根因与代码无关；② 它不去重，
    seed=5 实际抽出 3 组重复 OD，`bench_routing` 因此有 30 个重复查询，而
    `policies.compare` 靠字典键意外去重了——两组口径不一致。

    现在把抽样结果一次性固化进仓，运行时只读文件。文件缺失时才回退到抽样
    （并去重），用于首次生成。
    """
    key = "%d_%d" % (limit, seed)
    if OD_FILE.exists():
        book = json.loads(OD_FILE.read_text(encoding="utf-8"))
        if key in book:
            return [tuple(p) for p in book[key]]
    rng = np.random.default_rng(seed)
    names = sorted(net.nodes)
    out, seen = [], set()
    while len(out) < limit:
        a, b = rng.choice(names, 2, replace=False)
        if (str(a), str(b)) not in seen:
            seen.add((str(a), str(b)))
            out.append((str(a), str(b)))
    return out


def bench_routing(net: Network) -> dict:
    pairs = od_pairs(net)
    n = ok_d = ok_a = ok_p = ok_b = 0
    exp_d, exp_a, exp_r, exp_p, lat_ms = [], [], [], [], []
    for s, t in pairs:
        for dep in DEPARTURES:
            ref = label_correcting(net, s, t, dep)
            if ref.arrival == float("inf"):
                continue
            t0 = time.perf_counter()
            d = dijkstra(net, s, t, dep)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            a = astar(net, s, t, dep)
            pr = dijkstra(net, s, t, dep, prune=True)      # 上界剪枝
            bf = brute_force(net, s, t, dep)               # 简单路径穷举，不假设 FIFO
            n += 1
            ok_b += abs(bf.arrival - ref.arrival) < 1e-9
            ok_d += abs(d.arrival - ref.arrival) < 1e-9
            ok_a += abs(a.arrival - ref.arrival) < 1e-9
            ok_p += abs(pr.arrival - ref.arrival) < 1e-9
            exp_p.append(pr.expansions)
            # 路径与标签必须自洽：沿返回的路径真的走一遍应当同样到达
            assert abs(path_cost(net, d.path, dep) - d.arrival) < 1e-9, (s, t, dep)
            exp_d.append(d.expansions); exp_a.append(a.expansions); exp_r.append(ref.expansions)
    return {
        "queries": n,
        "dijkstra_optimal": "%d/%d" % (ok_d, n),
        "astar_optimal": "%d/%d" % (ok_a, n),
        "pruned_optimal": "%d/%d" % (ok_p, n),
        "brute_force_agrees": "%d/%d" % (ok_b, n),
        "expansions_mean": {"dijkstra": round(statistics.mean(exp_d), 2),
                            "astar": round(statistics.mean(exp_a), 2),
                            "pruned": round(statistics.mean(exp_p), 2),
                            "label_correcting": round(statistics.mean(exp_r), 2)},
        "astar_expansion_saving_pct": round(
            (1 - statistics.mean(exp_a) / statistics.mean(exp_d)) * 100, 2),
        "prune_expansion_saving_pct": round(
            (1 - statistics.mean(exp_p) / statistics.mean(exp_d)) * 100, 2),
        "latency_ms": {"p50": round(statistics.median(lat_ms), 4),
                       "p95": round(float(np.percentile(lat_ms, 95)), 4),
                       "p99": round(float(np.percentile(lat_ms, 99)), 4),
                       "max": round(max(lat_ms), 4)},
    }


def bench_threshold(net: Network) -> dict:
    pairs = od_pairs(net, limit=40, seed=6)
    rows = []
    for thr in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        n = ok = 0
        exp, loss = [], []
        for s, t in pairs:
            for dep in DEPARTURES[:6]:
                ref = label_correcting(net, s, t, dep)
                if ref.arrival == float("inf"):
                    continue
                r = dijkstra(net, s, t, dep, relax_threshold=thr)
                n += 1
                ok += abs(r.arrival - ref.arrival) < 1e-9
                exp.append(r.expansions)
                loss.append(max(0.0, r.arrival - ref.arrival))
        rows.append({"threshold_min": thr, "optimal": "%d/%d" % (ok, n),
                     "optimal_pct": round(ok / n * 100, 2),
                     "expansions_mean": round(statistics.mean(exp), 2),
                     "mean_loss_min": round(statistics.mean(loss), 4),
                     "max_loss_min": round(max(loss), 4)})
    # 把阈值开到天文数字：等于让算法彻底退化成「首次看到的标签即终值」。
    # 展开数**仍然不降**，这证明这个消融在数学上就不可能有正收益。
    degenerate = []
    for s, t in pairs:
        for dep in DEPARTURES[:6]:
            r = dijkstra(net, s, t, dep, relax_threshold=1e9)
            if r.arrival < float("inf") or True:
                degenerate.append(r.expansions)
    rows.append({"threshold_min": 1e9, "optimal": "(退化)", "optimal_pct": None,
                 "expansions_mean": round(statistics.mean(degenerate), 2),
                 "mean_loss_min": None, "max_loss_min": None})
    return {"rows": rows,
            "note": ("**这个消融是无效的，而且无效性是实现位置决定的。**阈值加在松弛条件 "
                     "`arr < cur - thr` 上，而节点首次被发现时 cur = INF，INF - thr 仍是 INF，"
                     "所以它永远挡不住首次入堆——展开数的下界恒等于可达节点数。把阈值开到 1e9 "
                     "（彻底退化）展开仍几乎不降，就是证据。真正能减少展开的是**上界剪枝**"
                     "（dijkstra(..., prune=True)），见 routing 组的 prune_expansion_saving_pct。")}


def bench_gps(net: Network) -> dict:
    path = ["west_gate", "hub_w", "hub_n", "hub_e", "dorm_c"]
    traj = gps.simulate(net, path, 470.0, hz=1.0)
    obs = gps.observe(traj)
    out = {"n_points": int(len(traj.t)), "path": path,
           "dwell_points": int((~traj.moving).sum()),
           "noise_sigma_m": gps.NOISE_M, "dwell_s": gps.DWELL_S,
           "raw_observation_error": gps.errors(traj, obs, np.ones(len(traj.t), bool))["all"]}
    for rate in (0.05, 0.15, 0.30):
        keep = gps.drop(traj, rate=rate)
        row = {"target_rate": rate, "actual_rate": round(float(1 - keep.mean()), 4),
               "longest_gap_points": int(_longest_gap(keep))}
        for mode in ("raw", "arclength"):
            row[mode] = gps.errors(traj, gps.compensate(net, traj, obs, keep, mode, path), keep)
        out["rate_%d" % int(rate * 100)] = row
    return out


def _longest_gap(keep) -> int:
    best = cur = 0
    for k in keep:
        cur = 0 if k else cur + 1
        best = max(best, cur)
    return best


def bench_congestion(net: Network, with_lstm: bool) -> dict:
    hist = congestion.make_history(net)
    models = ["persistence", "seasonal", "ridge"] + (["lstm"] if with_lstm else [])
    acc = [congestion.evaluate(hist, m) for m in models]

    # 第二把尺子：拿预测速度选路，放回**被预测的那一天的真实速度**里走。
    #
    # 这里改过一次口径。第一版结算用的是 `net.speed`（基础剖面），而预测器
    # 预测的是 hist[-1]（= 基础剖面 × 当日系数 + 事件 + 噪声）。两者不是同一个
    # 量，后果是：一个**完美预测第 20 天**的 oracle 在那个评测里仍会有非零多付，
    # 而输出越平滑、越接近基础剖面的模型越占便宜——这恰好解释了为什么 MAE 最差
    # 的 seasonal 在选路上反而最好。那个「预测得准 ≠ 路选得对」的结论因此**至少
    # 一部分是口径错配造成的**，不是纯粹的方法学洞见。现在结算真值改成 hist[-1]。
    truth_day = hist[-1].T                       # (n_edges, N_BUCKETS)
    saved = net.speed
    net.speed = truth_day
    pairs = od_pairs(net, limit=30, seed=9)
    # 出发时刻用**全部 10 个**，不再只取前 5 个。前 5 个是 7:00/7:30/8:00/8:30/12:00，
    # 晚高峰与深夜一个都没有，README 却让读者以为覆盖全天；顺带把样本量从 150 抬到 300。
    keys = [(s, t, dep) for s, t in pairs for dep in DEPARTURES]
    per_item = {}
    for m in models:
        grid = congestion.predicted_speed_grid(hist, m)          # (T, E)
        net.speed = grid.T
        chosen = {k: dijkstra(net, k[0], k[1], k[2]).path for k in keys}
        net.speed = truth_day
        vals = {}
        for k, p in chosen.items():
            s, t, dep = k
            best = label_correcting(net, s, t, dep).arrival
            got = path_cost(net, p, dep) if p else float("inf")
            if best < float("inf") and got < float("inf") and best > dep:
                vals[k] = (got - best) / (best - dep) * 100.0
        per_item[m] = vals

    common = sorted(set.intersection(*[set(v) for v in per_item.values()]))
    routed = [{"model": m, "n": len(common),
               "realised_excess_pct": round(statistics.mean([per_item[m][k] for k in common]), 4),
               "worst_excess_pct": round(max(per_item[m][k] for k in common), 4)}
              for m in models]

    # 配对自助：差距是真的还是噪声？重抽的是 (OD, 时刻) 这一维。
    # 用 campus_bus.stats 的纯标准库实现，不用 numpy——见该模块 docstring。
    best_model = min(models, key=lambda m: statistics.mean([per_item[m][k] for k in common]))
    boot = []
    for m in models:
        if m == best_model:
            continue
        d = [per_item[m][k] - per_item[best_model][k] for k in common]
        _, lo, hi = stats.paired_bootstrap(d)
        k_diff = stats.n_differing(d)
        boot.append({"worse_model": m, "vs": best_model,
                     "mean_diff_pct_points": round(sum(d) / len(d), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "verdict": stats.verdict(lo, hi),
                     "significant": bool(lo > 0),
                     # 两个模型选出同一条路的比例，以及**真正有分歧的配对数**。
                     # 后者是解读 CI 的前提：k 很小时 CI 下界会恰好等于 0，
                     # 那是离散性伪影而不是估计值（见 stats.n_differing）。
                     "n_differing": k_diff,
                     "identical_route_share": round(1 - k_diff / len(d), 4)})

    net.speed = saved
    return {"accuracy": acc, "routing": routed, "paired_bootstrap": boot,
            "note": ("速度 MAE 与「路选得好不好」不是一回事：MAE 更低的模型，"
                     "如果没改变选路，实现行程时间一分钟也省不下来。"
                     "配对自助（10000 次，重抽 OD×时刻）用来判断名次差是不是噪声。"
                     "结算真值是 hist[-1]（被预测的那一天），不是基础剖面——"
                     "第一版用错了，见 bench_congestion 里的注释。")}


def bench_policies(net: Network) -> dict:
    """分时到底值多少：高峰、平峰分开看。"""
    pairs = od_pairs(net, limit=60, seed=13)
    peak = policies.compare(net, pairs, [440, 460, 480, 1050, 1070])
    off = policies.compare(net, pairs, [120, 150, 210, 780, 810])
    return {"peak": peak, "offpeak": off,
            "note": ("选路时看到的边权不同，结算一律放回真实分时路网。"
                     "free_flow 只看限速，daily_mean 用全天均速，"
                     "peak_profile 把高峰那一档当全天常数，time_dep 才真按出发时刻查。")}


def bench_fleet(net: Network) -> dict:
    """用本网上一条环线的真实运行时间，算车队规模的台阶。"""
    # 一条覆盖全校的班车环线：西门 - 宿舍 - 教学 - 图书馆 - 东门 - 体育场 - 回西门
    loop = ["west_gate", "hub_w", "dorm_a", "hub_w", "hub_n", "teach_1", "hub_n",
            "library", "hub_e", "dorm_c", "hub_e", "east_gate", "hub_e", "hub_s",
            "teach_4", "hub_s", "stadium", "hub_s", "hub_w", "west_gate"]
    peak = path_cost(net, loop, 460.0) - 460.0
    off = path_cost(net, loop, 150.0) - 150.0
    # 环线是手写的节点序列，路网一改就可能断。断了要当场说清楚是哪一段，
    # 而不是把 inf 传下去让 fleet_size 抛 OverflowError（改 admin 那条边时就这么炸过）。
    if peak == float("inf") or off == float("inf"):
        missing = [(a, b) for a, b in zip(loop, loop[1:])
                   if not any(e.v == b for e in net.out[a])]
        raise ValueError("环线不连通，缺这些有向边：%s" % missing)
    return {"loop": loop,
            "cycle_peak_min": round(peak, 3), "cycle_offpeak_min": round(off, 3),
            "headway_min": 4.0, "layover_min": 3.0,
            "peak": fleet.breakeven(peak, 4.0, 3.0),
            "ladder_peak": fleet.ladder(peak, 4.0, 3.0),
            "note": ("车辆数是天花板函数，按台阶掉。行程时间降多少与少几辆车"
                     "不是线性关系——这也是原始简历里「12 → 10」需要单独交代的原因。")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", action="store_true", help="加跑 LSTM（需要 torch）")
    ap.add_argument("--check", action="store_true", help="只比对，不覆盖 results/")
    args = ap.parse_args()

    net = Network.load()
    fresh = {
        "network": {"nodes": net.n_nodes(), "edges": net.n_edges(),
                    "bucket_minutes": BUCKET_MIN,
                    # 采样器看得见的违反数（0）与连续意义上的真实违反：后者才是
                    # 事实，前者只是分辨率。两个都报，README 才不会再说错。
                    "fifo_violations_sampled": len(net.check_fifo()),
                    "boundary_drops": net.boundary_drop_count(),
                    "max_boundary_drop_min": round(net.max_boundary_drop(), 4)},
        "routing": bench_routing(net),
        "relax_threshold": bench_threshold(net),
        "gps": bench_gps(net),
        "congestion": bench_congestion(net, args.lstm),
        "policies": bench_policies(net),
        "fleet": bench_fleet(net),
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("bench_with_lstm.json" if args.lstm else "bench.json")
    if args.check:
        if not out.exists():
            print("missing", out); return 1
        old = json.loads(out.read_text(encoding="utf-8"))
        drift = compare(old, fresh)
        if drift:
            print("DRIFT (%d 处)：" % len(drift))
            for line in drift[:20]:
                print("  " + line)
            return 1
        print("ok: results 与新跑一致（序数严格、连续量容差 %g、延迟豁免）" % REL_TOL)
        return 0

    out.write_text(json.dumps(fresh, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    print(json.dumps({"routing": fresh["routing"], "fleet_peak": fresh["fleet"]["peak"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
