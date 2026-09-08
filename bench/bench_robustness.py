# -*- coding: utf-8 -*-
"""跨种子稳健性：换一张随机生成的路网，结论还成立吗？

所有结论都建立在**一个**合成路网上（`SEED = 20241001`），这是合成数据仓库最容易
被攻击的地方——"你是不是挑了个好看的种子"。这个脚本换 8 个种子重建整张网，
把三组结论各跑一遍，报**跨种子的区间**而不是单点值。

判据是**结论的方向**，不是具体数值：数值随种子变是正常的，结论跟着变才是问题。
对应的测试断言的也是方向（见 tests/test_all.py::Robustness）。

    python bench/bench_robustness.py            # -> results/robustness.json
    python bench/bench_robustness.py --check
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus import policies                                  # noqa: E402
from campus_bus.routing import dijkstra, label_correcting        # noqa: E402
from bench.bench_all import DEPARTURES, compare, od_pairs        # noqa: E402
from scripts.build_network import build                          # noqa: E402

SEEDS = [20241001, 7, 99, 2026, 31337, 4242, 8675309, 20260908]
DEP_PEAK = [440, 460, 480, 1050, 1070]
DEP_OFF = [120, 150, 210, 780, 810]


def one_seed(seed: int) -> dict:
    net = build(seed)
    pairs = od_pairs(net, limit=60, seed=13)
    pk = policies.compare(net, pairs, DEP_PEAK)
    of = policies.compare(net, pairs, DEP_OFF)

    # 求解器最优率 + 剪枝是否仍无损
    rp = od_pairs(net, limit=40, seed=6)
    n = ok_d = ok_p = 0
    exp0 = expP = expHuge = 0
    for s, t in rp:
        for dep in DEPARTURES[:6]:
            ref = label_correcting(net, s, t, dep)
            if ref.arrival == float("inf"):
                continue
            d = dijkstra(net, s, t, dep)
            p = dijkstra(net, s, t, dep, prune=True)
            n += 1
            ok_d += abs(d.arrival - ref.arrival) < 1e-9
            ok_p += abs(p.arrival - ref.arrival) < 1e-9
            exp0 += d.expansions
            expP += p.expansions
            expHuge += dijkstra(net, s, t, dep, relax_threshold=1e9).expansions
    return {
        "seed": seed,
        "fifo_sampled_violations": len(net.check_fifo()),
        "boundary_drops": net.boundary_drop_count(),
        "max_boundary_drop_min": round(net.max_boundary_drop(), 4),
        "peak_free_flow_mean_pct": pk["free_flow"]["mean_excess_pct"],
        "peak_free_flow_max_pct": pk["free_flow"]["max_excess_pct"],
        "peak_free_flow_worse": pk["free_flow"]["routes_worse_than_optimal"],
        "offpeak_free_flow_mean_pct": of["free_flow"]["mean_excess_pct"],
        "offpeak_peak_profile_mean_pct": of["peak_profile"]["mean_excess_pct"],
        "offpeak_peak_profile_max_pct": of["peak_profile"]["max_excess_pct"],
        "dijkstra_optimal": "%d/%d" % (ok_d, n),
        "pruned_optimal": "%d/%d" % (ok_p, n),
        "prune_saving_pct": round((1 - expP / exp0) * 100, 2),
        "threshold_1e9_saving_pct": round((1 - expHuge / exp0) * 100, 2),
    }


def span(rows, key):
    v = [r[key] for r in rows]
    return {"min": min(v), "median": round(statistics.median(v), 4), "max": max(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "robustness.json"))
    args = ap.parse_args()

    rows = [one_seed(s) for s in SEEDS]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))

    fresh = {
        "seeds": SEEDS,
        "rows": rows,
        "spans": {k: span(rows, k) for k in (
            "peak_free_flow_mean_pct", "peak_free_flow_max_pct",
            "offpeak_free_flow_mean_pct",
            "offpeak_peak_profile_mean_pct", "offpeak_peak_profile_max_pct",
            "prune_saving_pct", "threshold_1e9_saving_pct",
            "boundary_drops", "max_boundary_drop_min")},
        "invariants": {
            "dijkstra_always_optimal": all(r["dijkstra_optimal"].split("/")[0]
                                           == r["dijkstra_optimal"].split("/")[1] for r in rows),
            "pruning_always_lossless": all(r["pruned_optimal"].split("/")[0]
                                           == r["pruned_optimal"].split("/")[1] for r in rows),
            "offpeak_free_flow_always_exact": all(r["offpeak_free_flow_mean_pct"] == 0.0
                                                  for r in rows),
            "peak_free_flow_always_wrong_somewhere": all(
                int(r["peak_free_flow_worse"].split("/")[0]) >= 30 for r in rows),
            "threshold_never_saves_2pct": all(r["threshold_1e9_saving_pct"] < 2.0 for r in rows),
            "fifo_sampled_always_zero": all(r["fifo_sampled_violations"] == 0 for r in rows),
            "boundary_drops_always_nonzero": all(r["boundary_drops"] > 0 for r in rows),
        },
        "note": ("换 8 个种子重建路网。**判据是结论的方向，不是数值**——数值随种子变是"
                 "正常的。invariants 里每一条都是「在所有种子上都成立」的断言，"
                 "tests/test_all.py::Robustness 会读这个文件复核。"),
    }

    out = Path(args.out)
    if args.check:
        if not out.exists():
            print("missing", out); return 1
        drift = compare(json.loads(out.read_text(encoding="utf-8")), fresh)
        if drift:
            print("DRIFT (%d 处)：" % len(drift))
            for line in drift[:20]:
                print("  " + line)
            return 1
        print("ok: robustness.json 与新跑一致")
        return 0

    out.write_text(json.dumps(fresh, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", out)
    print(json.dumps(fresh["invariants"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
