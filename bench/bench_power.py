# -*- coding: utf-8 -*-
"""自助功效：把「不显著」背后那个离散性伪影算成一个数。

    python bench/bench_power.py              # 打印并写 results/power.json
    python bench/bench_power.py --check      # 与已提交结果比对

README 第 6 节原来写的是「这么小的 k，自助重抽**有可观概率**一个都没抽到，CI 下界
会恰好等于 0.0」。那句话是对的，但它是全仓唯一一条只有叙述、没有数的结论——在一个
通篇「每条结论对应一份可复跑产物」的仓里，「可观」两个字自己就是个洞。这个数当场
可算，而且算法一行：

    P(一个分歧配对都没抽到) = ((n - k) / n) ** n

`campus_bus.stats.paired_bootstrap` 重抽的是**配对**：每次有放回地抽 n 个配对索引。
只有 k 个配对的差值非 0，其余全是 0（两个模型选了同一条路，实现行程时间逐位相同）。
所以**只要这 n 次抽签一个都没落在那 k 个上，这次重抽的均值就恰好是 0**——概率就是
上面那个式子，与差值的具体大小无关，只与 (n, k) 有关。

于是自助分布在 0 这一点上**至少有 P 的质量**。而 95% 百分位区间的下尾只有 2.5%：
P > 2.5% 时下界不可能落在 0 以上，报出来就是 0.0。这不是「效应接近 0」的估计，
是采样分辨率的产物——和第 2 节里 `check_fifo` 那个 0 是同一类错误。

脚本做三件事：
1. 对已提交的每一组配对自助，算出这个概率（闭式 + 用 stats 里同一套重抽机制做的
   蒙特卡洛复核），并记下「预测下界会被 0 占住」与「产物里的下界确实是 0.0」是否
   一致——五组对照全部一致，两边都不是巧合；
2. 反解这个设计的分辨率：n=300 时 k 要多大，下尾才薄不过 0 上那块质量；
3. 反解规模：按观察到的分歧率，配对数要多少才不撞上这个伪影。

纯标准库（`random.Random`，与 `campus_bus/stats.py` 同路线），不引 numpy，跨机器
逐位一致。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUT = RESULTS / "power.json"

TAIL_ALPHA = 0.025        # 95% 百分位区间的单侧尾概率，与 stats.paired_bootstrap 的 alpha/2 一致
MC_DRAWS = 10000          # 蒙特卡洛复核的重抽次数，与 stats.B_DEFAULT 一致
MC_SEED = 20260913
SOURCES = ("bench.json", "bench_with_lstm.json")


def p_zero(n: int, k: int) -> float:
    """一次自助重抽里一个分歧配对都没抽到的概率（闭式）。"""
    return ((n - k) / n) ** n


def p_zero_mc(n: int, k: int, draws: int = MC_DRAWS, seed: int = MC_SEED) -> float:
    """同一件事用 stats.paired_bootstrap 的那套重抽机制真抽一遍。

    这里只关心「抽到的索引里有没有落在前 k 个上」——差值的具体数值不影响这个概率，
    所以不需要原始差值向量，把分歧配对记为索引 0..k-1 即可（可交换性）。
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        for _ in range(n):
            if rng.randrange(n) < k:
                break
        else:
            hits += 1
    return hits / draws


def min_k_for_thin_tail(n: int, tail: float = TAIL_ALPHA) -> int:
    """n 固定时，k 至少要多大，0 上那块质量才薄过下尾（下界才可能离开 0）。"""
    k = 1
    while k <= n and p_zero(n, k) >= tail:
        k += 1
    return k


def min_pairs_at_rate(rate: float, tail: float = TAIL_ALPHA) -> int:
    """分歧率固定时，配对数至少要多少，0 上那块质量才薄过下尾。"""
    if not 0.0 < rate < 1.0:
        return 0
    n = int(math.ceil(math.log(tail) / math.log(1.0 - rate)))
    while (1.0 - rate) ** n >= tail:
        n += 1
    return n


def run() -> dict:
    comparisons = []
    for src in SOURCES:
        art = json.loads((RESULTS / src).read_text(encoding="utf-8"))["congestion"]
        n_by_model = {row["model"]: row["n"] for row in art["routing"]}
        for b in art["paired_bootstrap"]:
            n, k = n_by_model[b["worse_model"]], b["n_differing"]
            p = p_zero(n, k)
            comparisons.append({
                "artefact": src,
                "worse_model": b["worse_model"],
                "vs": b["vs"],
                "n": n,
                "n_differing": k,
                "p_zero_analytic": round(p, 8),
                "p_zero_pct": round(p * 100.0, 2),
                "p_zero_mc": p_zero_mc(n, k),
                # 0 上的质量厚过下尾 -> 下界不可能落在 0 以上；这是预测
                "predicts_lower_bound_at_zero": p > TAIL_ALPHA,
                # 产物里实际报出来的下界是不是 0.0；这是事实
                "ci95_lo": b["ci95"][0],
                "lower_bound_is_zero": b["ci95"][0] == 0.0,
            })
    agree = all(c["predicts_lower_bound_at_zero"] == c["lower_bound_is_zero"]
                for c in comparisons)
    n_design = comparisons[-1]["n"]
    rate = comparisons[-1]["n_differing"] / n_design
    return {
        "tail_alpha": TAIL_ALPHA,
        "mc_draws": MC_DRAWS,
        "mc_seed": MC_SEED,
        "comparisons": comparisons,
        "prediction_matches_artefacts": agree,
        "design": {
            "n": n_design,
            "min_k_for_lower_bound_to_leave_zero": min_k_for_thin_tail(n_design),
            "observed_differing_rate": round(rate, 6),
            "min_pairs_at_observed_rate": min_pairs_at_rate(rate),
        },
        "note": ("P(自助重抽一个分歧配对都没抽到) = ((n-k)/n)**n。它大于 2.5% 时，"
                 "95% 百分位区间的下界不可能落在 0 以上——那是离散性伪影，"
                 "不是「效应接近 0」的估计。纯标准库，跨机器逐位一致。"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="重算一遍，与已提交的 results/power.json 比对")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    got = run()
    print(json.dumps(got, ensure_ascii=False, indent=1))
    if not got["prediction_matches_artefacts"]:
        print("闭式预测与产物里的 CI 下界对不上——README 第 6 节那段解释要重写")
        return 1

    if args.check:
        want = json.loads(OUT.read_text(encoding="utf-8"))
        if want != got:
            print("power.json 与新跑不一致：")
            for key in sorted(set(want) | set(got)):
                if want.get(key) != got.get(key):
                    print("  %s: 已提交 %r vs 新跑 %r" % (key, want.get(key), got.get(key)))
            return 1
        print("ok: power.json 与新跑一致")
        return 0

    Path(args.out or OUT).write_text(json.dumps(got, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
