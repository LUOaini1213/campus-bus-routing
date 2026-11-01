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
1. 对已提交的每一组配对自助，用三条互相独立的路算这件事：闭式一条；蒙特卡洛一条
   （`random.Random` + `randrange(n)`，与 `stats.paired_bootstrap` 是同一套重抽，
   但在本文件里重写了一遍——那个函数只返回分位数，数不到「重抽均值恰为 0」出现过
   几次）；再把 k 个 1 与 n−k 个 0 直接喂进 `campus_bus.stats.paired_bootstrap`
   真跑一遍，看它报出来的下界是不是真如闭式预测的那样停在 0.0。三条都记进
   `results/power.json`，还记下「预测下界会被 0 占住」与「产物里的下界确实是 0.0」
   是否一致——五组对照全部一致，两边都不是巧合；
2. 反解这个设计的分辨率：n=300 时 k 要多大，下尾才薄不过 0 上那块质量；
3. 反解规模：按观察到的分歧率，配对数要多少才不撞上这个伪影。

第 2、3 步是对**一组指定的对照**反解的（`DESIGN_KEY`，即 ridge-vs-LSTM），按名字
取，不按产物里的排列顺序取——否则 `bench_all.py` 那边换个模型顺序，README 里的
「1% 的分歧率」「368」就会悄悄换成另一组对照的数。

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

from campus_bus import stats  # noqa: E402  （要先把仓根放进 sys.path）

RESULTS = ROOT / "results"
OUT = RESULTS / "power.json"

TAIL_ALPHA = 0.025        # 95% 百分位区间的单侧尾概率，与 stats.paired_bootstrap 的 alpha/2 一致
MC_DRAWS = stats.B_DEFAULT   # 蒙特卡洛复核的重抽次数：直接取自 stats，不另抄一个 10000
MC_SEED = 20260913
SOURCES = ("bench.json", "bench_with_lstm.json")
# 反解设计分辨率（第 2、3 步）用哪一组对照。显式按 (产物, worse_model) 取：
# 产物里 paired_bootstrap 的排列顺序不是契约，不能拿它当选择器。
DESIGN_KEY = ("bench_with_lstm.json", "ridge")


def p_zero(n: int, k: int) -> float:
    """一次自助重抽里一个分歧配对都没抽到的概率（闭式）。"""
    return ((n - k) / n) ** n


def p_zero_mc(n: int, k: int, draws: int = MC_DRAWS, seed: int = MC_SEED) -> float:
    """同一套重抽机制（`random.Random` + `randrange(n)`）在这里重写一遍，真抽。

    为什么不直接调 `stats.paired_bootstrap`：那个函数只返回 (mean, lo, hi)，数不到
    「重抽均值恰为 0」出现了多少次，而这里要的正是这个频次。走 stats 那条真路径的
    是下面的 `stats_lower_bound()`。

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


def stats_lower_bound(n: int, k: int) -> float:
    """把 k 个 1、n−k 个 0 喂进 `stats.paired_bootstrap`，取它报出的 95% CI 下界。

    这是对闭式那句话的直接检验，而且走的是产物本身用的那个函数、那个默认种子、
    那个默认 B 与 alpha：0 上的质量厚过下尾时，下界应当**恰好**是 0.0；薄过下尾时
    就不该是。差值取 1 与 0 只是为了让「均值为 0」等价于「一个分歧配对都没抽到」，
    与真实差值的大小无关（同上，可交换性）。
    """
    _, lo, _ = stats.paired_bootstrap([1.0] * k + [0.0] * (n - k))
    return lo


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
            synth_lo = stats_lower_bound(n, k)
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
                # 同样的 (n, k) 真喂进 stats.paired_bootstrap 后报出的下界；这是真跑
                "synthetic_ci95_lo": round(synth_lo, 8),
                "synthetic_lower_bound_is_zero": synth_lo == 0.0,
                # 产物里实际报出来的下界是不是 0.0；这是事实
                "ci95_lo": b["ci95"][0],
                "lower_bound_is_zero": b["ci95"][0] == 0.0,
            })
    agree = all(c["predicts_lower_bound_at_zero"] == c["lower_bound_is_zero"]
                for c in comparisons)
    agree_bootstrap = all(
        c["predicts_lower_bound_at_zero"] == c["synthetic_lower_bound_is_zero"]
        for c in comparisons)
    by_key = {(c["artefact"], c["worse_model"]): c for c in comparisons}
    if DESIGN_KEY not in by_key:
        raise SystemExit("产物里没有 %s 的 %s 这一组对照，第 2、3 步无从算起"
                         % (DESIGN_KEY[0], DESIGN_KEY[1]))
    design_src = by_key[DESIGN_KEY]
    n_design = design_src["n"]
    rate = design_src["n_differing"] / n_design
    return {
        "tail_alpha": TAIL_ALPHA,
        "mc_draws": MC_DRAWS,
        "mc_seed": MC_SEED,
        "comparisons": comparisons,
        "prediction_matches_artefacts": agree,
        "closed_form_matches_paired_bootstrap": agree_bootstrap,
        "design": {
            "from_artefact": DESIGN_KEY[0],
            "from_worse_model": DESIGN_KEY[1],
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
    if not args.check:
        # --check 时不打这份 JSON：它有 80 行，进 CI 日志纯属噪声。
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
