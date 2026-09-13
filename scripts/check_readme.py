# -*- coding: utf-8 -*-
"""把 README 里出自 `results/` 的数字逐个回算一遍，对不上就报错。

    python scripts/check_readme.py

同门三个仓都有这套机制（navsim 的 `make_summary.py --check README.md`、
civil-buddy 的 `render_eval_table.py --check README.md`、malaysia 的
`tests/test_claims.py`），本仓一直没有——前两轮是用一次性脚本手工核的，核完就扔。
手工核的问题不是当时不准，是**下次改 README 时没人拦着**。

两条设计要点（都从 navsim 抄来）：
1. **断言"抓到的条目数 == 期望条目数"**。否则改 README 措辞导致某条不再匹配时，
   检查会静默跳过它，看起来还是绿的。这比漏检更危险。
2. **容差按显示精度给、计数与序数事实严格相等**。这条与 `bench_all.py --check`
   的口径**故意不同**：那边比的是「重跑一遍 vs 已提交结果」，两边都是浮点算出来的，
   所以要留 1e-6 吸收末位抖动；这边比的是「README 上的字面值 vs 已提交的 JSON」，
   两边都是固定文本，唯一合法的差异只有四舍五入。见 `close()`。
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def half_ulp(shown: str) -> float:
    """README 上写成 `shown` 的数，四舍五入本身能带来的最大误差。

    显示 `0.3145` 是四位小数，四舍五入最多差 0.00005；显示 `513` 最多差 0.5。
    """
    frac = shown.strip().split(".", 1)
    return 0.5 * 10 ** -(len(frac[1]) if len(frac) == 2 else 0)


def close(shown: str, expected: float) -> bool:
    """按**显示精度**判定 README 的字面值与产物是否一致。

    原来的写法是 `abs(a-b) <= 5e-3 * max(1.0, |a|, |b|)`。那个 `max(1.0, ...)` 地板
    让所有小于 1 的量的容差恒等于 0.005 绝对值——而 FIFO 最大下降 0.3145 min、
    配对自助 CI 下界 0.1167 pp 这些本仓最精细的结论恰好都小于 1，等于只核到两位
    小数：0.3145 写成 0.3190（错 1.4%）照样绿。而那个 5e-3 的相对项在另一头同样
    宽：平均展开 12.09 写成 12.15 也在容差内。护栏名义覆盖、实际失效。

    现在的容差**只有**显示精度这一项，没有相对项。理由是这个脚本比的两边都是固定
    文本（README 的字面值 vs 已提交的 JSON），没有任何浮点噪声可言，唯一合法的差异
    就是四舍五入本身：`0.44 ms` 对 0.436、`513 ms` 对 512.515 都是正确的展示值，
    落在 half_ulp 之内；再宽一点就只是在放行打字错误。

    需要吸收重跑抖动的是 `bench_all.py --check`（重跑 vs 已提交），那边留了 1e-6。
    """
    return abs(float(shown) - expected) <= half_ulp(shown)


def main() -> int:
    text = io.open(README, encoding="utf-8").read()
    bench, lstm = load("bench.json"), load("bench_with_lstm.json")
    scaling = load("scaling.json")["rows"]
    problems: list[str] = []
    checked = 0

    def want(label: str, pattern: str, expected, n_expected: int = 1, idx: int = 0):
        """在 README 里找 pattern，把第 idx 处匹配的捕获组逐个与 expected 比。

        `n_expected` 是 navsim 那条防呆：**匹配数不对就直接报错**，否则改 README
        措辞导致正则失配时，检查会静默跳过、看起来还是绿的。
        """
        nonlocal checked
        hits = re.findall(pattern, text)
        if len(hits) != n_expected:
            problems.append("%s: 期望匹配 %d 处，实际 %d 处 —— 正则与 README 脱节了"
                            % (label, n_expected, len(hits)))
            return
        got = hits[idx] if isinstance(hits[idx], tuple) else (hits[idx],)
        for g, e in zip(got, expected):
            checked += 1
            if isinstance(e, str):
                if g.strip() != e:
                    problems.append("%s: README %r vs 产物 %r" % (label, g.strip(), e))
            elif isinstance(e, int) and not isinstance(e, bool):
                # 计数（查询数、节点数、有分歧的配对数、车辆数）是序数事实，
                # 一变就说明实验真的变了，不是舍入——严格相等，不给容差。
                if g.strip() != str(e):
                    problems.append("%s: README %s vs 产物 %d（计数必须严格相等）"
                                    % (label, g.strip(), e))
            elif not close(g, float(e)):
                problems.append("%s: README %s vs 产物 %s" % (label, g, e))

    r = bench["routing"]
    want("路由 · 查询数与最优率", r"\*\*(\d+)\*\* 次查询", [r["queries"]])
    want("路由 · Dijkstra 最优", r"\| Dijkstra \| \*\*([\d/]+)\*\* \| ([\d.]+) \|",
         [r["dijkstra_optimal"], r["expansions_mean"]["dijkstra"]])
    want("路由 · A\\* 最优", r"\| A\\\* \| \*\*([\d/]+)\*\* \| \*\*([\d.]+)\*\* \|",
         [r["astar_optimal"], r["expansions_mean"]["astar"]])
    want("路由 · 上界剪枝", r"\| 上界剪枝 \| \*\*([\d/]+)\*\* \| ([\d.]+) \|",
         [r["pruned_optimal"], r["expansions_mean"]["pruned"]])
    want("路由 · A\\* 省下比例", r"少展开 \*\*([\d.]+)%\*\*", [r["astar_expansion_saving_pct"]])
    want("路由 · 剪枝省下比例", r"剪枝省下 \*\*([\d.]+)%\*\*", [r["prune_expansion_saving_pct"]])

    net = bench["network"]
    want("路网 · 规模", r"(\d+) 节点 / (\d+) 有向边的动态加权图", [net["nodes"], net["edges"]])
    want("FIFO · 连续违反", r"(\d+) 个槽边界里有 \*\*(\d+)\*\* 个",
         [16646, net["boundary_drops"]])
    want("FIFO · 最大下降", r"最大下降 \*\*([\d.]+)\*\* min", [net["max_boundary_drop_min"]])

    thr = {row["threshold_min"]: row for row in bench["relax_threshold"]["rows"]}
    want("阈值 · 退化对照", r"开到 \*\*1e9\*\*[^|]*?展开仍是 \*\*([\d.]+)\*\*",
         [thr[1e9]["expansions_mean"]])

    pk = bench["policies"]["peak"]
    of = bench["policies"]["offpeak"]
    want("策略 · free_flow 高峰", r"\| 只看限速 free_flow \| ([\d.]+)% \| ([\d.]+)% \| ([\d/]+) \|",
         [pk["free_flow"]["mean_excess_pct"], pk["free_flow"]["max_excess_pct"],
          pk["free_flow"]["routes_worse_than_optimal"]])
    want("策略 · peak_profile 平峰最差",
         r"\*\*平峰最差多花 ([\d.]+)%\*\*", [of["peak_profile"]["max_excess_pct"]])

    g = bench["gps"]
    want("GPS · 原始观测误差", r"原始观测平均误差\s*\*\*([\d.]+) m\*\*",
         [g["raw_observation_error"]["mean_m"]])
    want("GPS · 15% 两行", r"\| 15% \| 26 点 \| 朴素插值 \| ([\d.]+) m \| ([\d.]+) m \| \*\*([\d.]+) m\*\* \|",
         [g["rate_15"]["raw"]["kept"]["mean_m"], g["rate_15"]["raw"]["gap"]["mean_m"],
          g["rate_15"]["raw"]["gap"]["p95_m"]])
    want("GPS · 15% 匹配弧长",
         r"\| \| \| \*\*匹配\+弧长\*\* \| \*\*([\d.]+) m\*\* \| \*\*([\d.]+) m\*\* \| ([\d.]+) m \|",
         [g["rate_15"]["arclength"]["kept"]["mean_m"], g["rate_15"]["arclength"]["gap"]["mean_m"],
          g["rate_15"]["arclength"]["gap"]["p95_m"]], n_expected=3, idx=1)
    want("GPS · 5% 匹配弧长",
         r"\| \| \| \*\*匹配\+弧长\*\* \| \*\*([\d.]+) m\*\* \| \*\*([\d.]+) m\*\* \| ([\d.]+) m \|",
         [g["rate_5"]["arclength"]["kept"]["mean_m"], g["rate_5"]["arclength"]["gap"]["mean_m"],
          g["rate_5"]["arclength"]["gap"]["p95_m"]], n_expected=3, idx=0)
    want("GPS · 30% 匹配弧长",
         r"\| \| \| \*\*匹配\+弧长\*\* \| \*\*([\d.]+) m\*\* \| \*\*([\d.]+) m\*\* \| ([\d.]+) m \|",
         [g["rate_30"]["arclength"]["kept"]["mean_m"], g["rate_30"]["arclength"]["gap"]["mean_m"],
          g["rate_30"]["arclength"]["gap"]["p95_m"]], n_expected=3, idx=2)

    acc = {x["model"]: x for x in lstm["congestion"]["accuracy"]}
    rte = {x["model"]: x for x in lstm["congestion"]["routing"]}
    boot = {x["worse_model"]: x for x in lstm["congestion"]["paired_bootstrap"]}
    want("预测 · ridge", r"\| ridge[^|]*\| \*\*([\d.]+)（最好）\*\* \| ([\d.]+)% \|",
         [acc["ridge"]["mae_kmh"], rte["ridge"]["realised_excess_pct"]])
    want("预测 · persistence 显著性",
         r"persistence[^|]*\| ([\d.]+) \| ([\d.]+)% \| \+([\d.]+) pp \*\*\[([\d.]+), ([\d.]+)\]",
         [acc["persistence"]["mae_kmh"], rte["persistence"]["realised_excess_pct"],
          boot["persistence"]["mean_diff_pct_points"],
          boot["persistence"]["ci95"][0], boot["persistence"]["ci95"][1]])
    want("预测 · seasonal 显著性",
         r"\| seasonal[^|]*\| \*\*([\d.]+)（最差）\*\* \| ([\d.]+)% \| \+([\d.]+) pp "
         r"\[(-[\d.]+), ([\d.]+)\] 不显著 \| (\d+)/300 \|",
         [acc["seasonal"]["mae_kmh"], rte["seasonal"]["realised_excess_pct"],
          boot["seasonal"]["mean_diff_pct_points"],
          boot["seasonal"]["ci95"][0], boot["seasonal"]["ci95"][1],
          boot["seasonal"]["n_differing"]])
    want("预测 · ridge 显著性",
         r"\| ridge[^|]*\|[^|]*\|[^|]*\| \+([\d.]+) pp \[([\d.]+), ([\d.]+)\] 不显著 \| (\d+)/300 \|",
         [boot["ridge"]["mean_diff_pct_points"],
          boot["ridge"]["ci95"][0], boot["ridge"]["ci95"][1],
          boot["ridge"]["n_differing"]])
    want("预测 · LSTM 行", r"\| LSTM[^|]*\| ([\d.]+) \| ([\d.]+)% \|",
         [acc["lstm"]["mae_kmh"], rte["lstm"]["realised_excess_pct"]])
    want("预测 · 有分歧的配对数", r"300 个配对里只有 \*\*(\d+)\*\* 个让",
         [boot["seasonal"]["n_differing"]])
    want("预测 · ridge 与 LSTM 的差", r"两者只差 \*\*\+([\d.]+) pp\*\*",
         [boot["ridge"]["mean_diff_pct_points"]])
    want("预测 · ridge 的 CI 与分歧数",
         r"95% CI \*\*\[([\d.]+), ([\d.]+)\]\*\* 跨 0，300 个配对里只有 \*\*(\d+)\*\* 个有分歧",
         [boot["ridge"]["ci95"][0], boot["ridge"]["ci95"][1], boot["ridge"]["n_differing"]])

    # 结构性断言：README 里「（最好）」「（最差）」「名义最好」这三个标注必须与产物的
    # argmin / argmax 一致。逐点比数字拦不住「表格改了、上面的结论句没跟着改」——
    # 2026-09-13 修掉的那句「ridge 现在两项都最好」就是这么活下来的：表里 LSTM 的
    # 多付 0.0354% 明明比 ridge 的 0.05% 低，数字逐个都对得上，只有那句话是反的。
    def tagged_model(tag: str) -> list[str]:
        out = []
        for line in text.splitlines():
            m = re.match(r"\| ([A-Za-z]+)（", line)
            if m and tag in line:
                out.append(m.group(1).lower())
        return out

    def want_tag(label: str, tag: str, expected: str):
        nonlocal checked
        checked += 1
        got = tagged_model(tag)
        if got != [expected]:
            problems.append("%s: README 把 %s 标在 %s，产物说该是 %r"
                            % (label, tag, got or "（没有任何一行）", expected))

    want_tag("预测 · 「（最好）」必须落在 MAE 最小的模型上", "（最好）",
             min(acc.values(), key=lambda x: x["mae_kmh"])["model"])
    want_tag("预测 · 「（最差）」必须落在 MAE 最大的模型上", "（最差）",
             max(acc.values(), key=lambda x: x["mae_kmh"])["model"])
    want_tag("预测 · 「名义最好」必须落在多付最小的模型上", "名义最好",
             min(rte.values(), key=lambda x: x["realised_excess_pct"])["model"])

    # --- 功效那一段 ---------------------------------------------------------
    # 「有可观概率一个都没抽到」原来是全仓唯一一条只有叙述、没有数的结论。
    # bench/bench_power.py 现在把它算成数，这里把 README 的说法钉回产物。
    power = load("power.json")
    pw = {(c["artefact"], c["worse_model"]): c for c in power["comparisons"]}
    ridge_pw = pw[("bench_with_lstm.json", "ridge")]
    seas_pw = pw[("bench_with_lstm.json", "seasonal")]
    want("功效 · ridge 那一组的 k、n 与零分歧概率",
         r"ridge 那一组（k=(\d+), n=(\d+)）算出来是 \*\*([\d.]+)%\*\*："
         r"自助分布在 0 上至少有 ([\d.]+)% 的质量",
         [ridge_pw["n_differing"], ridge_pw["n"],
          ridge_pw["p_zero_pct"], ridge_pw["p_zero_pct"]])
    want("功效 · 下尾概率", r"而 95% 区间的下尾只有 \*\*([\d.]+)%\*\*",
         [power["tail_alpha"] * 100])
    want("功效 · seasonal 的 k 与零分歧概率",
         r"seasonal 的 k=(\d+) 把概率压到 \*\*([\d.]+)%\*\*",
         [seas_pw["n_differing"], seas_pw["p_zero_pct"]])
    want("功效 · seasonal 的下界没停在 0",
         r"它的下界就没有停在 0（(-[\d.]+)）", [seas_pw["ci95_lo"]])
    want("功效 · 预测与产物的一致率", r"\*\*一致率 (\d+)/(\d+)\*\*",
         [sum(c["predicts_lower_bound_at_zero"] == c["lower_bound_is_zero"]
              for c in power["comparisons"]), len(power["comparisons"])])
    want("功效 · 设计分辨率", r"n=(\d+) 时 k 要到 \*\*(\d+)\*\* 下界才可能离开 0",
         [power["design"]["n"], power["design"]["min_k_for_lower_bound_to_leave_zero"]])
    want("功效 · 反解的分歧率", r"若要在 (\d+)% 的分歧率上不撞这个伪影",
         [int(round(power["design"]["observed_differing_rate"] * 100))])
    want("功效 · 反解的配对数", r"配对数至少得 \*\*(\d+)\*\* 个",
         [power["design"]["min_pairs_at_observed_rate"]])

    sc = {row["nodes"]: row for row in scaling}
    want("规模 · 4900 节点堆加速", r"\| 4,900 \| [\d,]+ \| [\d.]+ ms \| [\d.]+ ms \| \*\*([\d.]+)×\*\*",
         [sc[4900]["heap_speedup_x"]])

    f = bench["fleet"]
    want("车队 · 周期与规模", r"运行 ([\d.]+) min、间隔 4 min、折返 3 min",
         [f["cycle_peak_min"]])
    want("车队 · 少一辆需降", r"少一辆车需要周期时间降 \*\*([\d.]+)%\*\*",
         [f["peak"]["reduction_needed_pct"]])
    # 车队台阶表原本一个格子都没进回算——26.81 改成 96.81 也是绿的。
    ladder = f["ladder_peak"]
    want("车队 · 台阶表首行", r"\| (\d+)% \| ([\d.]+) \| \*\*(\d+)\*\* \|",
         [int(ladder[0]["reduction_pct"]), ladder[0]["cycle_min"], ladder[0]["fleet"]])
    for i in range(1, 4):
        want("车队 · 台阶表第 %d 行" % (i + 1), r"\| (\d+)% \| ([\d.]+) \| (\d+) \|",
             [int(ladder[i]["reduction_pct"]), ladder[i]["cycle_min"], ladder[i]["fleet"]],
             n_expected=3, idx=i - 1)

    # --- HTTP 延迟表 ---------------------------------------------------------
    # 这张表撑着简历里「Docker 响应 <50 ms」那条，却是覆盖最差的一张：12 个数字
    # 一个都没进回算，实测把 17.7 写成 177、515 写成 5，check_readme 照样退出 0。
    # 延迟值本身跨机器不可比（`bench_all.py` 的 EXEMPT_SUFFIX 就是为此），但
    # 「README 的字面值 == 已提交 JSON」这件事与本机性能无关，CI 里比它是安全的。
    c1k, c16k, c16f = (load("service_c1_keep.json"), load("service_c16_keep.json"),
                       load("service_c16_fresh.json"))
    lat_keep = (r"\| (\d+) \| 复用 \| ([\d.]+) ms \| ([\d.]+) ms \| "
                r"\*\*([\d.]+) ms\*\* \| ([\d.]+) ms \|")
    for label, svc, idx in (("并发 1 复用", c1k, 0), ("并发 16 复用", c16k, 1)):
        L = svc["latency_ms"]
        want("延迟 · " + label, lat_keep,
             [svc["concurrency"], L["p50"], L["p95"], L["p99"], L["max"]],
             n_expected=2, idx=idx)
    Lf = c16f["latency_ms"]
    want("延迟 · 并发 16 每请求新建",
         r"\| (\d+) \| 每请求新建 \| ([\d.]+) ms \| ([\d.]+) ms \| "
         r"\*\*([\d.]+) ms\*\* \| ([\d.]+) ms \|",
         [c16f["concurrency"], Lf["p50"], Lf["p95"], Lf["p99"], Lf["max"]])
    want("延迟 · 前两行的 p99", r"p99 分别是 ([\d.]+) ms 与 ([\d.]+) ms",
         [c1k["latency_ms"]["p99"], c16k["latency_ms"]["p99"]])
    want("延迟 · 新建连接的 p99 与倍数",
         r"新建 TCP 连接时 p99 是 ([\d.]+) ms，比复用连接差 \*\*([\d.]+) 倍\*\*",
         [Lf["p99"], Lf["p99"] / c16k["latency_ms"]["p99"]])
    want("延迟 · 第三行的 p50", r"第三行的 p50 只有 ([\d.]+) ms", [Lf["p50"]])
    want("延迟 · 复用行的 max 孤点", r"复用连接那行的 max 也有一个 ([\d.]+) ms 的孤点",
         [c16k["latency_ms"]["max"]])
    want("延迟 · 求解本身的 p99", r"求解本身 p99 只有 \*\*([\d.]+) ms\*\*",
         [r["latency_ms"]["p99"]])

    # 三份 service_*.json 的元信息：README 的三行必须真的是三次不同配置的测量，
    # 而且量在同一张图上。这些不是 README 里的数字，单独计数。
    structural = 0

    def want_fact(label: str, got, expected):
        nonlocal structural
        structural += 1
        if got != expected:
            problems.append("%s: 产物是 %r，期望 %r" % (label, got, expected))

    want_fact("延迟 · 三行是三种不同配置",
              sorted((s["concurrency"], s["connection"]) for s in (c1k, c16k, c16f)),
              [(1, "keep-alive"), (16, "fresh-per-request"), (16, "keep-alive")])
    want_fact("延迟 · 三次测量的请求数一致",
              [s["requests"] for s in (c1k, c16k, c16f)], [c1k["requests"]] * 3)
    want_fact("延迟 · 三次都量在同一张图上",
              [(s["nodes"], s["edges"]) for s in (c1k, c16k, c16f)],
              [(net["nodes"], net["edges"])] * 3)
    for name, svc in (("c1_keep", c1k), ("c16_keep", c16k), ("c16_fresh", c16f)):
        L = svc["latency_ms"]
        want_fact("延迟 · %s 的分位数单调" % name,
                  [L["p50"] <= L["p95"] <= L["p99"] <= L["max"]], [True])

    # 自述与实现必须对齐：README 有两处宣称回算了「N 个数字」（中英各一）。
    # 这种自述一旦和实现脱节，就又成了一句不可核验的声明——这里把它也算进来，
    # N 必须等于这次真的核过的条数（含这两条自检本身）。
    declared = checked + 2
    want("自述 · 中文回算条数", r"README 的 (\d+) 个数字逐个回算", [declared])
    want("自述 · 英文回算条数", r"re-checks (\d+) numbers", [declared])

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if problems:
        print("README 与产物对不上（%d 处）：" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("ok: README 的 %d 个数字全部与 results/ 对得上（另有 %d 条结构性断言）"
          % (checked, structural))
    return 0


if __name__ == "__main__":
    sys.exit(main())
