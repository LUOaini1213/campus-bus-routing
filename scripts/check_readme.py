# -*- coding: utf-8 -*-
"""把 README 里的每个数字回算一遍，对不上就报错。

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

    sc = {row["nodes"]: row for row in scaling}
    want("规模 · 4900 节点堆加速", r"\| 4,900 \| [\d,]+ \| [\d.]+ ms \| [\d.]+ ms \| \*\*([\d.]+)×\*\*",
         [sc[4900]["heap_speedup_x"]])

    f = bench["fleet"]
    want("车队 · 周期与规模", r"运行 ([\d.]+) min、间隔 4 min、折返 3 min",
         [f["cycle_peak_min"]])
    want("车队 · 少一辆需降", r"少一辆车需要周期时间降 \*\*([\d.]+)%\*\*",
         [f["peak"]["reduction_needed_pct"]])

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if problems:
        print("README 与产物对不上（%d 处）：" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("ok: README 的 %d 个数字全部与 results/ 对得上" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
