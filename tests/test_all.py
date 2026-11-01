# -*- coding: utf-8 -*-
"""把关键声明锁进测试。全部离线、无网络、无模型权重。"""
from __future__ import annotations

import json
import random
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus import fleet, gps, stats                              # noqa: E402
from campus_bus.network import BUCKET_MIN, N_BUCKETS, Edge, Network   # noqa: E402
from campus_bus.routing import (astar, brute_force, dijkstra,         # noqa: E402
                                label_correcting, path_cost)

NET = Network.load()
DEPARTS = [150, 420, 460, 480, 720, 1020, 1070, 1200]


class NetworkShape(unittest.TestCase):
    def test_23_nodes_58_edges(self):
        self.assertEqual(NET.n_nodes(), 23)
        self.assertEqual(NET.n_edges(), 58)

    def test_committed_network_is_fifo(self):
        self.assertEqual(NET.check_fifo(), [])

    def test_check_fifo_actually_catches_a_violation(self):
        """构造一条长边 + 速度陡升，check_fifo 必须报错，enforce_fifo 必须修好。

        这条测试的存在是因为真网上 FIFO 从不违反（最坏单边行程 < 槽宽），
        没有它，那两个函数就等于没被验证过。
        """
        nodes = {"a": (0.0, 0.0), "b": (8000.0, 0.0)}          # 8 km，一条边
        edges = [Edge(0, "a", "b", 8000.0, 60.0, "main")]
        speed = np.full((1, N_BUCKETS), 10.0)                  # 8 km @10 km/h = 48 min
        speed[0][10:] = 60.0                                   # 陡升到 8 分钟
        bad = Network(nodes, edges, speed)
        self.assertTrue(bad.check_fifo(), "陡升的速度剖面应当违反 FIFO")
        self.assertGreater(bad.enforce_fifo(), 0)
        self.assertEqual(bad.check_fifo(), [])


class Routing(unittest.TestCase):
    def _pairs(self):
        names = sorted(NET.nodes)
        return [(a, b) for a in names[:8] for b in names[8:16] if a != b]

    def test_dijkstra_matches_the_reference_solver(self):
        for s, t in self._pairs():
            for dep in DEPARTS:
                ref = label_correcting(NET, s, t, dep)
                if ref.arrival == float("inf"):
                    continue
                self.assertAlmostEqual(dijkstra(NET, s, t, dep).arrival, ref.arrival, places=9,
                                       msg="%s->%s @%d" % (s, t, dep))

    def test_astar_matches_dijkstra_and_expands_no_more(self):
        tot_a = tot_d = 0
        for s, t in self._pairs():
            for dep in DEPARTS:
                d, a = dijkstra(NET, s, t, dep), astar(NET, s, t, dep)
                self.assertAlmostEqual(a.arrival, d.arrival, places=9)
                tot_a += a.expansions
                tot_d += d.expansions
        self.assertLess(tot_a, tot_d, "可采纳启发式不该展开得更多")

    def test_returned_path_reproduces_the_returned_arrival(self):
        for s, t in self._pairs():
            for dep in DEPARTS[:4]:
                r = dijkstra(NET, s, t, dep)
                if r.arrival == float("inf"):
                    continue
                self.assertAlmostEqual(path_cost(NET, r.path, dep), r.arrival, places=9)

    def test_route_is_time_dependent_at_all(self):
        """至少有一对 OD 在高峰与平峰选出不同的路——否则分时这套就是白做的。"""
        differ = 0
        for s, t in self._pairs():
            if dijkstra(NET, s, t, 460).path != dijkstra(NET, s, t, 150).path:
                differ += 1
        self.assertGreater(differ, 0)


class Admissibility(unittest.TestCase):
    """A* 的启发式必须真的不高估——第一版不是，靠运气没翻掉答案。"""

    def test_max_kmh_covers_the_whole_profile(self):
        top = max(float(max(row)) for row in NET.speed)
        self.assertGreaterEqual(NET.max_kmh, top - 1e-9,
                                "max_kmh 必须覆盖剖面里真实出现过的最高速")

    def test_heuristic_is_admissible(self):
        """穷举 (v, dst, 出发时刻) 断言 h(v) <= 真实剩余时间。

        第一版 `max_kmh` 取 max(free_kmh)=40，而生成器的 jitter 让实际速度到过
        42.73，于是 h 在若干平峰实例上高估——启发式不可采纳。当时没翻掉任何答案，
        但那是运气，不是保证。这条测试就是把运气换成保证。
        """
        names = sorted(NET.nodes)
        for dst in names[:8]:
            for v in names:
                if v == dst:
                    continue
                h = NET.euclid_m(v, dst) / 1000.0 / NET.max_kmh * 60.0
                for dep in (150.0, 460.0, 720.0, 1070.0):
                    r = dijkstra(NET, v, dst, dep)
                    if r.arrival == float("inf"):
                        continue
                    self.assertLessEqual(h, (r.arrival - dep) + 1e-9,
                                         "h 高估：%s->%s @%s" % (v, dst, dep))


class FifoIsLoadBearing(unittest.TestCase):
    """FIFO 不是装饰：违反时两个求解器会**一起**给出次优解。"""

    @staticmethod
    def _non_fifo_net():
        # a->c 有快慢两条路；c->b 早进极慢、晚进极快 —— 晚出发反而早到。
        nodes = {"a": (0., 0.), "d": (500., 800.), "c": (1000., 0.), "b": (4000., 0.)}
        edges = [Edge(0, "a", "c", 1000., 60., "main"),
                 Edge(1, "a", "d", 1000., 60., "main"),
                 Edge(2, "d", "c", 1000., 3., "main"),
                 Edge(3, "c", "b", 3000., 60., "main")]
        sp = np.zeros((4, N_BUCKETS))
        sp[0][:] = 60.0
        sp[1][:] = 60.0
        sp[2][:] = 3.0
        sp[3][:] = 3.0
        sp[3][int(20 // BUCKET_MIN):] = 90.0
        return Network(nodes, edges, sp)

    def test_both_solvers_fail_together_without_fifo(self):
        net = self._non_fifo_net()
        self.assertTrue(net.check_fifo(), "这个构造应当违反 FIFO")
        d = dijkstra(net, "a", "b", 0.0)
        l = label_correcting(net, "a", "b", 0.0)
        bf = brute_force(net, "a", "b", 0.0)
        self.assertAlmostEqual(d.arrival, 61.0, places=6)
        self.assertAlmostEqual(l.arrival, 61.0, places=6)      # 参照实现同样错
        self.assertAlmostEqual(bf.arrival, 23.0, places=6)     # 穷举才是真值
        self.assertEqual(bf.path, ["a", "d", "c", "b"])
        self.assertGreater(d.arrival - bf.arrival, 30.0,
                           "两个求解器一起错了 38 分钟——它们共享同一个假设")

    def test_enforcing_fifo_makes_them_agree_with_brute_force(self):
        net = self._non_fifo_net()
        net.enforce_fifo()
        self.assertEqual(net.check_fifo(), [])
        self.assertAlmostEqual(dijkstra(net, "a", "b", 0.0).arrival,
                               brute_force(net, "a", "b", 0.0).arrival, places=9)

    def test_committed_network_reports_its_true_fifo_status(self):
        """真网上 check_fifo 为 0，但连续 FIFO 违反是存在的——只是幅度有上界。

        这条测试锁住的是**叙述口径**：不许再说「结构上不可能违反」。
        """
        self.assertEqual(NET.check_fifo(), [])
        self.assertGreater(NET.boundary_drop_count(), 0,
                           "连续意义上的违反是存在的，别再说结构上不可能")
        self.assertLess(NET.max_boundary_drop(), BUCKET_MIN / 2,
                        "但幅度小于采样步长，所以采样器看不见——这才是真相")


class Reachability(unittest.TestCase):
    def test_every_ordered_pair_is_reachable(self):
        """`admin` 曾经入度为 0，导致 30 个查询被静默跳过、README 的算式对不上。"""
        names = sorted(NET.nodes)
        for n in names:
            self.assertTrue(any(e.v == n for e in NET.edges), "%s 入度为 0" % n)
        bad = [(s, t) for s in names for t in names
               if s != t and dijkstra(NET, s, t, 480.0).arrival == float("inf")]
        self.assertEqual(bad, [])


class PruningNotThreshold(unittest.TestCase):
    """真正能省展开的是上界剪枝；松弛阈值省不下任何东西，且这是恒真的。"""

    def test_pruning_saves_expansions_without_losing_optimality(self):
        names = sorted(NET.nodes)
        pairs = [(a, b) for a in names[:8] for b in names[8:16] if a != b]
        base = pruned = 0
        for s, t in pairs:
            for dep in DEPARTS[:4]:
                d = dijkstra(NET, s, t, dep)
                p = dijkstra(NET, s, t, dep, prune=True)
                self.assertAlmostEqual(p.arrival, d.arrival, places=9)
                base += d.expansions
                pruned += p.expansions
        self.assertLess(pruned, base * 0.95, "剪枝应当实打实少展开")

    def test_relax_threshold_cannot_help_by_construction(self):
        """阈值开到 1e9（算法彻底退化）展开仍不降——它加错了位置。"""
        names = sorted(NET.nodes)
        pairs = [(a, b) for a in names[:6] for b in names[6:12] if a != b]
        zero = huge = 0
        for s, t in pairs:
            for dep in DEPARTS[:3]:
                zero += dijkstra(NET, s, t, dep).expansions
                huge += dijkstra(NET, s, t, dep, relax_threshold=1e9).expansions
        self.assertGreater(huge, zero * 0.97,
                           "阈值即使开到 1e9 也省不下 3%，因为它挡不住首次入堆")


class Bootstrap(unittest.TestCase):
    """纯标准库的配对自助本身要被验证，不能只验证由它得出的结论。"""

    def test_constant_shift_gives_a_tight_interval_around_it(self):
        m, lo, hi = stats.paired_bootstrap([2.0] * 200)
        self.assertAlmostEqual(m, 2.0, places=9)
        self.assertAlmostEqual(lo, 2.0, places=9)
        self.assertAlmostEqual(hi, 2.0, places=9)
        self.assertEqual(stats.verdict(lo, hi), "CI excludes 0")

    def test_pure_noise_spans_zero(self):
        rng = random.Random(4)
        m, lo, hi = stats.paired_bootstrap([rng.gauss(0, 1) for _ in range(400)])
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertEqual(stats.verdict(lo, hi), "CI spans 0")

    def test_same_seed_same_answer_and_no_numpy(self):
        d = [0.1, -0.2, 0.35, 0.0, 0.9]
        self.assertEqual(stats.paired_bootstrap(d, seed=1),
                         stats.paired_bootstrap(d, seed=1))
        self.assertNotEqual(stats.paired_bootstrap(d, seed=1),
                            stats.paired_bootstrap(d, seed=2))
        src = (ROOT / "campus_bus" / "stats.py").read_text(encoding="utf-8")
        self.assertNotIn("import numpy", src,
                         "自助必须是纯标准库：numpy 的 Generator 流跨大版本不承诺稳定")

    def test_n_differing_counts_real_disagreements(self):
        self.assertEqual(stats.n_differing([0.0, 0.0, 1e-15, 0.5, -0.2]), 2)


class ThirdPartyOracle(unittest.TestCase):
    """把静态快照交给 networkx 独立求一遍。

    前面的正确性论证都是「我的实现 vs 我自己写的参照实现」——同一个人写的
    两份代码可能错在同一处。networkx 是第三方实现，它同意才算数。
    分时问题 networkx 解不了，所以把某个时刻的速度冻住做成静态图来比。
    """

    def test_matches_networkx_on_a_frozen_snapshot(self):
        import networkx as nx
        from campus_bus.network import Network as N
        for bucket_t in (150.0, 460.0, 1070.0):
            b = NET.bucket(bucket_t)
            frozen = N(NET.nodes, NET.edges,
                       [[float(NET.speed[e.eid][b])] * N_BUCKETS for e in NET.edges])
            g = nx.DiGraph()
            for e in NET.edges:                       # 权 = 该时刻的行程时间
                w = frozen.travel_time(e, 0.0)
                if not g.has_edge(e.u, e.v) or w < g[e.u][e.v]["weight"]:
                    g.add_edge(e.u, e.v, weight=w)
            names = sorted(NET.nodes)
            for s in names[:6]:
                for t in names[6:12]:
                    if s == t:
                        continue
                    mine = dijkstra(frozen, s, t, 0.0).arrival
                    try:
                        theirs = nx.dijkstra_path_length(g, s, t, weight="weight")
                    except nx.NetworkXNoPath:
                        self.assertEqual(mine, float("inf"))
                        continue
                    self.assertAlmostEqual(mine, theirs, places=9,
                                           msg="%s->%s @bucket %d" % (s, t, b))


class GPSCompensation(unittest.TestCase):
    PATH = ["west_gate", "hub_w", "hub_n", "hub_e", "dorm_c"]

    def test_dropout_is_bursty_not_independent(self):
        traj = gps.simulate(NET, self.PATH, 470.0)
        keep = gps.drop(traj, rate=0.15)
        gaps, cur = [], 0
        for k in keep:
            if k:
                if cur:
                    gaps.append(cur)
                cur = 0
            else:
                cur += 1
        self.assertGreater(max(gaps), 3, "成串丢失应当出现长于 3 点的空档")

    def test_trajectory_is_not_linear_in_time(self):
        """第一版模拟器是匀速直线的，于是线性插值天然精确、补偿显得毫无价值。

        这条测试守住那个教训：轨迹必须含停站（位置不动）与加减速（位移随时间
        非线性），否则整节 GPS 结论都是模拟器造出来的假象。
        """
        traj = gps.simulate(NET, self.PATH, 470.0)
        self.assertGreater(int((~traj.moving).sum()), 0, "应当有停站点")
        s = traj.s_on_edge[traj.edge_idx == traj.edge_idx[0]]
        step = np.diff(s)
        self.assertGreater(step.max() / max(step.min(), 1e-9), 2.0,
                           "边内推进速度应当变化（加减速），不是匀速")

    def test_map_matching_halves_the_error_on_observed_points(self):
        traj = gps.simulate(NET, self.PATH, 470.0)
        obs = gps.observe(traj)
        keep = gps.drop(traj, rate=0.15)
        raw = gps.errors(traj, gps.compensate(NET, traj, obs, keep, "raw", self.PATH), keep)
        arc = gps.errors(traj, gps.compensate(NET, traj, obs, keep, "arclength", self.PATH), keep)
        self.assertLess(arc["kept"]["mean_m"], raw["kept"]["mean_m"] * 0.6)
        self.assertLess(arc["gap"]["mean_m"], raw["gap"]["mean_m"])

    def test_isotonic_beats_cumulative_max(self):
        """cummax 也能保证单调，但它把对称噪声整流成前偏，丢失点误差更大。"""
        traj = gps.simulate(NET, self.PATH, 470.0)
        obs = gps.observe(traj)
        keep = gps.drop(traj, rate=0.15)
        arc = gps.compensate(NET, traj, obs, keep, "arclength", self.PATH)
        real = gps._pava
        try:
            gps._pava = lambda y: np.maximum.accumulate(y)      # 退回旧做法
            cummax = gps.compensate(NET, traj, obs, keep, "arclength", self.PATH)
        finally:
            gps._pava = real
        self.assertLess(gps.errors(traj, arc, keep)["gap"]["mean_m"],
                        gps.errors(traj, cummax, keep)["gap"]["mean_m"])

    def test_pava_is_monotone_and_length_preserving(self):
        for y in ([1, 3, 2, 4], [5, 4, 3, 2, 1], [2, 1, 4, 3, 6, 5], [1.0]):
            a = gps._pava(np.asarray(y, float))
            self.assertEqual(len(a), len(y))
            self.assertTrue(all(a[i] <= a[i + 1] + 1e-9 for i in range(len(a) - 1)))
            self.assertAlmostEqual(float(a.sum()), float(sum(y)), places=6,
                                   msg="保序回归应当保总和（cummax 不会）")


class Fleet(unittest.TestCase):
    def test_fleet_is_a_ceiling_function(self):
        self.assertEqual(fleet.fleet_size(24.0, 4.0, 0.0), 6)
        self.assertEqual(fleet.fleet_size(24.1, 4.0, 0.0), 7)      # 多 6 秒就多一辆

    def test_breakeven_reports_the_reduction_that_saves_a_vehicle(self):
        b = fleet.breakeven(24.54, 4.0, 3.0)
        self.assertEqual(b["fleet"], 7)
        self.assertEqual(b["next_fleet"], 6)
        self.assertAlmostEqual(b["cycle_needed_min"], 21.0, places=6)


class CommittedResults(unittest.TestCase):
    """results/bench.json 里的关键结论必须还成立。"""

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads((ROOT / "results" / "bench.json").read_text(encoding="utf-8"))

    def test_both_solvers_were_optimal_on_every_query(self):
        n = self.r["routing"]["queries"]
        self.assertEqual(self.r["routing"]["dijkstra_optimal"], "%d/%d" % (n, n))
        self.assertEqual(self.r["routing"]["astar_optimal"], "%d/%d" % (n, n))

    def test_relax_threshold_buys_nothing(self):
        rows = {r["threshold_min"]: r for r in self.r["relax_threshold"]["rows"]}
        base = rows[0.0]
        self.assertEqual(base["optimal_pct"], 100.0)
        worst = rows[1.0]
        self.assertLess(worst["optimal_pct"], 100.0)
        saved = (base["expansions_mean"] - worst["expansions_mean"]) / base["expansions_mean"]
        self.assertLess(saved, 0.02, "阈值若真能省下 >2% 展开，这个结论就要改写")

    def test_worst_mae_is_still_not_separable_on_routing(self):
        """MAE 与选路结果不是一回事——但这句话比我第一版写的弱得多。

        第一版断言「MAE 最好的 ridge 不是选路最好的」，而那是**结算真值用错**
        造成的：当时拿基础剖面结算，预测的却是 hist[-1]，越平滑的模型越占便宜。
        口径修正后，在这份不含 LSTM 的三模型结算（results/bench.json）里 ridge 两项
        都最好，那条断言当场变红——这条测试就是那次更正的化石。**注意口径**：
        README 第 6 节展示的是四模型表，那里选路上名义最好的是 LSTM（0.0354% 对
        ridge 的 0.05%），但两者只差 +0.0146 pp、配对自助跨 0，分不出来；所以「ridge
        两项都最好」只在这份三模型产物里成立，不能当成通用说法。现在还站得住的是
        更弱、也更有意思的一句：**seasonal 的 MAE 比 ridge 差 72%，在选路上却分不出来。**
        """
        acc = {r["model"]: r["mae_kmh"] for r in self.r["congestion"]["accuracy"]}
        boot = {b["worse_model"]: b for b in self.r["congestion"]["paired_bootstrap"]}
        self.assertGreater(acc["seasonal"], acc["ridge"] * 1.5, "seasonal 的 MAE 应当明显更差")
        self.assertFalse(boot["seasonal"]["significant"],
                         "但它在选路上与最好者的差跨 0，不显著")

    def test_the_routing_ranking_must_not_be_over_read(self):
        """名次差有没有超出噪声，由配对自助说了算，不由排序说了算。

        除了显著性，还必须报 `n_differing`：300 个配对里只有个位数真的有分歧时，
        「不显著」不等于「已证明两者等价」，而等于「这个实验没有区分它们的功效」。
        """
        boot = {b["worse_model"]: b for b in self.r["congestion"]["paired_bootstrap"]}
        self.assertTrue(boot["persistence"]["significant"], "persistence 明显更差，应当显著")
        self.assertEqual(boot["persistence"]["verdict"], "CI excludes 0")
        self.assertFalse(boot["seasonal"]["significant"], "seasonal 的差跨 0，不得当成名次讲")
        self.assertEqual(boot["seasonal"]["verdict"], "CI spans 0")
        for b in self.r["congestion"]["paired_bootstrap"]:
            self.assertIn("n_differing", b, "必须报真正有分歧的配对数，否则 CI 会被误读")
            self.assertLess(b["n_differing"], 0.1 * b.get("n", 300),
                            "有分歧的配对不足一成——功效不足，这一点要写在明处")


class Robustness(unittest.TestCase):
    """结论不随随机种子变——这是合成数据仓库最该先回答的质疑。

    断言的是**方向**不是数值：数值随种子变是正常的，结论跟着变才是问题。
    """

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads((ROOT / "results" / "robustness.json").read_text(encoding="utf-8"))

    def test_at_least_eight_seeds(self):
        self.assertGreaterEqual(len(self.r["seeds"]), 8)
        self.assertEqual(len(self.r["rows"]), len(self.r["seeds"]))

    def test_every_invariant_holds_on_every_seed(self):
        for name, ok in self.r["invariants"].items():
            self.assertTrue(ok, "跨种子不变量不成立：%s" % name)

    def test_the_headline_conclusion_survives_every_seed(self):
        """平峰 free_flow 恒为 0、高峰恒有几十条选错——两个方向都不许翻。"""
        for row in self.r["rows"]:
            self.assertEqual(row["offpeak_free_flow_mean_pct"], 0.0, row["seed"])
            worse = int(row["peak_free_flow_worse"].split("/")[0])
            self.assertGreaterEqual(worse, 30, "seed %s 高峰只错了 %d 条" % (row["seed"], worse))
            self.assertGreater(row["offpeak_peak_profile_max_pct"], 20.0, row["seed"])

    def test_fifo_story_is_the_same_on_every_seed(self):
        """采样看不见、连续违反始终存在——「结构上不可能」在任何种子上都不对。"""
        for row in self.r["rows"]:
            self.assertEqual(row["fifo_sampled_violations"], 0, row["seed"])
            self.assertGreater(row["boundary_drops"], 8000, row["seed"])
            self.assertLess(row["max_boundary_drop_min"], BUCKET_MIN / 2, row["seed"])


class ReadmeNumbers(unittest.TestCase):
    def test_readme_matches_the_committed_results(self):
        """README 里出自 results/ 的数字都由 scripts/check_readme.py 回算。

        同门三个仓都有这道；本仓前两轮是手工核的，核完就扔——手工核的问题不是
        当时不准，是下次改 README 时没人拦着。
        """
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_readme.py")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class Service(unittest.TestCase):
    def test_http_round_trip(self):
        import http.client

        from campus_bus.service import serve
        srv = serve(8799, "127.0.0.1")
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        c = http.client.HTTPConnection("127.0.0.1", 8799, timeout=5)
        try:
            c.request("GET", "/route?from=west_gate&to=dorm_c&depart=470")
            body = json.loads(c.getresponse().read())
            self.assertEqual(body["from"], "west_gate")
            self.assertEqual(body["path"][0], "west_gate")
            self.assertEqual(body["path"][-1], "dorm_c")
            self.assertGreater(body["travel_min"], 0)
            c.request("GET", "/route?from=nowhere&to=dorm_c")
            self.assertEqual(c.getresponse().status, 400)
        finally:
            # 先关客户端再停服务：反过来会让服务端在 readline 上撞到
            # ConnectionReset，把一串栈打进测试输出。
            c.close()
            srv.shutdown()
            srv.server_close()
            t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
