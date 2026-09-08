# -*- coding: utf-8 -*-
"""配对自助置信区间。**纯标准库，不用 numpy。**

为什么不用 numpy：`numpy.random.Generator`（PCG64）的**流**只在同一个 feature
release 内承诺稳定（NEP 19），跨大版本可以变。而这个仓要建成公开仓、CI 跑
ubuntu-latest 装的是最新 numpy——一旦 `Generator` 的实现变了，自助重抽的结果
就变，`bench_all.py --check` 直接红，而根因跟代码毫无关系。

`random.Random(seed)`（Mersenne Twister）是 CPython 的稳定契约，同一份输入在
任何机器、任何 Python 3.x 上逐位一致。同门的 navsim-ability-ladder 也是这么做的。

判决词由一行规则生成，不手写：CI 跨 0 就是不显著，没有商量余地。
"""
from __future__ import annotations

import random

B_DEFAULT = 10000
SEED_DEFAULT = 12345


def paired_bootstrap(diffs, seed: int = SEED_DEFAULT, b: int = B_DEFAULT,
                     alpha: float = 0.05):
    """对配对差值做自助，返回 (mean, lo, hi)。

    重抽的是**配对**（同一个 OD × 出发时刻上两个模型的差），不是两组独立样本
    ——配对能消掉「这条 OD 本来就难」这类共同变异。
    """
    d = [float(x) for x in diffs]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(d) / n
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * b)]
    hi = means[min(b - 1, int((1 - alpha / 2) * b))]
    return mean, lo, hi


def verdict(lo: float, hi: float) -> str:
    """唯一的判决口径。README 里每个「显著 / 不显著」都追到这一行。"""
    return "CI excludes 0" if (lo > 0 or hi < 0) else "CI spans 0"


def n_differing(diffs, tol: float = 1e-12) -> int:
    """有多少个配对真的不为 0。

    这个数是解读 CI 的前提。若 150 个配对里只有 1 个非 0，自助重抽有
    (1 - 1/150)^150 ≈ 37% 的概率一个都没抽到，CI 下界会**恰好等于 0**——
    那是离散性伪影，不是「效应接近 0」的估计。不报这个数，「不显著」
    三个字就会被读成「已证明两者等价」，而真相是这个实验没有区分它们的功效。
    """
    return sum(1 for x in diffs if abs(float(x)) > tol)
