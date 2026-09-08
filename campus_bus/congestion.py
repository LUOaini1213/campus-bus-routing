# -*- coding: utf-8 -*-
"""拥堵预测：给每条边预测下一个 15 分钟的速度，喂给路由做边权。

四个预测器，从笨到不笨排列，**目的是让评测决定用哪个，不是默认上最花哨的**：

  persistence   下一步 = 这一步（最强的免费基线，交通预测里经常打赢深度模型）
  seasonal      下一步 = 昨天同一时刻（抓住了日周期）
  ridge         滞后特征 + 时刻的 one-hot，岭回归
  lstm          单层 LSTM（需要 torch；CI 不装 torch，会自动跳过）

评价用两把尺子，第二把才是真正重要的：
  1. 速度 MAE / RMSE —— 预测本身准不准；
  2. **实现行程时间的多付比例** —— 拿预测速度算出来的路，放回真实速度里
     走一遍，比用真实速度算出的最优路多花多少。预测误差只有变成绕错路
     才值钱，MAE 低但路一样，就等于白预测。
"""
from __future__ import annotations

import numpy as np

from .network import BUCKET_MIN, N_BUCKETS

HORIZON = 3                          # 预测 3 个 5 分钟槽 = 15 分钟


def make_history(net, days: int = 21, seed: int = 11) -> np.ndarray:
    """(days, N_BUCKETS, n_edges) 的历史观测速度。

    = 剖面 × 当日整体系数（天气/活动）+ 偶发事件（某边某时段掉速）+ 观测噪声。
    事件是刻意加的：没有事件的话，seasonal 基线会强到没有对照价值。
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(net.speed).T                       # (N_BUCKETS, n_edges)
    hist = np.empty((days, N_BUCKETS, base.shape[1]))
    for d in range(days):
        day_factor = 1.0 + rng.normal(0.0, 0.05)
        x = base * day_factor
        for _ in range(rng.integers(2, 6)):              # 每天 2-5 起事件
            e = rng.integers(0, base.shape[1])
            b0 = rng.integers(0, N_BUCKETS - 12)
            span = rng.integers(4, 12)
            x[b0:b0 + span, e] *= rng.uniform(0.45, 0.75)
        hist[d] = np.maximum(3.0, x * (1.0 + rng.normal(0.0, 0.03, x.shape)))
    return hist


def _xy(hist: np.ndarray, lags: int = 6):
    """把历史摊成监督学习样本：滞后速度 + 时刻正余弦 -> HORIZON 步后的速度。"""
    days, T, E = hist.shape
    X, Y = [], []
    for d in range(1, days):
        for b in range(lags, T - HORIZON):
            lag = hist[d, b - lags + 1:b + 1, :]          # (lags, E)
            yday = hist[d - 1, b + HORIZON, :]            # 昨天同刻
            ang = 2 * np.pi * b / T
            for e in range(E):
                X.append(np.concatenate([lag[:, e], [yday[e], np.sin(ang), np.cos(ang)]]))
                Y.append(hist[d, b + HORIZON, e])
    return np.asarray(X), np.asarray(Y)


LAGS = 6


def evaluate(hist: np.ndarray, model: str = "ridge", seed: int = 3) -> dict:
    """按天切分训练/测试，返回 MAE / RMSE。

    **四个模型评在同一批样本上。**第一版不是这样：persistence 和 seasonal
    走的是各自的切片，样本数 129920 / 113680 / 113274 三者不同，MAE 之间
    根本没法比。现在统一用 `_xy` 造出的 (X, Y)，persistence 直接取最后一个
    滞后列、seasonal 取昨天同刻那一列——它们本来就是 X 的两个特征。
    """
    days = hist.shape[0]
    split = int(days * 0.7)
    tr, te = hist[:split], hist[split - 1:]              # 测试集保留一天的前文
    Xte, Yte = _xy(te)

    if model == "persistence":
        pred = Xte[:, LAGS - 1]
    elif model == "seasonal":
        pred = Xte[:, LAGS]
    elif model in ("ridge", "lstm"):
        Xtr, Ytr = _xy(tr)
        if model == "ridge":
            from sklearn.linear_model import Ridge
            pred = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte)
        else:
            pred = _lstm_fit_predict(Xtr, Ytr, Xte, seed)
    else:
        raise ValueError(model)

    err = np.abs(np.asarray(pred).ravel() - np.asarray(Yte).ravel())
    return {"model": model,
            "mae_kmh": round(float(err.mean()), 4),
            "rmse_kmh": round(float(np.sqrt((err ** 2).mean())), 4),
            "n": int(err.size)}


def _lstm_fit_predict(Xtr, Ytr, Xte, seed: int):
    """单层 LSTM。torch 缺席时由调用方跳过。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    lags = 6

    def shape(X):
        seq = torch.tensor(X[:, :lags], dtype=torch.float32).unsqueeze(-1)
        ctx = torch.tensor(X[:, lags:], dtype=torch.float32)
        return seq, ctx

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + 3, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, seq, ctx):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], ctx], dim=1)).squeeze(-1)

    seq, ctx = shape(Xtr)
    y = torch.tensor(Ytr, dtype=torch.float32)
    mu, sd = float(y.mean()), float(y.std() + 1e-6)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.SmoothL1Loss()
    n = len(y)
    for _ in range(8):
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(model(seq[j], ctx[j]), (y[j] - mu) / sd)
            loss.backward()
            opt.step()
    with torch.no_grad():
        s, c = shape(Xte)
        return (model(s, c) * sd + mu).numpy()


def predicted_speed_grid(hist: np.ndarray, model: str, seed: int = 3) -> np.ndarray:
    """用最后一天做一遍逐槽预测，返回 (N_BUCKETS, n_edges) 的预测速度网格。

    persistence/seasonal 直接构造；ridge/lstm 用前面的天训练后逐槽预测。
    """
    days, T, E = hist.shape
    if model == "seasonal":
        return hist[-2].copy()
    if model == "persistence":
        out = hist[-1].copy()
        out[HORIZON:] = hist[-1][:-HORIZON]
        return out
    Xtr, Ytr = _xy(hist[:-1])
    lags = 6
    rows = []
    for b in range(T):
        bb = min(max(b, lags - 1), T - 1)
        lag = hist[-1, bb - lags + 1:bb + 1, :]
        yday = hist[-2, bb, :]
        ang = 2 * np.pi * b / T
        rows.append(np.stack([np.concatenate([lag[:, e], [yday[e], np.sin(ang), np.cos(ang)]])
                              for e in range(E)]))
    X = np.concatenate(rows)
    if model == "ridge":
        from sklearn.linear_model import Ridge
        p = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(X)
    else:
        p = _lstm_fit_predict(Xtr, Ytr, X, seed)
    return np.maximum(3.0, p.reshape(T, E))
