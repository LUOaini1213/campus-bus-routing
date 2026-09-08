# -*- coding: utf-8 -*-
"""车队规模：从周期时间到车辆数，以及「12 → 10」需要多大的提速。

排班里最基本的一条关系：

    车辆数 = ceil( (环线运行时间 + 折返站停留) / 发车间隔 )

它是**天花板函数**，所以车辆数不随提速连续下降，而是一级一级掉。这也解释
了为什么「行程时间降 x%」和「少几辆车」不是一回事：跨过台阶才少一辆，
没跨过就一辆不少。下面这个函数就是把台阶画出来。
"""
from __future__ import annotations

import math


def fleet_size(cycle_min: float, headway_min: float, layover_min: float = 0.0) -> int:
    return int(math.ceil((cycle_min + layover_min) / headway_min))


def breakeven(cycle_min: float, headway_min: float, layover_min: float = 0.0) -> dict:
    """当前车队规模，以及少一辆车所需的周期时间与降幅。"""
    now = fleet_size(cycle_min, headway_min, layover_min)
    target = now - 1
    if target <= 0:
        return {"fleet": now, "next_step": None}
    need_cycle = target * headway_min - layover_min
    return {
        "fleet": now,
        "cycle_min": round(cycle_min, 2),
        "headway_min": headway_min,
        "layover_min": layover_min,
        "next_fleet": target,
        "cycle_needed_min": round(need_cycle, 2),
        "reduction_needed_pct": round((cycle_min - need_cycle) / cycle_min * 100, 2),
    }


def ladder(cycle_min: float, headway_min: float, layover_min: float = 0.0,
           span_pct: float = 40.0, step_pct: float = 1.0) -> list[dict]:
    """把周期时间从 0 压到 span_pct%，记录车辆数每次下降的位置。"""
    out, last = [], None
    pct = 0.0
    while pct <= span_pct + 1e-9:
        c = cycle_min * (1 - pct / 100.0)
        f = fleet_size(c, headway_min, layover_min)
        if f != last:
            out.append({"reduction_pct": round(pct, 2), "cycle_min": round(c, 2), "fleet": f})
            last = f
        pct += step_pct
    return out
