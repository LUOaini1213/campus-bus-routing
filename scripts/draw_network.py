# -*- coding: utf-8 -*-
"""画一张 SVG：路网 + 同一对起终点在高峰/平峰选出的两条不同的路。

手写 SVG，不引入 matplotlib —— 这张图要能在 CI 里无依赖地重画并比对。

    python scripts/draw_network.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus.network import Network                # noqa: E402
from campus_bus.routing import dijkstra               # noqa: E402

W, H, PAD = 900, 660, 46
SRC, DST = "west_gate", "dorm_c"
PEAK, OFF = 460.0, 150.0


def main() -> int:
    net = Network.load()
    xs = [p[0] for p in net.nodes.values()]
    ys = [p[1] for p in net.nodes.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = (W - 2 * PAD) / (x1 - x0)
    sy = (H - 2 * PAD) / (y1 - y0)

    def P(n):                                    # 世界坐标 -> 画布（y 翻转）
        x, y = net.nodes[n]
        return PAD + (x - x0) * sx, H - PAD - (y - y0) * sy

    r_peak = dijkstra(net, SRC, DST, PEAK)
    r_off = dijkstra(net, SRC, DST, OFF)
    # 必须是 list 不是 set：set 的迭代顺序随进程哈希种子变，同一份输入会画出
    # 行序不同的 SVG，CI 里那条「重画应当一致」的校验就形同虚设。
    peak_e = list(zip(r_peak.path, r_peak.path[1:]))
    off_e = list(zip(r_off.path, r_off.path[1:]))

    L = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
         'font-family="ui-sans-serif,system-ui,sans-serif">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#fbfbfd"/>' % (W, H)]

    kind_col = {"main": "#b9c2d0", "campus": "#cdd4de", "service": "#dfe4ea"}
    for e in net.edges:                          # 底层：全部边
        (ax, ay), (bx, by) = P(e.u), P(e.v)
        L.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"/>'
                 % (ax, ay, bx, by, kind_col[e.kind], 3.4 if e.kind == "main" else 2.0))

    def draw_route(edges, colour, width, dash=""):
        for a, b in edges:
            (ax, ay), (bx, by) = P(a), P(b)
            L.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                     'stroke-width="%.1f" stroke-linecap="round"%s/>'
                     % (ax, ay, bx, by, colour, width, dash))

    draw_route(off_e, "#2f9e6e", 7.0, ' stroke-dasharray="11 7"')
    draw_route(peak_e, "#d1495b", 7.0)

    for n in net.nodes:                          # 节点
        x, y = P(n)
        big = n in (SRC, DST)
        L.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="#fff" stroke-width="2"/>'
                 % (x, y, 8.5 if big else 5.0, "#1f2933" if big else "#7b8794"))
        L.append('<text x="%.1f" y="%.1f" font-size="%d" fill="#3e4c59" text-anchor="middle">%s</text>'
                 % (x, y - (14 if big else 10), 13 if big else 11, n))

    def legend(y, colour, dash, text):
        L.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="7" '
                 'stroke-linecap="round"%s/>' % (PAD, y, PAD + 46, y, colour, dash))
        L.append('<text x="%d" y="%d" font-size="14" fill="#1f2933">%s</text>'
                 % (PAD + 58, y + 5, text))

    legend(26, "#d1495b", "", "早高峰 07:40 出发 —— %s（%.2f min）"
           % (" → ".join(r_peak.path), r_peak.arrival - PEAK))
    legend(50, "#2f9e6e", ' stroke-dasharray="11 7"', "平峰 02:30 出发 —— %s（%.2f min）"
           % (" → ".join(r_off.path), r_off.arrival - OFF))
    L.append('<text x="%d" y="%d" font-size="12" fill="#7b8794">'
             '合成路网 23 节点 / 58 有向边；同一对起终点，出发时刻不同，最优路不同。</text>'
             % (PAD, H - 14))
    L.append("</svg>")

    out = ROOT / "docs" / "network.svg"
    out.parent.mkdir(exist_ok=True)
    # 显式 newline="\n"：Windows 上 write_text 会写出 CRLF、Linux 上写 LF，
    # 两边生成的同一张图会 diff 不过。
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote", out)
    print("peak :", " -> ".join(r_peak.path))
    print("off  :", " -> ".join(r_off.path))
    assert set(peak_e) != set(off_e), "两条路应当不同，否则这张图没意义"
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
