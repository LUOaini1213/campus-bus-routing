# -*- coding: utf-8 -*-
"""HTTP 端到端延迟：把服务起在本进程里，用真实 socket 压一遍。

原始简历写的是「Docker 部署响应 <50 ms」。那是一个**端到端**的数，
不是算法耗时——所以这里量的是 HTTP 往返，不是 `dijkstra()` 的函数耗时
（后者在 bench_all.py 里，是另一个数量级）。

    python bench/bench_service.py            # 单线程顺序压
    python bench/bench_service.py --concurrency 16
    python bench/bench_service.py --concurrency 16 --fresh-conn --backlog 5

`--backlog 5` reproduces socketserver's default listen queue, which the service
used before 2026-11-03; it is kept only to show where the old 0.5 s / 1 s tail
came from. Without it the server uses `campus_bus.service.LISTEN_BACKLOG`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import http.client
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_bus.service import net, serve                # noqa: E402

QUERIES = [("west_gate", "dorm_c", 470), ("south_gate", "lab_block", 1075),
           ("dorm_b", "east_gate", 500), ("stadium", "canteen_n", 150),
           ("admin", "teach_4", 730), ("gym", "west_gate", 1050)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fresh-conn", action="store_true",
                    help="每次请求新建 TCP 连接（默认复用，像真实客户端那样）")
    ap.add_argument("--backlog", type=int, default=None,
                    help="override the listen backlog (default: LISTEN_BACKLOG)")
    args = ap.parse_args()

    srv = serve(args.port, "127.0.0.1", backlog=args.backlog)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    local = threading.local()

    def conn() -> http.client.HTTPConnection:
        c = getattr(local, "c", None)
        if c is None or args.fresh_conn:
            if c is not None:
                c.close()
            c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=5)
            local.c = c
        return c

    def one(i: int) -> float:
        s, t, d = QUERIES[i % len(QUERIES)]
        path = "/route?from=%s&to=%s&depart=%d" % (s, t, d)
        t0 = time.perf_counter()
        c = conn()
        c.request("GET", path)
        json.loads(c.getresponse().read())
        return (time.perf_counter() - t0) * 1000.0

    for i in range(40):                                  # 预热，别把首次开销算进去
        one(i)
    if args.concurrency == 1:
        lat = [one(i) for i in range(args.n)]
    else:
        with ThreadPoolExecutor(args.concurrency) as pool:
            lat = list(pool.map(one, range(args.n)))
    srv.shutdown()
    srv.server_close()

    res = {"requests": args.n, "concurrency": args.concurrency,
           "connection": "fresh-per-request" if args.fresh_conn else "keep-alive",
           "listen_backlog": srv.request_queue_size,
           "platform": sys.platform,
           "nodes": net().n_nodes(), "edges": net().n_edges(),
           "latency_ms": {"p50": round(statistics.median(lat), 3),
                          "p95": round(float(np.percentile(lat, 95)), 3),
                          "p99": round(float(np.percentile(lat, 99)), 3),
                          "max": round(max(lat), 3)},
           "note": ("HTTP 往返，含 socket、JSON 编解码与服务端求解；"
                    "机器不同结果不同，这个数只对本机有意义。")}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
