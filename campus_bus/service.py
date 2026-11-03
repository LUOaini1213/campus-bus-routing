# -*- coding: utf-8 -*-
"""最小路由服务：标准库 http.server，零依赖，容器里跑得起来。

    python -m campus_bus.service --port 8080
    curl "http://127.0.0.1:8080/route?from=west_gate&to=dorm_c&depart=470"

用标准库而不是 FastAPI，是因为这层要证明的是**延迟**，不是框架。
路网在进程启动时加载一次并常驻，请求路径上只有一次堆 Dijkstra。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .network import Network
from .routing import astar, dijkstra

_NET: Network | None = None


def net() -> Network:
    global _NET
    if _NET is None:
        _NET = Network.load()
    return _NET


def handle_route(q: dict) -> tuple[int, dict]:
    n = net()
    src = (q.get("from") or [""])[0]
    dst = (q.get("to") or [""])[0]
    solver = (q.get("solver") or ["dijkstra"])[0]
    if src not in n.nodes or dst not in n.nodes:
        return 400, {"error": "unknown node", "known": sorted(n.nodes)[:5] + ["..."]}
    try:
        depart = float((q.get("depart") or ["480"])[0])
    except ValueError:
        return 400, {"error": "depart must be minutes past midnight"}
    # float() 认得 nan / inf / 1e400，它们解析成功却会在 Network.bucket 的
    # int(t // 5) 处抛 ValueError —— 在 BaseHTTPRequestHandler 里那意味着
    # 栈打进 stderr、**一个字节的响应都不发**、keep-alive 连接被切断，
    # 客户端只能等到超时。所以解析成功还不够，得是有限数。
    if not math.isfinite(depart):
        return 400, {"error": "depart must be finite", "got": (q.get("depart") or [""])[0]}
    if solver not in ("dijkstra", "astar"):
        # 旧版静默降级成 dijkstra，却在响应里原样回显 solver=bogus，
        # 客户端会以为自己用的是别的求解器。
        return 400, {"error": "unknown solver", "known": ["dijkstra", "astar"], "got": solver}
    fn = astar if solver == "astar" else dijkstra
    t0 = time.perf_counter()
    r = fn(n, src, dst, depart)
    took = (time.perf_counter() - t0) * 1000.0
    if r.arrival == float("inf"):
        return 404, {"error": "no path", "from": src, "to": dst}
    return 200, {"from": src, "to": dst, "depart_min": depart,
                 "arrival_min": round(r.arrival, 4),
                 "travel_min": round(r.arrival - depart, 4),
                 "path": r.path, "expansions": r.expansions,
                 "solver": solver, "solve_ms": round(took, 4)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):                                    # noqa: N802
        u = urlparse(self.path)
        if u.path == "/health":
            code, body = 200, {"ok": True, "nodes": net().n_nodes(), "edges": net().n_edges()}
        elif u.path == "/route":
            code, body = handle_route(parse_qs(u.query))
        else:
            code, body = 404, {"error": "try /route or /health"}
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):                           # 压测时别刷屏
        pass


LISTEN_BACKLOG = 128


class RouteServer(ThreadingHTTPServer):
    """`ThreadingHTTPServer` with a listen backlog sized for concurrent clients.

    socketserver's default `request_queue_size` is 5. With 16 concurrent clients
    opening connections, the accept queue overflows, connects are retried by the
    client, and the latency tail jumps in steps of roughly 0.5 s / 1 s. That is
    what the 2026-09 table's 515 ms / 1027 ms tail measured (README, latency
    section). `bench/bench_service.py --backlog 5` reproduces it on purpose.
    """
    request_queue_size = LISTEN_BACKLOG
    daemon_threads = True


def serve(port: int = 8080, host: str = "0.0.0.0",
          backlog: int | None = None) -> ThreadingHTTPServer:
    """Build the server (not started). `backlog` overrides the listen queue size;
    it exists only so the benchmark can reproduce the old default of 5."""
    net()                                                # 先把图加载好再开门
    if backlog is None:
        return RouteServer((host, port), Handler)
    cls = type("RouteServerBacklog%d" % backlog, (RouteServer,),
               {"request_queue_size": backlog})
    return cls((host, port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = serve(args.port, args.host)
    print("listening on %s:%d  (%d nodes / %d edges)" % (
        args.host, args.port, net().n_nodes(), net().n_edges()), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
