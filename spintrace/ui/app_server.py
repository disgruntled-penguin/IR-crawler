"""Local web app: search with originality, tracing a suspect to its source, and the per-site integrity table.

Everything is computed by the same modules the command line uses; this file only maps URLs to them.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import service

PAGE = Path(__file__).with_name("app.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/search":
                self._send(service.search(q.get("q", ""), use_g=q.get("g", "1") == "1",
                                          crawl_only=q.get("crawl", "1") == "1", k=int(q.get("k", 10))))
            elif u.path == "/api/trace":
                self._send(service.trace(q["doc"]))
            elif u.path == "/api/integrity":
                self._send(service.integrity())
            else:
                self._send({"error": "not found"}, code=404)
        except (KeyError, ValueError) as e:
            self._send({"error": str(e)}, code=400)

    def log_message(self, *args):
        pass


def serve(host="127.0.0.1", port=8766):
    threading.Thread(target=service.warm, daemon=True).start()
    print(f"SpinTrace app on http://{host}:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
