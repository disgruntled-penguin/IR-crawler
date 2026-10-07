"""Local web interface for the team judgments; a front end over the same sheets as `spintrace judge`.

Judges see both texts in full but never SpinTrace's score, verdict or the other judge's label, so the labels stay blind.
"""
import datetime as dt
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import store
from ..eval import judged, judgepack

PAGE = Path(__file__).with_name("judge.html")
KEYS = {"live": ("suspect", "source"), "hardneg": ("suspect", "earlier"), "search": ("doc_id",)}
ALLOWED = {"live": {"derived", "not_derived"}, "hardneg": {"derived", "not_derived"},
           "search": {"relevant", "not_relevant"}}


_packed = {}


def _pack():
    """Texts from evaldata/judge_pack.json, for a judge who has no crawl database."""
    if "docs" not in _packed:
        _packed["docs"] = judgepack.load()
    return _packed["docs"]


def _doc(con, doc_id):
    r = con.execute("SELECT url, site, title, published, text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    if not r:
        r = _pack().get(doc_id)
    if not r:
        return {"doc_id": doc_id, "missing": True}
    when = dt.datetime.fromtimestamp(r["published"]).strftime("%Y-%m-%d %H:%M") if r["published"] else None
    return {"doc_id": doc_id, "url": r["url"], "site": r["site"], "title": r["title"], "published": when,
            "text": r["text"]}


def progress():
    out = {}
    for sheet in KEYS:
        if judged.SHEETS[sheet].exists():
            rows = judged.read_rows(sheet)
            out[sheet] = {"rows": len(rows), **{f"judge{j}": sum(bool(r.get(f"judge{j}")) for r in rows) for j in (1, 2)}}
    return out


def item(sheet, judge, i=None):
    rows = judged.read_rows(sheet)
    col = f"judge{judge}"
    if i is None:
        i = next((k for k, r in enumerate(rows) if not r.get(col)), len(rows))
    if i >= len(rows):
        return {"i": i, "n": len(rows), "finished": True}
    r = rows[i]
    con = store.connect()
    docs = [_doc(con, r[k]) for k in KEYS[sheet]]
    con.close()
    return {"i": i, "n": len(rows), "label": r.get(col, ""), "query": r.get("query"), "docs": docs,
            "done": sum(bool(x.get(col)) for x in rows)}


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
            elif u.path == "/api/progress":
                self._send(progress())
            elif u.path == "/api/item":
                i = int(q["i"]) if q.get("i") else None
                self._send(item(q["sheet"], int(q["judge"]), i))
            else:
                self._send({"error": "not found"}, code=404)
        except (KeyError, ValueError, IndexError) as e:
            self._send({"error": str(e)}, code=400)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sheet, judge, i, value = body["sheet"], int(body["judge"]), int(body["i"]), body["label"]
            if sheet not in KEYS or judge not in (1, 2) or (value and value not in ALLOWED[sheet]):
                raise ValueError("bad label")
            judged.set_label(sheet, i, judge, value)
            self._send({"ok": True})
        except (KeyError, ValueError, IndexError, json.JSONDecodeError) as e:
            self._send({"error": str(e)}, code=400)

    def log_message(self, *args):
        pass


def serve(host="127.0.0.1", port=8765):
    print(f"Judging interface on http://{host}:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
