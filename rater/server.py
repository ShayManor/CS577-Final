import argparse
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from nuance.records import read_jsonl
from rater.db import RatingsDB
from rater.queue import assign_items, build_items, order_items

DIST = Path(__file__).parent / "web" / "dist"


def canonical(name) -> str:
    """Rater names ignore case and are kept as "Shay": "shay", "SHAY" and "Shay" are one rater."""
    s = str(name or "").strip()
    return s[:1].upper() + s[1:].lower()


def create_app(batch_dir: Path, db_path: Path, raters=("Shay", "Henry")) -> Flask:
    batch_dir = Path(batch_dir)
    app = Flask(__name__)
    items = order_items(build_items(read_jsonl(batch_dir / "clips.jsonl")))
    by_id = {it.item_id: it for it in items}
    # Known raters split the clips; any other name still gets the full queue
    queues = assign_items(items, [canonical(r) for r in raters])
    db = RatingsDB(db_path)

    @app.get("/api/next")
    def next_item():
        rater = canonical(request.args.get("rater"))
        if not rater:
            return jsonify(error="rater required"), 400
        done, flagged = db.done_item_ids(rater), db.flagged_clip_ids(rater)
        handled = 0
        queue = queues.get(rater, items)
        for it in queue:
            if it.item_id in done or it.clip_id in flagged:
                handled += 1
                continue
            return jsonify(item_id=it.item_id, clip_id=it.clip_id, instruction=it.instruction,
                           # ?v= changes the URL so browsers and Cloudflare drop clips they cached as missing
                           video_url=f"/clips/{it.clip_id}.mp4?v=2", position=handled + 1,
                           total=len(queue))
        return jsonify(done=True)

    @app.get("/api/prev")
    def prev_item():
        """The last scored item before `item_id` in the rater's queue, so its score can be overridden."""
        rater = canonical(request.args.get("rater"))
        if not rater:
            return jsonify(error="rater required"), 400
        queue, scores = queues.get(rater, items), db.scores(rater)
        ids = [it.item_id for it in queue]
        cur = request.args.get("item_id")
        for k in range((ids.index(cur) if cur in ids else len(ids)) - 1, -1, -1):
            it = queue[k]
            if it.item_id in scores:
                return jsonify(item_id=it.item_id, clip_id=it.clip_id, instruction=it.instruction,
                               video_url=f"/clips/{it.clip_id}.mp4?v=2", position=k + 1,
                               total=len(queue), score=scores[it.item_id])
        return jsonify(none=True)

    @app.post("/api/rate")
    def rate():
        body = request.get_json(silent=True) or {}
        rater, score = canonical(body.get("rater")), body.get("score")
        item = by_id.get(body.get("item_id"))
        if not rater or item is None or type(score) is not int or not 1 <= score <= 5:
            return jsonify(error="bad request"), 400
        if body.get("override") is True:
            return jsonify(ok=True) if db.rerate(rater, item, score) else (jsonify(error="not rated"), 404)
        if not db.rate(rater, item, score):
            return jsonify(error="duplicate"), 409
        return jsonify(ok=True)

    @app.post("/api/flag")
    def flag():
        body = request.get_json(silent=True) or {}
        rater = canonical(body.get("rater"))
        item = by_id.get(body.get("item_id"))
        if not rater or item is None:
            return jsonify(error="bad request"), 400
        db.flag(rater, item)
        return jsonify(ok=True)

    @app.get("/clips/<clip_id>.mp4")
    def clip(clip_id):
        return send_from_directory((batch_dir / "videos").resolve(), f"{clip_id}.mp4")

    @app.get("/")
    def index():
        if not (DIST / "index.html").exists():
            return "UI not built", 200, {"Content-Type": "text/plain; charset=utf-8"}
        return send_from_directory(DIST, "index.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(DIST / "assets", name)

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    create_app(Path(a.batch), Path(a.db)).run(host=a.host, port=a.port)
