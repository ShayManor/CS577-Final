from rater.server import create_app


def nxt(c, rater="a"):
    return c.get(f"/api/next?rater={rater}").get_json()


def rate(c, item_id, score=3, rater="a"):
    return c.post("/api/rate", json={"rater": rater, "item_id": item_id, "score": score})


def test_next_advances_after_rate(client):
    n1 = nxt(client)
    assert n1["position"] == 1 and n1["total"] == 60
    assert n1["video_url"] == f"/clips/{n1['clip_id']}.mp4?v=2"
    assert rate(client, n1["item_id"]).status_code == 200
    n2 = nxt(client)
    assert n2["item_id"] != n1["item_id"] and n2["position"] == 2


def test_resume_after_reload(batch_dir, db_path):
    c = create_app(batch_dir, db_path).test_client()
    seen = []
    for _ in range(3):
        n = nxt(c)
        seen.append(n["item_id"])
        rate(c, n["item_id"])
    c2 = create_app(batch_dir, db_path).test_client()
    n4 = nxt(c2)
    assert n4["item_id"] not in seen and n4["position"] == 4


def test_duplicate_rate_409(client):
    n = nxt(client)
    assert rate(client, n["item_id"]).status_code == 200
    assert rate(client, n["item_id"], 4).status_code == 409


def test_bad_score_400(client):
    n = nxt(client)
    for s in (0, 6, "3", None, 3.0, True):
        assert rate(client, n["item_id"], s).status_code == 400
    r = client.post("/api/rate", json={"rater": "a", "item_id": n["item_id"]})
    assert r.status_code == 400
    assert rate(client, n["item_id"], rater="").status_code == 400
    assert rate(client, "nope:0").status_code == 400


def test_flag_skips_clip(client):
    n = nxt(client)
    assert client.post("/api/flag", json={"rater": "a", "item_id": n["item_id"]}).status_code == 200
    for _ in range(57):
        m = nxt(client)
        assert m["clip_id"] != n["clip_id"]
        rate(client, m["item_id"])
    assert nxt(client) == {"done": True}
    assert nxt(client, "b")["total"] == 60


def test_done(client):
    for _ in range(60):
        rate(client, nxt(client)["item_id"])
    assert nxt(client) == {"done": True}


def test_raters_independent(client):
    n = nxt(client, "a")
    rate(client, n["item_id"], rater="a")
    assert nxt(client, "b")["item_id"] == n["item_id"]
    assert rate(client, n["item_id"], rater="b").status_code == 200


def test_clip_served_and_root(client):
    n = nxt(client)
    assert client.get(n["video_url"]).status_code == 200
    assert client.get("/").status_code == 200


def test_rater_name_ignores_case(client):
    n = nxt(client, rater="shay")
    assert rate(client, n["item_id"], rater="SHAY").status_code == 200
    assert rate(client, n["item_id"], rater="Shay").status_code == 409
    assert nxt(client, rater="Shay")["position"] == 2


def test_prev_overrides_rating(client, db_path):
    import sqlite3
    a = nxt(client)
    rate(client, a["item_id"], 2)
    b = nxt(client)
    rate(client, b["item_id"], 4)
    cur = nxt(client)
    p = client.get(f"/api/prev?rater=a&item_id={cur['item_id']}").get_json()
    assert p["item_id"] == b["item_id"] and p["score"] == 4 and p["position"] == 2
    p2 = client.get(f"/api/prev?rater=a&item_id={p['item_id']}").get_json()
    assert p2["item_id"] == a["item_id"] and p2["score"] == 2
    assert client.get(f"/api/prev?rater=a&item_id={a['item_id']}").get_json() == {"none": True}
    r = client.post("/api/rate", json={"rater": "a", "item_id": a["item_id"], "score": 5, "override": True})
    assert r.status_code == 200
    assert nxt(client)["item_id"] == cur["item_id"]
    rows = sqlite3.connect(db_path).execute("SELECT item_id, score FROM ratings ORDER BY score").fetchall()
    assert rows == [(b["item_id"], 4), (a["item_id"], 5)]
    r = client.post("/api/rate", json={"rater": "a", "item_id": cur["item_id"], "score": 5, "override": True})
    assert r.status_code == 404
