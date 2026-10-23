from rater.server import create_app


def nxt(c, rater="a"):
    return c.get(f"/api/next?rater={rater}").get_json()


def rate(c, item_id, score=3, rater="a"):
    return c.post("/api/rate", json={"rater": rater, "item_id": item_id, "score": score})


def test_next_advances_after_rate(client):
    n1 = nxt(client)
    assert n1["position"] == 1 and n1["total"] == 60
    assert n1["video_url"] == f"/clips/{n1['clip_id']}.mp4"
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
