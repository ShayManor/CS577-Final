import json
import sqlite3

import pytest

from nuance.reward_model import examples, load_clips, load_ratings, metrics, split
from nuance.train_policy import prompt_rows, sft_rows


def _clip(cid, did, ins="go faster", alts=("go slower",), scene="s1"):
    path = [[0, 0, 1.0], [0, 0, 0.0], [0, 0, -3.0]]
    return {"clip_id": cid, "batch_id": "b", "decision_id": did, "scene_id": scene, "path": path,
            "decision_position": path[1], "decision_yaw": 0.0, "start_speed": 0.6, "candidate_speed": 0.9,
            "instruction": ins, "alt_instructions": list(alts), "video": f"videos/{cid}.mp4",
            "front_frame": f"frames/{did}.png", "person": None, "referent": None}


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "rate"
    b.mkdir()
    clips = [_clip(f"c{i}", f"r{i // 3}", scene="s1" if i < 6 else "s2") for i in range(9)]
    (b / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    (b / "sources.json").write_text(json.dumps({"r0": "p/d0", "r1": "p/d1", "r2": "q/d5"}))
    db = tmp_path / "ratings.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE ratings(rater TEXT, item_id TEXT, clip_id TEXT, instruction TEXT, score INTEGER NULL,"
                  " flagged INTEGER, created_at TEXT)")
        rows = [("Shay", f"c{i}:0", f"c{i}", "go faster", 1 + i % 5, 0, "") for i in range(9)]
        rows += [("Shay", "c0:1", "c0", "go slower", 5, 0, ""), ("Shay", "c1:1", "c1", "go slower", None, 1, "")]
        c.executemany("INSERT INTO ratings VALUES (?,?,?,?,?,?,?)", rows)
    return b, db


def test_join_groups_and_flags(batch):
    b, db = batch
    clips = load_clips([b])
    assert clips["c4"]["group"] == "p/d1" and clips["c0"]["video"] == str(b / "videos/c0.mp4")
    exs = examples(load_ratings(db), clips)
    assert len(exs) == 10  # the flagged rating is dropped


def test_split_keeps_decisions_together(batch):
    b, db = batch
    exs = examples(load_ratings(db), load_clips([b]))
    tr, va = split(exs, 0.34, seed=1)
    assert not {e["group"] for e in tr} & {e["group"] for e in va}
    assert len(tr) + len(va) == len(exs) and va


def test_missing_clip_raises(batch):
    b, db = batch
    clips = load_clips([b])
    del clips["c3"]
    with pytest.raises(ValueError):
        examples(load_ratings(db), clips)


def test_metrics_perfect_and_reversed():
    exs = [{"clip_id": "a", "group": "g", "instruction": f"i{k}", "score": k, "rater": "Shay"} for k in (1, 3, 5)]
    good = metrics(exs, [1.0, 3.0, 5.0])
    assert good["spearman"] == pytest.approx(1.0) and good["pair_acc"] == 1.0 and good["pairs"] == 3
    assert metrics(exs, [5.0, 3.0, 1.0])["pair_acc"] == 0.0


def test_sft_and_prompt_rows(batch):
    b, db = batch
    clips = load_clips([b])
    rows = sft_rows(load_ratings(db), clips, exclude=set())
    assert {r["images"][0] for r in rows} <= {c["front_frame"] for c in clips.values()}
    assert all(json.loads(r["completion"][0]["content"])["goal"] == [3.0, 0.0] for r in rows)
    assert len(rows) == 4  # scores 4, 5 under "go faster" (c3, c4, c8) and c0 under "go slower"
    prompts = prompt_rows(clips, exclude={"s2"})
    assert len(prompts) == 2 * 2  # decisions r0, r1 in s1, each under its instruction and one alternative
    assert json.loads(prompts[0]["decision"])["scene_id"] == "s1"
