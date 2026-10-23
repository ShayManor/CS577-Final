import random

from nuance.config import PHRASES
from nuance.records import ClipRecord, make_clip_id, pick_alt_instructions, read_jsonl, write_jsonl


def _record(i):
    return ClipRecord(
        clip_id=make_clip_id("b", i, "near"), batch_id="b", decision_id=f"d{i}", action_type="speed",
        scene_id="s", path=[[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], decision_position=[0.5, 0.0, -0.5],
        decision_yaw=0.1, context={"clutter": "low"}, start_speed=0.5, candidate_speed=0.8,
        candidate_rank="near", instruction="much faster", alt_instructions=["slightly slower", "much slower"],
        video=f"videos/{i}.mp4", front_frame=f"frames/d{i}.png", fps=30, duration_s=8.0, t_decision_s=2.0)


def test_roundtrip(tmp_path):
    recs = [_record(i) for i in range(3)]
    p = tmp_path / "clips.jsonl"
    write_jsonl(p, recs)
    assert read_jsonl(p) == recs


def test_alt_instructions():
    for phrase in PHRASES:
        for seed in range(50):
            alts = pick_alt_instructions(phrase, random.Random(seed))
            assert len(alts) == 2 and len(set(alts)) == 2
            assert phrase not in alts
            assert set(alts) <= set(PHRASES)


def test_clip_id_stable_and_opaque():
    assert make_clip_id("b", 3, "faster") == make_clip_id("b", 3, "faster")
    cid = make_clip_id("b", 3, "faster")
    assert len(cid) == 10
    assert not any(w in cid for w in ("slower", "near", "faster"))
