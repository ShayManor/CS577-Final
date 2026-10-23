import pytest

from nuance.records import ClipRecord, write_jsonl
from rater.server import create_app

PHRASES = ["slightly faster", "much faster", "slightly slower", "much slower"]


def make_record(i):
    ins = PHRASES[i % 4]
    alts = [p for p in PHRASES if p != ins][:2]
    return ClipRecord(
        clip_id=f"clip{i:02d}", batch_id="b", decision_id=f"d{i // 3}", action_type="speed",
        scene_id="s", path=[[0, 0, 0], [1, 0, 0]], decision_position=[0, 0, 0],
        decision_yaw=0.0, context={}, start_speed=1.0, candidate_speed=1.2,
        candidate_rank="r", instruction=ins, alt_instructions=alts,
        video=f"videos/clip{i:02d}.mp4", front_frame="f.png", fps=10,
        duration_s=10.0, t_decision_s=2.0,
    )


@pytest.fixture
def records():
    return [make_record(i) for i in range(20)]


@pytest.fixture
def batch_dir(tmp_path, records):
    d = tmp_path / "batch"
    (d / "videos").mkdir(parents=True)
    write_jsonl(d / "clips.jsonl", records)
    for r in records:
        (d / "videos" / f"{r.clip_id}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42dummy")
    return d


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "ratings.sqlite"


@pytest.fixture
def client(batch_dir, db_path):
    return create_app(batch_dir, db_path).test_client()
