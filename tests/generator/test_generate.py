import json

import imageio
import numpy as np

from nuance import generate
from nuance.config import HOLD_S
from nuance.motion import trajectory
from nuance.records import read_jsonl


def test_small_batch(tmp_path):
    with open("configs/pilot_scenes.txt") as f:
        scenes = [f.readline().strip(), f.readline().strip()]
    scene_file = tmp_path / "scenes.txt"
    scene_file.write_text("\n".join(scenes) + "\n")
    out = tmp_path / "batch"
    generate.main(["--scenes", str(scene_file), "--n-decisions", "2", "--pool-per-scene", "5", "--seed", "0",
                   "--batch-id", "test", "--out", str(out)])
    records = read_jsonl(out / "clips.jsonl")
    assert len(records) == 6
    assert len(list((out / "videos").glob("*.mp4"))) == 6
    assert len(list((out / "frames").glob("*.png"))) == 2
    for r in records:
        assert r.action_type == "speed"
        assert len(r.alt_instructions) == 2
        assert r.duration_s <= 20.0
        # The clip ends HOLD_S after Spot reaches the end of the line; the walker is never faster than planned
        # (it tops out near 1.3 m/s, so fast candidates arrive late).
        traj = trajectory(np.array(r.path), r.start_speed, r.candidate_speed)
        assert np.allclose(traj.positions[-1], r.path[-1])
        assert r.duration_s >= 0.9 * traj.times[-1] + HOLD_S
        assert (out / r.video).exists() and (out / r.front_frame).exists()
    by_decision = {}
    for r in records:
        by_decision.setdefault(r.decision_id, []).append(r)
    assert len(by_decision) == 2
    for clips in by_decision.values():
        assert len(clips) == 3
        assert len({(c.scene_id, c.start_speed) for c in clips}) == 1
        # Same camera for all candidates: the first frame (shared pose, t=0) is identical across the 3 videos.
        firsts = []
        for c in clips:
            with imageio.get_reader(str(out / c.video), "ffmpeg") as reader:
                firsts.append(reader.get_data(0).astype(float))
        assert max(np.abs(f - firsts[0]).mean() for f in firsts) < 2.0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_decisions"] == 2 and manifest["batch_id"] == "test"
    assert manifest["config"]["CLIP_MAX_S"] == 20.0
