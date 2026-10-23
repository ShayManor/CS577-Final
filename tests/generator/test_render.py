import json
import random
import subprocess

import imageio
import numpy as np
import pytest

from nuance.config import CHASE_RES, SPOT_BASE_HEIGHT, TOPDOWN_RES
from nuance.motion import trajectory
from nuance.render import LEG_STAND, Motion, load_spot, render_clip, render_front, topdown_frame, world_to_pixel
from nuance.scenes import make_sim, sample_path


def _standing_motion(traj):
    """A Motion that glides along the trajectory in the standing pose (no physics), enough to test rendering."""
    rots = []
    for yaw in traj.yaws:
        f = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
        rots.append(np.stack([f, [0.0, 1.0, 0.0], np.cross(f, [0.0, 1.0, 0.0])], axis=1))
    return Motion(times=traj.times, positions=traj.positions + [0.0, SPOT_BASE_HEIGHT, 0.0], rotations=np.array(rots),
                  joints=np.tile(LEG_STAND, (len(traj.times), 1)), arc=traj.arc)


@pytest.fixture(scope="module")
def scene():
    with open("configs/pilot_scenes.txt") as f:
        scene_id = f.readline().strip()
    sim = make_sim(scene_id, seed=0)
    spot = load_spot(sim)
    path = sample_path(sim, 6.0, random.Random(0))
    traj = trajectory(path, 0.5, 1.0)
    yield sim, spot, path, _standing_motion(traj)
    sim.close()


@pytest.fixture(scope="module")
def clip(scene, tmp_path_factory):
    sim, spot, path, traj = scene
    out = tmp_path_factory.mktemp("render") / "clip.mp4"
    render_clip(sim, spot, traj, path, 1.0, out)
    return out


def test_video_codec_and_length(scene, clip):
    traj = scene[3]
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,nb_frames", "-of", "json", str(clip)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    assert s["codec_name"] == "h264"
    assert s["pix_fmt"] == "yuv420p"
    assert (s["width"], s["height"]) == (CHASE_RES[0] + TOPDOWN_RES, TOPDOWN_RES)
    assert int(s["nb_frames"]) == len(traj.times)


def test_spot_moves_in_frames(scene, clip):
    traj = scene[3]
    frames = []
    with imageio.get_reader(str(clip), "ffmpeg") as reader:
        for f in reader:
            frames = [frames[0] if frames else f, f]
    center, extent = topdown_frame(traj)
    u, v = world_to_pixel(traj.positions[:1], center, extent)[0]
    u += CHASE_RES[0]  # the top-down sits right of the chase view
    r = 20
    a = frames[0][max(v - r, 0):v + r, max(u - r, 0):u + r].astype(float)
    b = frames[-1][max(v - r, 0):v + r, max(u - r, 0):u + r].astype(float)
    assert np.abs(a - b).mean() > 10.0


def test_front_png(scene, tmp_path):
    sim, spot, path, traj = scene
    i = int(np.searchsorted(traj.times, 2.0))
    out = tmp_path / "front.png"
    render_front(sim, spot, traj.positions[i], traj.yaws[i], out)
    img = imageio.imread(out)
    assert img.shape[:2] == (480, 640)
    assert np.var(img.astype(float)) > 0
