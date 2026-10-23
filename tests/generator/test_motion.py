import numpy as np

from nuance.motion import trajectory


def _straight(n=30.0):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -n]])


def test_speed_profile():
    tr = trajectory(_straight(), 0.5, 1.0)
    i2 = int(np.argmin(np.abs(tr.times - 2.0)))
    assert abs(tr.times[i2] - 2.0) < 1e-9 and abs(tr.arc[i2] - 1.0) < 1e-6
    i10 = int(np.argmin(np.abs(tr.times - 10.0)))
    assert abs(tr.arc[i10] - 8.875) < 1 / 30


def test_stops_at_path_end():
    path = _straight(3.0)
    tr = trajectory(path, 0.5, 1.0)
    assert np.allclose(tr.positions[-1], path[-1])
    assert tr.times[-1] < 10.0
    assert tr.arc[-1] <= 3.0 + 1e-9 and tr.arc[-2] < 3.0


def test_max_duration():
    tr = trajectory(_straight(200.0), 0.3, 0.3)
    assert tr.times[-1] <= 20.0 and len(tr.times) <= 601


def test_yaw_convention():
    assert abs(trajectory(_straight(), 0.5, 1.0).yaws[0]) < 1e-6
    p = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    assert abs(trajectory(p, 0.5, 1.0).yaws[5] + np.pi / 2) < 1e-6


def test_yaw_smooth_on_turn():
    p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [4.0, 0.0, -4.0]])
    tr = trajectory(p, 0.8, 1.0)
    d = np.diff(tr.yaws)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(d).max() < 0.2
    assert abs(abs(tr.yaws[-1] - tr.yaws[0]) - np.pi / 2) < 0.1
