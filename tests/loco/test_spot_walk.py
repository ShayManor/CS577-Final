"""Run in the spot-loco env on ideas8 (needs the trained policy):
    ~/miniconda3/envs/spot-loco/bin/python -m pytest tests/loco -v
"""
import numpy as np
import pytest

from nuance.config import HOLD_S, SPOT_POLICY_DIR, T_DECISION_S
from nuance.spot_walk import STOP_DECEL, load_walker, walk_job

RES = 0.04


def _floor(bump_m: float = 0.0, at_m: float = 3.0):
    """Floor around a straight line along habitat -z (MuJoCo +y), with an optional threshold across it."""
    xs = np.arange(-2.0, 2.0 + 1e-9, RES)
    ys = np.arange(-3.0, 10.0 + 1e-9, RES)
    h = np.zeros((len(ys), len(xs)))
    h[(ys > at_m) & (ys < at_m + 0.1)] = bump_m
    return {"origin": [float(xs[0]), float(ys[0])], "res": RES, "heights": h.tolist()}


@pytest.fixture(scope="module")
def walker():
    return load_walker(SPOT_POLICY_DIR)


PATH = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -6.0]])


@pytest.mark.parametrize("v0,v1", [(0.5, 0.25), (0.5, 0.5), (0.8, 1.4)])
def test_tracks_speed_profile_on_flat_floor(walker, v0, v1):
    res = walk_job(walker, _floor(), PATH, v0, v1)
    assert not res["fell"]
    t, arc = res["times"], res["arc"]
    assert arc[-1] >= 6.0 - 0.05  # reaches the end of the line
    v = np.diff(arc) * 30
    assert abs(v[t[1:] < T_DECISION_S].mean() - v0) < 0.15 * v0
    cruise = (t[1:] > T_DECISION_S + 1.0) & (t[1:] < res["t_end"] - v1 / STOP_DECEL - 0.3)
    assert abs(v[cruise].mean() - v1) < 0.15 * v1
    assert np.abs(res["base_pos"][:, 0]).max() < 0.15  # stays near the line


def test_stops_and_holds_at_line_end(walker):
    res = walk_job(walker, _floor(), PATH, 0.8, 1.2)
    assert not res["fell"]
    t, arc = res["times"], res["arc"]
    assert t[-1] >= res["t_end"] + HOLD_S - 0.05
    hold = t > res["t_end"] + 0.5
    assert np.ptp(arc[hold]) < 0.1  # standing, not walking on
    assert abs(arc[-1] - 6.0) < 0.15
    assert 0.6 < res["v_post"] < 1.2


@pytest.mark.parametrize("v", [0.25, 1.0])
def test_steps_over_threshold(walker, v):
    res = walk_job(walker, _floor(bump_m=0.05), PATH, v, v)
    assert not res["fell"]
    assert res["arc"][-1] >= 6.0 - 0.05
