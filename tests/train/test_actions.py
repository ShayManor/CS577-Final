import json

import numpy as np
import pytest

from nuance.actions import MAX_SPEED, action_of, decision_of, format_action, parse_action, to_ego, to_world


def _record(path, k, yaw=0.0, v1=0.8):
    return {"scene_id": "s", "path": path, "decision_position": path[k], "decision_yaw": yaw, "start_speed": 0.6,
            "candidate_speed": v1, "person": None, "referent": {"kind": "speed"}}


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0, np.pi])
def test_ego_roundtrip(yaw):
    p = np.array([1.0, 0.2, -3.0])
    for xy in ([2.0, 0.0], [1.5, -0.7], [-1.0, 2.0]):
        assert to_ego(p, yaw, to_world(p, yaw, xy)) == pytest.approx(xy, abs=1e-9)


def test_frame_convention():
    # yaw 0 faces -z (habitat forward); left is -x
    p = np.zeros(3)
    assert to_world(p, 0.0, [1.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0])
    assert to_world(p, 0.0, [0.0, 1.0]) == pytest.approx([-1.0, 0.0, 0.0])


def test_decision_and_action_of_straight_line():
    path = [[0, 0, 2.0], [0, 0, 1.0], [0, 0, 0.0], [0, 0, -2.0], [0, 0, -4.0]]
    r = _record(path, 2)
    d = decision_of(r)
    assert d["pre"] == path[:3] and d["v0"] == 0.6
    a = action_of(r)
    assert a == {"via": [2.0, 0.0], "goal": [4.0, 0.0], "speed": 0.8}


def test_decision_of_takes_first_visit():
    # a "back" line returns through the decision point
    path = [[0, 0, 2.0], [0, 0, 0.0], [0, 0, -1.0], [0.0, 0, 0.0], [0, 0, 2.0]]
    assert len(decision_of(_record(path, 1))["pre"]) == 2


def test_action_reproduces_detour_side():
    path = [[0, 0, 1.0], [0, 0, 0.0], [-1.0, 0, -1.0], [0, 0, -2.0]]
    a = action_of(_record(path, 1))
    assert a["via"][1] > 0.5  # the detour bulges to Spot's left (-x)
    assert a["goal"] == [2.0, 0.0]


def test_parse_action():
    a = {"via": [1.0, 0.5], "goal": [3.0, -0.2], "speed": 0.9}
    assert parse_action(format_action(a)) == a
    assert parse_action("Sure: " + json.dumps(a) + " done") == a
    assert parse_action(json.dumps({**a, "speed": 5})) == {**a, "speed": MAX_SPEED}
    for bad in ("", "no json", '{"goal": [1, 2], "speed": 1}', json.dumps({**a, "goal": [100, 0]}),
                json.dumps({**a, "speed": -1}), json.dumps({**a, "via": "x"}), "{not json}"):
        assert parse_action(bad) is None
