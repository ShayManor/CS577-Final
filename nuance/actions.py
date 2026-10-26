"""The policy's action at a decision point, its prompt, and the decision a rated clip was cut at.

An action is JSON {"via": [forward, left], "goal": [forward, left], "speed": v}: two points in meters in Spot's frame at
the decision point (forward along its heading, left to its left) and the speed in m/s after the decision point. Spot
walks the navmesh's shortest route through via to goal (nuance.rollout). Plain numpy, so the habitat env (python 3.9)
and the training env both import it.
"""
import json
import re
from typing import Optional

import numpy as np

from nuance.config import MIN_SPEED

MAX_SPEED = 1.2  # nuance.candidates.WALK_MAX_SPEED: the walking policy tops out near here
MAX_REACH_M = 9.0  # nuance.candidates.MAX_POST_M

POLICY_PROMPT = (
    "You control a Spot robot. The image is Spot's front camera. Spot is walking straight ahead at {v0:.1f} m/s. "
    'You are told: "{instruction}". Choose the route Spot takes from here and its speed. Reply with JSON only: '
    '{{"via": [forward, left], "goal": [forward, left], "speed": meters_per_second}}. via and goal are points in '
    "meters in Spot's frame (forward is straight ahead, left is to Spot's left); Spot walks through via to goal and "
    "stops there.")


def heading(yaw: float) -> np.ndarray:
    return np.array([-np.sin(yaw), -np.cos(yaw)])


def to_world(p, yaw: float, xy) -> np.ndarray:
    """Spot-frame (forward, left) at decision point p (habitat xyz) to a habitat point at p's height."""
    h = heading(yaw)
    left = np.array([h[1], -h[0]])  # nuance.grounding.left_of
    xz = np.array([p[0], p[2]]) + xy[0] * h + xy[1] * left
    return np.array([xz[0], p[1], xz[1]])


def to_ego(p, yaw: float, q) -> list:
    h = heading(yaw)
    left = np.array([h[1], -h[0]])
    d = np.array([q[0] - p[0], q[2] - p[2]])
    return [round(float(d @ h), 1), round(float(d @ left), 1)]


def _pt(v) -> Optional[list]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        v = [float(v[0]), float(v[1])]
    except (TypeError, ValueError):
        return None
    return v if all(np.isfinite(v)) and np.hypot(*v) <= MAX_REACH_M else None


def parse_action(text: str) -> Optional[dict]:
    """The action in a completion (from its first "{" to its last "}"), or None if there is none or it is out of
    range."""
    m = re.search(r"\{.*\}", text, re.S)
    if m is None:
        return None
    try:
        a = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(a, dict):
        return None
    via, goal = _pt(a.get("via")), _pt(a.get("goal"))
    try:
        speed = float(a.get("speed"))
    except (TypeError, ValueError):
        return None
    if via is None or goal is None or not np.isfinite(speed) or speed <= 0:
        return None
    return {"via": via, "goal": goal, "speed": float(np.clip(speed, MIN_SPEED, MAX_SPEED))}


def format_action(a: dict) -> str:
    return json.dumps({"via": a["via"], "goal": a["goal"], "speed": round(float(a["speed"]), 2)})


def decision_of(r: dict) -> dict:
    """The decision point a clip record (nuance.records.ClipRecord as a dict) was cut at: its walk-in ends at the
    decision position, which is a vertex of the clip's line (generate_instructions writes the line as walk-in plus
    continuation)."""
    path = np.asarray(r["path"], dtype=float)
    d = np.linalg.norm(path - np.asarray(r["decision_position"]), axis=1)
    k = int(np.argmax(d < 1e-3)) if (d < 1e-3).any() else int(np.argmin(d))  # first: "back" lines return past it
    return {"scene_id": r["scene_id"], "pre": path[:k + 1].tolist(), "yaw": float(r["decision_yaw"]),
            "v0": float(r["start_speed"]), "person": r.get("person"), "referent": r.get("referent")}


def action_of(r: dict) -> dict:
    """The action that reproduces a clip's candidate: its line's end, the point half-way along its continuation, and
    the commanded speed."""
    path = np.asarray(r["path"], dtype=float)
    k = len(decision_of(r)["pre"]) - 1
    post = path[k:]
    seg = np.linalg.norm(np.diff(post, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    half = cum[-1] / 2
    i = min(int(np.searchsorted(cum, half, side="right")) - 1, len(seg) - 1)
    f = (half - cum[i]) / max(seg[i], 1e-9) if len(seg) else 0.0
    via = post[i] + f * (post[i + 1] - post[i]) if len(seg) else post[0]
    p, yaw = post[0], float(r["decision_yaw"])
    return {"via": to_ego(p, yaw, via), "goal": to_ego(p, yaw, post[-1]),
            "speed": round(float(r["candidate_speed"]), 2)}
