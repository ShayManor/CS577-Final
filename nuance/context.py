import csv
import functools
import os

import magnum as mn
import numpy as np

from nuance.config import (AHEAD_M, AHEAD_SAMPLE_M, CONTEXT_RADIUS_M, NARROWING_RATIO, NAVMESH_AGENT_RADIUS,
                           OPEN_CLEARANCE_M, SCENE_DATASET, TIGHT_CLEARANCE_M, TURN_DEG)
from nuance.motion import HEADING_WINDOW_M

EXCLUDED_CATEGORIES = {"door", "window", "curtain", "carpet", "rug", "stairs"}
MAX_CLEARANCE_M = 2.0
# Clutter = objects standing on the decision point's floor (bottom within this height of it) with a real footprint,
# which leaves out books on shelves, lamps on ceilings and pictures on walls.
CLUTTER_MAX_BOTTOM_DY = 0.3
CLUTTER_MIN_FOOTPRINT_M2 = 0.04
CATEGORY_CSV = os.path.join(os.path.dirname(SCENE_DATASET), "metadata", "fpmodels-with-decomposed.csv")


@functools.lru_cache(maxsize=1)
def _categories() -> dict:
    with open(CATEGORY_CSV) as f:
        return {r["id"]: r["main_category"] for r in csv.DictReader(f)}


def _semantic_objects(sim):
    """(category, center, bottom y, footprint m^2) of every scene object, from its world-space bounding box."""
    rom = sim.get_rigid_object_manager()
    out = []
    for handle in rom.get_object_handles():
        obj = rom.get_object_by_handle(handle)
        bb = obj.root_scene_node.cumulative_bb
        m = obj.root_scene_node.absolute_transformation()
        pts = np.array([m.transform_point(mn.Vector3(x, y, z)) for x in (bb.min.x, bb.max.x)
                        for y in (bb.min.y, bb.max.y) for z in (bb.min.z, bb.max.z)], dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        name = _categories().get(handle.split("_:")[0].split("/")[-1], "")
        out.append((name, (lo + hi) / 2, lo[1], (hi[0] - lo[0]) * (hi[2] - lo[2])))
    return out


def _cum(path):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def _pos(path, cum, s):
    s = np.clip(s, 0.0, cum[-1])
    return np.array([np.interp(s, cum, path[:, k]) for k in range(3)])


def _heading(path, cum, s):
    d = _pos(path, cum, s + HEADING_WINDOW_M) - _pos(path, cum, s - HEADING_WINDOW_M)
    d = np.array([d[0], d[2]])
    return d / max(np.linalg.norm(d), 1e-9)


def _clearance(sim, p):
    # Distance to the edge of the Spot-eroded navmesh plus the erosion radius = distance to the obstacle.
    return float(sim.pathfinder.distance_to_closest_obstacle(p, MAX_CLEARANCE_M)) + NAVMESH_AGENT_RADIUS


def _clutter_count(sim, p):
    n = 0
    for name, c, bottom, footprint in _semantic_objects(sim):
        if name in EXCLUDED_CATEGORIES or footprint < CLUTTER_MIN_FOOTPRINT_M2:
            continue
        if abs(bottom - p[1]) > CLUTTER_MAX_BOTTOM_DY:
            continue
        if np.hypot(c[0] - p[0], c[2] - p[2]) <= CONTEXT_RADIUS_M:
            n += 1
    return n


def measure_context(sim, path: np.ndarray, arc_length: float) -> dict:
    path = np.asarray(path, dtype=float)
    cum = _cum(path)
    s0 = float(np.clip(arc_length, 0.0, cum[-1]))
    p = _pos(path, cum, s0)
    clearance = _clearance(sim, p)
    clutter_count = _clutter_count(sim, p)
    if clearance < TIGHT_CLEARANCE_M and clutter_count <= 2:
        location = "corridor"
    elif clearance < OPEN_CLEARANCE_M:
        location = "near_furniture"
    else:
        location = "open"

    ss = np.arange(s0, min(s0 + AHEAD_M, cum[-1]) + 1e-9, AHEAD_SAMPLE_M)
    h0 = _heading(path, cum, s0)
    turn_deg = max(float(np.degrees(np.arccos(np.clip(h0 @ _heading(path, cum, s), -1.0, 1.0)))) for s in ss)
    ahead_clear = np.array([_clearance(sim, _pos(path, cum, s)) for s in ss])
    min_ahead = float(ahead_clear.min())
    p20 = float(np.percentile(ahead_clear, 20))
    if turn_deg > TURN_DEG:
        ahead = "turn"
    elif p20 < TIGHT_CLEARANCE_M and p20 < NARROWING_RATIO * clearance:
        ahead = "narrowing"
    else:
        ahead = "open"

    return {"clearance": clearance, "clutter_count": clutter_count, "location": location,
            "turn_deg": turn_deg, "min_ahead_clearance": min_ahead, "ahead_clearance_p20": p20, "ahead": ahead}
