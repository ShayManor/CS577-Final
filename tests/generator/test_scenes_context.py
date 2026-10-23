import random

import numpy as np
import pytest

from nuance import context
from nuance.config import NAVMESH_AGENT_RADIUS
from nuance.context import measure_context
from nuance.scenes import body_clear, make_sim, sample_path


@pytest.fixture(scope="module")
def sim():
    with open("configs/pilot_scenes.txt") as f:
        scene_id = f.readline().strip()
    s = make_sim(scene_id, seed=0)
    yield s
    s.close()


def test_spot_navmesh_radius(sim):
    assert sim.pathfinder.nav_mesh_settings.agent_radius == pytest.approx(0.4)


def test_sample_path_same_floor_and_length(sim):
    rng = random.Random(0)
    for _ in range(10):
        path = sample_path(sim, 6.0, rng)
        assert path is not None
        length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        assert length == pytest.approx(6.0)
        assert path[:, 1].max() - path[:, 1].min() <= 0.3


def test_sample_path_impossible_returns_none(sim):
    assert sample_path(sim, 1000.0, random.Random(0)) is None


def test_context_keys_and_types(sim):
    path = sample_path(sim, 6.0, random.Random(1))
    ctx = measure_context(sim, path, 1.0)
    assert set(ctx) == {"clearance", "clutter_count", "location", "turn_deg", "min_ahead_clearance",
                        "ahead_clearance_p20", "ahead"}
    assert ctx["location"] in {"corridor", "near_furniture", "open"}
    assert ctx["ahead"] in {"turn", "narrowing", "open"}
    assert isinstance(ctx["clutter_count"], int) and ctx["clutter_count"] >= 0


def test_clutter_zero_without_objects(sim, monkeypatch):
    monkeypatch.setattr(context, "_semantic_objects", lambda s: [])
    path = sample_path(sim, 6.0, random.Random(2))
    assert measure_context(sim, path, 1.0)["clutter_count"] == 0


def test_clearance_is_physical(sim):
    rng = random.Random(3)
    for _ in range(5):
        path = sample_path(sim, 6.0, rng)
        for arc in (0.0, 1.0, 2.5):
            ctx = measure_context(sim, path, arc)
            assert ctx["clearance"] >= NAVMESH_AGENT_RADIUS
            assert ctx["min_ahead_clearance"] >= NAVMESH_AGENT_RADIUS
            assert ctx["min_ahead_clearance"] <= ctx["ahead_clearance_p20"]


def test_body_clear_sampled_path_passes(sim):
    rng = random.Random(4)
    assert any(body_clear(sim, sample_path(sim, 6.0, rng)) for _ in range(20))


def test_body_clear_fails_into_obstacle(sim):
    # Straight segment ending on the navmesh edge, pointing at the nearest obstacle: Spot's front is inside it.
    pf = sim.pathfinder
    pf.seed(5)
    n = 0
    while n < 5:
        p = np.array(pf.get_random_navigable_point(), dtype=float)
        o = np.array(pf.closest_obstacle_surface_point(p, 2.0).hit_pos, dtype=float)
        if not np.isfinite(o).all() or np.linalg.norm((o - p)[[0, 2]]) < 0.5:
            continue
        assert not body_clear(sim, np.stack([p, o]))
        n += 1
