import math
import random

import habitat_sim
import magnum as mn
import numpy as np

from nuance.config import (BODY_CHECK_STEP_M, BODY_MIN_CLEARANCE, CHASE_DEPTH_RES, CHASE_HFOV, CHASE_RES,
                           FRONT_CAM_FORWARD, FRONT_CAM_HEIGHT, FRONT_HFOV, FRONT_RES, LIGHT_SIDE, LIGHT_SIDE_ELEV_DEG,
                           LIGHT_TOP, NAVMESH_AGENT_HEIGHT, NAVMESH_AGENT_RADIUS, SAME_FLOOR_DY, SCENE_DATASET,
                           SPOT_HALF_LENGTH, TOPDOWN_CAM_HEIGHT, TOPDOWN_RES)
from nuance.context import _cum, _heading, _pos

def make_sim(scene_id: str, seed: int = 0) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id
    backend.scene_dataset_config_file = SCENE_DATASET
    backend.enable_physics = True
    backend.gpu_device_id = 0

    top = habitat_sim.CameraSensorSpec()
    top.uuid = "topdown"
    top.sensor_type = habitat_sim.SensorType.COLOR
    top.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    top.resolution = [TOPDOWN_RES, TOPDOWN_RES]
    top.position = [0.0, TOPDOWN_CAM_HEIGHT, 0.0]
    top.orientation = [-math.pi / 2, 0.0, 0.0]

    front = habitat_sim.CameraSensorSpec()
    front.uuid = "front"
    front.sensor_type = habitat_sim.SensorType.COLOR
    front.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    front.resolution = [FRONT_RES[1], FRONT_RES[0]]
    front.hfov = FRONT_HFOV
    front.position = [0.0, FRONT_CAM_HEIGHT, -FRONT_CAM_FORWARD]

    chase = habitat_sim.CameraSensorSpec()
    chase.uuid = "chase"
    chase.sensor_type = habitat_sim.SensorType.COLOR
    chase.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    chase.resolution = [CHASE_RES[1], CHASE_RES[0]]
    chase.hfov = CHASE_HFOV
    chase.position = [0.0, 0.0, 0.0]  # position and pitch come from the agent pose, set per frame

    chase_depth = habitat_sim.CameraSensorSpec()
    chase_depth.uuid = "chase_depth"
    chase_depth.sensor_type = habitat_sim.SensorType.DEPTH
    chase_depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    chase_depth.resolution = [CHASE_DEPTH_RES[1], CHASE_DEPTH_RES[0]]
    chase_depth.hfov = CHASE_HFOV
    chase_depth.position = [0.0, 0.0, 0.0]

    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=[top, front, chase, chase_depth])
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    e = math.radians(LIGHT_SIDE_ELEV_DEG)
    lights = [((0.0, -1.0, 0.0), LIGHT_TOP)] + [
        ((math.cos(e) * math.cos(a), -math.sin(e), math.cos(e) * math.sin(a)), LIGHT_SIDE)
        for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
    sim.set_light_setup([habitat_sim.gfx.LightInfo(vector=mn.Vector4(*v, 0.0), color=mn.Color3(c, c, c),
                                                   model=habitat_sim.gfx.LightPositionModel.Global)
                         for v, c in lights])

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = NAVMESH_AGENT_RADIUS
    settings.agent_height = NAVMESH_AGENT_HEIGHT
    settings.include_static_objects = True
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        sim.close()
        raise RuntimeError(f"navmesh recompute failed for {scene_id}")
    sim.pathfinder.seed(seed)
    return sim


def sample_path(sim, min_length: float, rng: random.Random, max_tries: int = 200):
    pf = sim.pathfinder
    pf.seed(rng.randrange(2 ** 31))
    for _ in range(max_tries):
        a = pf.get_random_navigable_point()
        b = pf.get_random_navigable_point()
        if abs(a[1] - b[1]) > SAME_FLOOR_DY:
            continue
        sp = habitat_sim.ShortestPath()
        sp.requested_start = a
        sp.requested_end = b
        if not pf.find_path(sp) or sp.geodesic_distance < min_length:
            continue
        path = np.array(sp.points, dtype=float)
        # Cut the path at min_length: the clip runs until Spot reaches the end of the line.
        cum = _cum(path)
        k = int(np.searchsorted(cum, min_length))
        path = np.vstack([path[:k], _pos(path, cum, min_length)])
        if path[:, 1].max() - path[:, 1].min() <= SAME_FLOOR_DY:
            return path
    return None


def _physical_clearance(pf, q) -> float:
    if pf.is_navigable(q):
        return float(pf.distance_to_closest_obstacle(q, 2.0)) + NAVMESH_AGENT_RADIUS
    snap = np.array(pf.snap_point(q), dtype=float)
    if not np.isfinite(snap).all():
        return -np.inf
    return NAVMESH_AGENT_RADIUS - float(np.hypot(q[0] - snap[0], q[2] - snap[2]))


def body_clear(sim, path: np.ndarray) -> bool:
    """True if Spot's front and rear ends keep BODY_MIN_CLEARANCE from obstacles along the whole path."""
    path = np.asarray(path, dtype=float)
    cum = _cum(path)
    for s in np.append(np.arange(0.0, cum[-1], BODY_CHECK_STEP_M), cum[-1]):
        c = _pos(path, cum, s)
        h = _heading(path, cum, s)
        fwd = np.array([h[0], 0.0, h[1]])
        for q in (c + SPOT_HALF_LENGTH * fwd, c - SPOT_HALF_LENGTH * fwd):
            if _physical_clearance(sim.pathfinder, q) < BODY_MIN_CLEARANCE:
                return False
    return True
