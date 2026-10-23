"""What a decision point's surroundings contain: floor objects, walls, rooms, and what Spot's front camera sees."""
import csv
import functools
import json
import os
from dataclasses import dataclass

import habitat_sim
import magnum as mn
import numpy as np

from nuance.config import (FRONT_CAM_FORWARD, FRONT_CAM_HEIGHT, FRONT_HFOV, NAVMESH_AGENT_HEIGHT,
                           NAVMESH_AGENT_RADIUS, SCENE_DATASET)
from nuance.context import CATEGORY_CSV

HSSD_DIR = os.path.dirname(SCENE_DATASET)
NAV_ABOVE_FLOOR_M = 0.16  # HSSD navmesh points float this far above the rendered floor
FLOOR_OBJECT_MAX_BOTTOM_M = 0.3
FLOOR_OBJECT_MIN_TOP_M = 0.3
FLOOR_OBJECT_MIN_FOOTPRINT_M2 = 0.04
NOT_OBSTACLES = {"door", "window", "curtain", "carpet", "rug", "stairs", "picture", "mirror", "wall_lamp",
                 "ceiling_lamp", "wall_clock"}
WALL_RAY_HEIGHT_M = 0.5  # above the floor: under windowsills, over skirting boards
SPOT_HALF_WIDTH_M = 0.25
# Standing people from HSSD (Bertie, Paul), placed on the line for the person instructions. ("Mary standing" stands
# on tiptoe leaning forward and reads as hovering.)
PEOPLE = ("6e0d3b722da8f57e0c9752e729423d2fc84d1b3d", "edd178f7d2a5cc9af9fc52f9fdaa649c742e19f8")
OUTDOOR = {"outdoor", "porch/terrace/deck/driveway", "balcony"}
DOORWAY_M = 0.5  # points outside every room outline but this close to one are in its doorway


@functools.lru_cache(maxsize=1)
def _categories() -> dict:
    with open(CATEGORY_CSV) as f:
        return {r["id"]: r["main_category"] for r in csv.DictReader(f)}


@dataclass
class Obj:
    oid: int
    category: str
    lo: np.ndarray  # world AABB
    hi: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    def dist_xz(self, pts) -> np.ndarray:
        """Distance in the floor plane from each point to the footprint (0 inside)."""
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        dx = np.maximum(np.maximum(self.lo[0] - p[:, 0], p[:, 0] - self.hi[0]), 0.0)
        dz = np.maximum(np.maximum(self.lo[2] - p[:, 2], p[:, 2] - self.hi[2]), 0.0)
        return np.hypot(dx, dz)


def _world_aabb(obj):
    bb = obj.root_scene_node.cumulative_bb
    m = obj.root_scene_node.absolute_transformation()
    pts = np.array([m.transform_point(mn.Vector3(x, y, z)) for x in (bb.min.x, bb.max.x)
                    for y in (bb.min.y, bb.max.y) for z in (bb.min.z, bb.max.z)], dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def scene_objects(sim) -> list:
    rom = sim.get_rigid_object_manager()
    out = []
    for handle in rom.get_object_handles():
        obj = rom.get_object_by_handle(handle)
        lo, hi = _world_aabb(obj)
        out.append(Obj(obj.object_id, _categories().get(handle.split("_:")[0].split("/")[-1], ""), lo, hi))
    return out


def floor_objects(objs: list, floor_y: float) -> list:
    """Objects standing on this floor that Spot has to walk around."""
    return [o for o in objs if o.category not in NOT_OBSTACLES and abs(o.lo[1] - floor_y) < FLOOR_OBJECT_MAX_BOTTOM_M
            and o.hi[1] - floor_y > FLOOR_OBJECT_MIN_TOP_M
            and (o.hi[0] - o.lo[0]) * (o.hi[2] - o.lo[2]) > FLOOR_OBJECT_MIN_FOOTPRINT_M2]


def floor_y(nav_point) -> float:
    return float(nav_point[1]) - NAV_ABOVE_FLOOR_M


def floor_at(sim, p) -> float:
    """Height of the rendered floor under navmesh point p, found by a ray cast down (the navmesh floats above it by
    a scene-dependent amount); floor_y(p) if the ray finds nothing."""
    dist, _ = ray(sim, [p[0], p[1] + 0.5, p[2]], [0.0, -1.0, 0.0], 1.5)
    return float(p[1] + 0.5 - dist) if np.isfinite(dist) else floor_y(p)


def has_ceiling(sim, p, max_m: float = 4.0) -> bool:
    dist, _ = ray(sim, [p[0], floor_at(sim, p) + 1.0, p[2]], [0.0, 1.0, 0.0], max_m)
    return bool(np.isfinite(dist))


def left_of(heading_xz) -> np.ndarray:
    return np.array([heading_xz[1], -heading_xz[0]])


def ray(sim, origin, direction, max_dist: float):
    """(distance, object_id) of the first hit, or (inf, None)."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    hits = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(*origin), mn.Vector3(*d)), max_dist)
    if not hits.has_hits():
        return np.inf, None
    h = hits.hits[0]
    return float(h.ray_distance), int(h.object_id)


def wall_distance(sim, p, direction_xz, max_dist: float = 3.0, height: float = WALL_RAY_HEIGHT_M):
    """Distance from navmesh point p to a wall along a floor direction; inf if the ray hits nothing or an object."""
    o = np.array([p[0], floor_at(sim, p) + height, p[2]])
    dist, oid = ray(sim, o, [direction_xz[0], 0.0, direction_xz[1]], max_dist)
    return dist if oid == habitat_sim.stage_id else np.inf


def front_camera(p, yaw: float, sim=None) -> np.ndarray:
    fwd = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
    fy = floor_at(sim, p) if sim is not None else floor_y(p)
    return np.array([p[0], fy + FRONT_CAM_HEIGHT, p[2]]) + FRONT_CAM_FORWARD * fwd


def in_view(cam, yaw: float, target, margin_deg: float = 8.0) -> bool:
    fwd = np.array([-np.sin(yaw), -np.cos(yaw)])
    d = np.array([target[0] - cam[0], target[2] - cam[2]])
    ang = np.degrees(np.arccos(np.clip(fwd @ d / max(np.linalg.norm(d), 1e-9), -1, 1)))
    return ang <= FRONT_HFOV / 2 - margin_deg


def sees_point(sim, cam, target) -> bool:
    d = np.asarray(target, dtype=float) - cam
    dist = float(np.linalg.norm(d))
    hit, _ = ray(sim, cam, d, dist)
    return hit >= dist - 0.1


def sees_object(sim, cam, yaw: float, obj: Obj, min_points: int = 2) -> bool:
    """True if rays from the camera reach the object itself at at least min_points of its center and upper
    corners (a ray that misses everything does not count: the box corners can lie in empty space)."""
    c = obj.center
    top = min(obj.hi[1], c[1] + 0.3)
    pts = [c] + [np.array([x, top, z]) for x in (obj.lo[0] + 0.05, obj.hi[0] - 0.05)
                 for z in (obj.lo[2] + 0.05, obj.hi[2] - 0.05)]
    n = 0
    for t in pts:
        if not in_view(cam, yaw, t):
            continue
        d = t - cam
        dist = float(np.linalg.norm(d))
        _, oid = ray(sim, cam, d, dist + 0.3)
        if oid == obj.oid:
            n += 1
    return n >= min_points


def wall_flush(sim, obj: Obj, fy: float, gap_m: float = 0.3) -> bool:
    """True if a wall runs within gap_m of a side of the object's footprint: such an object is not something to
    walk around."""
    c = obj.center
    for k, sign in ((0, 1), (0, -1), (2, 1), (2, -1)):
        o = np.array([c[0], fy + WALL_RAY_HEIGHT_M, c[2]])
        o[k] = (obj.hi if sign > 0 else obj.lo)[k] + sign * 0.02
        d = np.zeros(3)
        d[k] = sign
        dist, oid = ray(sim, o, d, gap_m)
        if oid == habitat_sim.stage_id:
            return True
    return False


def sees_sky(sim, cam, yaw: float, max_frac: float = 0.05) -> bool:
    """True if more than max_frac of rays over the upper part of the camera's view escape the scene: open sky, which
    renders black (courtyards, porches and windows onto the outdoors the room outlines miss)."""
    n = escaped = 0
    for el in np.radians([5, 15, 25, 35]):
        for az in np.radians(np.linspace(-FRONT_HFOV / 2, FRONT_HFOV / 2, 9)):
            a = yaw + az
            d = np.array([-np.sin(a) * np.cos(el), np.sin(el), -np.cos(a) * np.cos(el)])
            dist, _ = ray(sim, cam, d, 40.0)
            n += 1
            escaped += not np.isfinite(dist)
    return escaped / n > max_frac


def free_standing(obj: Obj, others: list, allowed=(), gap_m: float = 0.2) -> bool:
    """No other floor object within gap_m of the object's footprint (except categories in allowed, e.g. the chairs
    around a table), so it reads as one thing to keep clear of."""
    for o in others:
        if o is obj or o.category in allowed or o.category == "carpet":
            continue
        g = np.hypot(*np.maximum(np.maximum(obj.lo - o.hi, o.lo - obj.hi), 0.0)[[0, 2]])
        if g < gap_m:
            return False
    return True


@dataclass
class Region:
    label: str
    poly: np.ndarray  # (n, 2) x, z
    floor_y: float

    def contains(self, p) -> bool:
        if abs(floor_y(p) - self.floor_y) > 1.0:
            return False
        x, z = p[0], p[2]
        xs, zs = self.poly[:, 0], self.poly[:, 1]
        inside = False
        for i in range(len(xs)):
            j = i - 1
            if (zs[i] > z) != (zs[j] > z) and x < (xs[j] - xs[i]) * (z - zs[i]) / (zs[j] - zs[i]) + xs[i]:
                inside = not inside
        return inside

    def edge_distance(self, p) -> float:
        q = np.array([p[0], p[2]])
        a, b = self.poly, np.roll(self.poly, -1, axis=0)
        ab = b - a
        t = np.clip(((q - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0, 1)
        return float(np.min(np.linalg.norm(a + t[:, None] * ab - q, axis=1)))

    @property
    def area(self) -> float:
        x, z = self.poly[:, 0], self.poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))

    @property
    def centroid(self) -> np.ndarray:
        return self.poly.mean(axis=0)


@functools.lru_cache(maxsize=None)
def regions(scene_id: str) -> tuple:
    path = f"{HSSD_DIR}/semantics/scenes/{scene_id}.semantic_config.json"
    if not os.path.exists(path):
        return ()
    with open(path) as f:
        anns = json.load(f).get("region_annotations", [])
    return tuple(Region(a["label"], np.array([[q[0], q[2]] for q in a["poly_loop"]], dtype=float),
                        float(a["floor_height"])) for a in anns)


def region_of(scene_id: str, p):
    inside = [r for r in regions(scene_id) if r.contains(p)]
    return min(inside, key=lambda r: r.area) if inside else None


def indoor(scene_id: str, p) -> bool:
    """Inside a room, or in a doorway between room outlines; the outdoors renders with a black sky."""
    r = region_of(scene_id, p)
    if r is not None:
        return r.label not in OUTDOOR
    return any(r.label not in OUTDOOR and abs(floor_y(p) - r.floor_y) < 1.0 and r.edge_distance(p) < DOORWAY_M
               for r in regions(scene_id))


def recompute_navmesh(sim) -> None:
    s = habitat_sim.NavMeshSettings()
    s.set_defaults()
    s.agent_radius = NAVMESH_AGENT_RADIUS
    s.agent_height = NAVMESH_AGENT_HEIGHT
    s.include_static_objects = True
    if not sim.recompute_navmesh(sim.pathfinder, s):
        raise RuntimeError("navmesh recompute failed")


def add_person(sim, model: str, at, yaw: float):
    """Stand an HSSD person at navmesh point `at`, facing yaw; the caller recomputes the navmesh."""
    otm = sim.get_object_template_manager()
    tid = otm.load_configs(f"{HSSD_DIR}/objects/{model[0]}/{model}.object_config.json")[0]
    obj = sim.get_rigid_object_manager().add_object_by_template_id(tid)
    obj.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
    obj.translation = mn.Vector3(at[0], floor_y(at) - obj.root_scene_node.cumulative_bb.min.y, at[2])
    obj.motion_type = habitat_sim.physics.MotionType.STATIC
    return obj


def remove_object(sim, obj) -> None:
    sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
