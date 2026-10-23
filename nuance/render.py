import math
from dataclasses import dataclass
from pathlib import Path

import habitat_sim
import imageio
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

from nuance.config import (CHASE_ALPHA, CHASE_AZIMUTHS_DEG, CHASE_DEPTH_RES, CHASE_DISTANCES_M, CHASE_HEIGHTS_M, CHASE_HFOV,
                           CHASE_ANCHOR_ALPHA, CHASE_FORE_FRAC, CHASE_FORE_MAX_AZ_DEG, CHASE_FORE_RATIO, CHASE_LOOK_AT_M, CHASE_MAX_PITCH_DEG, CHASE_MOVING_MPS,
                           CHASE_NEAR_FRAC, CHASE_NEAR_M, CHASE_OCCLUDE_TOL_M, CHASE_RESELECT_FRAMES,
                           CHASE_HEADING_AHEAD_M, CHASE_VOID_FRAC, FPS, SPOT_BASE_HEIGHT, T_DECISION_S,
                           TOPDOWN_MARGIN_M, TOPDOWN_MIN_EXTENT_M, TOPDOWN_RES)

# Yellow copy of hab_spot_arm made by scripts/make_yellow_spot.py (the original meshes are teal).
SPOT_URDF = "data/robots/spot_yellow/urdf/hab_spot_arm.urdf"
# habitat-lab SpotRobot params; joint order is arm (7), gripper (1), legs (4 x hx, hy, kn).
ARM_STOW = [0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0]
GRIPPER_CLOSED = [0.0]
LEG_STAND = [0.0, 0.7, -1.5] * 4
HIDDEN = mn.Vector3(1e4, 1e4, 1e4)
# Navmesh points float above the rendered floor (0.16 m in HSSD) and miss thresholds and rugs; the floor is
# found by a ray cast down from GROUND_RAY_UP_M above them.
GROUND_RAY_UP_M = 0.5
# Floor height map handed to the MuJoCo walker: cells within FLOOR_NEAR_M of the line take the ray-cast floor (so
# thresholds and rugs are there), cells farther out copy the nearest line point so furniture tops never become walls.
FLOOR_RES_M = 0.04
FLOOR_MARGIN_M = 2.0
FLOOR_NEAR_M = 0.6
# From the decision point on, both views get a border in the post-decision line color.
DECISION_BORDER_PX = 10
# The path is drawn as a UI line: a dark outline around a light core, white before the decision point and
# magenta after it, so the decision point is visible without a marker that could pass for an object.
PATH_PRE_RGB = np.array([235, 235, 235], dtype=float)
PATH_POST_RGB = np.array([255, 40, 200], dtype=float)
OUTLINE_RGB = np.array([20, 20, 20], dtype=float)
PATH_CORE_PX = 3
PATH_OUTLINE_PX = 5
SPOT_OUTLINE_PX = 4
SPOT_OUTLINE_RGB = np.array([255, 255, 255], dtype=float)
SPOT_DIFF_THRESH = 8
# Half-width of the box around Spot's center that is re-composited each frame (Spot is 1.1 m long).
SPOT_BOX_M = 1.0
# Outside the mesh and inside cut-off walls the top-down render is pure black; paint it a flat grey.
VOID_THRESH = 6
VOID_RGB = np.array([70, 70, 70], dtype=float)


def load_spot(sim):
    aom = sim.get_articulated_object_manager()
    spot = aom.add_articulated_object_from_urdf(SPOT_URDF, fixed_base=True)
    spot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    spot.joint_positions = ARM_STOW + GRIPPER_CLOSED + LEG_STAND
    return spot


def place_spot(spot, position: np.ndarray, yaw: float) -> None:
    p = np.asarray(position, dtype=float)
    spot.translation = mn.Vector3(p[0], p[1] + SPOT_BASE_HEIGHT, p[2])
    # URDF forward is local +x; yaw's forward is (-sin yaw, 0, -cos yaw), i.e. +x turned by yaw + pi/2.
    spot.rotation = mn.Quaternion.rotation(mn.Rad(yaw + math.pi / 2), mn.Vector3.y_axis())


@dataclass
class Motion:
    """Spot's simulated motion for one clip (nuance.spot_walk), at FPS, in habitat coordinates."""
    times: np.ndarray
    positions: np.ndarray  # base origin
    rotations: np.ndarray  # (N, 3, 3) base frame -> world; base frame is x forward, y up, -z left
    joints: np.ndarray  # (N, 12) legs, hab_spot_arm order
    arc: np.ndarray  # progress along the line

    @property
    def yaws(self) -> np.ndarray:
        f = self.rotations[:, :, 0]
        return np.arctan2(-f[:, 0], -f[:, 2])


def load_motion(npz: Path) -> Motion:
    z = np.load(npz)
    return Motion(times=z["times"], positions=z["base_pos"], rotations=z["base_rot"], joints=z["joints"],
                  arc=z["arc"])


def place_motion(spot, motion: Motion, i: int) -> None:
    spot.joint_positions = ARM_STOW + GRIPPER_CLOSED + [float(x) for x in motion.joints[i]]
    spot.translation = mn.Vector3(*motion.positions[i])
    r = motion.rotations[i]
    spot.rotation = mn.Quaternion.from_matrix(mn.Matrix3x3(mn.Vector3(*r[:, 0]), mn.Vector3(*r[:, 1]),
                                                           mn.Vector3(*r[:, 2])))


def topdown_frame(traj):
    """Camera center and extent covering a Trajectory or a Motion."""
    xz = traj.positions[:, [0, 2]]
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    extent = max(float((hi - lo).max()) + 2 * TOPDOWN_MARGIN_M, TOPDOWN_MIN_EXTENT_M)
    return (lo + hi) / 2, extent


def world_to_pixel(points: np.ndarray, center: np.ndarray, extent: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    uv = (pts[:, [0, 2]] - center) / extent * TOPDOWN_RES + TOPDOWN_RES / 2
    return np.floor(uv).astype(int)


def _set_agent(sim, position, yaw: float, pitch: float = 0.0) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = (quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
                      * quat_from_angle_axis(pitch, np.array([1.0, 0.0, 0.0])))
    sim.get_agent(0).set_state(state)


def _draw(sim, uuid: str) -> np.ndarray:
    sensor = sim._sensors[uuid]
    sensor.draw_observation()
    return np.array(sensor.get_observation())[..., :3]


def _disk(r: int) -> list:
    return [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r]


def _stamp(color: np.ndarray, alpha: np.ndarray, pts: np.ndarray, r: int, rgb: np.ndarray) -> None:
    n = color.shape[0]
    for du, dv in _disk(r):
        u, v = pts[:, 0] + du, pts[:, 1] + dv
        ok = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        alpha[v[ok], u[ok]] = 1.0
        color[v[ok], u[ok]] = rgb


def _dense(path: np.ndarray, center: np.ndarray, extent: float) -> np.ndarray:
    n = TOPDOWN_RES
    uv = (np.asarray(path, dtype=float)[:, [0, 2]] - center) / extent * n + n / 2
    pts = [uv[:1]]
    for a, b in zip(uv[:-1], uv[1:]):
        k = max(int(np.ceil(np.linalg.norm(b - a) * 2)), 1)
        pts.append(a + (b - a) * np.linspace(0, 1, k + 1)[1:, None])
    return np.floor(np.concatenate(pts)).astype(int)


def _overlay(path: np.ndarray, decision_arc: float, center: np.ndarray, extent: float):
    """Alpha mask and color image of the path line, two-toned at the decision point."""
    path = np.asarray(path, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    k = int(np.searchsorted(cum, decision_arc))
    split = np.interp(decision_arc, cum, np.arange(len(cum)))
    dp = path[int(split)] + (path[min(int(split) + 1, len(path) - 1)] - path[int(split)]) * (split - int(split))
    pre = _dense(np.vstack([path[:k], dp]), center, extent)
    post = _dense(np.vstack([dp, path[k:]]), center, extent)
    n = TOPDOWN_RES
    alpha = np.zeros((n, n))
    color = np.zeros((n, n, 3))
    allpts = np.concatenate([pre, post])
    _stamp(color, alpha, allpts, PATH_OUTLINE_PX, OUTLINE_RGB)
    _stamp(color, alpha, pre, PATH_CORE_PX, PATH_PRE_RGB)
    _stamp(color, alpha, post, PATH_CORE_PX, PATH_POST_RGB)
    return alpha[..., None], color


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = np.zeros_like(mask)
    vs, us = np.nonzero(mask)
    if not len(vs):
        return out
    # Work on Spot's bounding box only: a full-frame dilation costs more than the render.
    v0, v1 = max(vs.min() - r, 0), min(vs.max() + r + 1, mask.shape[0])
    u0, u1 = max(us.min() - r, 0), min(us.max() + r + 1, mask.shape[1])
    m = np.pad(mask[v0:v1, u0:u1], r)
    h, w = v1 - v0, u1 - u0
    for du, dv in _disk(r):
        out[v0:v1, u0:u1] |= m[r + dv:r + dv + h, r + du:r + du + w]
    return out


def _sees(sim, cam: np.ndarray, targets: list) -> bool:
    """True if nothing lies between cam and every target (rays cast from the targets, so a camera inside a wall
    counts as blocked). Spot must be hidden while this runs."""
    for t in targets:
        d = cam - t
        dist = float(np.linalg.norm(d))
        hits = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(*t), mn.Vector3(*(d / dist))), dist)
        if hits.has_hits():
            return False
    return True


def _aim(cam: np.ndarray, look: np.ndarray) -> tuple:
    d = look - cam
    return float(np.arctan2(-d[0], -d[2])), float(np.arctan2(d[1], np.hypot(d[0], d[2])))


def _view_quality(sim, cam: np.ndarray, look: np.ndarray, targets: list) -> tuple:
    """From the depth view from cam toward look: (fraction of the view nearer than CHASE_NEAR_M, fraction showing
    nothing, number of targets in frame with nothing rendered in front of them, fraction of the view nearer than
    CHASE_FORE_RATIO of the distance to look). Spot must be hidden."""
    yaw, pitch = _aim(cam, look)
    _set_agent(sim, cam, yaw, pitch)
    sensor = sim._sensors["chase_depth"]
    sensor.draw_observation()
    depth = np.array(sensor.get_observation())
    w, h = CHASE_DEPTH_RES
    f = w / 2 / np.tan(np.radians(CHASE_HFOV) / 2)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    seen = 0
    for t in targets:
        x, y, z = rot.T @ (t - cam)  # camera frame: x right, y up, looking along -z
        if z >= -1e-3:
            continue
        u, v = int(w / 2 + f * x / -z), int(h / 2 - f * y / -z)
        if 0 <= u < w and 0 <= v < h and not 0 < depth[v, u] < -z - CHASE_OCCLUDE_TOL_M:
            seen += 1
    fore = float(((depth > 0) & (depth < CHASE_FORE_RATIO * np.linalg.norm(look - cam))).mean())
    return float(((depth > 0) & (depth < CHASE_NEAR_M)).mean()), float((depth <= 0).mean()), seen, fore


def _clear_view(sim, cam: np.ndarray, look: np.ndarray, targets: list) -> tuple:
    """(the depth view has no wall at the lens, is not mostly empty, and shows every target unoccluded; it also has
    nothing filling the space before Spot)."""
    near, void, seen, fore = _view_quality(sim, cam, look, targets)
    clear = near <= CHASE_NEAR_FRAC and void <= CHASE_VOID_FRAC and seen == len(targets)
    return clear, clear and fore <= CHASE_FORE_FRAC


def _path_ahead(path: np.ndarray, s: float) -> np.ndarray:
    """Point at arc s along path, continued straight past its end."""
    path = np.asarray(path, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    d = (path[-1] - path[-2]) / max(cum[-1] - cum[-2], 1e-9)
    return np.array([np.interp(s, cum, path[:, k]) for k in range(3)]) + max(s - cum[-1], 0.0) * d


def _chase_poses(sim, spot, motion: Motion, path: np.ndarray) -> list:
    """Per frame (agent position, yaw, pitch) of a camera that follows Spot from behind and keeps it in sight. Behind
    means opposite the direction to the line CHASE_HEADING_AHEAD_M ahead of Spot, so the camera starts swinging round
    before a turn instead of after it. The camera eases between poses around Spot (bearing, distance, height), so in a
    turn it swings round Spot instead of across it. Views with nothing before Spot are preferred within
    CHASE_FORE_MAX_AZ_DEG of behind; past that a side view is worse than a wall at the edge of the frame."""
    candidates = [(dist, h, np.radians(az)) for az in CHASE_AZIMUTHS_DEG for dist in CHASE_DISTANCES_M
                  for h in CHASE_HEIGHTS_M
                  if np.degrees(np.arctan2(h - CHASE_LOOK_AT_M, dist)) <= CHASE_MAX_PITCH_DEG]
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    speed = np.gradient(np.asarray(motion.arc, dtype=float)) * FPS
    spot.translation = HIDDEN
    poses, anchor, heading, k, rel = [], None, None, None, None
    for i, (pos, yaw, s) in enumerate(zip(motion.positions, motion.yaws, motion.arc)):
        pos = np.asarray(pos, dtype=float) - np.array([0.0, SPOT_BASE_HEIGHT, 0.0])  # the floor below Spot
        fwd = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
        ahead = _path_ahead(path, s + CHASE_HEADING_AHEAD_M) - pos
        aim = np.arctan2(ahead[2], ahead[0]) if np.hypot(ahead[0], ahead[2]) > 0.3 else np.arctan2(fwd[2], fwd[0])
        heading = aim if heading is None else wrap(heading + CHASE_ALPHA * wrap(aim - heading))
        back = -np.array([np.cos(heading), 0.0, np.sin(heading)])
        target = pos + np.array([0.0, CHASE_LOOK_AT_M, 0.0])
        side = np.array([-fwd[2], 0.0, fwd[0]])
        rays = [target, target + 0.5 * fwd, target - 0.5 * fwd, target + 0.2 * side, target - 0.2 * side,
                target + np.array([0.0, 0.2, 0.0])]
        check = lambda c: _clear_view(sim, c, target, rays) if _sees(sim, c, rays) else (False, False)

        def place(c):
            dist, h, az = c
            ca, sa = np.cos(az), np.sin(az)
            b = np.array([ca * back[0] - sa * back[2], 0.0, sa * back[0] + ca * back[2]])
            return pos + dist * b + np.array([0.0, h, 0.0])

        moving = speed[i] > CHASE_MOVING_MPS
        clear, unblocked = check(place(candidates[k])) if k is not None else (False, False)
        # Standing Spot keeps its camera unless that view is lost.
        if not clear or (moving and (i % CHASE_RESELECT_FRAMES == 0 or not unblocked)):
            checks = [check(place(c)) for c in candidates]
            k = next((j for j, c in enumerate(candidates)
                      if abs(c[2]) <= np.radians(CHASE_FORE_MAX_AZ_DEG) and checks[j][1]), None)
            if k is None:
                k = next((j for j in range(len(candidates)) if checks[j][0]), None)
            if k is None:  # nothing passes both checks (e.g. Spot against a wall or door): no wall at the lens,
                # then the most of Spot in view, then the least empty view
                def quality(j):
                    near, void, seen, fore = _view_quality(sim, place(candidates[j]), target, rays)
                    return near <= CHASE_NEAR_FRAC, seen, -fore, -void, -j
                k = max(range(len(candidates)), key=quality)
        goal = place(candidates[k]) - pos
        goal = np.array([np.arctan2(goal[2], goal[0]), np.hypot(goal[0], goal[2]), goal[1]])
        rel = goal if rel is None else rel + CHASE_ALPHA * np.array([wrap(goal[0] - rel[0]), *(goal[1:] - rel[1:])])
        anchor = pos if anchor is None else anchor + CHASE_ANCHOR_ALPHA * (pos - anchor)  # smooths the gait's sway
        cam = anchor + np.array([rel[1] * np.cos(rel[0]), rel[2], rel[1] * np.sin(rel[0])])
        if not check(cam)[0] and (check(place(candidates[k]))[0] or not _sees(sim, cam, rays)):
            rel = goal  # cut instead of gliding through a wall
            cam = anchor + place(candidates[k]) - pos
        poses.append((cam.copy(), *_aim(cam, anchor + np.array([0.0, CHASE_LOOK_AT_M, 0.0]))))
    return poses


def _floor_y(sim, p: np.ndarray) -> float:
    """Height of the rendered floor below navmesh point p (p's own height if the ray finds nothing)."""
    hits = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(p[0], p[1] + GROUND_RAY_UP_M, p[2]),
                                            mn.Vector3(0.0, -1.0, 0.0)), 2 * GROUND_RAY_UP_M)
    return float(p[1] + GROUND_RAY_UP_M - hits.hits[0].ray_distance) if hits.has_hits() else float(p[1])


def floor_grid(sim, spot, path: np.ndarray) -> dict:
    """Floor height map around the line for nuance.spot_walk, on a grid in MuJoCo's x / y (= habitat x / -z)."""
    spot.translation = HIDDEN
    path = np.asarray(path, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    samples = np.array([[np.interp(t, cum, path[:, k]) for k in range(3)]
                        for t in np.arange(0.0, cum[-1] + 1e-9, FLOOR_RES_M)])
    line_floor = np.array([_floor_y(sim, q) for q in samples])
    mx, my = samples[:, 0], -samples[:, 2]
    xs = np.arange(mx.min() - FLOOR_MARGIN_M, mx.max() + FLOOR_MARGIN_M + 1e-9, FLOOR_RES_M)
    ys = np.arange(my.min() - FLOOR_MARGIN_M, my.max() + FLOOR_MARGIN_M + 1e-9, FLOOR_RES_M)
    heights = np.empty((len(ys), len(xs)))
    for j, y in enumerate(ys):
        d2 = (mx[None, :] - xs[:, None]) ** 2 + (my[None, :] - y) ** 2
        k = d2.argmin(axis=1)
        for i, x in enumerate(xs):
            near = d2[i, k[i]] <= FLOOR_NEAR_M ** 2
            heights[j, i] = _floor_y(sim, np.array([x, samples[k[i], 1], -y])) if near else line_floor[k[i]]
    return {"origin": [float(xs[0]), float(ys[0])], "res": FLOOR_RES_M, "heights": heights.tolist()}


def _border(img: np.ndarray, w: int, rgb: np.ndarray) -> None:
    img[:w], img[-w:], img[:, :w], img[:, -w:] = rgb, rgb, rgb, rgb


def render_clip(sim, spot, motion: Motion, path: np.ndarray, decision_arc: float, out_video: Path,
                frame=None) -> None:
    center, extent = frame if frame is not None else topdown_frame(motion)
    floor_y = float(motion.positions[:, 1].min()) - SPOT_BASE_HEIGHT
    so = sim._sensors["topdown"]._sensor_object
    spec = so.specification()
    spec.ortho_scale = 1.0 / extent
    so.set_projection_params(spec)
    _set_agent(sim, [center[0], floor_y, center[1]], 0.0)

    spot.translation = HIDDEN
    background = _draw(sim, "topdown").astype(np.int16)
    void = (background.max(axis=-1) < VOID_THRESH)[..., None]
    alpha, color = _overlay(path, decision_arc, center, extent)

    # Only Spot changes between frames: composite the static scene once, then redo a box around Spot.
    base = np.where(void, VOID_RGB, background.astype(float))
    base = (base * (1 - alpha) + color * alpha).astype(np.uint8)
    half = int(np.ceil(SPOT_BOX_M / extent * TOPDOWN_RES))
    n = TOPDOWN_RES

    writer = imageio.get_writer(str(out_video), fps=FPS, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=16, quality=None,
                                output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    chase = _chase_poses(sim, spot, motion, path)
    try:
        for i, (t, (cam, cam_yaw, cam_pitch)) in enumerate(zip(motion.times, chase)):
            place_motion(spot, motion, i)
            pos = motion.positions[i]
            _set_agent(sim, [center[0], floor_y, center[1]], 0.0)
            frame = _draw(sim, "topdown")
            u, v = world_to_pixel(np.asarray(pos)[None], center, extent)[0]
            box = np.s_[max(v - half, 0):min(v + half, n), max(u - half, 0):min(u + half, n)]
            f, bg = frame[box], background[box]
            # Spot occludes the overlay: keep rendered pixels where Spot differs from the empty scene.
            spot_mask = np.abs(f.astype(np.int16) - bg).max(axis=-1) > SPOT_DIFF_THRESH
            ring = _dilate(spot_mask, SPOT_OUTLINE_PX) & ~spot_mask
            img = base.copy()
            crop = img[box]
            crop[spot_mask] = f[spot_mask]
            crop[ring] = SPOT_OUTLINE_RGB
            _set_agent(sim, cam, cam_yaw, cam_pitch)
            views = [_draw(sim, "chase").copy(), img]
            if t >= T_DECISION_S - 1e-9:
                for view in views:
                    _border(view, DECISION_BORDER_PX, PATH_POST_RGB)
            writer.append_data(np.concatenate(views, axis=1))
    finally:
        writer.close()


def render_front(sim, spot, position: np.ndarray, yaw: float, out_png: Path) -> None:
    spot.translation = HIDDEN
    position = np.array(position, dtype=float)
    position[1] = _floor_y(sim, position)
    place_spot(spot, position, yaw)
    _set_agent(sim, position, yaw)
    imageio.imwrite(str(out_png), _draw(sim, "front"))
