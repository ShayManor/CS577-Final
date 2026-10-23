"""Walk a physically simulated Spot along a clip's line in MuJoCo and record its motion for rendering.

Runs in the spot-loco env (python 3.11, mujoco, jax, brax, playground), not the habitat env:
    python -m nuance.spot_walk jobs.json out_dir
jobs.json holds the shared floor height map and one job per candidate (see generate.py, generate_instructions.py). Each job writes
out_dir/<name>.npz with, at FPS, the clip times, Spot's base position and orientation in habitat coordinates, its 12
leg joint angles (fl, fr, hl, hr x hx, hy, kn, the same order and convention as hab_spot_arm), and its progress
along the line.

The policy is MuJoCo Playground's Spot joystick policy (scripts/train_spot_policy.py). It takes a body-frame
velocity command; a path follower turns the line and the clip's speed profile into that command each step.
"""
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import spot_constants as consts
from ml_collections import config_dict

from nuance.config import FPS, HOLD_S, RAMP_S, SPOT_MAX_SPEED, SPOT_POLICY_DIR as POLICY_DIR, T_DECISION_S

WARMUP_S = 2.0  # Spot starts standing this long before the clip, walking up to the start speed off-screen
SETTLE_S = 0.5  # of the warm-up, time standing still before it starts walking
T_LIMIT_S = 30.0
K_ARC = 1.0  # speed command correction per meter Spot is behind its planned progress
K_YAW = 2.0
K_LAT = 1.0
LOOKAHEAD_M = (0.4, 0.3)  # a + b * speed
MAX_YAW_RATE = 1.2
MAX_LAT = 0.3
FALL_UP_Z = 0.85
END_TOL_M = 0.02
STOP_DECEL = 1.2  # m/s^2: Spot slows to a stop at the end of the line
STOP_CREEP = 0.15  # m/s: speed cap floor so Spot still reaches the end of the line
MIN_CRUISE_S = 0.5
SPEED_WINDOW_S = 0.3
PROJECT_BACK_M, PROJECT_AHEAD_M = 0.5, 1.5  # arc window around Spot's last progress searched for its position

# Habitat is y-up (x right, z back); MuJoCo is z-up. mj = M @ habitat, and Spot's local frames (habitat: x forward,
# y up, -z left; MuJoCo: x forward, y left, z up) map the same way, so R_habitat = M.T @ R_mj @ M.
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

SCENE_XML = """
<mujoco model="spot on nuance floor">
  <include file="spot_mjx_feetonly.xml"/>
  <asset>
    <hfield name="floor" nrow="{nrow}" ncol="{ncol}" size="{rx} {ry} {elev} 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="floor" pos="{cx} {cy} {base}" contype="1" conaffinity="0"
      priority="1" friction="0.6" condim="3"/>
  </worldbody>
  <include file="sensor.xml"/>
</mujoco>
"""


def arc_at(t, v0, v1, t_decision=T_DECISION_S, ramp=RAMP_S):
    """Planned distance along the line at clip time t (negative t: walking up to the start of the line)."""
    t = np.asarray(t, dtype=float)
    s = v0 * np.minimum(t, t_decision)
    r = np.clip(t - t_decision, 0.0, ramp)
    s = s + v0 * r + 0.5 * (v1 - v0) * r * r / ramp
    return s + v1 * np.maximum(t - t_decision - ramp, 0.0)


class Line:
    """The clip's line in MuJoCo xy, extended backwards along its first segment for the warm-up walk."""

    def __init__(self, path_h: np.ndarray, back_m: float):
        xy = (np.asarray(path_h, dtype=float) @ M.T)[:, :2]
        d0 = (xy[1] - xy[0]) / np.linalg.norm(xy[1] - xy[0])
        self.xy = np.vstack([xy[0] - back_m * d0, xy])
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)]) - back_m  # arc 0 = start of the clip's line
        self.total = float(self.cum[-1])

    def point(self, s):
        return np.array([np.interp(s, self.cum, self.xy[:, k]) for k in range(2)])

    def end_heading(self) -> float:
        d = self.xy[-1] - self.xy[-2]
        return float(np.arctan2(d[1], d[0]))

    def ahead(self, s):
        """point(s), continued straight past the end of the line so steering lines Spot up with the last segment."""
        h = self.end_heading()
        return self.point(s) + max(s - self.total, 0.0) * np.array([np.cos(h), np.sin(h)])

    def project(self, p, s_prev):
        """(arc, signed left offset) of the closest point to p on the part of the line near arc s_prev, so a line
        that comes back near itself (a U-turn) does not snap onto a later segment."""
        best = (np.inf, 0.0, 0.0)
        for i in range(len(self.xy) - 1):
            if self.cum[i + 1] < s_prev - PROJECT_BACK_M or self.cum[i] > s_prev + PROJECT_AHEAD_M:
                continue
            a, b = self.xy[i], self.xy[i + 1]
            ab = b - a
            u = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
            q = a + u * ab
            dist = float(np.linalg.norm(p - q))
            if dist < best[0]:
                left = float(np.cross(ab / np.linalg.norm(ab), p - q))
                best = (dist, self.cum[i] + u * (self.cum[i + 1] - self.cum[i]), left)
        return best[1], best[2]


def _model(floor: dict, env_cfg) -> mujoco.MjModel:
    h = np.asarray(floor["heights"], dtype=float)  # rows along mj y, cols along mj x
    res = float(floor["res"])
    nrow, ncol = h.shape
    lo, hi = float(h.min()), float(h.max())
    elev = max(hi - lo, 1e-3)
    x0, y0 = floor["origin"]
    xml = SCENE_XML.format(nrow=nrow, ncol=ncol, rx=(ncol - 1) * res / 2, ry=(nrow - 1) * res / 2, elev=elev,
                           cx=x0 + (ncol - 1) * res / 2, cy=y0 + (nrow - 1) * res / 2, base=lo)
    m = mujoco.MjModel.from_xml_string(xml, assets=spot_base.get_assets())
    m.hfield_data[:] = ((h - lo) / elev).ravel()
    m.opt.timestep = env_cfg.sim_dt
    # Same PD gains as training (SpotEnv.__init__).
    m.dof_damping[6:] = env_cfg.Kd
    m.actuator_gainprm[:, 0] = env_cfg.Kp
    m.actuator_biasprm[:, 1] = -env_cfg.Kp
    return m


def _policy(policy_dir: str, obs_size):
    cfg = json.loads((Path(policy_dir) / "config.json").read_text())
    # params.pkl is written by our own scripts/train_spot_policy.py; never load one from elsewhere.
    with open(Path(policy_dir) / "params.pkl", "rb") as f:
        params = pickle.load(f)
    nets = ppo_networks.make_ppo_networks(obs_size, 12, preprocess_observations_fn=running_statistics.normalize,
                                          **cfg["network"])
    return jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True)), cfg


def _sensor(m, d, name):
    s = m.sensor(name)
    return d.sensordata[s.adr[0]:s.adr[0] + s.dim[0]].copy()


def walk(m, policy, env_cfg, default_pose, priv_size, floor: dict, line: Line, v0: float, v1: float) -> dict:
    d = mujoco.MjData(m)
    n_sub = int(round(env_cfg.ctrl_dt / env_cfg.sim_dt))
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    # Stand where the warm-up walk begins, facing along the line, feet on the floor. The warm-up speeds up evenly
    # from 0 to v0 over walk_s, so it covers v0 * walk_s / 2 and reaches the start of the line at v0 at t = 0.
    walk_s = WARMUP_S - SETTLE_S
    s0 = -v0 * walk_s / 2
    start = line.point(s0)
    heading = line.point(s0 + 0.1) - start
    psi = float(np.arctan2(heading[1], heading[0]))
    d.qpos[:3] = [start[0], start[1], 0.0]
    d.qpos[3:7] = [np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)]
    d.qpos[7:] = default_pose
    mujoco.mj_forward(m, d)
    feet_z = min(_sensor(m, d, f"{f}_pos")[2] for f in consts.FEET_SITES)  # feet height relative to the body
    h = np.asarray(floor["heights"], dtype=float)
    i = int(np.clip(round((start[0] - floor["origin"][0]) / floor["res"]), 0, h.shape[1] - 1))
    j = int(np.clip(round((start[1] - floor["origin"][1]) / floor["res"]), 0, h.shape[0] - 1))
    d.qpos[2] = h[j, i] - feet_z + 0.03  # settles onto the floor while standing still
    mujoco.mj_forward(m, d)

    targets = default_pose.copy()
    last_act = np.zeros(12)
    hist = np.zeros(36)
    key = jax.random.PRNGKey(0)
    rec = {"t": [], "pos": [], "quat": [], "joints": [], "arc": []}
    t, fell, t_end, t_brake = -WARMUP_S, False, None, None
    s_act = s0
    while t < T_LIMIT_S:
        p = d.qpos[:2].copy()
        s_act, left = line.project(p, s_act)
        if t >= 0:
            rec["t"].append(t)
            rec["pos"].append(d.qpos[:3].copy())
            rec["quat"].append(d.qpos[3:7].copy())
            rec["joints"].append(d.qpos[7:].copy())
            rec["arc"].append(s_act)
            if t_end is None and s_act >= line.total - END_TOL_M:
                t_end = t
            if t_end is not None and t >= t_end + HOLD_S:
                break
        up = _sensor(m, d, "upvector")
        if up[2] < FALL_UP_Z:
            fell = True
            print(f"fell at t={t:.2f} s, arc {s_act:.2f} of {line.total:.2f}, line offset {left:.2f} m", flush=True)
            break

        q = d.qpos[3:7]
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        if t < -WARMUP_S + SETTLE_S:
            cmd = np.zeros(3)
        elif t_end is not None:  # stand, turning in place to face along the end of the line
            err = (line.end_heading() - yaw + np.pi) % (2 * np.pi) - np.pi
            cmd = np.array([0.0, 0.0, float(np.clip(K_YAW * err, -MAX_YAW_RATE, MAX_YAW_RATE))])
        else:
            if t < 0:
                tw = t + walk_s  # time since the warm-up walk began
                v_des, s_des = v0 * tw / walk_s, s0 + v0 * tw * tw / (2 * walk_s)
            else:
                v_des = float((arc_at(t + 1e-3, v0, v1) - arc_at(t, v0, v1)) / 1e-3)
                s_des = float(arc_at(t, v0, v1))
            v = v_des + K_ARC * (s_des - s_act)
            stop = np.sqrt(2 * STOP_DECEL * max(line.total - s_act, 0.0)) + STOP_CREEP
            if t_brake is None and t >= 0 and stop < v:
                t_brake = t
            v = float(np.clip(v, 0.0, min(SPOT_MAX_SPEED, stop)))
            target = line.ahead(s_act + LOOKAHEAD_M[0] + LOOKAHEAD_M[1] * v)
            err = float(np.arctan2(target[1] - p[1], target[0] - p[0])) - yaw
            err = (err + np.pi) % (2 * np.pi) - np.pi
            cmd = np.array([v, float(np.clip(-K_LAT * left, -MAX_LAT, MAX_LAT)),
                            float(np.clip(K_YAW * err, -MAX_YAW_RATE, MAX_YAW_RATE))])

        joints = d.qpos[7:].copy()
        hist = np.roll(hist, 12)
        hist[:12] = joints - targets
        feet = np.concatenate([_sensor(m, d, f"{f}_pos") for f in consts.FEET_SITES])
        state = np.concatenate([_sensor(m, d, "gyro"), up, joints - default_pose, hist, feet, last_act, cmd])
        obs = {"state": state.astype(np.float32), "privileged_state": np.zeros(priv_size, np.float32)}
        key, sub = jax.random.split(key)
        act, _ = policy(obs, sub)
        act = np.asarray(act, dtype=float)
        targets = np.clip(default_pose + act * env_cfg.action_scale, lo, hi)
        d.ctrl[:] = targets
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        last_act = act
        t += env_cfg.ctrl_dt
    return {k: np.array(v) for k, v in rec.items()} | {"fell": fell, "t_end": t_end, "t_brake": t_brake}


def _resample(rec: dict) -> dict:
    """Control-rate record -> FPS, in habitat coordinates."""
    t = rec["t"]
    tf = np.arange(int(np.floor(t[-1] * FPS)) + 1) / FPS
    lerp = lambda a: np.stack([np.interp(tf, t, a[:, k]) for k in range(a.shape[1])], axis=-1)
    quat = lerp(rec["quat"])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rot = []
    for w, x, y, z in quat:
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        rot.append(M.T @ r @ M)
    return {"times": tf, "base_pos": lerp(rec["pos"]) @ M, "base_rot": np.array(rot),
            "joints": lerp(rec["joints"]), "arc": np.interp(tf, t, rec["arc"])}


@dataclass
class Walker:
    policy: object
    env_cfg: object
    default_pose: np.ndarray
    priv_size: int


def load_walker(policy_dir: str = POLICY_DIR) -> Walker:
    cfg = json.loads((Path(policy_dir) / "config.json").read_text())
    env_cfg = config_dict.ConfigDict(cfg["env_config"])
    env = registry.load(cfg["env"], config=env_cfg)
    obs_size = dict(env.observation_size)
    policy, _ = _policy(policy_dir, obs_size)
    return Walker(policy=policy, env_cfg=env_cfg, default_pose=np.array(env.mj_model.keyframe("home").qpos[7:]),
                  priv_size=obs_size["privileged_state"][0])


def walk_job(w: Walker, floor: dict, path: np.ndarray, v0: float, v1: float) -> dict:
    """Walk one candidate; the resampled motion plus "fell", "t_end" and "v_post"."""
    m = _model(floor, w.env_cfg)
    line = Line(np.asarray(path, dtype=float), back_m=v0 * (WARMUP_S - SETTLE_S) / 2 + 0.3)
    rec = walk(m, w.policy, w.env_cfg, w.default_pose, w.priv_size, floor, line, v0, v1)
    if rec["fell"] or len(rec["t"]) < 2 or rec["t_end"] is None:
        return {"fell": True}
    # Achieved speed after the decision point: the mean from the end of the speed ramp until braking for the end of
    # the line, or on runs too short for that (under MIN_CRUISE_S), the speed over SPEED_WINDOW_S at the ramp's end.
    t_end = rec["t_end"]
    c0, c1 = T_DECISION_S + RAMP_S, rec["t_brake"] if rec["t_brake"] is not None else t_end
    arc = lambda tt: float(np.interp(tt, rec["t"], rec["arc"]))
    if c1 - c0 >= MIN_CRUISE_S:
        v_post = (arc(c1) - arc(c0)) / (c1 - c0)
    else:
        c = min(c0, t_end - SPEED_WINDOW_S / 2)
        v_post = (arc(c + SPEED_WINDOW_S / 2) - arc(c - SPEED_WINDOW_S / 2)) / SPEED_WINDOW_S
    return {"fell": False, **_resample(rec), "t_end": t_end, "v_post": v_post}


def main(jobs_file: str, out_dir: str) -> None:
    spec = json.loads(Path(jobs_file).read_text())
    w = load_walker(spec.get("policy_dir", POLICY_DIR))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for job in spec["jobs"]:
        res = walk_job(w, spec["floor"], np.array(spec["path"]), job["v0"], job["v1"])
        np.savez(out / f"{job['name']}.npz", **res)
        print(job["name"], "fell" if res["fell"] else f"{res['times'][-1]:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
