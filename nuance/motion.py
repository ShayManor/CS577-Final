from dataclasses import dataclass

import numpy as np

from nuance.config import CLIP_MAX_S, FPS, POST_DECISION_M, RAMP_S, T_DECISION_S

HEADING_WINDOW_M = 0.3


@dataclass
class Trajectory:
    times: np.ndarray
    positions: np.ndarray
    yaws: np.ndarray
    arc: np.ndarray


def _arc_at(t, v0, v1, td, ramp):
    t = np.asarray(t, dtype=float)
    s = v0 * np.minimum(t, td)
    r = np.clip(t - td, 0.0, ramp)
    s = s + v0 * r + 0.5 * (v1 - v0) * r * r / ramp
    return s + v1 * np.maximum(t - td - ramp, 0.0)


def trajectory(path: np.ndarray, v0: float, v1: float, t_decision: float = T_DECISION_S,
               ramp_s: float = RAMP_S, fps: int = FPS, t_max: float = CLIP_MAX_S) -> Trajectory:
    path = np.asarray(path, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total = cum[-1]

    def pos(s):
        return np.stack([np.interp(s, cum, path[:, k]) for k in range(3)], axis=-1)

    t = np.arange(int(round(t_max * fps)) + 1) / fps
    s = _arc_at(t, v0, v1, t_decision, ramp_s)
    hit = np.nonzero(s >= total)[0]
    if len(hit):
        t, s = t[:hit[0] + 1], s[:hit[0] + 1]
        s[-1] = total
    d = pos(np.minimum(s + HEADING_WINDOW_M, total)) - pos(np.maximum(s - HEADING_WINDOW_M, 0.0))
    yaws = np.arctan2(-d[:, 0], -d[:, 2])
    return Trajectory(times=t, positions=pos(s), yaws=yaws, arc=s)


def required_path_length(v0: float) -> float:
    return v0 * T_DECISION_S + POST_DECISION_M
