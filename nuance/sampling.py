import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from nuance import config as C


def sample_start_speed(rng: random.Random) -> float:
    return rng.uniform(*C.START_SPEED_RANGE)


def start_speed_bucket(v0: float) -> str:
    return "slow" if v0 < C.START_SPEED_SPLIT else "fast"


def sample_candidates(v0: float, rng: random.Random) -> dict[str, float]:
    while True:
        c = {k: min(max(v0 * rng.uniform(*band), C.MIN_SPEED), C.SPOT_MAX_SPEED)
             for k, band in C.CANDIDATE_BANDS.items()}
        if c["near"] / c["slower"] >= C.MIN_ADJ_RATIO and c["faster"] / c["near"] >= C.MIN_ADJ_RATIO:
            return c


@dataclass
class DecisionCandidate:
    scene_id: str
    path: np.ndarray
    arc_length: float
    position: np.ndarray
    yaw: float
    start_speed: float
    candidates: dict[str, float]
    context: dict


def assign_clutter_buckets(pool: list[DecisionCandidate]) -> None:
    med = float(np.median([d.context["clutter_count"] for d in pool]))
    for d in pool:
        d.context["clutter"] = "high" if d.context["clutter_count"] >= med else "low"


def stratum(dc: DecisionCandidate) -> tuple[str, str, str, str]:
    return (dc.context["clutter"], dc.context["location"], dc.context["ahead"],
            start_speed_bucket(dc.start_speed))


def select_stratified(pool: list[DecisionCandidate], n: int, rng: random.Random) -> list[DecisionCandidate]:
    remaining = list(pool)
    strata_count: Counter = Counter()
    scene_count: Counter = Counter()
    out = []
    for _ in range(n):
        if not remaining:
            break
        present = {stratum(d) for d in remaining}
        lo = min(strata_count[s] for s in present)
        cands = [d for d in remaining if strata_count[stratum(d)] == lo]
        lo_scene = min(scene_count[d.scene_id] for d in cands)
        cands = [d for d in cands if scene_count[d.scene_id] == lo_scene]
        pick = rng.choice(cands)
        remaining = [d for d in remaining if d is not pick]
        strata_count[stratum(pick)] += 1
        scene_count[pick.scene_id] += 1
        out.append(pick)
    return out


def assign_phrases(n: int, rng: random.Random) -> list[str]:
    ph = [C.PHRASES[i % len(C.PHRASES)] for i in range(n)]
    rng.shuffle(ph)
    return ph
