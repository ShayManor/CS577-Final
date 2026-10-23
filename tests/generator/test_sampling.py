import random
from collections import Counter

import numpy as np

from nuance import config as C
from nuance.sampling import DecisionCandidate, assign_phrases, sample_candidates, select_stratified, stratum


def test_candidates_bounds_and_separation():
    for seed in range(1000):
        rng = random.Random(seed)
        v0 = rng.uniform(*C.START_SPEED_RANGE)
        c = sample_candidates(v0, rng)
        assert set(c) == {"slower", "near", "faster"}
        for v in c.values():
            assert C.MIN_SPEED <= v <= C.SPOT_MAX_SPEED
        assert c["slower"] < c["near"] < c["faster"]
        assert c["near"] / c["slower"] >= C.MIN_ADJ_RATIO
        assert c["faster"] / c["near"] >= C.MIN_ADJ_RATIO


def _dc(scene, clutter, location, ahead, v0):
    return DecisionCandidate(
        scene_id=scene, path=np.zeros((2, 3)), arc_length=10.0, position=np.zeros(3), yaw=0.0,
        start_speed=v0, candidates={}, context={"clutter": clutter, "location": location, "ahead": ahead})


def test_select_balances_strata():
    pool = []
    for k, (cl, loc, ah, v0) in enumerate([("low", "open", "straight", 0.4), ("low", "tight", "turn", 0.9),
                                           ("high", "open", "turn", 0.4), ("high", "tight", "straight", 0.9)]):
        pool += [_dc(f"s{i % 3}", cl, loc, ah, v0) for i in range(10)]
    sel = select_stratified(pool, 8, random.Random(0))
    assert len(sel) == 8 and len({id(d) for d in sel}) == 8
    assert set(Counter(stratum(d) for d in sel).values()) == {2}


def test_select_balances_scenes():
    pool = [_dc(f"s{i % 4}", "low", "open", "straight", 0.4) for i in range(40)]
    sel = select_stratified(pool, 8, random.Random(0))
    assert set(Counter(d.scene_id for d in sel).values()) == {2}


def test_assign_phrases_balanced():
    n = 4 * len(C.PHRASES) + 1
    ph = assign_phrases(n, random.Random(0))
    assert len(ph) == n
    assert sorted(Counter(ph).values()) == [4] * (len(C.PHRASES) - 1) + [5]
