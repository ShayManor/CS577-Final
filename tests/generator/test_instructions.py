import json
import random

import numpy as np

from nuance.candidates import SPEED_SETS, SPEED_TARGET, pick3, self_approach, speed_set
from nuance.generate_instructions import pick_alts
from nuance.grounding import Region
from nuance.instructions import FALLBACK, SPECS, SPEED_KIND
from nuance.candidates import GENERATORS


def test_every_approved_instruction_has_a_spec():
    review = json.load(open("configs/instruction_review.json"))
    approved = {t for t, v in review.items() if v == "approve"}
    assert approved == set(SPECS)
    assert {s.family for s in SPECS.values()} <= set(GENERATORS)


def test_alternatives_are_known_and_distinct():
    for k, v in FALLBACK.items():
        assert k in SPECS and set(v) <= set(SPECS)
    for text in SPECS:
        names = SPEED_SETS[SPEED_KIND[text]][0] if SPECS[text].family == "speed" else ()
        alts = pick_alts(text, random.Random(0), names)
        assert len(alts) <= 2 and len(set(alts)) == len(alts) and text not in alts
        if SPECS[text].family == "speed":
            # each alternative is answered by one of the candidates, and is not a synonym of the instruction
            assert alts and all(SPEED_TARGET[SPEED_KIND[a]] in names for a in alts)
            assert all(SPEED_KIND[a] != SPEED_KIND[text] for a in alts)


def test_speed_sets_are_spread_and_feasible():
    rng = random.Random(0)
    for kind in SPEED_SETS:
        for names in SPEED_SETS[kind]:
            assert SPEED_TARGET[kind] in names
            assert any(speed_set(kind, v0, rng, names) for v0 in np.linspace(0.5, 1.0, 11)), (kind, names)
        ok = [speed_set(kind, v0, rng) for v0 in np.linspace(0.3, 1.0, 15)]
        for c in filter(None, ok):
            v = sorted(c.values())
            assert min(b - a for a, b in zip(v[:-1], v[1:])) >= 0.1 and v[0] >= 0.25 and v[-1] <= 1.2


def test_pick3_spacing():
    items = [0.1, 0.15, 0.4, 0.5, 0.9]
    assert pick3(items, key=lambda x: x, step=0.2, span=0.6) == [0.1, 0.5, 0.9]
    assert pick3([0.1, 0.2, 0.3], key=lambda x: x, step=0.2, span=0.6) is None


def test_self_approach_flags_a_folded_line():
    straight = np.array([[0, 0, 0], [6, 0, 0]], dtype=float)
    folded = np.array([[0, 0, 0], [3, 0, 0], [3, 0, 0.2], [0, 0, 0.2]], dtype=float)
    assert self_approach(straight) == np.inf or self_approach(straight) > 1.0
    assert self_approach(folded) < 0.3


def test_region_contains_and_edge_distance():
    r = Region("bedroom", np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float), 0.0)
    nav_y = 0.16  # navmesh points float above the floor
    assert r.contains([2, nav_y, 1]) and not r.contains([5, nav_y, 1])
    assert abs(r.edge_distance([4.3, nav_y, 1]) - 0.3) < 1e-9
    assert abs(r.area - 12) < 1e-9
