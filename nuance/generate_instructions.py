"""Clips for every approved instruction, each at decision points where the instruction makes sense.

    python -m nuance.generate_instructions run --scenes configs/pilot_scenes.txt --per-instruction 1 \
        --batch-id ctx-001 --out data/batches/ctx-001 [--workers 4]

run does five steps, each also callable on its own:
  plan    per scene: for every instruction, decision points where it is grounded (nuance.candidates), with
          the three candidate lines
  assign  per instruction: which scene's decision points to render, spread over scenes, and the alternative
          instructions each clip is also rated under
  render  per scene: Spot walks each candidate line (nuance.spot_walk), then the clips and front frame are rendered
  merge   all scenes' clip records into one manifest
  sheet   one review image per decision point: the front frame, and each candidate's chase and top-down views
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nuance import config as cfg
from nuance import grounding as G
from nuance.candidates import (AROUND, SEATS, _corners, SPEED_TARGET, say, _grouped, _members, SPEED_V0_BY_KIND, WALK_MAX_SPEED, Decision, _lateral_room, at, dense, ground, heading_at, length,
                               max_turn_deg, shortest, v3, valid)
from nuance.config import FPS, HOLD_S, T_DECISION_S
from nuance.context import measure_context
from nuance.instructions import FALLBACK, FURNITURE, SPECS, SPEED_KIND, same_kind
from nuance.records import ClipRecord, make_clip_id, write_jsonl
from nuance.scenes import body_clear, sample_path

CONTEXT_SPEED_RANGE = (0.5, 0.9)  # context clips: slow enough to see, fast enough to cover the route in a clip
CONTEXT_POST_M = 6.0
SPEED_POST_M = 4.5  # long enough to see a fast candidate for a few seconds
SPEED_V0_RANGE = (0.6, 1.0)  # slower starts make every candidate a crawl
BACK_SPEED_RANGE = (0.65, 0.8)  # a longer walk-in, so "the way you came" is established, slow enough to U-turn
MAX_TURN_AT_DECISION_DEG = 30.0
FRONT_FREE_M = 1.5
END_TOL_M = 0.35  # a walk must end this close to the end of its line
OFF_NAVMESH_M = 0.25  # Spot's base may stray this far off the (Spot-sized) navmesh
STUCK_S = 3.0  # a walk taking this much longer than planned got stuck
UPRIGHT_MIN = 0.9  # cosine between Spot's body up axis and vertical (about 25 degrees of tilt)
TRIES = int(os.environ.get("NUANCE_TRIES", 2000))  # lower for quick targeted re-plans
PERSON_TRIES = 150
PLACE_TRIES = 150  # each place try already searches ~200 goals
OPTIONS_PER_SCENE = 2


def _dec_from_path(scene_id, path, v0, all_objs) -> Decision:
    s0 = v0 * T_DECISION_S
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    k = int(np.searchsorted(cum, s0))
    p = at(path, s0)
    pre = np.vstack([path[:k], p])
    post = np.vstack([p, path[k:]])
    h = heading_at(path, s0)
    yaw = float(np.arctan2(-h[0], -h[1]))
    return Decision(scene_id=scene_id, pre=pre, post=post, yaw=yaw, v0=v0,
                    objs=G.floor_objects(all_objs, G.floor_y(p)), all_objs=all_objs)


AIMED = {"around", "pass", "length"}  # families that always need a target on the way
AIMED_REFERENTS = {"obstacle", "obstacle_or_person", "table", "chair", "furniture", "wall", "thing_hit"}


def _aims(spec) -> bool:
    return spec.family in AIMED or (spec.family == "clear" and spec.referent in AIMED_REFERENTS)


def _aimed_decision(sim, scene_id, rng, spec, all_objs, attempts: int = 5):
    for _ in range(attempts):
        dec = _aimed_once(sim, scene_id, rng, spec, all_objs)
        if dec is not None:
            return dec
    return None


def _aimed_once(sim, scene_id, rng, spec, all_objs):
    """A decision point whose line runs past what the instruction is about, which random lines rarely do: through
    the gap between two free-standing objects ("around"), just beside one object (its edge 0.15-0.6 m off Spot's
    straight line, or on it for "thing_hit"), or along a wall 0.5-1.4 m to one side."""
    pf = sim.pathfinder
    v0 = rng.uniform(*CONTEXT_SPEED_RANGE)
    p0 = np.array(pf.get_random_navigable_point(), dtype=float)
    if spec.referent == "wall":
        dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 8, False)]
        side = [d for d in dirs if 0.5 <= G.wall_distance(sim, p0, d, 1.4) <= 1.4]
        if not side:
            return None
        d = rng.choice(side)
        targets = [(p0[[0, 2]], np.array([-d[1], d[0]]) * k) for k in (1, -1)]
    else:
        objs = [o for o in G.floor_objects(all_objs, G.floor_y(p0)) if o.category and o.category != "person"
                and G.indoor(scene_id, np.r_[o.center[0], p0[1], o.center[2]])]
        # only what can be a referent: standing free, not against a wall (object_referent rejects the rest)
        fy = G.floor_at(sim, p0)
        floor = objs
        # "the chair" and "the table" are said as whatever single thing is there (nuance.candidates.ground)
        if spec.referent in ("obstacle", "obstacle_or_person", "furniture", "table", "chair", "thing_hit"):
            objs = _grouped(objs, floor)  # a table and its chairs are passed as one
        objs = [o for o in objs if G.free_standing(o, [x for x in floor if x not in _members(o)],
                                                   allowed=SEATS if o.category == "table" else ())
                and not _flush(sim, scene_id, o, fy)]
        if spec.referent == "furniture":
            objs = [o for o in objs if o.category in FURNITURE]
        if not objs:
            return None
        a = rng.choice(objs)
        if spec.family == "around":
            def gap(b):
                return float(np.hypot(*np.maximum(np.maximum(a.lo - b.hi, b.lo - a.hi), 0.0)[[0, 2]]))
            pals = [b for b in objs if b is not a and 0.9 <= gap(b) <= 1.5]
            if not pals:
                return None
            b = rng.choice(pals)
            u = (b.center - a.center)[[0, 2]]
            # the middle of the gap, between the facing sides of the two footprints
            da, db = a.dist_xz(b.center[None])[0], b.dist_xz(a.center[None])[0]
            n = np.linalg.norm(u)
            g = a.center[[0, 2]] + (n - db + (n - da)) / 2 / n * u
            across = np.array([-u[1], u[0]]) / n
            targets = [(g, across), (g, -across)]
        else:
            targets = []
            for t in rng.sample(list(np.linspace(0, 2 * np.pi, 12, False)), 12):
                h = np.array([np.cos(t), np.sin(t)])
                lat = np.array([-h[1], h[0]])
                half = 0.5 * (abs(lat[0]) * (a.hi[0] - a.lo[0]) + abs(lat[1]) * (a.hi[2] - a.lo[2]))
                k = rng.choice([-1, 1])
                if spec.family == "pass":
                    # "the other side" is open floor too
                    far = a.center[[0, 2]] - k * (half + 0.65) * lat
                    if not pf.is_navigable(np.r_[far[0], p0[1], far[1]].astype(np.float32)):
                        continue
                # "go around the table": the straight line runs into it
                off = rng.uniform(-0.3, 0.1) if spec.referent == "thing_hit" else rng.uniform(0.15, 0.6)
                targets.append((a.center[[0, 2]] + k * (half + off) * lat, h))
    s0 = v0 * T_DECISION_S
    y = p0[1]
    # passing on the far side needs room to rejoin the line past the object
    aheads = {"length": (4.0,), "pass": (4.0, 3.4, 2.8)}.get(spec.family, (2.8,))
    for g, h in targets:
        g3, h3 = np.r_[g[0], y, g[1]], v3(h)
        for back, ahead in [(b, a) for b in (2.5, 2.0, 3.0) for a in aheads]:
            start, p, end = g3 - (back + s0) * h3, g3 - back * h3, g3 + ahead * h3
            # the line and Spot's warm-up run-up behind it (see _good_spot) are open floor
            line = np.linspace(start, p, 12)
            run_up = [start - t * h3 for t in np.arange(0.1, 0.3 + 0.75 * v0 + 0.5, 0.1)]
            if not all(pf.is_navigable(q.astype(np.float32)) for q in [*line, *run_up]):
                continue
            e = np.array(pf.snap_point(end.astype(np.float32)), dtype=float)
            if not np.isfinite(e).all() or np.hypot(*(e - end)[[0, 2]]) > 0.6:
                continue
            post = shortest(sim, p, e)
            if post is None or length(post) > ahead + 4.2:
                continue
            path = np.vstack([line[:-1], post])
            if body_clear(sim, line):
                return _dec_from_path(scene_id, path, v0, all_objs)
    return None


_FLUSH = {}


def _flush(sim, scene_id, o, fy) -> bool:
    key = (scene_id, o.oid, len(_members(o)))  # a table alone and with its chairs differ
    if key not in _FLUSH:
        _FLUSH[key] = G.wall_flush(sim, o, fy)
    return _FLUSH[key]


def _good_spot(sim, dec: Decision) -> bool:
    """A decision point a rater can read: indoors under a ceiling, Spot not in the middle of a turn, and the front
    camera looking down free floor rather than into a wall."""
    pts = dense(dec.pre, 0.25)
    if not all(G.indoor(dec.scene_id, q) for q in pts) or not G.has_ceiling(sim, dec.p):
        return False
    if G.sees_sky(sim, G.front_camera(dec.p, dec.yaw, sim), dec.yaw):
        return False
    s0 = length(dec.pre)
    line = np.vstack([dec.pre, dec.post[1:]])
    h_in, h_out = heading_at(line, max(s0 - 0.8, 0.3), 0.3), heading_at(line, s0 + 0.8, 0.3)
    if np.degrees(np.arccos(np.clip(h_in @ h_out, -1, 1))) > MAX_TURN_AT_DECISION_DEG:
        return False
    free, _ = G.ray(sim, G.front_camera(dec.p, dec.yaw, sim), v3(dec.heading), FRONT_FREE_M)
    if np.isfinite(free):
        return False
    # Spot's warm-up run-up behind the start of the line (nuance.spot_walk) must be open floor too, or the chase
    # camera starts behind a door or inside a wall
    h0 = heading_at(dec.pre, 0.2, 0.2)
    back = [dec.pre[0] - t * v3(h0) for t in np.arange(0.0, 0.3 + 0.75 * dec.v0 + 0.5, 0.1)]
    return all(sim.pathfinder.is_navigable(q.astype(np.float32)) for q in back)


def _corner_decision(sim, scene_id, rng, all_objs):
    """Spot walking roughly toward a wall-to-wall corner 2.5-5 m ahead that it can see: random lines almost never
    face one."""
    pf = sim.pathfinder
    corners = [c for c, _ in _corners(sim, scene_id)]
    for c in rng.sample(corners, min(len(corners), 5)):
        for _ in range(20):
            p = np.array(pf.get_random_navigable_point_near(c.astype(np.float32), 5.0, 50), dtype=float)
            v = (c - p)[[0, 2]]
            if not np.isfinite(p).all() or abs(p[1] - c[1]) > 0.2 or not 2.5 <= np.linalg.norm(v) <= 5.0:
                continue
            t = np.arctan2(v[1], v[0]) + np.radians(rng.uniform(-20, 20))
            h3 = v3([np.cos(t), np.sin(t)])
            v0 = rng.uniform(*CONTEXT_SPEED_RANGE)
            start = p - v0 * T_DECISION_S * h3
            line = np.linspace(start, p, 10)
            run_up = [start - k * h3 for k in np.arange(0.1, 0.3 + 0.75 * v0 + 0.5, 0.1)]
            if not all(pf.is_navigable(q.astype(np.float32)) for q in [*line, *run_up]) or not body_clear(sim, line):
                continue
            post = shortest(sim, p, c)
            if post is None or not G.sees_point(sim, G.front_camera(p, float(np.arctan2(-h3[0], -h3[2])), sim),
                                                c + np.r_[0, 0.3, 0]):
                continue
            return _dec_from_path(scene_id, np.vstack([line[:-1], post]), v0, all_objs)
    return None


def sample_decision(sim, scene_id, rng, spec, all_objs):
    family = spec.family
    if family == "place" and spec.metric == "corner" and rng.random() < 0.8:
        dec = _corner_decision(sim, scene_id, rng, all_objs)
    elif _aims(spec) and rng.random() < 0.8:
        dec = _aimed_decision(sim, scene_id, rng, spec, all_objs)
    else:
        if family == "speed":
            v0 = rng.uniform(*(spec.v0 or SPEED_V0_BY_KIND.get(tuple(spec.metric.split("/")), SPEED_V0_RANGE)))
            path = sample_path(sim, v0 * T_DECISION_S + SPEED_POST_M, rng)
            if path is None or not body_clear(sim, path):
                return None
        else:
            v0 = rng.uniform(*(BACK_SPEED_RANGE if family == "back" else CONTEXT_SPEED_RANGE))
            path = sample_path(sim, v0 * T_DECISION_S + CONTEXT_POST_M, rng)
            if path is None:
                return None
        dec = _dec_from_path(scene_id, path, v0, all_objs)
    if dec is None or not _good_spot(sim, dec):
        return None
    if family == "speed":
        # the speed change plays out on a straight stretch, not in a turn or a hook at the end
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dec.post, axis=0), axis=1))])
        if (max_turn_deg(dec.post[:max(2, int(np.searchsorted(cum, 3.0)) + 1)]) > 25 or max_turn_deg(dec.post) > 45
                or not valid(sim, dec, dec.post)):
            return None
    return dec


def _with_person(sim, dec: Decision, rng, n: int = 1):
    """Stand n people on open floor about 2.5 m ahead on the line, with room to pass them on at least one side;
    re-route the line around them. (objs, placements, new dec) or None."""
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dec.post, axis=0), axis=1))])
    if length(dec.post) < 4.0 or max_turn_deg(dec.post[:max(2, int(np.searchsorted(cum, 3.5)) + 1)]) > 30:
        return None
    s = rng.uniform(2.0, 3.0)
    left = G.left_of(heading_at(dec.post, s))
    base = at(dec.post, s)
    if max(_lateral_room(sim, base, left)) < 1.3:
        return None
    spots = [base + rng.uniform(-0.3, 0.3) * v3(left)]
    if n == 2:
        spots.append(spots[0] + rng.choice([-1, 1]) * rng.uniform(0.7, 0.9) * v3(left) + rng.uniform(-0.3, 0.3)
                     * v3(heading_at(dec.post, s)))
    for q in spots:
        # on open floor: not inside or against furniture, and on the navmesh
        if not sim.pathfinder.is_navigable(q.astype(np.float32)) or any(o.dist_xz(q[None])[0] < 0.45 for o in dec.objs):
            return None
    places = [{"model": rng.choice(G.PEOPLE), "at": [float(x) for x in q],
               "yaw": float(dec.yaw + rng.uniform(-0.6, 0.6))} for q in spots]
    objs = [G.add_person(sim, pl["model"], np.array(pl["at"]), pl["yaw"]) for pl in places]
    G.recompute_navmesh(sim)
    post = shortest(sim, dec.p, dec.post[-1])
    if post is None or not sim.pathfinder.is_navigable(dec.p.astype(np.float32)):
        _drop_person(sim, objs)
        return None
    all_objs = G.scene_objects(sim)
    dec = Decision(scene_id=dec.scene_id, pre=dec.pre, post=post, yaw=dec.yaw, v0=dec.v0,
                   objs=G.floor_objects(all_objs, G.floor_y(dec.p)), all_objs=all_objs)
    return objs, places, dec


def _drop_person(sim, objs):
    for obj in objs:
        G.remove_object(sim, obj)
    G.recompute_navmesh(sim)


def _arr(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def plan_scene(scene_id: str, instructions: list, seed: int, out: Path) -> None:
    from nuance.scenes import make_sim
    sim = make_sim(scene_id, seed=seed)
    t0 = time.time()
    rows, found = [], Counter()
    try:
        all_objs = G.scene_objects(sim)
        for text in instructions:
            spec = SPECS[text]
            if text in AROUND:
                spec = replace(spec, referent="thing_hit")  # aim the line into something (see candidates.ground)
            rng = random.Random(f"{seed}/{scene_id}/{text}")
            person = spec.referent in ("person", "people")
            tries = PERSON_TRIES if person else PLACE_TRIES if spec.family == "place" else TRIES
            for _ in range(tries):
                if found[text] >= OPTIONS_PER_SCENE:
                    break
                dec = sample_decision(sim, scene_id, rng, spec, all_objs)
                if dec is None:
                    continue
                objs = place = None
                if person:
                    got = _with_person(sim, dec, rng, 2 if spec.referent == "people" else 1)
                    if got is None:
                        continue
                    objs, place, dec = got
                try:
                    res = ground(sim, dec, text, rng)
                finally:
                    if objs is not None:
                        _drop_person(sim, objs)
                if res is None:
                    continue
                ref, cands = res
                found[text] += 1
                rows.append({"scene_id": scene_id, "instruction": text, "family": spec.family, "referent": ref,
                             "v0": dec.v0, "yaw": dec.yaw, "pre": _arr(dec.pre), "base": _arr(full_base(dec)),
                             "person": place,
                             "candidates": [{"name": c.name, "path": _arr(c.path), "v1": float(c.v1),
                                             "metric": float(c.metric)} for c in cands]})
            print(f"plan {scene_id} '{text}': {found[text]} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        sim.close()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{scene_id}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def full_base(dec: Decision) -> np.ndarray:
    return np.vstack([dec.pre, dec.post[1:]])


# cannot tell the candidates apart (every candidate avoids the obstacle or goes around the table, which is off the line; both sides are "the side") / needs a gesture
NOT_ALTS = {"keep to the side", "go over there", "avoid the obstacle", "go around the table"}
SKIP = {"go over there"}  # "there" needs a pointing gesture the clips cannot show


# the opposite request, which one of the candidates answers (the generic choice drops these as same-size synonyms)
OPPOSITE = {"stay to the left": ["stay to the right"], "stay to the right": ["stay to the left"],
            "keep to the side": ["stay to the left", "stay to the right"],
            "take the long way around": ["take the shortest way"], "take the shortest way": ["take the long way around"]}


def pick_alts(text: str, rng: random.Random, names=(), v0=None) -> list:
    """Up to two other instructions to rate the same clips under. Speed: phrases that one of the candidates (names)
    answers, one in another direction and one in the same direction at another size. Others: instructions about
    the same referent, preferring a different size, so the alternatives are not synonyms."""
    if SPECS[text].family == "speed":
        mine = SPEED_KIND[text]
        # a phrase that only makes sense at some start speeds ("you're going too fast") is used only there
        fits = [p for p, k in SPEED_KIND.items() if p != text and k != mine and SPEED_TARGET[k] in names
                and (v0 is None or not SPECS[p].v0 or SPECS[p].v0[0] <= v0 <= SPECS[p].v0[1])]
        out = []
        for pool in ([p for p in fits if SPEED_KIND[p][0] != mine[0]], [p for p in fits if SPEED_KIND[p][0] == mine[0]]):
            if pool:
                out.append(rng.choice(pool))
        return out
    if text in OPPOSITE:
        return OPPOSITE[text]
    alts = [t for t in same_kind(text) if t not in NOT_ALTS]
    rng.shuffle(alts)
    other = [t for t in alts if SPECS[t].metric != SPECS[text].metric]
    if SPECS[text].family == "place":
        other = []
    # one of the same size is a synonym and tells raters nothing new, so alternatives are contrasts only
    out = other[:2]
    return (out + [t for t in FALLBACK.get(text, []) if t not in out and t not in NOT_ALTS
                   and (SPECS[t].family, SPECS[t].metric) != (SPECS[text].family, SPECS[text].metric)])[:2]


DEDUP_M, DEDUP_DEG = 1.0, 45.0  # decision points this close, facing this similarly, in one scene are the same one


def _near(a: dict, b: dict, m: float = 2.0) -> bool:
    return np.hypot(*(np.array(a["pre"][-1]) - np.array(b["pre"][-1]))[[0, 2]]) < m


def _same_spot(a: dict, b: dict) -> bool:
    pa, pb = np.array(a["pre"][-1]), np.array(b["pre"][-1])
    turn = abs((a["yaw"] - b["yaw"] + np.pi) % (2 * np.pi) - np.pi)
    return np.hypot(*(pa - pb)[[0, 2]]) < DEDUP_M and np.degrees(turn) < DEDUP_DEG


def _same_target(a: dict, b: dict) -> bool:
    """Both go to the same marked point (a corner, a doorway), even from different spots."""
    ta, tb = a["referent"].get("door"), b["referent"].get("door")
    return ta is not None and tb is not None and np.hypot(ta[0] - tb[0], ta[2] - tb[2]) < DEDUP_M


def assign(scenes: list, instructions: list, per_instruction: int, seed: int, out: Path) -> list:
    options = defaultdict(list)
    for s in scenes:
        p = out / "plans" / f"{s}.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                options[r["instruction"]].append(r)
    rng = random.Random(seed)
    load = Counter()
    chosen, missing = [], {}
    # spots already in another batch's clips (NUANCE_AVOID, a clips.jsonl) count as taken
    avoid = [{"scene_id": r["scene_id"], "pre": [r["decision_position"]], "yaw": r["decision_yaw"],
              "family": r["action_type"]} for r in map(json.loads, open(os.environ["NUANCE_AVOID"]))
             ] if os.environ.get("NUANCE_AVOID") else []
    # instructions with the fewest options pick first, so the common ones do not take their only spots
    for text in sorted(instructions, key=lambda t: len(options.get(t, []))):
        # a spot another batch already has for this family adds nothing
        opts = [o for o in options.get(text, []) if not any(
            c["scene_id"] == o["scene_id"] and c["family"] == o["family"] and _same_spot(o, c) for c in avoid)]
        by_scene = defaultdict(list)
        for o in opts:
            by_scene[o["scene_id"]].append(o)
        picked = []
        used = Counter()
        while len(picked) < per_instruction and any(by_scene.values()):
            # a spot no chosen decision point already uses, then least-used scene for this instruction, then
            # least-loaded scene overall
            def key(o):
                s = o["scene_id"]
                return (any(c["scene_id"] == s and (_same_spot(o, c) or c["family"] == o["family"] and _near(o, c))
                            for c in chosen + picked + avoid), used[s], load[s], rng.random())
            o = min((o for v in by_scene.values() for o in v), key=key)
            s = o["scene_id"]
            by_scene[s].remove(o)
            if any(c["scene_id"] == s and (_same_spot(o, c) or _same_target(o, c)) for c in picked):
                continue  # another decision point for this instruction at the same spot or the same target
            o = {**o, "backups": by_scene[s][:]}
            used[s] += 1
            load[s] += 1
            picked.append(o)
        if len(picked) < per_instruction:
            missing[text] = per_instruction - len(picked)
        chosen += picked
    order = {t: i for i, t in enumerate(instructions)}
    chosen.sort(key=lambda o: order[o["instruction"]])
    for i, o in enumerate(chosen):
        o["decision_index"] = i
        o["alt_instructions"] = pick_alts(o["instruction"], random.Random(f"{seed}/alts/{i}"),
                                          [c["name"] for c in o["candidates"]], o["v0"])
    with open(out / "assigned.json", "w") as f:
        json.dump({"decisions": chosen, "missing": missing}, f)
    print(f"assigned {len(chosen)} decisions; missing {missing}", flush=True)
    return chosen


def _walk(sim, spot, path: np.ndarray, jobs: list, workdir: Path):
    from nuance.render import floor_grid
    spec = {"policy_dir": cfg.SPOT_POLICY_DIR, "path": np.asarray(path).tolist(), "floor": floor_grid(sim, spot, path),
            "jobs": jobs}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "jobs.json").write_text(json.dumps(spec))
    return [cfg.LOCO_PYTHON, "-m", "nuance.spot_walk", str(workdir / "jobs.json"), str(workdir)]


def walk_candidates(sim, spot, d: dict, tmp: Path):
    """({name: Motion}, None) for the decision's candidates, or (None, why) if Spot fell or walked badly on any."""
    from nuance.render import load_motion
    cands = d["candidates"]
    cmds = []
    if d["family"] == "speed":
        cmds.append(_walk(sim, spot, np.array(cands[0]["path"]), [{"name": c["name"], "v0": d["v0"], "v1": c["v1"]}
                                                                   for c in cands], tmp / "speed"))
    else:
        for c in cands:
            cmds.append(_walk(sim, spot, np.array(c["path"]), [{"name": c["name"], "v0": d["v0"], "v1": c["v1"]}],
                              tmp / c["name"]))
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    with ThreadPoolExecutor(len(cmds)) as pool:
        list(pool.map(lambda c: subprocess.run(c, check=True, stdout=subprocess.DEVNULL, env=env), cmds))
    motions = {}
    for c in cands:
        f = (tmp / "speed" if d["family"] == "speed" else tmp / c["name"]) / f"{c['name']}.npz"
        if bool(np.load(f)["fell"]):
            return None, f"fell on '{c['name']}'"
        motions[c["name"]] = load_motion(f)
        motions[c["name"]].v_post = float(np.load(f)["v_post"])
        why = _motion_problem(sim, np.array(c["path"]), motions[c["name"]], d["v0"], c["v1"])
        if why:
            return None, f"'{c['name']}' {why}"
    return motions, None


def _motion_problem(sim, path: np.ndarray, m, v0: float, v1: float):
    """Why Spot's simulated walk does not show the candidate (it only follows the line; nothing stops it walking
    into furniture or walls if it drifts), or None."""
    from nuance.config import SPOT_BASE_HEIGHT
    ground = m.positions - np.array([0.0, SPOT_BASE_HEIGHT, 0.0])
    end = path[-1]
    if np.hypot(*(ground[-1] - end)[[0, 2]]) > END_TOL_M:
        return "did not reach the end of its line"
    for q in ground[::5]:
        sn = np.array(sim.pathfinder.snap_point(np.r_[q[0], end[1], q[2]].astype(np.float32)), dtype=float)
        if not np.isfinite(sn).all() or np.hypot(*(sn - q)[[0, 2]]) > OFF_NAVMESH_M:
            return "left the walkable floor"
    if (m.rotations[:, 1, 1] < UPRIGHT_MIN).any():
        return "tipped or crouched over"  # base up axis well off vertical
    s0 = v0 * T_DECISION_S
    expect = T_DECISION_S + max(length(path) - s0, 0.0) / max(min(v1, WALK_MAX_SPEED), 1e-3) + HOLD_S
    if m.times[-1] > expect + STUCK_S:
        return f"took {m.times[-1]:.1f} s for an expected {expect:.1f} s"
    return None


def render_scene(scene_id: str, batch_id: str, seed: int, out: Path) -> None:
    from nuance.render import load_spot, render_clip, render_front, topdown_frame
    from nuance.scenes import make_sim
    decisions = [d for d in json.load(open(out / "assigned.json"))["decisions"] if d["scene_id"] == scene_id]
    if not decisions:
        return
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    sim = make_sim(scene_id, seed=seed)
    spot = load_spot(sim)
    t0 = time.time()
    records, failed = [], []
    try:
        for d0 in decisions:
            i = d0["decision_index"]
            decision_id = f"d{i:03d}"
            done = False
            for d in [d0] + d0.get("backups", []):
                with tempfile.TemporaryDirectory() as tmp:
                    motions, why = walk_candidates(sim, spot, d, Path(tmp))
                if motions is None:
                    print(f"rejected {decision_id} {scene_id} '{d['instruction']}': {why}; trying a backup", flush=True)
                    continue
                people = [G.add_person(sim, p["model"], np.array(p["at"]), p["yaw"]) for p in d["person"] or []]
                try:
                    pre = np.array(d["pre"])
                    front = f"frames/{decision_id}.png"
                    render_front(sim, spot, pre[-1], d["yaw"], out / front)
                    # the top-down covers every candidate's walk and the referent
                    ref_pts = [np.array(q) for o in d["referent"].get("objects", []) for q in
                               _box_corners(o["oid"], sim)] + [np.array(q) for p in d["person"] or [] for q in [p["at"]]]
                    frame = topdown_frame(SimpleNamespace(positions=np.concatenate(
                        [m.positions for m in motions.values()] + ([np.array(ref_pts)] if ref_pts else []))))
                    context = measure_context(sim, np.array(d["base"]), length(pre))
                    boxes = []
                    if d["referent"].get("objects"):
                        by_id = {o.oid: o for o in G.scene_objects(sim)}
                        want = {o["oid"] for o in d["referent"]["objects"]}
                        boxes = [[_arr(by_id[k].lo), _arr(by_id[k].hi)] for k in want if k in by_id and not d["person"]]
                    for person in people:
                        lo, hi = G._world_aabb(person)
                        boxes.append([_arr(lo), _arr(hi)])
                    (out / "frames" / f"{decision_id}.json").write_text(json.dumps(
                        {"center": _arr(frame[0]), "extent": float(frame[1]), "boxes": boxes,
                         "door": d["referent"].get("door"), "wall": d["referent"].get("wall")}))
                    for c in d["candidates"]:
                        clip_id = make_clip_id(batch_id, i, c["name"])
                        video = f"videos/{clip_id}.mp4"
                        m = motions[c["name"]]
                        render_clip(sim, spot, m, np.array(c["path"]), length(pre), out / video, frame=frame)
                        records.append(ClipRecord(
                            clip_id=clip_id, batch_id=batch_id, decision_id=decision_id, action_type=d["family"],
                            scene_id=scene_id, path=c["path"], decision_position=list(pre[-1]), decision_yaw=d["yaw"],
                            context=context, start_speed=d["v0"], candidate_speed=c["v1"], candidate_rank=c["name"],
                            instruction=say(d["instruction"], d["referent"]),
                            alt_instructions=[say(a, d["referent"]) for a in d0["alt_instructions"]], video=video,
                            front_frame=front, fps=FPS, duration_s=float(m.times[-1]), t_decision_s=T_DECISION_S,
                            referent=d["referent"], candidate_metric=c["metric"], person=d["person"],
                            achieved_speed=m.v_post))
                finally:
                    for person in people:
                        G.remove_object(sim, person)
                done = True
                break
            if not done:
                failed.append(d0["instruction"])
            print(f"rendered {decision_id} {scene_id} '{d0['instruction']}' ({time.time() - t0:.0f} s)", flush=True)
    finally:
        sim.close()
    (out / "parts").mkdir(exist_ok=True)
    write_jsonl(out / "parts" / f"{scene_id}.jsonl", records)
    (out / "parts" / f"{scene_id}.failed.json").write_text(json.dumps(failed))


def _box_corners(oid: int, sim) -> list:
    o = next((x for x in G.scene_objects(sim) if x.oid == oid), None)
    return [] if o is None else [[o.lo[0], o.lo[1], o.lo[2]], [o.hi[0], o.lo[1], o.hi[2]]]


def merge(scenes: list, batch_id: str, seed: int, out: Path) -> None:
    from nuance.records import read_jsonl
    records, failed = [], []
    for s in scenes:
        if (out / "parts" / f"{s}.jsonl").exists():
            records += read_jsonl(out / "parts" / f"{s}.jsonl")
            failed += json.loads((out / "parts" / f"{s}.failed.json").read_text())
    records.sort(key=lambda r: r.decision_id)  # stable: candidates keep their low-to-high order
    write_jsonl(out / "clips.jsonl", records)
    assigned = json.load(open(out / "assigned.json"))
    manifest = {
        "batch_id": batch_id, "seed": seed, "scenes": scenes, "created_at": datetime.now(timezone.utc).isoformat(),
        "n_decisions": len({r.decision_id for r in records}), "n_clips": len(records),
        "decisions_per_instruction": dict(Counter(i for _, i in {(r.decision_id, r.instruction) for r in records})),
        "unplanned": assigned["missing"], "fell": failed,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(cfg).items() if k.isupper()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"merged {len(records)} clips; unplanned {assigned['missing']}; fell {failed}", flush=True)


def sheet(out: Path) -> None:
    """review/dNNN.png per decision: the front frame, then per candidate a chase frame half-way through the part
    after the decision point and the final top-down frame, with the referent outlined in cyan."""
    import imageio
    from PIL import Image, ImageDraw
    from nuance.config import TOPDOWN_RES
    from nuance.records import read_jsonl
    from nuance.render import world_to_pixel
    by = defaultdict(list)
    for r in read_jsonl(out / "clips.jsonl"):
        by[r.decision_id].append(r)
    (out / "review").mkdir(exist_ok=True)
    H, LABEL = 320, 22
    for did, rs in sorted(by.items()):
        r0 = rs[0]
        meta_f = out / "frames" / f"{did}.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else None
        front = Image.open(out / r0.front_frame).convert("RGB")
        rows = [("front camera at the decision point", [front.resize((round(front.width * H / front.height), H))])]
        for r in rs:
            with imageio.get_reader(str(out / r.video), "ffmpeg") as rd:
                n = rd.count_frames()
                k = int((r.t_decision_s + (n / r.fps - r.t_decision_s) / 2) * r.fps)
                mid, last = rd.get_data(min(n - 1, k)), rd.get_data(n - 1)
                # filmstrip: 8 frames spread over the clip, 4 x 2, each labelled with its time
                strip = Image.new("RGB", (4 * 600, 2 * 256))
                sd = ImageDraw.Draw(strip)
                for j, f in enumerate(np.linspace(0, n - 1, 8).astype(int)):
                    im = Image.fromarray(rd.get_data(int(f))).resize((600, 256))
                    strip.paste(im, ((j % 4) * 600, (j // 4) * 256))
                    sd.rectangle([(j % 4) * 600, (j // 4) * 256, (j % 4) * 600 + 70, (j // 4) * 256 + 14], fill=(0, 0, 0))
                    sd.text(((j % 4) * 600 + 3, (j // 4) * 256 + 1), f"t={f / r.fps:.1f}s", fill=(255, 255, 255))
                strip.save(out / "review" / f"{did}_{r.candidate_rank}.jpg", quality=85)
            w = mid.shape[1]
            chase = Image.fromarray(mid[:, : w - mid.shape[0]])
            top = Image.fromarray(last[:, w - last.shape[0]:])
            if meta:
                dr = ImageDraw.Draw(top)
                c, e = np.array(meta["center"]), meta["extent"]
                sc = top.width / TOPDOWN_RES
                for lo, hi in meta["boxes"]:
                    (u0, v0), (u1, v1) = world_to_pixel(np.array([lo, hi]), c, e) * sc
                    dr.rectangle([min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)], outline=(0, 255, 255), width=4)
                if meta.get("wall"):
                    (u0, v0), (u1, v1) = world_to_pixel(np.array(meta["wall"]), c, e) * sc
                    dr.line([u0, v0, u1, v1], fill=(0, 255, 255), width=6)
                if meta.get("door"):
                    u, v = world_to_pixel(np.array([meta["door"]]), c, e)[0] * sc
                    dr.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 255), width=4)
            label = (f"candidate '{r.candidate_rank}'   metric={r.candidate_metric:.2f}   v1={r.candidate_speed:.2f} m/s"
                     f"   achieved={r.achieved_speed or 0:.2f} m/s   duration={r.duration_s:.1f} s   clip {r.clip_id}")
            rows.append((label, [chase.resize((round(chase.width * H / chase.height), H)), top.resize((H, H))]))
        width = max(sum(t.width for t in tiles) for _, tiles in rows)
        head = [f"{did}   scene {r0.scene_id}   family {r0.action_type}   start speed {r0.start_speed:.2f} m/s",
                f'INSTRUCTION: "{r0.instruction}"      also rated under: {r0.alt_instructions}',
                f"referent: {json.dumps(r0.referent)}" + ("   (person placed for this decision)" if r0.person else ""),
                "per candidate: chase view half-way after the decision point | final top-down (white = before the "
                "decision point, magenta = after; cyan box = referent)"]
        top_h = 16 * len(head) + 8
        img = Image.new("RGB", (width, top_h + (H + LABEL) * len(rows)), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.multiline_text((6, 4), "\n".join(head), fill=(0, 0, 0), spacing=4)
        y = top_h
        for label, tiles in rows:
            dr.text((6, y + 5), label, fill=(0, 0, 0))
            x = 0
            for t in tiles:
                img.paste(t, (x, y + LABEL))
                x += t.width
            y += H + LABEL
        img.save(out / "review" / f"{did}.png")
    print(f"sheets for {len(by)} decisions in {out / 'review'}", flush=True)


def _instructions(arg: str) -> list:
    if arg == "all":
        return [t for t in SPECS if t not in SKIP]
    if arg == "context":
        return [t for t, s in SPECS.items() if s.family != "speed" and t not in SKIP]
    if arg == "speed":
        return [t for t, s in SPECS.items() if s.family == "speed"]
    return [t.strip() for t in open(arg) if t.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "plan", "assign", "render", "merge", "sheet"])
    ap.add_argument("--scenes", default="configs/pilot_scenes.txt")
    ap.add_argument("--scene")
    ap.add_argument("--instructions", default="all", help="all, context, speed, or a file with one per line")
    ap.add_argument("--per-instruction", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    scenes = [s.strip() for s in open(a.scenes) if s.strip()]
    instructions = _instructions(a.instructions)
    if a.step == "plan":
        plan_scene(a.scene, instructions, a.seed, out / "plans")
    elif a.step == "assign":
        assign(scenes, instructions, a.per_instruction, a.seed, out)
    elif a.step == "render":
        render_scene(a.scene, a.batch_id, a.seed, out)
    elif a.step == "merge":
        merge(scenes, a.batch_id, a.seed, out)
    elif a.step == "sheet":
        sheet(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

        def each(step, extra):
            def one(s):
                log = open(out / f"{step}-{s}.log", "w")
                return subprocess.run([sys.executable, "-m", "nuance.generate_instructions", step, "--scene", s,
                                       "--out", str(out), "--seed", str(a.seed), "--scenes", a.scenes,
                                       "--instructions", a.instructions, *extra], stdout=log, stderr=subprocess.STDOUT).returncode
            with ThreadPoolExecutor(a.workers) as pool:
                codes = dict(zip(scenes, pool.map(one, scenes)))
            bad = {s: c for s, c in codes.items() if c}
            if bad:
                print(f"{step} failed for {bad}; see {out}/{step}-<scene>.log", flush=True)
        each("plan", [])
        assign(scenes, instructions, a.per_instruction, a.seed, out)
        each("render", ["--batch-id", a.batch_id])
        merge(scenes, a.batch_id, a.seed, out)
        sheet(out)


if __name__ == "__main__":
    main()
