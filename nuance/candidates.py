"""Three candidate continuations of Spot's line after a decision point, varied along what an instruction is about.

Every generator takes a Decision and returns (referent, [Candidate] * 3) or None when the instruction does not make
sense at that decision point (no such referent, it is ambiguous or out of view, or the candidates cannot differ enough).
A candidate's path is the full line (the shared walk-in up to the decision point, then its own continuation).
"""
import random
from dataclasses import dataclass, field, replace

import habitat_sim
import numpy as np

from nuance import grounding as G
from nuance.config import MIN_SPEED, SPOT_HALF_LENGTH
from nuance.context import _cum, _pos
from nuance.instructions import FURNITURE, SPECS
from nuance.scenes import body_clear

AHEAD = (0.8, 4.5)  # arc window after the decision point in which a referent counts as "on the way"
NEAR_PATH_M = 1.2  # a referent is on the way if the line passes this close to it
IN_WAY_M = 0.5  # "the obstacle" is in the way: within this of the straight line ahead of Spot
HIT_M = 0.4  # "go around the table" / "avoid the obstacle": the straight line would walk Spot into it
AROUND = ("go around the table", "avoid the obstacle")
UNIQUE_M = 2.5  # other objects of the referent's category this close to the line must be clearly farther
MAX_TURN_DEG = 120.0
MAX_POST_M = 9.0
CLEAR_STEP_M = 0.25  # neighbouring candidates differ by at least this much clearance
CLEAR_SPAN_M = 0.5
CLEAR_MIN_M = 0.05  # closer than this to the referent is a collision
WIDE_LARGE_M = 0.9  # "wide berth", "way back", "plenty of room": the widest candidate keeps at least this much
OTHERS_MIN_M = 0.35  # a candidate may not trade the referent for brushing something else
GOAL_CLEAR_M = 0.55  # place goals keep this much room around them unless the instruction asks for a tight spot
SAME_KIND_M = 3.5  # "the table": no other table within this of it
FIRST_TURN_DEG = 60.0  # a detour eases out of the line instead of turning sharply away from it
OFFSETS_M = (-0.4, -0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.3, 1.6)
WAYPOINT_SPREAD_M = 0.9
SNAP_TOL_M = 0.08
SELF_GAP_M = 0.35
FLOOR_STEP_M = 0.06  # a route whose floor rises more than this above the decision point's floor climbs onto an object
GOAL_APART_M = 1.5  # place candidates end at least this far apart
PLACE_MAX_M = 6.5

# Speed candidates, as multiples of the start speed, per change class; each speed phrase gets the class it asks for,
# a class that differs in size, and one that differs in direction (or "same").
SPEED_RATIO = {"same": (0.97, 1.03), "small_up": (1.18, 1.3), "mid_up": (1.4, 1.7), "big_up": (1.9, 2.6),
               "small_down": (0.78, 0.88), "mid_down": (0.55, 0.7), "big_down": (0.25, 0.45)}
# Per phrase kind, the candidate sets to choose from; each holds the class the phrase asks for, so raters see it
# against both a change of size and a change of direction across decision points.
SPEED_SETS = {("faster", "small"): (("same", "small_up", "mid_up"), ("mid_down", "same", "small_up")),
              ("faster", "large"): (("same", "mid_up", "big_up"), ("mid_down", "same", "big_up")),
              ("faster", "unspecified"): (("mid_down", "same", "mid_up"), ("small_down", "same", "mid_up")),
              ("faster", "max"): (("same", "mid_up", "max"),),
              ("slower", "small"): (("big_down", "small_down", "same"), ("small_down", "same", "mid_up")),
              ("slower", "large"): (("big_down", "small_down", "same"), ("big_down", "same", "mid_up")),
              ("slower", "unspecified"): (("mid_down", "same", "mid_up"), ("mid_down", "same", "small_up")),
              ("hold", "none"): (("mid_down", "same", "mid_up"), ("mid_down", "same", "small_up"),
                                 ("small_down", "same", "mid_up"))}
SPEED_TARGET = {("faster", "small"): "small_up", ("faster", "large"): "big_up", ("faster", "unspecified"): "mid_up",
                ("faster", "max"): "max", ("slower", "small"): "small_down", ("slower", "large"): "big_down",
                ("slower", "unspecified"): "mid_down", ("hold", "none"): "same"}
MIN_SPEED_GAP = 0.12  # m/s between any two candidates, so they look different
# Start speeds with room for the classes a speed kind needs below Spot's top walking speed and above its slowest.
SPEED_V0_BY_KIND = {("faster", "large"): (0.45, 0.6), ("faster", "max"): (0.45, 0.65), ("slower", "large"): (0.7, 1.0)}
WALK_MAX_SPEED = 1.2  # fastest target the walking policy reaches (nuance.spot_walk tops out near 1.2-1.3 m/s)


@dataclass
class Decision:
    scene_id: str
    pre: np.ndarray  # walk-in, ending at the decision point
    post: np.ndarray  # the sampled line's continuation, starting at the decision point
    yaw: float
    v0: float
    objs: list = field(default_factory=list)  # floor objects
    all_objs: list = field(default_factory=list)

    @property
    def p(self) -> np.ndarray:
        return self.pre[-1]

    @property
    def heading(self) -> np.ndarray:
        return np.array([-np.sin(self.yaw), -np.cos(self.yaw)])


@dataclass
class Candidate:
    name: str
    path: np.ndarray
    v1: float
    metric: float


# ---- geometry helpers -------------------------------------------------------------------------------------------

def dense(path: np.ndarray, step: float = 0.05) -> np.ndarray:
    cum = _cum(path)
    return np.array([_pos(path, cum, s) for s in np.arange(0.0, cum[-1] + 1e-9, step)])


def length(path) -> float:
    return float(_cum(np.asarray(path))[-1])


def at(path, s) -> np.ndarray:
    path = np.asarray(path)
    return _pos(path, _cum(path), s)


def heading_at(path, s, w: float = 0.3) -> np.ndarray:
    d = at(path, s + w) - at(path, s - w)
    d = np.array([d[0], d[2]])
    return d / max(np.linalg.norm(d), 1e-9)


def v3(xz) -> np.ndarray:
    return np.array([xz[0], 0.0, xz[1]])


def snap(sim, q):
    s = np.array(sim.pathfinder.snap_point(np.asarray(q, dtype=np.float32)), dtype=float)
    if not np.isfinite(s).all() or np.hypot(*(s - q)[[0, 2]]) > SNAP_TOL_M or abs(s[1] - q[1]) > 0.3:
        return None
    return s


def shortest(sim, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    if not sim.pathfinder.find_path(sp) or len(sp.points) < 2:
        return None
    return np.array(sp.points, dtype=float)


def chain(sim, pts):
    out = [np.asarray(pts[0], dtype=float)[None]]
    for a, b in zip(pts[:-1], pts[1:]):
        leg = shortest(sim, a, b)
        if leg is None:
            return None
        out.append(leg[1:])
    path = np.concatenate(out)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-3]
    return path[keep]


def max_turn_deg(path, w: float = 0.3) -> float:
    """Sharpest heading change over any 2w of the line (vertex angles miss turns spread over short segments)."""
    path = np.asarray(path)
    L = length(path)
    if L < 2 * w + 0.1:
        return 0.0
    hs = [heading_at(path, s, 0.1) for s in np.arange(0.1, L - 0.1, 0.1)]
    k = max(1, int(round(2 * w / 0.1)))
    return max((float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))) for a, b in zip(hs[:-k], hs[k:])), default=0.0)


def self_approach(path, min_arc: float = 1.5) -> float:
    """Closest the line comes to a part of itself at least min_arc earlier along it; a line that folds onto itself
    leaves Spot's path follower (nuance.spot_walk) unsure which part it is on."""
    pts = dense(path, 0.1)[:, [0, 2]]
    k = int(round(min_arc / 0.1))
    if len(pts) <= k:
        return np.inf
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    i, j = np.triu_indices(len(pts), k)
    return float(d[i, j].min())


def between_free(sim, a, b, n: int = 16) -> float:
    """Fraction of matched point pairs of lines a and b (at equal fractions of their length) joined by free floor:
    1 when they run through the same free space, low when an obstacle lies between them."""
    La, Lb = length(a), length(b)
    ok = 0
    for f in np.linspace(0.1, 0.9, n):
        p, q = at(a, f * La), at(b, f * Lb)
        ok += all(sim.pathfinder.is_navigable((p + t * (q - p)).astype(np.float32)) for t in np.linspace(0, 1, 8))
    return ok / n


def full(dec: Decision, post) -> np.ndarray:
    return np.vstack([dec.pre, np.asarray(post)[1:]])


def valid(sim, dec: Decision, post, reverse: bool = False, max_len: float = MAX_POST_M) -> bool:
    if post is None or len(post) < 2 or not (0.2 <= length(post) <= max_len):
        return False
    if abs(np.asarray(post)[:, 1].max() - dec.p[1]) > 0.3:
        return False
    pts = dense(post, 0.3)
    if not all(G.indoor(dec.scene_id, q) for q in pts):
        return False
    if self_approach(full(dec, post)) < SELF_GAP_M:
        return False
    first = heading_at(post, 0.35, 0.3)
    if not reverse and (dec.heading @ first < np.cos(np.radians(100)) or max_turn_deg(post) > MAX_TURN_DEG):
        return False
    f0 = G.floor_at(sim, dec.p)
    if any(G.floor_at(sim, q) - f0 > FLOOR_STEP_M for q in pts):
        return False
    if not all(G.has_ceiling(sim, q) for q in pts[::3]):
        return False
    return body_clear(sim, full(dec, post))


def straight(dec: Decision, d: float) -> np.ndarray:
    p = dec.p
    h = dec.heading
    return np.array([p, p + d * v3(h)])


def pick3(items, key, step=CLEAR_STEP_M, span=CLEAR_SPAN_M, apart=None, pos=None):
    """Low, middle and high items by key, at least `step` apart and spanning at least `span`; if `apart`, their
    positions pos(item) are also at least that far apart. None if impossible."""
    items = sorted(items, key=key)
    if len(items) < 3:
        return None

    def far(x, y):
        return apart is None or np.linalg.norm(np.asarray(pos(x)) - np.asarray(pos(y))) >= apart
    for lo in items[:max(1, len(items) // 3)]:
        his = [x for x in items[::-1] if key(x) - key(lo) >= max(span, 2 * step) and far(lo, x)]
        if not his:
            continue
        hi = his[0]
        mid_target = (key(lo) + key(hi)) / 2
        mids = [x for x in items if key(x) - key(lo) >= step and key(hi) - key(x) >= step and far(x, lo) and far(x, hi)]
        if mids:
            return [lo, min(mids, key=lambda x: abs(key(x) - mid_target)), hi]
    return None


# ---- referents --------------------------------------------------------------------------------------------------

def _on_way(dec: Decision, objs, near=NEAR_PATH_M):
    """(obj, closest-approach arc, distance) for objects the line passes within `near` of, ahead of Spot."""
    pts = dense(dec.post)
    arcs = np.arange(len(pts)) * 0.05
    ok = (arcs >= AHEAD[0]) & (arcs <= AHEAD[1])
    out = []
    for o in objs:
        d = o.dist_xz(pts)
        d = np.where(ok, d, np.inf)
        k = int(np.argmin(d))
        if d[k] <= near:
            out.append((o, float(arcs[k]), float(d[k])))
    return sorted(out, key=lambda t: t[2])


def _ahead_ray(dec: Decision) -> np.ndarray:
    return np.array([dec.p + t * v3(dec.heading) for t in np.arange(AHEAD[0], AHEAD[1], 0.1)])


def _category(o) -> str:
    return o.category or "?"


SEATS = ("chair", "stool", "bench")  # pulled up to a table, they belong to it


@dataclass(eq=False)
class Group(G.Obj):
    """A table and the chairs at it, one thing to keep clear of: its box covers them all."""
    members: tuple = ()


RUGS = ("carpet", "rug")
NAMES = {"carpet": "rug", "chest_of_drawers": "dresser", "potted_plant": "plant", "washer_dryer": "washer",
         "tv": "TV", "table_lamp": "lamp", "floor_lamp": "lamp"}


def noun(o) -> str:
    """What a person calls the object: "the table" for a table and the chairs at it."""
    return NAMES.get(o.category, o.category.replace("_", " "))


def say(text: str, referent: dict) -> str:
    """text as said at this decision point: "the obstacle" becomes what the referent is ("the table", "the wall")."""
    objs = referent.get("objects") or [{}]
    if referent.get("name"):
        # "the chair" / "the table" stand for whatever single thing is there
        for word in ("the obstacle", "the chair", "the table"):
            text = text.replace(word, f"the {referent['name']}")
        return text
    n = "wall" if referent.get("kind") == "wall" else objs[0].get("category") and NAMES.get(
        objs[0]["category"], objs[0]["category"].replace("_", " "))
    return text.replace("the obstacle", f"the {n}") if n else text


def _rugs(dec: Decision) -> list:
    fy = G.floor_y(dec.p)
    return [o for o in dec.all_objs if o.category in RUGS and abs(o.lo[1] - fy) < 0.15
            and (o.hi[0] - o.lo[0]) * (o.hi[2] - o.lo[2]) > 0.5]


def _members(o) -> tuple:
    return getattr(o, "members", ()) or (o,)


def _grouped(pool: list, objs: list) -> list:
    """pool with each free-standing table and the chairs at it merged into a Group: a chair in a set around a table
    is not "the obstacle", the table and its chairs are."""
    out, taken = [], set()
    for t in pool:
        if t.category != "table" or not G.free_standing(t, objs, allowed=SEATS):
            continue
        chairs = [c for c in objs if c.category in SEATS and
                  np.hypot(*np.maximum(np.maximum(t.lo - c.hi, c.lo - t.hi), 0.0)[[0, 2]]) < 0.35]
        if not chairs:
            continue
        m = (t, *chairs)
        out.append(Group(t.oid, "table", np.min([x.lo for x in m], axis=0), np.max([x.hi for x in m], axis=0), m))
        taken |= {id(x) for x in m}
    return out + [o for o in pool if id(o) not in taken]


def _pool(dec: Decision, kind: str) -> list:
    if kind in ("person", "people"):
        return [o for o in dec.objs if o.category == "person"]
    if kind == "obstacle_or_person":
        return _grouped([o for o in dec.objs if o.category], dec.objs)
    if kind == "obstacle":
        return _grouped([o for o in dec.objs if o.category and o.category != "person"], dec.objs)
    if kind == "furniture":
        return _grouped([o for o in dec.objs if o.category in FURNITURE], dec.objs)
    if kind in ("thing", "thing_hit"):
        return _grouped([o for o in dec.objs if o.category and o.category != "person"], dec.objs) + _rugs(dec)
    return [o for o in dec.objs if o.category == kind]


def object_referent(sim, dec: Decision, kind: str):
    """The single object `kind` refers to and its closest-approach arc on the line; None if it is absent, ambiguous,
    out of view, or standing against a wall (then it is not something to walk around)."""
    cam = G.front_camera(dec.p, dec.yaw, sim)
    hits = _on_way(dec, _pool(dec, kind))
    if kind in ("obstacle", "obstacle_or_person", "thing", "thing_hit"):
        # the referent is in the way: on Spot's straight line ahead
        ray = _ahead_ray(dec)
        hits = [h for h in hits if h[0].dist_xz(ray).min() <= (HIT_M if kind == "thing_hit" else IN_WAY_M)]
    if not hits:
        return None
    o, s, d = hits[0]
    if kind in ("obstacle", "obstacle_or_person"):
        # nothing else is nearly as much in the way: on Spot's straight line ahead, about as close
        ray = _ahead_ray(dec)
        mine = o.dist_xz(ray).min()
        if any(x not in _members(o) and x.dist_xz(ray).min() < min(mine + 0.25, IN_WAY_M) for x in dec.objs
               if x.category != "carpet"):
            return None
    elif kind in ("thing", "thing_hit"):
        # named ("the rug", "the table"), so it is the only one called that around: nothing else of that noun near it
        # or nearly as close to the line
        same = [x for x in dec.objs + _rugs(dec) if noun(x) == noun(o) and x not in _members(o)]
        if any(h[2] < d + 0.3 for h in _on_way(dec, same, UNIQUE_M)):
            return None
        if any(o.dist_xz(x.center[None])[0] < SAME_KIND_M for x in same):
            return None
    elif kind not in ("furniture", "people"):
        # "the chair" is the one chair around: no other one near it or nearly as close to the line
        same = _on_way(dec, [x for x in dec.objs if x.category == o.category], UNIQUE_M)
        if any(h[0] is not o and h[2] < d + 0.3 for h in same):
            return None
        if any(x is not o and x.category == o.category and x.dist_xz(o.center[None])[0] < SAME_KIND_M
               for x in dec.objs):
            return None
    # a table (set) is in view when it or one of the chairs at it is: the chairs hide the table's middle
    if not any(G.sees_object(sim, cam, dec.yaw, m) for m in _members(_grouped([o], dec.objs)[0])):
        return None
    if o.category in RUGS:
        return o, s  # a rug lies flat: against a wall or under furniture it is still a rug to keep off
    if o.category != "person" and G.wall_flush(sim, o, G.floor_at(sim, dec.p)):
        return None
    # free-standing: a stool tucked under an island, a chair in a row at a table or a side table against a sofa is
    # part of something bigger
    if not G.free_standing(o, [x for x in dec.objs if x not in _members(o)],
                           allowed=SEATS if o.category == "table" else ()):
        return None
    return o, s


def _away_normal(dec: Decision, s: float, target_xz) -> np.ndarray:
    q = at(dec.post, s)
    left = G.left_of(heading_at(dec.post, s))
    side = (np.asarray(target_xz) - q[[0, 2]]) @ left
    return -left if side > 0 else left


def _offset_routes(sim, dec: Decision, s: float, normal_xz, offsets=OFFSETS_M, spread=WAYPOINT_SPREAD_M):
    """Routes that shift sideways by each offset around arc s of the line, then rejoin it at its end, through the
    same free space as the line."""
    L = length(dec.post)
    # ease out to the full offset over ~1 m before the referent and back over ~1 m after it
    ramp = [(a, f) for a, f in ((s - spread - 0.9, 0.5), (s - spread, 1.0), (s + spread, 1.0), (s + spread + 0.9, 0.5))
            if 0.4 <= a <= L - 0.4]
    if not ramp:
        ramp = [(min(max(s, 0.4), L - 0.4), 1.0)]
    out = []
    for d in offsets:
        wps = []
        for a, f in ramp:
            q = snap(sim, at(dec.post, a) + f * d * v3(normal_xz))
            if q is None:
                break
            wps.append(q)
        else:
            post = chain(sim, [dec.p, *wps, dec.post[-1]])
            if (valid(sim, dec, post) and between_free(sim, dec.post, post) >= 0.9
                    and max_turn_deg(post[:max(2, int(np.searchsorted(_cum(post), 1.2)) + 1)]) <= FIRST_TURN_DEG):
                out.append((d, post))
    return out


def _clearance_to(objs, post) -> float:
    pts = dense(post)
    return float(min(o.dist_xz(pts).min() for o in objs)) - G.SPOT_HALF_WIDTH_M


def _clear_pick(routes, large: bool, min_m: float = CLEAR_MIN_M):
    routes = [r for r in routes if r[0] >= min_m]
    picked = pick3(routes, key=lambda r: r[0])
    if picked is None or (large and picked[2][0] < WIDE_LARGE_M):
        return None
    return picked


# ---- families ---------------------------------------------------------------------------------------------------

def speed_set(kind, v0: float, rng: random.Random, names=None):
    """{class: speed} for a speed phrase's (direction, size) at start speed v0 (from one of its candidate sets, or
    the given class names), or None if Spot cannot show them."""
    out = {}
    for name in names or rng.choice(SPEED_SETS[kind]):
        if name == "max":
            v = WALK_MAX_SPEED
        else:
            lo, hi = SPEED_RATIO[name]
            pad = 0.2 * (hi - lo)  # stay off the class edges, where neighbouring classes blur together
            lo, hi = max(lo + pad, MIN_SPEED / v0), min(hi - pad, WALK_MAX_SPEED / v0)
            if lo > hi:
                return None
            v = v0 * rng.uniform(lo, hi)
        out[name] = v
    vs = sorted(out.values())
    if min(b - a for a, b in zip(vs[:-1], vs[1:])) < MIN_SPEED_GAP:
        return None
    if kind == ("faster", "max") and WALK_MAX_SPEED / v0 < 1.6:
        return None
    return out


def gen_speed(sim, dec: Decision, spec, rng):
    kind = tuple(spec.metric.split("/"))
    c = speed_set(kind, dec.v0, rng)
    if c is None:
        return None
    path = full(dec, dec.post)
    return {"kind": "speed", "class": SPEED_TARGET[kind]}, [Candidate(k, path, v, v / dec.v0) for k, v in c.items()]


def gen_clear(sim, dec: Decision, spec, rng):
    if spec.referent == "wall":
        return _clear_wall(sim, dec, spec)
    if spec.referent == "corner":
        return _clear_corner(sim, dec)
    if spec.referent in ("furniture", "people"):
        hits = _on_way(dec, _pool(dec, spec.referent))
        cam = G.front_camera(dec.p, dec.yaw, sim)
        if (not hits or (spec.referent == "people" and len(hits) < 2)
                or not G.sees_object(sim, cam, dec.yaw, _members(hits[0][0])[0])
                or (spec.referent == "furniture" and G.wall_flush(sim, hits[0][0], G.floor_at(sim, dec.p)))):
            return None
        objs, s = [h[0] for h in hits], hits[0][1]
    else:
        ref = object_referent(sim, dec, spec.referent)
        if ref is None and spec.referent == "thing":
            return _clear_wall(sim, dec, spec)  # nothing unique in the way, but a wall beside it: "the wall"
        if ref is None:
            return None
        objs, s = [ref[0]], ref[1]
    n = _away_normal(dec, s, objs[0].center[[0, 2]])
    near_others = [o for o in dec.objs if o.category != "carpet" and not any(o in _members(x) for x in objs)]
    # a candidate may not come closer to anything else than OTHERS_MIN_M, or than the line already does
    floor_ = min(OTHERS_MIN_M, _clearance_to(near_others, dec.post) - 0.05) if near_others else -np.inf
    rug = objs[0].category in RUGS
    offsets, spread = OFFSETS_M, WAYPOINT_SPREAD_M
    if rug:
        # a rug is walked over or around: shift by up to its width past it, along all of its length
        half = (objs[0].hi - objs[0].lo)[[0, 2]] / 2
        h = heading_at(dec.post, s)
        offsets = tuple(np.arange(0.0, 2 * (np.abs(n) @ half) + 1.0, 0.2))
        spread = max(spread, float(np.abs(h) @ half) + 0.2)
    routes = []
    for _, post in _offset_routes(sim, dec, s, n, offsets=offsets, spread=spread):
        others = _clearance_to(near_others, post) if near_others else np.inf
        if others >= floor_:
            routes.append((_clearance_to(objs, post), post))
    picked = _clear_pick(routes, large=spec.metric == "large", min_m=-np.inf if rug else CLEAR_MIN_M)
    if picked is None:
        return None
    ref = {"kind": spec.referent, "objects": [{"oid": m.oid, "category": _category(m)} for o in objs for m in _members(o)]}
    if spec.referent in ("thing", "thing_hit"):
        ref["name"] = noun(objs[0])
    return ref, [Candidate(n, full(dec, post), dec.v0, c) for n, (c, post) in zip(("close", "mid", "wide"), picked)]


def _lateral_room(sim, q, left_xz, max_m: float = 2.0):
    """Navigable distance from q to the left and to the right."""
    out = []
    for sign in (1, -1):
        d = 0.0
        while d < max_m:
            t = q + (d + 0.1) * sign * v3(left_xz)
            if not sim.pathfinder.is_navigable(t.astype(np.float32)):
                break
            d += 0.1
        out.append(d)
    return out


def _lanes(sim, dec: Decision, toward_sign: int, end_s: float = 4.2, margin: float = 0.05):
    """Routes along lanes parallel to the line ahead, each held until it stops end_s on: (offset, route).
    Offsets are signed along toward_sign * left and bounded by the free space on both sides."""
    if length(dec.post) < end_s or max_turn_deg(dec.post[:max(2, np.searchsorted(_cum(dec.post), end_s) + 1)]) > 20:
        return None
    arcs = (1.4, 2.8, end_s)
    lefts = [G.left_of(heading_at(dec.post, a)) for a in arcs]
    rooms = [_lateral_room(sim, at(dec.post, a), l) for a, l in zip(arcs, lefts)]
    lmin, rmin = min(r[0] for r in rooms), min(r[1] for r in rooms)
    routes = []
    for d in np.linspace(-rmin + margin, lmin - margin, 21):
        wps = [snap(sim, at(dec.post, a) + d * v3(l)) for a, l in zip(arcs, lefts)]
        if any(w is None for w in wps):
            continue
        r = chain(sim, [dec.p, *wps])
        if valid(sim, dec, r) and length(r) <= end_s + 1.0 and between_free(sim, dec.post[:2], r) >= 0.9:
            routes.append((float(d * toward_sign), r))
    return routes, lmin, rmin


def _fit_line(pts):
    """Direction and max residual of the best straight line through points (xz)."""
    xz = np.asarray(pts)[:, [0, 2]]
    c = xz.mean(axis=0)
    _, _, vt = np.linalg.svd(xz - c)
    d = vt[0]
    res = np.abs((xz - c) @ np.array([-d[1], d[0]]))
    return d, float(res.max())


def _clear_wall(sim, dec: Decision, spec):
    """A single straight wall runs beside the line ahead (and no second wall on the other side); routes along it at
    three distances that straddle Spot's current one, held until they stop."""
    end_s = 4.2
    if length(dec.post) < end_s:
        return None
    arcs = np.arange(0.8, end_s + 1e-9, 0.4)
    found = None
    for sign in (1, -1):
        hits, opposite = [], 0
        for a in arcs:
            q = at(dec.post, a)
            side = sign * G.left_of(heading_at(dec.post, a))
            d = G.wall_distance(sim, q, side, 1.8)
            if np.isfinite(d):
                hits.append(q + d * v3(side))
            opposite += np.isfinite(G.wall_distance(sim, q, -side, 2.2))
        if len(hits) >= len(arcs) - 1 and opposite <= 1:
            found = sign, hits
            break
    if found is None:
        return None
    sign, hits = found
    d_wall, res = _fit_line(hits)
    h = heading_at(dec.post, 2.5, 1.5)
    if res > 0.1 or abs(d_wall @ h) < np.cos(np.radians(20)):
        return None
    got = _lanes(sim, dec, sign, end_s)
    if got is None:
        return None
    side = sign * G.left_of(h)

    def wall_clear(r):
        ds = [G.wall_distance(sim, q, side, 3.0) for q in dense(r, 0.3) if np.linalg.norm(q - dec.p) > 1.4]
        ds = [d for d in ds if np.isfinite(d)]
        return float(np.mean(ds)) - G.SPOT_HALF_WIDTH_M if ds else np.nan

    now = wall_clear(dec.post[:max(2, np.searchsorted(_cum(dec.post), end_s) + 1)])
    routes = [(c, r) for c, r in ((wall_clear(r), r) for _, r in got[0]) if np.isfinite(c) and c >= CLEAR_MIN_M]
    picked = pick3(routes, key=lambda r: r[0], step=0.25, span=0.6)
    if picked is None or not np.isfinite(now) or picked[0][0] > now - 0.2 or picked[2][0] < now + 0.2:
        return None
    ends = [hits[0], hits[-1]]
    return ({"kind": "wall", "side": "left" if sign > 0 else "right", "wall": [_r(q) for q in ends],
             "now_m": now},
            [Candidate(n, full(dec, p), dec.v0, c) for n, (c, p) in zip(("close", "mid", "wide"), picked)])


def _r(q):
    return [round(float(x), 3) for x in q]


def _clear_corner(sim, dec: Decision):
    """The line turns, within 3 m, around a tall corner on the inside of the turn; routes take the turn tight to
    wide."""
    post = dec.post
    best, s_best = 0.0, None
    for s in np.arange(AHEAD[0], min(3.0, length(post) - 0.7), 0.1):
        h_in, h_out = heading_at(post, s - 0.6, 0.15), heading_at(post, s + 0.6, 0.15)
        ang = float(np.degrees(np.arccos(np.clip(h_in @ h_out, -1, 1))))
        if ang > best:
            best, s_best = ang, s
    if s_best is None or best < 60:
        return None
    h_in, h_out = heading_at(post, s_best - 0.6, 0.15), heading_at(post, s_best + 0.6, 0.15)
    inside = (h_out - h_in) / np.linalg.norm(h_out - h_in)
    apex = at(post, s_best)
    fy = G.floor_at(sim, apex)
    dist, oid = G.ray(sim, [apex[0], fy + 0.5, apex[2]], v3(inside), 1.3)
    high, _ = G.ray(sim, [apex[0], fy + 1.0, apex[2]], v3(inside), 1.3)
    if not (np.isfinite(dist) and np.isfinite(high)):
        return None  # nothing tall on the inside of the turn
    corner = apex[[0, 2]] + dist * inside
    if not G.sees_point(sim, G.front_camera(dec.p, dec.yaw, sim), np.r_[corner[0], fy + 0.8, corner[1]] - 0.1 * v3(inside)):
        return None
    routes = []
    for d in (-0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        wps = [snap(sim, at(post, s_best + t) - f * d * v3(inside)) for t, f in ((-0.6, 0.6), (0.0, 1.0), (0.6, 0.6))]
        if any(w is None for w in wps):
            continue
        r = chain(sim, [dec.p, *wps, post[-1]])
        if valid(sim, dec, r):
            c = float(np.min(np.linalg.norm(dense(r)[:, [0, 2]] - corner, axis=1))) - G.SPOT_HALF_WIDTH_M
            routes.append((c, r))
    picked = _clear_pick(routes, large=False)
    if picked is None:
        return None
    kind = "wall" if oid == habitat_sim.stage_id else "object"
    return ({"kind": "corner", "turn_deg": best, "corner_is": kind, "door": _r(np.r_[corner[0], apex[1], corner[1]])},
            [Candidate(n, full(dec, p), dec.v0, c) for n, (c, p) in zip(("tight", "mid", "wide"), picked)])


def gen_side(sim, dec: Decision, spec, rng):
    """Walls bound the space ahead on both sides; routes move to its right, middle or left and stay there."""
    h = heading_at(dec.post, 2.5, 1.5)
    q = at(dec.post, 2.5)
    if not (np.isfinite(G.wall_distance(sim, q, G.left_of(h), 3.0))
            and np.isfinite(G.wall_distance(sim, q, -G.left_of(h), 3.0))):
        return None
    got = _lanes(sim, dec, 1, margin=0.25)
    if got is None:
        return None
    routes, lmin, rmin = got
    if lmin < 0.6 or rmin < 0.6:
        return None
    # symmetric about the line: right, middle and left by the same amount, up to 80% of the narrower side
    a = 0.8 * min(lmin, rmin)
    if a < 0.4 or not routes:
        return None
    picked = [min(routes, key=lambda r: abs(r[0] - t)) for t in (-a, 0.0, a)]
    if any(abs(p[0] - t) > 0.15 for p, t in zip(picked, (-a, 0.0, a))):
        return None
    return ({"kind": "free_space", "left_m": lmin, "right_m": rmin},
            [Candidate(n, full(dec, p), dec.v0, c) for n, (c, p) in zip(("right", "middle", "left"), picked)])


def gen_pass(sim, dec: Decision, spec, rng):
    ref = object_referent(sim, dec, "obstacle")
    if ref is None:
        return None
    o, s = ref
    # "the other side" needs a side already taken: Spot's straight line ahead passes beside the object, not into it
    if o.dist_xz(_ahead_ray(dec)).min() < 0.1:
        return None
    n = _away_normal(dec, s, o.center[[0, 2]])
    near = [(_clearance_to([o], p), p) for _, p in _offset_routes(sim, dec, s, n, offsets=(0.0, 0.25, 0.5, 0.8))]
    near = [x for x in near if x[0] >= CLEAR_MIN_M]
    c = np.array([o.center[0], dec.p[1], o.center[2]])
    far = []
    h = heading_at(dec.post, s)
    size = (o.hi - o.lo)[[0, 2]] / 2
    across, along = np.abs(n) @ size, np.abs(h) @ size  # half the footprint across and along the line
    for extra in (0.45, 0.65, 0.9, 1.2):
        side = c - (across + extra) * v3(n)
        # waypoints past the object's ends: inside them the route doubles back around its near end
        ws = [snap(sim, side + t * v3(h)) for t in (-along - 0.2, along + 0.2)]
        if any(w is None for w in ws):
            continue
        r = chain(sim, [dec.p, *ws, dec.post[-1]])
        if (valid(sim, dec, r) and length(r) <= length(dec.post) + 4.0 and _side(dec, s, o, r) < 0
                and _clearance_to([o], r) >= CLEAR_MIN_M):
            far.append(r)
    if not near or not far:
        return None
    tight = near[0]
    wide = [x for x in near if x[0] >= tight[0] + 0.4]
    if wide:
        picked = [("near_side", tight[1]), ("near_side_wide", wide[-1][1]), ("far_side", far[0])]
    elif len(far) >= 2:
        picked = [("near_side", tight[1]), ("far_side", far[0]), ("far_side_wide", far[-1])]
    else:
        return None
    return ({"kind": "obstacle", "objects": [{"oid": o.oid, "category": _category(o)}]},
            [Candidate(name, full(dec, r), dec.v0, _clearance_to([o], r)) for name, r in picked])


def _side(dec: Decision, s: float, o, route) -> float:
    """+1 if the route passes the object on the same side as the sampled line, -1 if on the other side."""
    def side_of(path):
        pts = dense(path)
        k = int(np.argmin(o.dist_xz(pts)))
        q = pts[k]
        nxt = pts[min(k + 2, len(pts) - 1)] - pts[max(k - 2, 0)]
        left = G.left_of(nxt[[0, 2]] / max(np.linalg.norm(nxt[[0, 2]]), 1e-9))
        return np.sign((o.center[[0, 2]] - q[[0, 2]]) @ left)
    return float(side_of(dec.post) * side_of(route))


def gen_around(sim, dec: Decision, spec, rng):
    """Spot heads through the gap between two free-standing objects; routes go through it and around each side."""
    hits = _on_way(dec, [o for o in dec.objs if o.category], near=1.0)
    cam = G.front_camera(dec.p, dec.yaw, sim)
    fy = G.floor_at(sim, dec.p)
    for i, (a, sa, _) in enumerate(hits):
        for b, sb, _ in hits[i + 1:]:
            if abs(sa - sb) > 1.0:
                continue
            s = (sa + sb) / 2
            q = at(dec.post, s)
            left = G.left_of(heading_at(dec.post, s))
            sa_side = (a.center[[0, 2]] - q[[0, 2]]) @ left
            sb_side = (b.center[[0, 2]] - q[[0, 2]]) @ left
            if sa_side * sb_side >= 0:
                continue  # the line does not pass between them
            gap = float(np.hypot(*np.maximum(np.maximum(a.lo - b.hi, b.lo - a.hi), 0.0)[[0, 2]]))
            if gap > 1.5:
                continue
            if not (G.sees_object(sim, cam, dec.yaw, a) and G.sees_object(sim, cam, dec.yaw, b)):
                continue
            if G.wall_flush(sim, a, fy) or G.wall_flush(sim, b, fy):
                continue
            arounds = []
            for o, side in ((a, np.sign(sa_side)), (b, np.sign(sb_side))):
                half = 0.5 * np.hypot(o.hi[0] - o.lo[0], o.hi[2] - o.lo[2])
                out = side * left
                c = np.array([o.center[0], dec.p[1], o.center[2]])
                for extra in (0.55, 0.8, 1.1):
                    w = snap(sim, c + (half + extra) * v3(out))
                    if w is None:
                        continue
                    r = chain(sim, [dec.p, w, dec.post[-1]])
                    if valid(sim, dec, r) and length(r) <= length(dec.post) + 4.0 and _side(dec, s, o, r) < 0:
                        arounds.append((o, side, r))
                        break
            names = ["around_" + _category(a), "around_" + _category(b)]
            if len(arounds) == 1:
                # only one side is open: around it, and around it wide (at least 0.5 m farther out)
                o, side, r0 = arounds[0]
                half = 0.5 * np.hypot(o.hi[0] - o.lo[0], o.hi[2] - o.lo[2])
                c = np.array([o.center[0], dec.p[1], o.center[2]])
                w = snap(sim, c + (half + 1.6) * v3(side * left))
                r = chain(sim, [dec.p, w, dec.post[-1]]) if w is not None else None
                if not (valid(sim, dec, r) and _clearance_to([o], r) >= _clearance_to([o], r0) + 0.5):
                    continue
                arounds.append((o, side, r))
                names = ["around_" + _category(o), "around_" + _category(o) + "_wide"]
            elif len(arounds) < 2:
                continue
            ref = {"kind": "gap", "objects": [{"oid": a.oid, "category": _category(a)},
                                              {"oid": b.oid, "category": _category(b)}], "gap_m": gap}
            return ref, [Candidate("through", full(dec, dec.post), dec.v0, 0.0),
                         Candidate(names[0], full(dec, arounds[0][2]), dec.v0, 1.0),
                         Candidate(names[1], full(dec, arounds[1][2]), dec.v0, 1.0)]
    return None


def gen_length(sim, dec: Decision, spec, rng):
    """Routes to the same end of three clearly different lengths; the longer ones go around something that the
    shortest one does not."""
    base = dec.post
    L = length(base)
    if L < 3.0:
        return None
    mid = at(base, L / 2)
    routes = [(L, base)]
    pf = sim.pathfinder
    for _ in range(40):
        w = np.array(pf.get_random_navigable_point_near(mid.astype(np.float32), 3.5, 50), dtype=float)
        if not np.isfinite(w).all() or abs(w[1] - mid[1]) > 0.3:
            continue
        r = chain(sim, [dec.p, w, base[-1]])
        if (r is not None and length(r) <= MAX_POST_M and valid(sim, dec, r)
                and between_free(sim, base, r) <= 0.7):
            routes.append((length(r), r))
    picked = pick3(routes, key=lambda r: r[0], step=0.8, span=1.8)
    if picked is None or picked[0][0] > L + 0.05:
        return None
    return ({"kind": "routes"},
            [Candidate(n, full(dec, p), dec.v0, m) for n, (m, p) in zip(("short", "medium", "long"), picked)])


def _straight_run(sim, dec: Decision, max_m: float = 10.0):
    """End of the navigable straight line ahead of Spot, the distance to it, and what stops it (ray hit)."""
    p = dec.p
    h = v3(dec.heading)
    d = 0.0
    while d < max_m and sim.pathfinder.is_navigable((p + (d + 0.05) * h).astype(np.float32)):
        d += 0.05
    if d < 0.3:
        return None
    e = p + d * h
    hit, oid = G.ray(sim, [e[0], G.floor_at(sim, e) + G.WALL_RAY_HEIGHT_M, e[2]], h, 1.5)
    return e, d, hit, oid


STOPS = {"small": (0.4, 1.3, 3.0), "unspecified": (0.3, 1.5, 3.5)}  # m after the decision point


def gen_stop(sim, dec: Decision, spec, rng):
    run = _straight_run(sim, dec)
    if run is None:
        return None
    e, d, hit, oid = run
    if spec.referent == "end":
        if not 4.0 <= d <= 7.0 or not np.isfinite(hit):
            return None
        stops = (d / 3, 2 * d / 3, d)
    elif spec.metric == "large":
        if d < 5.0:
            return None
        stops = (0.4, 1.5, min(d, 6.0))
    else:
        if d < 4.0:
            return None
        stops = STOPS[spec.metric or "unspecified"]
    cands = []
    for name, s in zip(("short", "mid", "far"), stops):
        post = straight(dec, s)
        # the line ends at the navmesh edge, where Spot's nose would touch; stop where body_clear allows
        while not valid(sim, dec, post) and s > 0.3:
            s -= 0.1
            post = straight(dec, s)
        if not valid(sim, dec, post):
            return None
        cands.append(Candidate(name, full(dec, post), dec.v0, s))
    if cands[1].metric - cands[0].metric < 0.8 or cands[2].metric - cands[1].metric < 0.8:
        return None
    return ({"kind": spec.referent, "run_m": d}, cands)


def gen_approach(sim, dec: Decision, spec, rng):
    run = _straight_run(sim, dec)
    if run is None:
        return None
    e, d, hit, oid = run
    if not np.isfinite(hit) or d + hit > 4.5:
        return None
    cam = G.front_camera(dec.p, dec.yaw, sim)
    if spec.referent == "wall":
        # the wall is what Spot sees straight ahead at eye height too, not a low object in front of it
        eye, eye_oid = G.ray(sim, cam, v3(dec.heading), d + hit + 0.3)
        if oid != habitat_sim.stage_id or eye_oid != habitat_sim.stage_id:
            return None
        # and it faces Spot squarely across its width: not a corner or a jamb beside an opening
        left = v3(G.left_of(dec.heading))
        for side in (-0.4, 0.4):
            q = e + side * left
            h2, o2 = G.ray(sim, [q[0], G.floor_at(sim, e) + G.WALL_RAY_HEIGHT_M, q[2]], v3(dec.heading), hit + 0.6)
            if o2 != habitat_sim.stage_id or abs(h2 - hit) > 0.15:
                return None
        ref = {"kind": "wall", "door": _r(e + hit * v3(dec.heading))}
    else:
        o = next((x for x in dec.all_objs if x.oid == oid), None)
        if spec.referent == "thing":
            # whatever single thing Spot is walking at, said by its name
            if o is None or not o.category or o.category == "person" or o.category in RUGS:
                return None
        elif o is None or o.category != spec.referent:
            return None
        if not G.sees_object(sim, cam, dec.yaw, o, 3):
            return None
        if any(x is not o and noun(x) == noun(o) and x.dist_xz(o.center[None])[0] < SAME_KIND_M
               for x in dec.objs):
            return None
        ref = {"kind": spec.referent, "objects": [{"oid": o.oid, "category": o.category}]}
        if spec.referent == "thing":
            ref["name"] = noun(o)
    to_ref = d + hit  # from the decision point to the referent, along the heading
    cands = []
    # gaps between Spot's nose and the referent; the walk stops up to ~0.15 m past the line's end (nuance.spot_walk)
    for name, gap in (("far", 1.6), ("just_before", 0.7), ("closest", 0.25)):
        s = to_ref - SPOT_HALF_LENGTH - gap
        if s < 0.6:
            return None
        post = straight(dec, s)
        while not valid(sim, dec, post) and name == "closest" and s > 0.6:
            s -= 0.05
            post = straight(dec, s)
        if not valid(sim, dec, post):
            return None
        cands.append(Candidate(name, full(dec, post), dec.v0, to_ref - SPOT_HALF_LENGTH - s))
    if cands[1].metric - cands[2].metric < 0.25:
        return None
    return ref, cands


# ---- place ------------------------------------------------------------------------------------------------------

def _goals(sim, dec: Decision, rng, n: int = 120, cone_deg: float = 100.0, radius: float = 6.0,
           min_clear: float = GOAL_CLEAR_M):
    """(goal, route) pairs: reachable points ahead of Spot with min_clear of room around them."""
    out = []
    pf = sim.pathfinder
    for _ in range(n):
        g = np.array(pf.get_random_navigable_point_near(dec.p.astype(np.float32), radius, 50), dtype=float)
        if not np.isfinite(g).all() or abs(g[1] - dec.p[1]) > 0.3:
            continue
        v = (g - dec.p)[[0, 2]]
        if np.linalg.norm(v) < 1.5:
            continue
        if np.degrees(np.arccos(np.clip(dec.heading @ v / np.linalg.norm(v), -1, 1))) > cone_deg:
            continue
        if _clear_at(sim, g) < min_clear:
            continue
        r = shortest(sim, dec.p, g)
        if r is not None and length(r) <= PLACE_MAX_M and valid(sim, dec, r):
            out.append((g, r))
    return out


def _clear_at(sim, g) -> float:
    return float(sim.pathfinder.distance_to_closest_obstacle(g.astype(np.float32), 3.0)) + G.NAVMESH_AGENT_RADIUS


def _openness(sim, g) -> float:
    """Free space around a spot, as a rater sees it: the mean clearance at it and 0.7 m around it."""
    pts = [g] + [g + 0.7 * np.r_[np.cos(a), 0, np.sin(a)] for a in np.linspace(0, 2 * np.pi, 6, False)]
    return float(np.mean([_clear_at(sim, q) if sim.pathfinder.is_navigable(q.astype(np.float32)) else 0.0
                          for q in pts]))


def _place_cands(dec, picked, names):
    return [Candidate(n, full(dec, r), dec.v0, float(m)) for n, (m, g, r) in zip(names, picked)]


def _pick_goals(scored, step, span, apart=GOAL_APART_M):
    return pick3(scored, key=lambda x: x[0], step=step, span=span, apart=apart, pos=lambda x: x[1][[0, 2]])


def gen_place(sim, dec: Decision, spec, rng):
    m = spec.metric
    cam = G.front_camera(dec.p, dec.yaw, sim)
    if m == "open":
        goals = [(_openness(sim, g), g, r) for g, r in _goals(sim, dec, rng, cone_deg=75, min_clear=0.0)]
        goals = [x for x in goals if G.in_view(cam, dec.yaw, x[1]) and G.sees_point(sim, cam, x[1] + np.r_[0, 0.3, 0])]
        picked = _pick_goals(goals, step=0.4, span=1.0, apart=2.0)
        if picked is None or picked[2][0] < 1.6 or picked[0][0] > 0.5:
            return None
        return {"kind": "open"}, _place_cands(dec, picked, ("cramped", "mid", "open"))
    if m == "aside":
        return _aside(sim, dec)
    if m == "visible":
        return _visible(sim, dec, rng)
    if m == "wall":
        here = G.region_of(dec.scene_id, dec.p)
        if here is None or G.wall_distance(sim, dec.p, G.left_of(dec.heading), 1.0) < 1.0 or \
                G.wall_distance(sim, dec.p, -G.left_of(dec.heading), 1.0) < 1.0:
            return None  # Spot already beside a wall: "the wall" would be that one
        goals = []
        for g, r in _goals(sim, dec, rng, cone_deg=120, min_clear=0.0):
            w = min(G.wall_distance(sim, g, [np.cos(a), np.sin(a)], 3.0) for a in np.linspace(0, 2 * np.pi, 8, False))
            goals.append((min(w, 3.0), g, r))
        # "the wall" is one Spot can see: the spot by it is in view, and against a wall, not in an opening
        goals = [x for x in goals if here.contains(x[1]) and G.in_view(cam, dec.yaw, x[1])]
        goals = [x for x in goals if x[0] > 0.75 or (G.sees_point(sim, cam, x[1] + np.r_[0, 0.3, 0])
                                                     and sum(_lateral_room(sim, x[1], [1.0, 0.0])) > 1.4
                                                     and sum(_lateral_room(sim, x[1], [0.0, 1.0])) > 1.4)]
        picked = _pick_goals(goals, step=0.6, span=1.4, apart=2.0)
        if picked is None or picked[0][0] > 0.75 or picked[1][0] < 1.2:
            return None
        return {"kind": "wall"}, _place_cands(dec, picked, ("by_wall", "mid", "away"))
    if spec.referent in ("window", "door"):
        return _near_object(sim, dec, rng, spec.referent)
    if spec.referent == "doorway":
        return _next_room(sim, dec, rng)
    return _room(sim, dec, rng, m)


def _near_object(sim, dec, rng, category):
    """Goals near the only window (door) in view ahead, at a middle distance from it, and far from it."""
    cam = G.front_camera(dec.p, dec.yaw, sim)
    seen = [o for o in dec.all_objs if o.category in ("window", "door")
            and np.hypot(*(o.center - dec.p)[[0, 2]]) < 8.0 and G.sees_object(sim, cam, dec.yaw, o, 2)]
    mine = [o for o in seen if o.category == category]
    if len(mine) != 1:
        return None
    o = mine[0]

    def bearing(x):
        v = (x.center - dec.p)[[0, 2]]
        return np.degrees(np.arctan2(v @ G.left_of(dec.heading), v @ dec.heading))
    if any(x is not o and abs(bearing(x) - bearing(o)) < 30 for x in seen):
        return None  # glass doors next to the window (or windows next to the door) make "the window" ambiguous
    if not (2.0 < np.hypot(*(o.center - dec.p)[[0, 2]]) < 6.0 and G.in_view(cam, dec.yaw, o.center, margin_deg=15)):
        return None
    goals = [(float(o.dist_xz(g[None])[0]), g, r) for g, r in _goals(sim, dec, rng, n=200)]
    picked = _pick_goals(goals, step=1.0, span=2.5)
    if picked is None or picked[0][0] > 1.0:
        return None
    return ({"kind": category, "objects": [{"oid": o.oid, "category": category}]},
            _place_cands(dec, picked, ("near", "mid", "far")))


def _doorways(sim, dec, rng, goals):
    """(doorway point, goal, route, room beyond) for routes that leave Spot's room through a narrow opening in view."""
    here = G.region_of(dec.scene_id, dec.p)
    if here is None:
        return here, []
    cam = G.front_camera(dec.p, dec.yaw, sim)
    out = []
    for g, r in goals:
        other = G.region_of(dec.scene_id, g)
        if other is None or other is here or other.label == here.label:
            continue
        pts = dense(r)
        k = next((i for i, q in enumerate(pts) if not here.contains(q)), None)
        if k is None:
            continue
        door = pts[k]
        left = G.left_of(heading_at(r, k * 0.05))
        width = sum(_lateral_room(sim, door, left))
        # a real opening: wall on both sides at door height, not a gap between furniture in an open-plan room
        jambs = all(np.isfinite(G.wall_distance(sim, door, sgn * left, 1.0, height=1.2)) for sgn in (1, -1))
        ahead = (door - dec.p)[[0, 2]]
        if (width <= 1.4 and jambs and G.in_view(cam, dec.yaw, door, margin_deg=15)
                and ahead @ dec.heading > np.cos(np.radians(45)) * np.linalg.norm(ahead)
                and G.sees_point(sim, cam, door + np.r_[0, 0.8, 0])
                and length(r) - k * 0.05 >= 1.2 and all(not here.contains(q) for q in pts[k:])):
            out.append((door, g, r, other.label))
    return here, out


def _next_room(sim, dec, rng):
    goals = _goals(sim, dec, rng, n=200)
    here, through = _doorways(sim, dec, rng, goals)
    if not through:
        return None
    door, g, r, label = through[0]
    by_door = [x for x in goals if G.region_of(dec.scene_id, x[0]) is here and np.hypot(*(x[0] - door)[[0, 2]]) < 1.3]
    # staying is stopping where Spot is, not a walk somewhere else in the room
    halt = straight(dec, 0.5)
    if not by_door or not valid(sim, dec, halt):
        return None
    return ({"kind": "doorway", "from": here.label, "to": label, "door": door.tolist()},
            [Candidate("stay", full(dec, halt), dec.v0, 0.0),
             Candidate("at_doorway", full(dec, by_door[0][1]), dec.v0, 1.0),
             Candidate("through", full(dec, r), dec.v0, 2.0)])


NOT_ROOMS = {"hallway", "closet", "toilet", "entryway/foyer/lobby", "garage"}  # no middle, corner or view to speak of


def _room(sim, dec, rng, metric):
    here = G.region_of(dec.scene_id, dec.p)
    if here is None or (metric != "far" and (here.area < 10 or here.label in NOT_ROOMS)):
        return None
    cam = G.front_camera(dec.p, dec.yaw, sim)
    if metric != "far":
        # Spot is inside the room (not on its threshold), and for "the middle" looking into it
        c = here.centroid
        if here.edge_distance(dec.p) < 0.8 or (metric in ("center", "view") and not G.in_view(
                cam, dec.yaw, np.r_[c[0], dec.p[1], c[1]], margin_deg=5)):
            return None
    goals = [(g, r) for g, r in _goals(sim, dec, rng, n=200, cone_deg=110, min_clear=0.0 if metric == "corner" else
                                        GOAL_CLEAR_M) if here.contains(g)]
    if metric == "center":
        c = here.centroid
        scored = [(float(np.hypot(*(g[[0, 2]] - c))), g, r) for g, r in goals]
        picked = _pick_goals(scored, step=0.8, span=2.0)
        if picked is None or picked[0][0] > 0.8:
            return None
        return ({"kind": "room", "label": here.label, "door": _r(np.r_[c[0], dec.p[1], c[1]])},
                _place_cands(dec, picked, ("center", "mid", "edge")))
    if metric == "corner":
        # the one corner of the room outline that is in view ahead
        def bearing(v):
            w = v - dec.p[[0, 2]]
            return abs(np.degrees(np.arctan2(w @ G.left_of(dec.heading), w @ dec.heading)))
        corners = [v for v in here.poly if np.hypot(*(v - dec.p[[0, 2]])) > 1.5 and bearing(v) < 25]
        if not corners:
            return None
        v = min(corners, key=bearing)  # "the corner" is the one Spot is facing
        scored = [(float(np.min(np.hypot(*(here.poly - g[[0, 2]]).T))), g, r) for g, r in goals]
        def walled(g):
            # a wall-to-wall corner: walls close by in two directions about square to each other
            hit = [a for a in np.linspace(0, 2 * np.pi, 8, False)
                   if np.isfinite(G.wall_distance(sim, g, [np.cos(a), np.sin(a)], 0.9, height=1.2))]
            return any(60 <= np.degrees(abs((a - b + np.pi) % (2 * np.pi) - np.pi)) <= 120 for a in hit for b in hit)
        near = [x for x in scored if np.hypot(*(x[1][[0, 2]] - v)) < 1.1 and walled(x[1])]
        rest = [x for x in scored if x[0] > 1.5]
        if not near or not rest:
            return None
        picked = _pick_goals([min(near, key=lambda x: x[0])] + rest, step=0.8, span=2.0)
        if picked is None or picked[0][0] > 1.1:
            return None
        return {"kind": "room", "label": here.label, "door": _r(np.r_[v[0], dec.p[1], v[1]])}, \
            _place_cands(dec, picked, ("corner", "mid", "open"))
    if metric == "far":
        # the far end is the visible one straight ahead; nearer goals lie on the same line, not off to a side
        scored = [(float((g - dec.p)[[0, 2]] @ dec.heading), g, r) for g, r in goals
                  if G.sees_point(sim, cam, g + np.r_[0, 0.3, 0])
                  and abs((g - dec.p)[[0, 2]] @ G.left_of(dec.heading)) <= 1.0]
        if not scored or max(s[0] for s in scored) < 4.0:
            return None
        picked = _pick_goals(scored, step=1.2, span=2.5)
        if picked is None:
            return None
        return {"kind": "room", "label": here.label}, _place_cands(dec, picked, ("near", "mid", "far_end"))
    if metric == "view":
        pf = sim.pathfinder
        probes = []
        for _ in range(300):
            q = np.array(pf.get_random_navigable_point_near(np.r_[here.centroid[0], dec.p[1], here.centroid[1]]
                                                            .astype(np.float32), 6.0, 50), dtype=float)
            if np.isfinite(q).all() and here.contains(q):
                probes.append(q + np.r_[0, 0.5, 0])
            if len(probes) >= 40:
                break
        if len(probes) < 20:
            return None
        scored = [(np.mean([G.sees_point(sim, g + np.r_[0, 0.6, 0], q) for q in probes]), g, r) for g, r in goals]
        picked = _pick_goals(scored, step=0.25, span=0.6, apart=2.0)
        if picked is None:
            return None
        return {"kind": "room", "label": here.label}, _place_cands(dec, picked, ("low_view", "mid", "full_view"))
    return None


def _aside(sim, dec):
    """Spot is in a passage; candidates carry on, stop on the spot, or step sideways out of the passage."""
    if length(dec.post) < 3.2:
        return None
    stop = at(dec.post, 0.5)
    go = dec.post[-1]
    s = 0.5
    q = at(dec.post, s)
    left = G.left_of(heading_at(dec.post, s))
    room = _lateral_room(sim, q, left)
    side = 1 if room[0] >= room[1] else -1
    if max(room) < 0.9 or min(room) > 1.5:
        return None  # nowhere to step to, or no passage to clear
    if not all(np.isfinite(G.wall_distance(sim, dec.p, sgn * G.left_of(dec.heading), 2.5)) for sgn in (1, -1)):
        return None  # a passage between walls, not a spot in an open room
    g = snap(sim, q + side * min(max(room) - 0.05, 1.4) * v3(left))
    if g is None:
        return None
    v = (g - dec.p)[[0, 2]]
    if np.degrees(np.arccos(np.clip(v @ dec.heading / np.linalg.norm(v), -1, 1))) < 50:
        return None  # a step aside is mostly sideways
    if G.region_of(dec.scene_id, g) is not G.region_of(dec.scene_id, dec.p):
        return None  # and stays in the passage, not into another room
    routes = [("continue", chain(sim, [dec.p, go])), ("stop", chain(sim, [dec.p, stop])),
              ("aside", chain(sim, [dec.p, g]))]
    if not all(valid(sim, dec, r) for _, r in routes):
        return None
    return ({"kind": "aside", "side": "left" if side > 0 else "right"},
            [Candidate(n, full(dec, r), dec.v0, float(i)) for i, (n, r) in enumerate(routes)])


def _visible(sim, dec, rng):
    cam = G.front_camera(dec.p, dec.yaw, sim)
    by = {"left": [], "center": [], "right": []}
    for g, r in _goals(sim, dec, rng, n=200, cone_deg=45):
        v = (g - dec.p)[[0, 2]]
        if not (2.5 <= np.linalg.norm(v) <= 6.0) or not G.sees_point(sim, cam, g + np.r_[0, 0.3, 0]):
            continue
        b = float(np.degrees(np.arctan2(v @ G.left_of(dec.heading), v @ dec.heading)))
        key = "left" if b > 18 else "right" if b < -18 else "center" if abs(b) < 8 else None
        if key:
            by[key].append((b, g, r))
    if not all(by.values()):
        return None
    return ({"kind": "visible"},
            [Candidate(k, full(dec, by[k][0][2]), dec.v0, by[k][0][0]) for k in ("left", "center", "right")])


def gen_back(sim, dec: Decision, spec, rng):
    left = G.left_of(dec.heading)
    fwd3, left3 = v3(dec.heading), v3(left)
    h0 = heading_at(dec.pre, 0.3)
    back = None
    for sign in (1, -1):
        # A rounded turn, then back towards where Spot came from on a lane beside the walk-in.
        lane = sign * 0.9 * left3
        pts = [snap(sim, dec.p + 0.45 * fwd3 + sign * 0.45 * left3), snap(sim, dec.p + 0.15 * fwd3 + lane)]
        for back_m in (2.0, 1.0, 0.0):
            pts_end = snap(sim, dec.pre[0] - back_m * v3(h0) + lane)
            if pts_end is not None:
                break
        if any(w is None for w in pts) or pts_end is None:
            continue
        r = chain(sim, [dec.p, *pts, pts_end])
        if valid(sim, dec, r, reverse=True) and length(r) <= length(dec.pre) + 5.0:
            back = r
            break
    if back is None or length(back) < 2.0:
        return None
    side = None
    for sign in (1, -1):
        g = snap(sim, dec.p + 0.8 * fwd3 + sign * 2.0 * left3)
        if g is not None:
            r = chain(sim, [dec.p, g])
            if valid(sim, dec, r) and length(r) <= 1.3 * np.hypot(0.8, 2.0):
                side = r
                break
    if side is None or length(dec.post) < 2.5:
        return None
    onward = dec.post[:max(2, int(np.searchsorted(_cum(dec.post), 4.0)) + 1)]
    return ({"kind": "back"}, [Candidate("onward", full(dec, onward), dec.v0, 0.0),
                               Candidate("sideways", full(dec, side), dec.v0, 1.0),
                               Candidate("back", full(dec, back), dec.v0, 2.0)])


GENERATORS = {"speed": gen_speed, "clear": gen_clear, "side": gen_side, "pass": gen_pass, "around": gen_around,
              "length": gen_length, "stop": gen_stop, "approach": gen_approach, "place": gen_place, "back": gen_back}


def _sky_free(sim, c: Candidate) -> bool:
    """No open sky in Spot's view half-way along the candidate's continuation or at its end."""
    L = length(c.path)
    for s in (0.6 * L, L - 0.2):
        h = heading_at(c.path, s, 0.2)
        if G.sees_sky(sim, G.front_camera(at(c.path, s), float(np.arctan2(-h[0], -h[1])), sim),
                      float(np.arctan2(-h[0], -h[1]))):
            return False
    return True


def ground(sim, dec: Decision, text: str, rng: random.Random):
    spec = SPECS[text]
    if text in AROUND:
        spec = replace(spec, referent="thing_hit")  # something Spot would walk into, said as what it is; never a wall
    elif (spec.referent == "obstacle" and "the obstacle" in text) or (
            spec.referent in ("chair", "table") and spec.family in ("clear", "approach")):
        spec = replace(spec, referent="thing")  # said as what it is: "the plant", "the rug", "the wall"
    res = GENERATORS[spec.family](sim, dec, spec, rng)
    if res is None or not all(_sky_free(sim, c) for c in res[1]):
        return None
    return res
