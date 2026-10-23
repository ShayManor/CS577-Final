"""Every approved instruction and what a decision point needs before the instruction makes sense there.

family picks the candidate generator (nuance.candidates); referent is what the instruction talks about and must be
present, unambiguous and visible at the decision point (nuance.grounding).
  speed     the line is fixed, the candidates are three speeds
  clear     same goal, routes that pass the referent at three clearances
  side      same goal, routes along the left side, the middle and the right side of the free space
  pass      same goal, routes past the referent on its near side (tight and wide) and on its far side
  around    same goal, a route through the gap between two objects and routes around them
  length    same goal, routes of three clearly different lengths
  stop      the line is fixed, the candidates stop at three distances along it
  approach  the line runs at the referent, the candidates stop at three gaps in front of it
  place     three goals that score low, middle and high on the instruction's metric
  back      routes back the way Spot came, to the side, and onward
"""
import json
import os
from dataclasses import dataclass

from nuance.config import PHRASES


@dataclass(frozen=True)
class Spec:
    family: str
    referent: str = "none"
    metric: str = ""
    v0: tuple = ()  # start speed range the instruction needs, if not the family's


# Speed phrases that only make sense at some start speeds.
SPEED_V0 = {"you're going too fast": (0.8, 1.0), "you're going too slow": (0.4, 0.6), "take it easy": (0.7, 1.0)}


FURNITURE = {"table", "chair", "couch", "cabinet", "chest_of_drawers", "shelves", "bed", "bench", "stool", "wardrobe",
             "counter", "stand", "filing_cabinet"}

CONTEXT = {
    # clearance from an object on the way
    "give the obstacle a little more room": Spec("clear", "obstacle"),
    "a bit more space around that": Spec("clear", "obstacle"),
    "a slightly wider berth": Spec("clear", "obstacle"),
    "don't cut it so close": Spec("clear", "obstacle"),
    "give the obstacle a wide berth": Spec("clear", "obstacle"),
    "avoid the obstacle": Spec("clear", "obstacle"),
    "keep your distance": Spec("clear", "obstacle_or_person"),
    "stay a touch further from the table": Spec("clear", "table"),
    "keep way back from the table": Spec("clear", "table"),
    "go around the table": Spec("clear", "table"),
    "stay clear of the chair": Spec("clear", "chair"),
    "give the chair more room": Spec("clear", "chair"),
    "stay well clear of the furniture": Spec("clear", "furniture"),
    "steer clear of the furniture": Spec("clear", "furniture"),
    "give the person plenty of room": Spec("clear", "person"),
    "watch out for the person": Spec("clear", "person"),
    "don't brush past the person": Spec("clear", "person"),
    "give people space": Spec("clear", "people"),
    # clearance from a wall running beside the line
    "drift a little away from the wall": Spec("clear", "wall"),
    "a little closer to the wall": Spec("clear", "wall"),
    "hug the wall a bit more": Spec("clear", "wall"),
    "stay as far from the wall as you can": Spec("clear", "wall"),
    "hug the wall": Spec("clear", "wall"),
    "go right up next to the wall": Spec("clear", "wall"),
    "keep away from the wall": Spec("clear", "wall"),
    # clearance from the inside of a turn
    "tighten up the turn a little": Spec("clear", "corner"),
    "swing wide around the corner": Spec("clear", "corner"),
    # where in the free space
    "stay to the right": Spec("side"),
    "stay to the left": Spec("side"),
    "keep to the side": Spec("side"),
    "pass on the other side": Spec("pass", "obstacle"),
    "go around it, not through": Spec("around", "gap"),
    "take the long way around": Spec("length"),
    "take the shortest way": Spec("length"),
    # where to stop on the line
    "go a bit further": Spec("stop"),
    "go a little further and stop": Spec("stop"),
    "go a lot further": Spec("stop"),
    "stop right here": Spec("stop"),
    "go all the way to the end": Spec("stop", "end"),
    "stop just before the table": Spec("approach", "table"),
    "stop just short of the wall": Spec("approach", "wall"),
    # where to go
    "go to the empty space": Spec("place", metric="open"),
    "head to the open area": Spec("place", metric="open"),
    "go wherever there's more room": Spec("place", metric="open"),
    "go to the middle of the room": Spec("place", "room", "center"),
    "go to the corner": Spec("place", "room", "corner"),
    "go toward the window": Spec("place", "window", "near"),
    "go to the far end": Spec("place", "room", "far"),
    "go through the doorway": Spec("place", "doorway", "next_room"),
    "go into the next room": Spec("place", "doorway", "next_room"),
    "wait by the door": Spec("place", "door", "near"),
    "park by the wall": Spec("place", metric="wall"),
    "go where you can see the whole room": Spec("place", "room", "view"),
    "get out of the way": Spec("place", metric="aside"),
    "move aside": Spec("place", metric="aside"),
    "go over there": Spec("place", metric="visible"),
    "go back the way you came": Spec("back"),
    "turn around": Spec("back"),
}

def _speed_kinds() -> dict:
    """Speed phrase -> (direction, size) from the reviewed candidate list: faster / slower / hold, and small /
    large / unspecified; "go as fast as you can" asks for Spot's top speed."""
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "instruction_candidates.json")
    kinds = {c["text"]: (c["direction"], c["size"]) for c in json.load(open(path))}
    kinds["go as fast as you can"] = ("faster", "max")
    kinds["pick up the pace"] = ("faster", "unspecified")  # reviewers read it as a moderate speed-up
    return {p: kinds[p] for p in PHRASES}


def _sized(context: dict) -> dict:
    """Clear and stop specs carry the instruction's size (small / large / unspecified) as their metric."""
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "instruction_candidates.json")
    size = {c["text"]: c["size"] for c in json.load(open(path))}
    return {t: Spec(s.family, s.referent, size[t]) if s.family in ("clear", "stop") else s for t, s in context.items()}


SPEED_KIND = _speed_kinds()
SPECS = {**{p: Spec("speed", metric="/".join(SPEED_KIND[p]), v0=SPEED_V0.get(p, ())) for p in PHRASES},
         **_sized(CONTEXT)}


# Alternative instructions that make sense wherever text does, for kinds with fewer than two same_kind partners.
FALLBACK = {
    "stay clear of the chair": ["don't cut it so close"],
    "give the chair more room": ["don't cut it so close"],
    "stay well clear of the furniture": ["don't cut it so close"],
    "steer clear of the furniture": ["a slightly wider berth"],
    "tighten up the turn a little": ["don't cut it so close"],
    "swing wide around the corner": ["don't cut it so close"],
    "keep your distance": ["don't cut it so close", "a slightly wider berth"],
    "give people space": ["keep your distance", "don't cut it so close"],
    "pass on the other side": ["give the obstacle a wide berth", "avoid the obstacle"],
    "go around it, not through": ["take the long way around", "take the shortest way"],
    "go all the way to the end": ["go a lot further", "go a bit further"],
    "stop just before the table": ["go a little further and stop", "stop right here"],
    "stop just short of the wall": ["go all the way to the end", "go a little further and stop"],
    "go to the middle of the room": ["go where you can see the whole room"],
    "go to the corner": ["go to the middle of the room"],
    "go where you can see the whole room": ["go to the middle of the room"],
    "go through the doorway": ["stop right here"],
    "go into the next room": ["stop right here"],
    "get out of the way": ["stop right here", "go a lot further"],
    "move aside": ["stop right here", "go a lot further"],
    "go to the far end": ["go a lot further", "go a bit further"],
}


def same_kind(text: str) -> list:
    """Instructions that can share text's candidates: same family and referent (and metric, for place)."""
    s = SPECS[text]
    return [t for t, o in SPECS.items() if t != text and o.family == s.family and o.referent == s.referent
            and (s.family != "place" or o.metric == s.metric)]
