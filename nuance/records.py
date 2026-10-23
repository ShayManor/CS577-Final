import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from nuance.config import PHRASES


@dataclass
class ClipRecord:
    clip_id: str
    batch_id: str
    decision_id: str
    action_type: str
    scene_id: str
    path: list[list[float]]
    decision_position: list[float]
    decision_yaw: float
    context: dict
    start_speed: float
    candidate_speed: float
    candidate_rank: str
    instruction: str
    alt_instructions: list[str]
    video: str
    front_frame: str
    fps: int
    duration_s: float
    t_decision_s: float
    # Context instructions (nuance.generate_instructions): what the instruction refers to, the candidate's value on
    # the dimension its family varies (clearance, stop distance, ...), and the people placed for it, if any.
    referent: Optional[dict] = None
    candidate_metric: Optional[float] = None
    person: Optional[list] = None
    achieved_speed: Optional[float] = None  # Spot's mean simulated speed from the decision point to the line's end


def make_clip_id(batch_id: str, decision_index: int, rank: str) -> str:
    return hashlib.sha1(f"{batch_id}/{decision_index}/{rank}".encode()).hexdigest()[:10]


def pick_alt_instructions(instruction: str, rng: random.Random) -> list[str]:
    return rng.sample([p for p in PHRASES if p != instruction], 2)


def write_jsonl(path: Path, records: list[ClipRecord]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_jsonl(path: Path) -> list[ClipRecord]:
    with open(path) as f:
        return [ClipRecord(**json.loads(line)) for line in f if line.strip()]
