import random
from dataclasses import dataclass

from nuance.records import ClipRecord


@dataclass
class Item:
    item_id: str
    clip_id: str
    instruction: str
    group: str


def build_items(records: list[ClipRecord]) -> list[Item]:
    items = []
    for r in records:
        for k, ins in enumerate([r.instruction, *r.alt_instructions]):
            items.append(Item(f"{r.clip_id}:{k}", r.clip_id, ins, r.decision_id))
    return items


def order_items(items: list[Item], seed: int = 0, min_gap: int = 5) -> list[Item]:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    last: dict[str, int] = {}
    out: list[Item] = []
    while pool:
        n = len(out)
        left: dict[str, int] = {}
        for it in pool:
            left[it.group] = left.get(it.group, 0) + 1
        ok = [i for i, it in enumerate(pool) if n - last.get(it.group, -min_gap) >= min_gap]
        pick = max(ok, key=lambda i: (left[pool[i].group], -i)) if ok else None
        if pick is None:
            pick = max(range(len(pool)), key=lambda i: (n - last[pool[i].group], -i))
        it = pool.pop(pick)
        last[it.group] = n
        out.append(it)
    return out
