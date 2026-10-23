import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rater.queue import Item

SCHEMA = """CREATE TABLE IF NOT EXISTS ratings(
    rater TEXT, item_id TEXT, clip_id TEXT, instruction TEXT, score INTEGER NULL,
    flagged INTEGER, created_at TEXT, UNIQUE(rater, item_id))"""


class RatingsDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def _insert(self, rater: str, item: Item, score, flagged: int) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO ratings VALUES (?,?,?,?,?,?,?)",
                (rater, item.item_id, item.clip_id, item.instruction, score, flagged, now),
            )
            return cur.rowcount == 1

    def rate(self, rater: str, item: Item, score: int) -> bool:
        return self._insert(rater, item, score, 0)

    def flag(self, rater: str, item: Item) -> None:
        with self._conn() as c:
            c.execute("UPDATE ratings SET flagged=1 WHERE rater=? AND item_id=?", (rater, item.item_id))
        self._insert(rater, item, None, 1)

    def done_item_ids(self, rater: str) -> set[str]:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT item_id FROM ratings WHERE rater=?", (rater,))}

    def flagged_clip_ids(self, rater: str) -> set[str]:
        with self._conn() as c:
            return {r[0] for r in c.execute(
                "SELECT clip_id FROM ratings WHERE rater=? AND flagged=1", (rater,))}
