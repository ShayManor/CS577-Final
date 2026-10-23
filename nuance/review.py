import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

from nuance.records import read_jsonl

ROW_H = 240
RANKS = ("slower", "near", "faster")


def _frame_at(video: Path, t: float, fps: int) -> np.ndarray:
    with imageio.get_reader(str(video), "ffmpeg") as reader:
        return reader.get_data(int(round(t * fps)))


def _fit(img: np.ndarray) -> Image.Image:
    im = Image.fromarray(np.asarray(img)[..., :3])
    return im.resize((round(im.width * ROW_H / im.height), ROW_H))


def contact_sheet(batch: Path) -> Path:
    records = read_jsonl(batch / "clips.jsonl")
    decisions = {}
    for r in records:
        decisions.setdefault(r.decision_id, {})[r.candidate_rank] = r
    rows = []
    for decision_id in sorted(decisions):
        clips = decisions[decision_id]
        first = clips[RANKS[0]]
        tiles = [_fit(imageio.imread(batch / first.front_frame))]
        tiles += [_fit(_frame_at(batch / clips[k].video, first.t_decision_s, first.fps)) for k in RANKS]
        row = Image.new("RGB", (sum(t.width for t in tiles), ROW_H))
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += t.width
        c = first.context
        label = (f"{decision_id} {first.scene_id} '{first.instruction}' v0={first.start_speed:.2f} "
                 f"{' '.join(f'{k}={clips[k].candidate_speed:.2f}' for k in RANKS)} | "
                 f"{c['clutter']} {c['location']} {c['ahead']}")
        d = ImageDraw.Draw(row)
        d.rectangle([0, 0, d.textlength(label) + 8, 14], fill=(0, 0, 0))
        d.text((4, 1), label, fill=(255, 255, 255))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), ROW_H * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * ROW_H))
    out = batch / "contact_sheet.png"
    sheet.save(out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    print(contact_sheet(Path(ap.parse_args(argv).batch)))


if __name__ == "__main__":
    main()
