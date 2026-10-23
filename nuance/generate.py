import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nuance import config as cfg
from nuance.config import FPS, T_DECISION_S
from nuance.context import measure_context
from nuance.motion import required_path_length, trajectory
from nuance.records import ClipRecord, make_clip_id, pick_alt_instructions, write_jsonl
from nuance.render import floor_grid, load_motion, load_spot, render_clip, render_front, topdown_frame
from nuance.sampling import (DecisionCandidate, assign_clutter_buckets, assign_phrases, sample_candidates,
                             sample_start_speed, select_stratified, stratum)
from nuance.scenes import body_clear, make_sim, sample_path

RANKS = ("slower", "near", "faster")
MAX_TRIES_PER_CANDIDATE = 50


def _decision_index(traj) -> int:
    return min(int(np.searchsorted(traj.times, T_DECISION_S - 1e-9)), len(traj.times) - 1)


def walk_spot(sim, spot, d: DecisionCandidate, fail_file: Path):
    """Simulate Spot walking the decision's line at each candidate speed (nuance.spot_walk in the spot-loco env).

    Returns {rank: Motion}, or None if Spot fell at any speed; the walk's input is then kept in fail_file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        jobs = {"policy_dir": cfg.SPOT_POLICY_DIR, "path": d.path.tolist(), "floor": floor_grid(sim, spot, d.path),
                "jobs": [{"name": rank, "v0": d.start_speed, "v1": d.candidates[rank]} for rank in RANKS]}
        (Path(tmp) / "jobs.json").write_text(json.dumps(jobs))
        subprocess.run([cfg.LOCO_PYTHON, "-m", "nuance.spot_walk", f"{tmp}/jobs.json", tmp], check=True,
                       stdout=subprocess.DEVNULL, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        motions = {}
        for rank in RANKS:
            z = np.load(f"{tmp}/{rank}.npz")
            if bool(z["fell"]):
                fail_file.parent.mkdir(parents=True, exist_ok=True)
                fail_file.write_text(json.dumps(jobs))
                return None
            motions[rank] = load_motion(Path(tmp) / f"{rank}.npz")
    return motions


def build_pool(sim, scene_id: str, n: int, rng: random.Random, stats: Counter) -> list:
    pool = []
    tries = 0
    while len(pool) < n:
        tries += 1
        if tries > MAX_TRIES_PER_CANDIDATE * n:
            raise RuntimeError(f"{scene_id}: only {len(pool)}/{n} candidates after {tries - 1} tries")
        v0 = sample_start_speed(rng)
        cands = sample_candidates(v0, rng)
        path = sample_path(sim, required_path_length(v0), rng)
        if path is None:
            stats["no_path"] += 1
            continue
        if not body_clear(sim, path):
            stats["body_rejected"] += 1
            continue
        stats["accepted"] += 1
        traj = trajectory(path, v0, v0)
        i = _decision_index(traj)
        arc = v0 * T_DECISION_S
        pool.append(DecisionCandidate(scene_id=scene_id, path=path, arc_length=arc, position=traj.positions[i],
                                      yaw=float(traj.yaws[i]), start_speed=v0, candidates=cands,
                                      context=measure_context(sim, path, arc)))
    return pool


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-decisions", type=int, required=True)
    ap.add_argument("--pool-per-scene", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.scenes) as f:
        scenes = [line.strip() for line in f if line.strip()]
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()

    pool, stats = [], Counter()
    for k, scene_id in enumerate(scenes):
        sim = make_sim(scene_id, seed=args.seed)
        try:
            pool += build_pool(sim, scene_id, args.pool_per_scene, rng, stats)
        finally:
            sim.close()
        print(f"pool {k + 1}/{len(scenes)} {scene_id}: {dict(stats)} ({time.time() - t0:.0f} s)", flush=True)

    assign_clutter_buckets(pool)
    chosen = select_stratified(pool, args.n_decisions, rng)
    phrases = assign_phrases(len(chosen), rng)

    records = []
    for scene_id in scenes:
        todo = [(i, d) for i, d in enumerate(chosen) if d.scene_id == scene_id]
        if not todo:
            continue
        sim = make_sim(scene_id, seed=args.seed)
        try:
            spot = load_spot(sim)
            for i, d in todo:
                decision_id = f"d{i:03d}"
                front = f"frames/{decision_id}.png"
                render_front(sim, spot, d.position, d.yaw, out / front)
                motions = walk_spot(sim, spot, d, out / "falls" / f"{decision_id}.json")
                if motions is None:
                    stats["fell"] += 1
                    print(f"skipped {decision_id} {scene_id}: Spot fell in simulation", flush=True)
                    continue
                # One camera for all candidates of a decision, so zoom does not differ between them.
                frame = topdown_frame(max(motions.values(), key=lambda m: m.arc[-1]))
                for rank in RANKS:
                    clip_id = make_clip_id(args.batch_id, i, rank)
                    video = f"videos/{clip_id}.mp4"
                    motion = motions[rank]
                    render_clip(sim, spot, motion, d.path, d.arc_length, out / video, frame=frame)
                    records.append(ClipRecord(
                        clip_id=clip_id, batch_id=args.batch_id, decision_id=decision_id, action_type="speed",
                        scene_id=scene_id, path=d.path.tolist(), decision_position=[float(x) for x in d.position],
                        decision_yaw=d.yaw, context=d.context, start_speed=d.start_speed,
                        candidate_speed=d.candidates[rank], candidate_rank=rank, instruction=phrases[i],
                        alt_instructions=pick_alt_instructions(phrases[i], rng), video=video, front_frame=front,
                        fps=FPS, duration_s=float(motion.times[-1]), t_decision_s=T_DECISION_S))
                print(f"rendered {decision_id} {scene_id} ({time.time() - t0:.0f} s)", flush=True)
        finally:
            sim.close()

    records.sort(key=lambda r: (r.decision_id, RANKS.index(r.candidate_rank)))
    write_jsonl(out / "clips.jsonl", records)
    strata = Counter(stratum(d) for d in chosen)
    manifest = {
        "batch_id": args.batch_id, "seed": args.seed, "scenes": scenes, "n_decisions": len(chosen),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strata": [{"clutter": c, "location": loc, "ahead": a, "start_speed": v, "count": n}
                   for (c, loc, a, v), n in sorted(strata.items())],
        "phrases": dict(Counter(phrases)),
        "pool": dict(stats),
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(cfg).items() if k.isupper()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done: {len(records)} clips in {time.time() - t0:.0f} s; pool {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
