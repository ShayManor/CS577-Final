"""Time the simulator per GRPO rollout: what one sampled action costs if it is walked in MuJoCo and rendered in Habitat.

Run from the repo root on ideas8 with the repo on PYTHONPATH:
    PYTHONPATH=. ~/miniconda3/envs/nuance/bin/python scripts/bench_sim_in_loop.py rollouts --out data/bench/single
    PYTHONPATH=. ~/miniconda3/envs/nuance/bin/python scripts/bench_sim_in_loop.py throughput --workers 4 --out data/bench/w4
    python scripts/bench_sim_in_loop.py plot --bench data/bench --out figures/bench_sim_in_loop.png  (local, matplotlib)
rollouts: one process walks and renders decisions sampled as nuance.generate samples them, one candidate speed each,
timing the floor height map, the walk as generate.py runs it (a nuance.spot_walk subprocess), the walk in a persistent
walker (load_walker once, then walk_job per rollout), and render_clip (split into chase camera search, frame drawing and
H.264 encoding). throughput: N rollouts processes with the persistent walker, started together after warm-up.
walk-server runs in the spot-loco env; rollouts starts it.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

RANKS = ("slower", "near", "faster")


def walk_server(argv) -> None:
    from nuance.config import SPOT_POLICY_DIR
    from nuance.spot_walk import load_walker, walk_job

    proto, sys.stdout = sys.stdout, sys.stderr  # walk() prints falls; keep stdout for replies
    t0 = time.perf_counter()
    w = load_walker(SPOT_POLICY_DIR)
    print(json.dumps({"ready_s": time.perf_counter() - t0}), file=proto, flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        spec = json.loads(Path(req["jobs"]).read_text())
        job = spec["jobs"][0]
        t = time.perf_counter()
        res = walk_job(w, spec["floor"], np.array(spec["path"]), job["v0"], job["v1"])
        walk_s = time.perf_counter() - t
        np.savez(Path(req["out"]) / f"{job['name']}.npz", **res)
        print(json.dumps({"walk_s": walk_s, "fell": bool(res["fell"])}), file=proto, flush=True)


class _TimedWriter:
    def __init__(self, writer, acc):
        self.writer, self.acc = writer, acc

    def append_data(self, img):
        t = time.perf_counter()
        self.writer.append_data(img)
        self.acc["encode_s"] += time.perf_counter() - t

    def close(self):
        t = time.perf_counter()
        self.writer.close()
        self.acc["encode_s"] += time.perf_counter() - t


def _instrument(acc: dict) -> None:
    """Split render_clip's time into the chase camera search and H.264 encoding (the rest is drawing frames)."""
    import imageio
    from nuance import render

    chase, get_writer = render._chase_poses, imageio.get_writer

    def timed_chase(*a, **k):
        t = time.perf_counter()
        out = chase(*a, **k)
        acc["chase_s"] += time.perf_counter() - t
        return out

    render._chase_poses = timed_chase
    imageio.get_writer = lambda *a, **k: _TimedWriter(get_writer(*a, **k), acc)


def rollouts(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="default: the first scene in configs/pilot_scenes.txt")
    ap.add_argument("--n", type=int, default=12, help="timed rollouts")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subprocess", action="store_true", help="skip the spot_walk subprocess timing")
    ap.add_argument("--barrier", default=None, help="touch <barrier>.ready_<seed> after warm-up, wait for <barrier>.go")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from nuance import config as cfg
    from nuance.generate import build_pool
    from nuance.render import floor_grid, load_motion, load_spot, render_clip, topdown_frame
    from nuance.scenes import make_sim

    scene = args.scene or Path("configs/pilot_scenes.txt").read_text().split()[0]
    out = Path(args.out)
    work = out / f"work_{args.seed}"
    work.mkdir(parents=True, exist_ok=True)
    acc = Counter()
    _instrument(acc)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    rng = random.Random(args.seed)

    t0 = time.perf_counter()
    sim = make_sim(scene, seed=args.seed)
    spot = load_spot(sim)
    make_sim_s = time.perf_counter() - t0
    pool = build_pool(sim, scene, args.warmup + args.n, rng, Counter())
    t0 = time.perf_counter()
    server = subprocess.Popen([cfg.LOCO_PYTHON, __file__, "walk-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env=env)
    server_ready_s = json.loads(server.stdout.readline())["ready_s"]
    server_start_s = time.perf_counter() - t0
    rows = []
    try:
        for k, d in enumerate(pool):
            if k == args.warmup and args.barrier:
                Path(f"{args.barrier}.ready_{args.seed}").touch()
                while not Path(f"{args.barrier}.go").exists():
                    time.sleep(0.05)
            rank = RANKS[k % len(RANKS)]
            row = {"k": k, "warmup": k < args.warmup, "scene": scene, "rank": rank, "v0": d.start_speed,
                   "v1": d.candidates[rank], "t_start": time.time()}
            t_all = time.perf_counter()
            t = time.perf_counter()
            floor = floor_grid(sim, spot, d.path)
            row["floor_s"] = time.perf_counter() - t
            jobs = {"policy_dir": cfg.SPOT_POLICY_DIR, "path": d.path.tolist(), "floor": floor,
                    "jobs": [{"name": rank, "v0": d.start_speed, "v1": d.candidates[rank]}]}
            jobs_file = work / "jobs.json"
            jobs_file.write_text(json.dumps(jobs))
            if not args.no_subprocess:
                t = time.perf_counter()
                subprocess.run([cfg.LOCO_PYTHON, "-m", "nuance.spot_walk", str(jobs_file), str(work)], check=True,
                               stdout=subprocess.DEVNULL, env=env)
                row["walk_subprocess_s"] = time.perf_counter() - t
            t = time.perf_counter()
            server.stdin.write(json.dumps({"jobs": str(jobs_file), "out": str(work)}) + "\n")
            server.stdin.flush()
            reply = json.loads(server.stdout.readline())
            row["walk_persistent_s"] = time.perf_counter() - t
            row["walk_job_s"] = reply["walk_s"]
            row["fell"] = reply["fell"]
            if not reply["fell"]:
                motion = load_motion(work / f"{rank}.npz")
                acc.clear()
                t = time.perf_counter()
                render_clip(sim, spot, motion, d.path, d.arc_length, work / "clip.mp4", frame=topdown_frame(motion))
                row["render_s"] = time.perf_counter() - t
                row["chase_s"], row["encode_s"] = acc["chase_s"], acc["encode_s"]
                row["clip_s"], row["frames"] = float(motion.times[-1]), len(motion.times)
                if k == args.warmup:
                    os.replace(work / "clip.mp4", out / f"clip_{args.seed}.mp4")  # kept for the reward model bench
            row["rollout_s"] = time.perf_counter() - t_all - row.get("walk_subprocess_s", 0.0)
            row["t_end"] = time.time()
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        server.stdin.close()
        server.wait()
        sim.close()
        for f in work.iterdir():
            f.unlink()
        work.rmdir()
    with open(out / f"rollouts_{args.seed}.jsonl", "w") as f:
        f.write(json.dumps({"meta": {"scene": scene, "make_sim_s": make_sim_s, "server_start_s": server_start_s,
                                     "server_ready_s": server_ready_s}}) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _gpu_mem_mib() -> int:
    """Memory in use on GPU 0 by everyone (Habitat's EGL contexts are missing from --query-compute-apps)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True).stdout
    return int(q.strip())


def throughput(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--n", type=int, default=10, help="timed rollouts per worker")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = Path("configs/pilot_scenes.txt").read_text().split()
    barrier = out / "barrier"
    procs = [subprocess.Popen([sys.executable, __file__, "rollouts", "--scene", scenes[i % len(scenes)], "--n",
                               str(args.n), "--warmup", str(args.warmup), "--seed", str(i), "--no-subprocess",
                               "--barrier", str(barrier), "--out", str(out)], stdout=subprocess.DEVNULL)
             for i in range(args.workers)]
    base = peak = _gpu_mem_mib()
    while not all(Path(f"{barrier}.ready_{i}").exists() for i in range(args.workers)):
        if any(p.poll() not in (None, 0) for p in procs):
            raise RuntimeError("a worker failed before warm-up ended")
        time.sleep(0.5)
    t0 = time.time()
    Path(f"{barrier}.go").touch()
    while any(p.poll() is None for p in procs):
        try:
            peak = max(peak, _gpu_mem_mib())
        except (ValueError, OSError):
            pass
        time.sleep(1.0)
    t1 = time.time()
    if any(p.returncode for p in procs):
        raise RuntimeError("a worker failed")
    done = sum(1 for i in range(args.workers) for line in open(out / f"rollouts_{i}.jsonl")
               if '"warmup": false' in line and '"fell": false' in line)
    res = {"workers": args.workers, "rollouts": done, "wall_s": t1 - t0, "rollouts_per_s": done / (t1 - t0),
           "gpu_mib_before": base, "peak_gpu_mib": peak}
    (out / "throughput.json").write_text(json.dumps(res) + "\n")
    for i in range(args.workers):
        Path(f"{barrier}.ready_{i}").unlink()
    Path(f"{barrier}.go").unlink()
    print(json.dumps(res), flush=True)


def _stats(x) -> tuple:
    return float(np.median(x)), float(np.percentile(x, 90))


def plot(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = Path(args.bench)
    rows = [json.loads(line) for line in open(bench / "single" / "rollouts_0.jsonl")][1:]
    rows = [r for r in rows if not r["warmup"] and not r["fell"]]
    stages = [("Floor map + walk", [r["floor_s"] + r["walk_persistent_s"] for r in rows], "#e07a5f"),
              ("Camera search", [r["chase_s"] for r in rows], "#3d5a80"),
              ("Draw frames", [r["render_s"] - r["chase_s"] - r["encode_s"] for r in rows], "#81b29a"),
              ("Encode", [r["encode_s"] for r in rows], "#f2cc8f")]
    tp = sorted((json.loads((d / "throughput.json").read_text()) for d in bench.glob("w*")
                 if (d / "throughput.json").exists()), key=lambda r: r["workers"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    bottom = 0.0
    for name, x, c in stages:
        m = float(np.median(x))
        a.bar(0, m, bottom=bottom, color=c, width=0.6)
        a.text(0.36, bottom + m / 2, f"{name}  {m:.1f} s", va="center", fontsize=9)
        bottom += m
    a.set_xlim(-0.5, 2.2)
    a.set_xticks([])
    a.set_ylabel("Seconds per rollout (median)")
    w = [r["workers"] for r in tp]
    b.plot(w, [3600 * r["rollouts_per_s"] for r in tp], "o-", color="#3d5a80")
    b.set_xticks(w)
    b.set_xlabel("Worker processes")
    b.set_ylabel("Rollouts per hour")
    b.set_ylim(0, None)
    best = max(tp, key=lambda r: r["rollouts_per_s"])
    b.annotate(f"64k rollouts: {64000 / best['rollouts_per_s'] / 3600:.0f} h", (best["workers"], 3600 * best["rollouts_per_s"]),
               xytext=(0, -22), textcoords="offset points", ha="right", fontsize=9)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    a.spines["bottom"].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    {"walk-server": walk_server, "rollouts": rollouts, "throughput": throughput, "plot": plot}[sys.argv[1]](sys.argv[2:])
