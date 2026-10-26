"""Policy actions walked and rendered as the clips raters score, for the reward model inside GRPO (nuance.train_policy).

RolloutPool (training env, stdlib only) runs N workers, each `python -m nuance.rollout worker` in the habitat env with a
persistent walker (`nuance.spot_walk serve` in the spot-loco env). A request is one decision point (nuance.actions
.decision_of) with a group of actions; per action the worker returns the clip, or why there is none:
  plan    via and goal (nuance.actions.to_world) snapped to the navmesh, the shortest route through them, checked like a
          candidate line (nuance.candidates.valid); people at the decision point stand on the navmesh while it plans
  walk    Spot walks the line in MuJoCo from the start speed to the action's speed (checked like generate_instructions)
  render  render_clip, framed like generate_instructions.render_scene

Smoke test on ideas8 (renders the rated candidates of one decision again from their actions):
    PYTHONPATH=. ~/miniconda3/envs/nuance/bin/python -m nuance.rollout smoke --batch data/batches/ctx-rate-02 \
        --out data/rollout-smoke
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HAB_PYTHON = os.path.expanduser(os.environ.get("NUANCE_HAB_PYTHON", "~/miniconda3/envs/nuance/bin/python"))
REPO = Path(__file__).resolve().parent.parent
SNAP_MAX_M = 0.5  # a via or goal farther than this from the navmesh is not somewhere Spot can go


class RolloutPool:
    def __init__(self, workers: int):
        env = {**os.environ, "PYTHONPATH": str(REPO), "JAX_PLATFORMS": "cpu"}
        self.procs = [subprocess.Popen([HAB_PYTHON, "-m", "nuance.rollout", "worker"], cwd=REPO, env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                      for _ in range(workers)]
        for p in self.procs:
            json.loads(p.stdout.readline())  # ready

    def run(self, groups: list) -> list:
        """groups: [(decision, [action], out_dir)]; per group, per action {"video": path or None, "why": str or None}.
        Each worker takes the next group in its current scene if there is one (a scene switch reloads the sim)."""
        pending = list(enumerate(groups))
        results = [None] * len(groups)
        lock = threading.Lock()

        def drive(proc):
            scene = None
            while True:
                with lock:
                    if not pending:
                        return
                    j = next((j for j, (_, g) in enumerate(pending) if g[0]["scene_id"] == scene), 0)
                    i, (decision, actions, out_dir) = pending.pop(j)
                scene = decision["scene_id"]
                proc.stdin.write(json.dumps({"decision": decision, "actions": actions, "out_dir": str(out_dir)}) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("rollout worker died")
                results[i] = json.loads(line)

        threads = [threading.Thread(target=drive, args=(p,)) for p in self.procs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if any(r is None for r in results):
            raise RuntimeError("a rollout worker failed")
        return results

    def close(self) -> None:
        for p in self.procs:
            p.stdin.close()
            p.wait()


# ---- worker (habitat env) -----------------------------------------------------------------------------------------

class _Worker:
    def __init__(self):
        from nuance import config as cfg
        self.sim = self.spot = self.scene = None
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        self.walker = subprocess.Popen([cfg.LOCO_PYTHON, "-m", "nuance.spot_walk", "serve"], cwd=REPO, env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        json.loads(self.walker.stdout.readline())

    def _scene(self, scene_id: str) -> None:
        from nuance.render import load_spot
        from nuance.scenes import make_sim
        if scene_id != self.scene:
            if self.sim is not None:
                self.sim.close()
            self.sim = make_sim(scene_id)
            self.spot, self.scene = load_spot(self.sim), scene_id

    def _plan(self, dec, a):
        import numpy as np
        from nuance.actions import to_world
        from nuance.candidates import chain, full, valid
        p = dec.p
        pts = [p]
        for xy in (a["via"], a["goal"]):
            q = to_world(p, dec.yaw, xy)
            s = np.array(self.sim.pathfinder.snap_point(q.astype(np.float32)), dtype=float)
            if not np.isfinite(s).all() or np.hypot(*(s - q)[[0, 2]]) > SNAP_MAX_M:
                return None, "off the walkable floor"
            pts.append(s)
        post = chain(self.sim, pts)
        if not valid(self.sim, dec, post, reverse=True):  # reverse: turning back is allowed (the "back" instructions)
            return None, "not a valid line"
        return full(dec, post), None

    def _walk(self, path, v0: float, v1: float, tmp: Path):
        from nuance import config as cfg
        from nuance.generate_instructions import _motion_problem
        from nuance.render import floor_grid, load_motion
        import numpy as np
        spec = {"policy_dir": cfg.SPOT_POLICY_DIR, "path": path.tolist(), "floor": floor_grid(self.sim, self.spot, path),
                "jobs": [{"name": "walk", "v0": v0, "v1": v1}]}
        (tmp / "jobs.json").write_text(json.dumps(spec))
        self.walker.stdin.write(json.dumps({"jobs": str(tmp / "jobs.json"), "out": str(tmp)}) + "\n")
        self.walker.stdin.flush()
        json.loads(self.walker.stdout.readline())
        z = np.load(tmp / "walk.npz")
        if bool(z["fell"]):
            return None, "fell"
        m = load_motion(tmp / "walk.npz")
        why = _motion_problem(self.sim, path, m, v0, v1)
        return (None, why) if why else (m, None)

    def handle(self, req: dict) -> dict:
        import numpy as np
        from types import SimpleNamespace
        from nuance import grounding as G
        from nuance.candidates import Decision, length
        from nuance.generate_instructions import _box_corners
        from nuance.render import render_clip, topdown_frame
        d, out = req["decision"], Path(req["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        self._scene(d["scene_id"])
        pre = np.array(d["pre"], dtype=float)
        dec = Decision(scene_id=d["scene_id"], pre=pre, post=pre[-1:], yaw=d["yaw"], v0=d["v0"])
        people = d.get("person") or []

        def add_people():
            return [G.add_person(self.sim, q["model"], np.array(q["at"]), q["yaw"]) for q in people]

        def drop(objs):
            for o in objs:
                G.remove_object(self.sim, o)

        # plan with the people on the navmesh; walk without them (render_scene's walks never saw them either)
        objs = add_people()
        if objs:
            G.recompute_navmesh(self.sim)
        try:
            plans = [self._plan(dec, a) for a in req["actions"]]
        finally:
            drop(objs)
        results = []
        ref_pts = [np.array(q) for o in (d.get("referent") or {}).get("objects", []) for q in
                   _box_corners(o["oid"], self.sim)] + [np.array(q["at"]) for q in people]
        with tempfile.TemporaryDirectory() as tmp:
            walks = []
            for k, ((path, why), a) in enumerate(zip(plans, req["actions"])):
                w = Path(tmp) / str(k)
                w.mkdir()
                walks.append(self._walk(path, d["v0"], a["speed"], w) if path is not None else (None, why))
        objs = add_people()
        try:
            for k, ((path, _), (m, why)) in enumerate(zip(plans, walks)):
                if m is None:
                    results.append({"video": None, "why": why})
                    continue
                frame = topdown_frame(SimpleNamespace(positions=np.concatenate(
                    [m.positions] + ([np.array(ref_pts)] if ref_pts else []))))
                video = out / f"{k}.mp4"
                render_clip(self.sim, self.spot, m, path, length(pre), video, frame=frame)
                results.append({"video": str(video), "why": None})
        finally:
            drop(objs)
            if objs:
                G.recompute_navmesh(self.sim)
        return {"results": results}


def worker() -> None:
    proto, sys.stdout = sys.stdout, sys.stderr  # habitat logs to stdout; it carries only replies
    w = _Worker()
    print("{}", file=proto, flush=True)
    for line in sys.stdin:
        try:
            reply = w.handle(json.loads(line))
        except Exception as e:  # one bad decision must not stall training; every action of it gets no clip
            n = len(json.loads(line)["actions"])
            reply = {"results": [{"video": None, "why": f"error: {e!r}"}] * n}
        print(json.dumps(reply), file=proto, flush=True)


def smoke(argv) -> None:
    import argparse
    from nuance.actions import action_of, decision_of
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--decision", default=None, help="decision_id; default: the first")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    recs = [json.loads(line) for line in open(Path(args.batch) / "clips.jsonl")]
    did = args.decision or recs[0]["decision_id"]
    recs = [r for r in recs if r["decision_id"] == did]
    actions = [action_of(r) for r in recs]
    pool = RolloutPool(args.workers)
    try:
        res = pool.run([(decision_of(recs[0]), actions, Path(args.out))])[0]["results"]
    finally:
        pool.close()
    for r, a, x in zip(recs, actions, res):
        print(json.dumps({"clip": r["clip_id"], "rank": r["candidate_rank"], "action": a, **x}))


if __name__ == "__main__":
    worker() if sys.argv[1] == "worker" else smoke(sys.argv[2:])
