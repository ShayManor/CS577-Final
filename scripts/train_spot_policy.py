"""Train MuJoCo Playground's Spot joystick walking policy for the speeds NUANCE clips use.

Runs in the spot-loco conda env (python 3.11, jax[cuda12], mujoco, playground) on ideas8:
    ~/miniconda3/envs/spot-loco/bin/python scripts/train_spot_policy.py --out data/policies/spot_joystick
Writes params.pkl (brax PPO params) and config.json (the env config the policy was trained with).
"""
import argparse
import functools
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground.config import locomotion_params

ENV = "SpotFlatTerrainJoystick"
# Clips only walk forward along a line and turn to follow it: forward speed up to Spot's top speed (1.6 m/s),
# a little sideways slack, and turning fast enough for the line's corners.
COMMANDS = {"lin_vel_x": [0.0, 1.6], "lin_vel_y": [-0.3, 0.3], "ang_vel_yaw": [-1.2, 1.2]}

# Rough terrain: home floors are flat except for thresholds (narrow raised strips) and rugs or level changes
# (raised patches), up to BUMP_MAX_M. Spot spawns on a flat pad at the center and walks onto them.
TERRAIN_HALF_M = 10.0
TERRAIN_RES_M = 0.05
BUMP_MAX_M = 0.06
N_STRIPS = 250
N_PATCHES = 120
PAD_M = 1.0
ROUGH_NJMAX = 96
ROUGH_CONTACTS_PER_ENV = 12  # naconmax is the contact budget summed over all parallel robots
ROUGH_XML = """
<mujoco model="spot rough home floor">
  <include file="spot_mjx_feetonly.xml"/>
  <asset>
    <hfield name="floor" file="nuance_floor.png" size="{half} {half} {elev} 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="floor" contype="1" conaffinity="0" priority="1" friction="0.6"
      condim="3"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home" qpos="0 0 0.46 1 0 0 0 0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8"
      ctrl="0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8"/>
  </keyframe>
</mujoco>
"""


def rough_floor(rng: np.random.Generator) -> np.ndarray:
    n = int(round(2 * TERRAIN_HALF_M / TERRAIN_RES_M)) + 1
    h = np.zeros((n, n))
    for _ in range(N_STRIPS):
        w, length = rng.uniform(0.05, 0.15), rng.uniform(0.8, 2.0)
        if rng.random() < 0.5:
            w, length = length, w
        x, y = rng.uniform(-TERRAIN_HALF_M, TERRAIN_HALF_M, 2)
        _raise(h, x, y, w, length, rng.uniform(0.01, BUMP_MAX_M))
    for _ in range(N_PATCHES):
        x, y = rng.uniform(-TERRAIN_HALF_M, TERRAIN_HALF_M, 2)
        _raise(h, x, y, rng.uniform(1.0, 3.0), rng.uniform(1.0, 3.0), rng.uniform(0.005, BUMP_MAX_M))
    c, r = n // 2, int(PAD_M / TERRAIN_RES_M)
    h[c - r:c + r + 1, c - r:c + r + 1] = 0.0
    return h


def _raise(h, x, y, wx, wy, dz):
    n = h.shape[0]
    to = lambda v: int(round((v + TERRAIN_HALF_M) / TERRAIN_RES_M))
    i0, i1 = max(to(x - wx / 2), 0), min(to(x + wx / 2), n - 1)
    j0, j1 = max(to(y - wy / 2), 0), min(to(y + wy / 2), n - 1)
    h[j0:j1 + 1, i0:i1 + 1] = np.maximum(h[j0:j1 + 1, i0:i1 + 1], dz)


def use_rough_terrain(out: Path) -> None:
    """Point Playground's Spot joystick task at a rough floor instead of its flat plane."""
    h = rough_floor(np.random.default_rng(0))
    img = (np.flipud(h) / BUMP_MAX_M * 255).round().astype(np.uint8)  # MuJoCo reads PNG row 0 as +y
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    np.save(out / "terrain.npy", h)
    xml = out / "scene_rough.xml"
    xml.write_text(ROUGH_XML.format(half=TERRAIN_HALF_M, elev=BUMP_MAX_M))
    get_assets = spot_base.get_assets
    spot_base.get_assets = lambda: get_assets() | {"nuance_floor.png": buf.getvalue()}
    spot_joystick.consts.task_to_xml = lambda task: xml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--terrain", choices=["flat", "rough"], default="rough")
    # Rough-terrain contact buffers do not fit 8192 parallel robots in the RTX 3060's 12 GB.
    ap.add_argument("--num-envs", type=int, default=4096)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.terrain == "rough":
        use_rough_terrain(out)

    env_cfg = registry.get_default_config(ENV)
    for k, v in COMMANDS.items():
        env_cfg.command_config[k] = v
    if args.terrain == "rough":
        # A foot on a height field makes several contacts; the flat-plane limits drop them ("nefc overflow").
        env_cfg.njmax = ROUGH_NJMAX
        env_cfg.naconmax = ROUGH_CONTACTS_PER_ENV * args.num_envs
    env = registry.load(ENV, config=env_cfg)
    eval_env = registry.load(ENV, config=env_cfg)

    ppo_cfg = locomotion_params.brax_ppo_config(ENV)
    if args.num_timesteps:
        ppo_cfg.num_timesteps = args.num_timesteps
    ppo_cfg.num_envs = args.num_envs
    params = dict(ppo_cfg)
    net = params.pop("network_factory")
    t0 = time.time()

    def progress(step, metrics):
        print(f"step {step} reward {metrics.get('eval/episode_reward', float('nan')):.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    make_inference_fn, policy_params, _ = ppo.train(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training, progress_fn=progress,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net), **params)

    with open(out / "params.pkl", "wb") as f:
        pickle.dump(policy_params, f)
    (out / "config.json").write_text(json.dumps({"env": ENV, "terrain": args.terrain, "env_config": env_cfg.to_dict(),
                                                  "network": {k: list(v) if isinstance(v, tuple) else v
                                                              for k, v in net.items()}}, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
