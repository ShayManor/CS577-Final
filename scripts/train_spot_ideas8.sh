#!/usr/bin/env bash
# Train the Spot walking policy on ideas8, detached, with MuJoCo Warp's per-step solver warnings filtered out of
# the log (they otherwise write ~1 GB every 10 minutes). Run from the repo root: bash scripts/train_spot_ideas8.sh
set -euo pipefail
OUT=${1:-data/policies/spot_joystick}
STEPS=${2:-150000000}
setsid nohup bash -c "$HOME/miniconda3/envs/spot-loco/bin/python scripts/train_spot_policy.py --out $OUT \
  --num-timesteps $STEPS 2>&1 | grep --line-buffered -v -e 'iterations limit reached' -e 'To disable the print' \
  > data/train_spot.log" < /dev/null > /dev/null 2>&1 &
echo "training -> $OUT, log data/train_spot.log"
