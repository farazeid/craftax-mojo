#!/usr/bin/env bash
# Phase 2 sweep: end-to-end PPO, JAX vs Mojo, on one GPU. Writes
# results/train/<label>-<impl>-<run>.json. Run inside the docker image:
#
#   LABEL=l40s bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="${LABEL:-$(hostname -s)}"
OUT=results/train
mkdir -p "$OUT" build
STEPS="${STEPS:-5e7}"
SEEDS="${SEEDS:-0 1 2}"

mojo build -I . --fp-mode contract=off train/ppo_mojo.mojo -o build/ppo_mojo

# baxtub's configuration (512 envs), several seeds.
for seed in $SEEDS; do
  ./build/ppo_mojo --total-steps "$STEPS" --seed "$seed" --out "$OUT/$LABEL-mojo-512-s$seed.json" | head -1
  python train/ppo_jax.py --total-steps "$STEPS" --seed "$seed" --out "$OUT/$LABEL-jax-512-s$seed.json" | head -1
done

# More envs per batch (same steps per env, 8x the batch).
./build/ppo_mojo --total-steps "$STEPS" --num-envs 4096 --out "$OUT/$LABEL-mojo-4096-s0.json" | head -1
python train/ppo_jax.py --total-steps "$STEPS" --num-envs 4096 --seed 0 --out "$OUT/$LABEL-jax-4096-s0.json" | head -1

# Full float32 matmuls instead of TF32 (both sides).
./build/ppo_mojo --total-steps 1e7 --precision highest --out "$OUT/$LABEL-mojo-512-fp32.json" | head -1
python train/ppo_jax.py --total-steps 1e7 --precision highest --out "$OUT/$LABEL-jax-512-fp32.json" | head -1
echo done
