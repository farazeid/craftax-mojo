#!/usr/bin/env bash
# Full benchmark sweep on one machine. Writes JSON lines to
# results/raw/<label>.jsonl. Intended to run inside the docker image on the
# FLAIR nodes (MOJO=mojo PYTHON=python) or locally via uv.
#
#   LABEL=l40s SKIP_CPU=1 bash scripts/bench_all.sh        # NVIDIA node
#   LABEL=m4pro SKIP_JAX_GPU=1 bash scripts/bench_all.sh   # Mac (no JAX GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
export MOJO="${MOJO:-uv run mojo}"
PY="${PYTHON:-uv run python}"
LABEL="${LABEL:-$(hostname -s)}"
OUT="results/raw/${LABEL}.jsonl"
mkdir -p results/raw
: > "$OUT"

GPU_SIZES="${GPU_SIZES:-1,16,256,1024,4096,16384,65536,262144,1048576}"
CPU_SIZES="${CPU_SIZES:-1,16,256,1024,4096,16384}"
JAX_GPU_SIZES="${JAX_GPU_SIZES:-1,16,256,1024,4096,16384,65536,262144,1048576}"
JAX_CPU_SIZES="${JAX_CPU_SIZES:-1,16,256,1024,4096}"

if [[ "$(uname)" == "Darwin" && -z "${SDKROOT:-}" ]]; then
  export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"  # see build.sh
fi
bash scripts/build.sh

echo "== correctness: GPU vs CPU trajectories"
$MOJO run -I . --fp-mode contract=off tests/test_gpu_cpu.mojo 2048 500 | tee -a results/raw/${LABEL}.check.txt

steps_for() { local n=$1; local s=$(( (1 << 24) / n )); (( s < 50 )) && s=50; (( s > 2000 )) && s=2000; echo $s; }

run_mojo() {
  local dev=$1 sizes=$2
  for n in ${sizes//,/ }; do
    ./build/bench_mojo "$dev" "$(steps_for "$n")" "$n" | grep '^{' | tee -a "$OUT"
  done
}

echo "== mojo"
if [[ "${SKIP_GPU:-0}" != 1 ]]; then
  run_mojo gpu "$GPU_SIZES"
  run_mojo gpu-naive "${NAIVE_SIZES:-1024,16384,262144}"
fi
[[ "${SKIP_CPU:-0}" != 1 ]] && run_mojo cpu "$CPU_SIZES"

[[ "${SKIP_JAX:-0}" == 1 ]] && { echo "wrote $OUT"; exit 0; }
echo "== jax"
if [[ "${SKIP_GPU:-0}" != 1 && "${SKIP_JAX_GPU:-0}" != 1 ]]; then
  $PY bench/bench_jax.py --mode optimistic --sizes "$JAX_GPU_SIZES" | tee -a "$OUT"
  $PY bench/bench_jax.py --mode autoreset --sizes "$JAX_GPU_SIZES" | tee -a "$OUT"
fi
if [[ "${SKIP_CPU:-0}" != 1 ]]; then
  # JAX on CPU is slow (auto-reset ~9k steps/s on an M4 Pro): fixed, short runs.
  for mode in optimistic autoreset; do
    JAX_PLATFORMS=cpu $PY bench/bench_jax.py --mode $mode --sizes "$JAX_CPU_SIZES" \
      --steps "${JAX_CPU_STEPS:-100}" --repeats 2 | tee -a "$OUT"
  done
fi
echo "wrote $OUT"
