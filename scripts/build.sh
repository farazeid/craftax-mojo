#!/usr/bin/env bash
# Build the Python extension and the benchmark binary into build/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# macOS 27's CommandLineTools SDK ships .tbd stubs Mojo's linker can't parse;
# link against Xcode's SDK instead (`xcrun --sdk macosx` resolves to Xcode
# once xcode-select points there).
if [[ "$(uname)" == "Darwin" && -z "${SDKROOT:-}" ]]; then
  SDKROOT="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
  export SDKROOT
fi

MOJO="${MOJO:-uv run mojo}"
# No FMA contraction: keeps CPU and GPU builds bit-identical (ptxas and the
# CPU backend otherwise fuse different multiply-adds in world generation).
FLAGS=(-I . --fp-mode contract=off)
$MOJO build "${FLAGS[@]}" python/craftax_mojo_ext.mojo --emit shared-lib -o build/craftax_mojo_ext.so
if [[ -f bench/bench_mojo.mojo ]]; then
  $MOJO build "${FLAGS[@]}" bench/bench_mojo.mojo -o build/bench_mojo
fi
echo "built: $(ls build)"
