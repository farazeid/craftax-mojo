#!/usr/bin/env bash
# Build the Python extension and the benchmark binary into build/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# macOS 27's CommandLineTools SDK ships .tbd stubs Mojo's linker can't parse;
# link against Xcode's SDK instead.
if [[ "$(uname)" == "Darwin" && -z "${SDKROOT:-}" ]]; then
  SDKROOT="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
  export SDKROOT
fi

MOJO="${MOJO:-uv run mojo}"
# xcrun --sdk macosx resolves to Xcode (not the CLT) once xcode-select points there.
$MOJO build -I . python/craftax_mojo_ext.mojo --emit shared-lib -o build/craftax_mojo_ext.so
if [[ -f bench/bench_mojo.mojo ]]; then
  $MOJO build -I . bench/bench_mojo.mojo -o build/bench_mojo
fi
echo "built: $(ls build)"
