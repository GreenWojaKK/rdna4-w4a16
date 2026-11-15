#!/bin/bash
# Builds the device kernels, then checks the record reader's formatting, lints and tests. Runs under Linux or WSL.
# The checks on the device, tools/check_kernels.py and tools/check_dev.py, need a GPU and the d3-ultra-INT4 checkpoint
# and run separately.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
"$root/tools/build_kernels.sh"
export PATH="$HOME/.cargo/bin:$PATH"
# The reader's build stays out of the folder a build under Windows uses.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/rdna4-w4a16-record}"
cd "$root/record"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
