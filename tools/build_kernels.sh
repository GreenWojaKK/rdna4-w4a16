#!/bin/bash
# Builds the device kernels and their developer mode into dist/librdna4_w4a16.so with the HIP compiler of the ROCm SDK
# that a Python environment holds. RDNA4_W4A16_PYTHON names that environment's interpreter, `python3` unless set.
# RDNA4_W4A16_GPU_ARCH names the device architecture, gfx1201 unless set.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python="${RDNA4_W4A16_PYTHON:-python3}"
sdk=$("$python" -I -c 'import os, _rocm_sdk_core; print(os.path.dirname(_rocm_sdk_core.__file__))')
arch="${RDNA4_W4A16_GPU_ARCH:-gfx1201}"
mkdir -p "$root/dist"
objects=$(mktemp -d)
trap 'rm -rf "$objects"' EXIT
export ROCM_PATH="$sdk" HIP_DEVICE_LIB_PATH="$sdk/lib/llvm/amdgcn/bitcode" DEVICE_LIB_PATH="$sdk/lib/llvm/amdgcn/bitcode"
flags=(-DRDNA4_W4A16_BUILD -fvisibility=hidden -O3 -std=c++20 -fPIC -I"$sdk/include")
"$sdk/bin/hipcc" "${flags[@]}" -fno-gpu-rdc --offload-arch="$arch" -c "$root/kernels/int4_linear.hip" -o "$objects/int4_linear.o"
# The developer mode runs on the host only, and marks its spans and timed products as ROCTx ranges.
c++ "${flags[@]}" -D__HIP_PLATFORM_AMD__ -DRDNA4_W4A16_ROCTX=1 -Wall -Wextra -c "$root/kernels/instrument.cpp" -o "$objects/instrument.o"
# The SDK holds its libraries under their versioned names only. A process that loads this library has the HIP runtime
# loaded already; the run path lets it find ROCTx beside it.
c++ -shared "$objects/int4_linear.o" "$objects/instrument.o" "$sdk"/lib/libamdhip64.so.[0-9]* "$sdk"/lib/libroctx64.so.[0-9]* \
  -Wl,-rpath,"$sdk/lib" -o "$root/dist/librdna4_w4a16.so"
