# Builds the device kernels and their developer mode into dist\rdna4_w4a16.dll with the HIP compiler of the ROCm SDK
# that a Python environment holds. RDNA4_W4A16_PYTHON names that environment's interpreter, `python` unless set.
# RDNA4_W4A16_GPU_ARCH names the device architecture, gfx1201 unless set. Needs the Visual Studio C++ build tools and the
# Windows SDK, which the compiler finds on its own. A process that loads the DLL finds the HIP runtime and the DLLs it
# needs in the SDK's bin folder, which goes on its PATH.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:RDNA4_W4A16_PYTHON) { $env:RDNA4_W4A16_PYTHON } else { 'python' }
$sdk = & $python -I -c 'import os, _rocm_sdk_core; print(os.path.dirname(_rocm_sdk_core.__file__))'
if ($LASTEXITCODE -ne 0) { throw "the ROCm SDK is not importable from $python" }
$arch = if ($env:RDNA4_W4A16_GPU_ARCH) { $env:RDNA4_W4A16_GPU_ARCH } else { 'gfx1201' }
$dist = Join-Path $root 'dist'
New-Item -ItemType Directory -Force $dist | Out-Null
$objects = Join-Path ([IO.Path]::GetTempPath()) ("rdna4_w4a16_" + [Guid]::NewGuid())
New-Item -ItemType Directory $objects | Out-Null
$env:HIP_PATH = $sdk
$env:ROCM_PATH = $sdk
$env:HIP_DEVICE_LIB_PATH = "$sdk\lib\llvm\amdgcn\bitcode"
$flags = @('-O3', '-std=c++20', "-I$sdk\include")

# A compiler's warnings on stderr are not failures; its exit code tells.
function Run([string]$program, [string[]]$arguments) {
    $ErrorActionPreference = 'Continue'
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { throw "$program failed with exit code $LASTEXITCODE" }
}

try {
    Run "$sdk\bin\hipcc.exe" ($flags + @('-fno-gpu-rdc', "--offload-arch=$arch", '-c', "$root\kernels\int4_linear.hip", '-o', "$objects\int4_linear.o"))
    # The developer mode runs on the host only. The SDK holds no ROCTx, so its spans are not marked for profilers. The
    # C runtime's advice to use its own variants of standard functions such as fopen is turned off.
    Run "$sdk\lib\llvm\bin\clang++.exe" ($flags + @('-D__HIP_PLATFORM_AMD__', '-D_CRT_SECURE_NO_WARNINGS', '-Wall', '-Wextra', '-c', "$root\kernels\instrument.cpp", '-o', "$objects\instrument.o"))
    Run "$sdk\lib\llvm\bin\clang++.exe" @('-shared', '-fuse-ld=lld', "$objects\int4_linear.o", "$objects\instrument.o", "$sdk\lib\amdhip64.lib", '-o', "$dist\rdna4_w4a16.dll")
} finally {
    Remove-Item -Recurse -Force $objects
}
