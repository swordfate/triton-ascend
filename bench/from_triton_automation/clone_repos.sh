#!/bin/bash
# Clone external Triton kernel repos for bench reference.
# Run this script ONCE on the NPU server before using bench/operators/.
#
# Usage:
#   bash bench/from_triton_automation/clone_repos.sh
#
# The cloned repos are gitignored — they stay local only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cloning external Triton kernel repos ==="
echo "Target directory: $SCRIPT_DIR"
echo ""

# LigerKernel (private, needs auth)
if [ ! -d LigerKernel ]; then
    echo "[1/4] Cloning LigerKernel..."
    git clone https://zhy0212:dpWy8HTyejsDuqKQLiTyBqCw@gitcode.com/TritonAscendTest/LigerKernel.git
else
    echo "[1/4] LigerKernel already exists, skipping."
fi

# Q2TritonKernel (private, needs auth)
if [ ! -d Q2TritonKernel ]; then
    echo "[2/4] Cloning Q2TritonKernel..."
    git clone https://zhiliangtang0727:hpyNCursTuTNMY1sq4bNxNTw@gitcode.com/TritonAscendTest/Q2TritonKernel.git
else
    echo "[2/4] Q2TritonKernel already exists, skipping."
fi

# mojo_opset (public, GitHub)
if [ ! -d mojo_opset ]; then
    echo "[3/4] Cloning mojo_opset..."
    git clone -b master https://github.com/XPU-Forces/mojo_opset.git
else
    echo "[3/4] mojo_opset already exists, skipping."
fi

# triton-ascend-kernels (public, gitcode)
if [ ! -d triton-ascend-kernels ]; then
    echo "[4/4] Cloning triton-ascend-kernels..."
    git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
else
    echo "[4/4] triton-ascend-kernels already exists, skipping."
fi

echo ""
echo "=== All repos ready ==="
echo ""
echo "Now you can run the costmodel benchmarks:"
echo "  bash bench/operators/run_all.sh"
