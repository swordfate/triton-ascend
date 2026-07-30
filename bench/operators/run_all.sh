#!/bin/bash
# Run all costmodel operator benchmarks.
#
# Usage:
#   bash bench/operators/run_all.sh              # default top_k=5
#   TRITON_COSTMODEL_TOP_K=3 bash bench/operators/run_all.sh

set -euo pipefail

TOP_K="${TRITON_COSTMODEL_TOP_K:-5}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$0")/../.."  # repo root

echo "============================================"
echo "Costmodel Operator Benchmarks (top_k=${TOP_K})"
echo "============================================"
echo ""

for bench in \
    bench/operators/bench_gelu.py \
    bench/operators/bench_silu.py \
    bench/operators/bench_swiglu.py \
    bench/operators/bench_matmul.py \
    bench/operators/bench_group_gemm.py \
    bench/operators/bench_sdpa.py \
    bench/operators/bench_quant.py \
    bench/operators/bench_lightning_indexer.py \
    bench/operators/bench_int8_gemm.py \
    bench/operators/bench_convolution.py \
    bench/operators/bench_diffution_attention.py \
    bench/operators/bench_fused_ce.py; do
    echo ""
    echo "############################################"
    echo "# $(basename $bench)"
    echo "############################################"
    echo ""
    TRITON_COSTMODEL_TOP_K="${TOP_K}" python "$bench"
    echo ""
done

echo "============================================"
echo "All benchmarks complete."
echo "Results saved to bench/operators/results_*.json"
echo "============================================"
