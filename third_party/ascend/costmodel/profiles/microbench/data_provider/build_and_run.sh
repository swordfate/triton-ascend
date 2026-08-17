#!/usr/bin/env bash
# Build + run the Ascend SIMT probes ON THE BOARD (dav-c310):
#   meas             -> SIMT launch overhead (empty-launch / SIMT-scan / SIMD-scan)
#   tput             -> saturated peak throughput SIMD vs SIMT + #independent warps
#   simt_memory      -> SIMT UB memory throughput
#   simt_gm_memory   -> SIMT global memory throughput (contiguous baseline)
#   simt_gm_memory_pattern -> SIMT global memory pattern sweep (contiguous/stride/gather)
#   simt_shuffle     -> SIMT shuffle dependent latency and ILP4 throughput
#   simt_predicate   -> SIMT masked/predicated execution sweep
#   transition       -> SIMD/SIMT boundary harness
# Run this in a LOGIN shell on triton_a5 from the directory holding the sources:
#     bash -lc './build_and_run.sh'
#     bash -lc './build_and_run.sh simt_predicate'
# (a login shell is required so conda + CANN env are set up; do NOT hack LD_LIBRARY_PATH).
set -e

# --- environment (wj_autoscope conda for toolchain, CANN env for ccec + runtime) ---
conda activate wj_autoscope 2>/dev/null || true
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh

# Template headers (RegBase/VecUtils.h, RegBase/Cumulative/SIMTCumsumCore.h, ...).
# Override by exporting INC=... if your checkout lives elsewhere.
INC="${INC:-/home/c00946898/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/lib/Template/include}"
TK="${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME unset - did set_env.sh run?}"

echo "INC = $INC"
echo "TK  = $TK"
if [ -f "$INC/RegBase/VecUtils.h" ]; then
  :
elif [ -f "$INC/Vector/VecUtils.h" ]; then
  :
else
  echo "!! neither RegBase/VecUtils.h nor Vector/VecUtils.h found under INC"
  exit 1
fi

# build one probe: <name>.cce -> <name>.o (device) and <name>_host.cpp -> <name>_host (host)
build_probe() {
  local name="$1"
  echo "--- building $name.o (device) ---"
  ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
       -I"$INC" "$name.cce" -o "$name.o"
  echo "--- building ${name}_host (host) ---"
  g++ -O2 "${name}_host.cpp" -o "${name}_host" \
      -I"$TK/x86_64-linux/pkg_inc" -I"$TK/include" \
      -L"$TK/lib64" -lruntime -lascendcl
}

run_probe() {
  local name="$1"
  echo "=== running $name ==="
  "./${name}_host"
  echo
}

# Default to SIMT-only probes that compile with the current AscendNPU-IR header
# layout.  The legacy SIMD probes (tput/concur2/meas/transition) need the old
# RegBase/VecUtils.h API and are only built when explicitly requested.
targets="${*:-simt_predicate simt_gm_memory_pattern simt_gm_memory simt_shuffle simt_memory}"
for t in $targets; do
  build_probe "$t"
done
for t in $targets; do
  run_probe "$t"
done
