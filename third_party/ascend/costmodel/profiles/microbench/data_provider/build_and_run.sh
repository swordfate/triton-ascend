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

# --- environment (kaixin conda for toolchain, CANN env for ccec + runtime) ---
conda activate kaixin 2>/dev/null || true
if [[ -f /data/kaixin/set_env.sh ]]; then
  source /data/kaixin/set_env.sh >/dev/null 2>&1
else
  source /home/kaixin/set_env.sh >/dev/null 2>&1
fi

# Template headers (RegBase/VecUtils.h, RegBase/Cumulative/SIMTCumsumCore.h, ...).
# Override by exporting INC=... if your catfood checkout lives elsewhere.
INC="${INC:-/data/kaixin/AscendNPU-IR-Dev/bishengir/lib/Template/include}"
TK="${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME unset - did set_env.sh run?}"

echo "INC = $INC"
echo "TK  = $TK"
[ -f "$INC/RegBase/VecUtils.h" ] || { echo "!! RegBase/VecUtils.h not found under INC"; exit 1; }

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

targets="${*:-tput concur2 meas simt_memory simt_gm_memory simt_gm_memory_pattern simt_shuffle simt_predicate transition}"
for t in $targets; do
  build_probe "$t"
done
for t in $targets; do
  run_probe "$t"
done
