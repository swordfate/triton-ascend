#!/usr/bin/env bash
set -eo pipefail
# Build the real-board SYS_CNT probes and the common host/wrapper.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ~/env_ascend.sh
INC="${INC:-$HOME/AscendNPU-IR-triton/bishengir/lib/Template/include}"

build_cce () {
  local dir="$1" name="$2"
  ( cd "$dir"
    ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
      -I"$INC" "$name.cce" -o "$name.o" )
}

build_cce "$ROOT/load/scalar_o1"  load_scalar_o1_syscnt
build_cce "$ROOT/load/scalar_o4"  load_scalar_o4_syscnt
build_cce "$ROOT/store/scalar_o1" store_scalar_o1_syscnt
build_cce "$ROOT/store/scalar_o4" store_scalar_o4_syscnt
build_cce "$ROOT/compute/scalar_add" scalar_add_syscnt
build_cce "$ROOT/load_vec_compute/scalar_o1_o4" load_vec_compute_syscnt

g++ -O2 compute/scalar_add/scalar_add_host.cpp -o compute/scalar_add/scalar_add_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl
g++ -O2 syscnt/syscnt_host.cpp -o syscnt/syscnt_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl
g++ -O2 syscnt/syscnt_rep_host.cpp -o syscnt/syscnt_rep_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl
chmod +x syscnt/syscnt_runner.sh syscnt/syscnt_compute_runner.sh
echo "SYSCNT_BUILD_DONE"
