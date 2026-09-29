#!/usr/bin/env bash
set -eo pipefail
# Build the scalar-add CAModel probes.
cd "$(dirname "$0")"
source ~/env_ascend.sh
INC="${INC:-$HOME/AscendNPU-IR-triton/bishengir/lib/Template/include}"

ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" scalar_add.cce -o scalar_add.o

g++ -O2 scalar_add_host.cpp -o scalar_add_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl

echo "BUILD DONE: $(pwd)/scalar_add.o $(pwd)/scalar_add_host"
