// Repeated-launch host for real-board SYS_CNT probes.
// usage: syscnt_rep_host <obj> <func> <reps> <memset_gm 0|1>
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

struct Args { void *out; void *gm; int iters; int mode; };

static std::vector<char> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

#define CHECK(call) do { auto e = (call); if (e != 0) { \
  std::fprintf(stderr, "%s failed: %d\n", #call, (int)e); std::exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <obj> <func> <reps> <memset_gm 0|1>\n", argv[0]);
    return 2;
  }
  int reps = std::atoi(argv[3]);
  int do_memset = std::atoi(argv[4]);
  std::vector<char> bin = readFile(argv[1]);
  if (bin.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

  CHECK(aclInit(nullptr));
  CHECK(rtSetDevice(0));
  aclrtBinary binary = aclrtCreateBinary(bin.data(), bin.size());
  if (!binary) { std::fprintf(stderr, "aclrtCreateBinary failed\n"); return 1; }
  aclrtBinHandle handle = nullptr;
  CHECK(aclrtBinaryLoad(binary, &handle));
  aclrtFuncHandle fn = nullptr;
  CHECK(aclrtBinaryGetFunction(handle, argv[2], &fn));

  rtStream_t stream;
  CHECK(rtStreamCreate(&stream, 0));
  void *out = nullptr;
  void *gm = nullptr;
  constexpr size_t GM_BYTES = 16 * 1024 * 1024;
  CHECK(rtMalloc(&out, 4096, RT_MEMORY_HBM, 0));
  CHECK(rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0));

  aclrtLaunchKernelAttr attr = {};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attr.value.dynUBufSize = 192 * 1024;
  aclrtLaunchKernelCfg cfg = {&attr, 1};

  for (int r = 0; r < reps; ++r) {
    if (do_memset) CHECK(rtMemset(gm, GM_BYTES, 0, GM_BYTES));
    CHECK(rtMemset(out, 4096, 0, 4096));
    Args args{out, gm, 1, 0};
    CHECK(aclrtLaunchKernelWithHostArgs(fn, 1, stream, &cfg,
                                        &args, sizeof(args), nullptr, 0));
    CHECK(rtStreamSynchronize(stream));
    long long vals[4] = {0};
    CHECK(rtMemcpy(vals, sizeof(vals), out, sizeof(vals),
                   RT_MEMCPY_DEVICE_TO_HOST));
    std::printf("REP %d %s v0=%lld v1=%lld\n", r, argv[2], vals[0], vals[1]);
    std::fflush(stdout);
  }

  CHECK(rtFree(gm));
  CHECK(rtFree(out));
  CHECK(rtStreamDestroy(stream));
  CHECK(aclrtBinaryUnLoad(handle));
  CHECK(aclrtDestroyBinary(binary));
  CHECK(rtDeviceReset(0));
  CHECK(aclFinalize());
  return 0;
}
