// Generic host for real-board SYS_CNT probes.
// usage: syscnt_host <obj> <func> [func ...]
// Launches each func once with args (out, gm, iters=1, mode=0), then prints
// the first 8 long-long outputs of `out`.
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
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <obj> <func> [func ...]\n", argv[0]);
    return 2;
  }
  std::vector<char> bin = readFile(argv[1]);
  if (bin.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

  CHECK(aclInit(nullptr));
  CHECK(rtSetDevice(0));
  aclrtBinary binary = aclrtCreateBinary(bin.data(), bin.size());
  if (!binary) { std::fprintf(stderr, "aclrtCreateBinary failed\n"); return 1; }
  aclrtBinHandle handle = nullptr;
  CHECK(aclrtBinaryLoad(binary, &handle));

  rtStream_t stream;
  CHECK(rtStreamCreate(&stream, 0));
  void *out = nullptr;
  void *gm = nullptr;
  constexpr size_t GM_BYTES = 16 * 1024 * 1024;
  CHECK(rtMalloc(&out, 4096, RT_MEMORY_HBM, 0));
  CHECK(rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0));
  CHECK(rtMemset(gm, GM_BYTES, 0, GM_BYTES));

  aclrtLaunchKernelAttr attr = {};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attr.value.dynUBufSize = 192 * 1024;
  aclrtLaunchKernelCfg cfg = {&attr, 1};

  for (int i = 2; i < argc; ++i) {
    aclrtFuncHandle fn = nullptr;
    CHECK(aclrtBinaryGetFunction(handle, argv[i], &fn));
    Args args{out, gm, 1, 0};
    CHECK(rtMemset(out, 4096, 0, 4096));
    CHECK(aclrtLaunchKernelWithHostArgs(fn, 1, stream, &cfg,
                                        &args, sizeof(args), nullptr, 0));
    CHECK(rtStreamSynchronize(stream));
    long long vals[8] = {0};
    CHECK(rtMemcpy(vals, sizeof(vals), out, sizeof(vals),
                   RT_MEMCPY_DEVICE_TO_HOST));
    std::printf("LAUNCH %d %s", i - 2, argv[i]);
    for (int j = 0; j < 8; ++j) std::printf(" v%d=%lld", j, vals[j]);
    std::printf("\n");
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
