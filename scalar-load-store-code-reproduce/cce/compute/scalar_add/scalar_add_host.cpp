// Host for the scalar-add CAModel probes.
// usage: scalar_add_host <obj> <kernel> <K> <nlane> <nwarp> <iters> <mode>
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

struct Args {
  void *out;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

#define CHECK(call) do { auto e = (call); if (e != 0) { \
  std::fprintf(stderr, "%s failed: %d\n", #call, (int)e); std::exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s <obj> <kernel> <K> <nlane> <nwarp> <iters> <mode>\n", argv[0]);
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> bin((std::istreambuf_iterator<char>(f)), {});
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
  CHECK(rtMalloc(&out, 256, RT_MEMORY_HBM, 0));
  Args args{out, std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
            std::atoi(argv[6]), std::atoi(argv[7])};
  aclrtLaunchKernelAttr attr = {};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attr.value.dynUBufSize = 192 * 1024;
  aclrtLaunchKernelCfg cfg = {&attr, 1};
  CHECK(aclrtLaunchKernelWithHostArgs(fn, 1, stream, &cfg, &args, sizeof(args), nullptr, 0));
  CHECK(rtStreamSynchronize(stream));
  long long v = 0;
  CHECK(rtMemcpy(&v, sizeof(v), out, sizeof(v), RT_MEMCPY_DEVICE_TO_HOST));
  std::printf("RESULT %s K=%s nlane=%s nwarp=%s iters=%s mode=%s out0=%lld\n",
              argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], v);
  CHECK(rtFree(out));
  CHECK(rtStreamDestroy(stream));
  CHECK(aclrtBinaryUnLoad(handle));
  CHECK(aclrtDestroyBinary(binary));
  CHECK(rtDeviceReset(0));
  CHECK(aclFinalize());
  return 0;
}
