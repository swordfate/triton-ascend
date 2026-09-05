// Host driver for simt_scalar_gm_dep.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;
constexpr size_t GM_BYTES = 64ULL * 1024 * 1024;
constexpr size_t GM_INTS = GM_BYTES / sizeof(int);

static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary);
  s.seekg(0, ios::end);
  size_t n = s.tellg();
  s.seekg(0);
  char *b = new char[n];
  s.read(b, n);
  *sz = n;
  return b;
}

static void *reg(const char *bin, char **buf) {
  uint32_t sz;
  *buf = readBin(bin, &sz);
  rtDevBinary_t b;
  b.data = *buf;
  b.length = sz;
  b.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  b.version = 0;
  void *h = nullptr;
  rtDevBinaryRegister(&b, &h);
  return h;
}

struct Args {
  void *out;
  void *gm;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
  int ops;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, void *gm,
                      int K, int nwarp, int iters, int mode, int ops) {
  Args args{dout, gm, K, 32, nwarp, iters, mode, ops};
  rtArgsEx_t ai = {};
  ai.args = &args;
  ai.argsSize = sizeof(args);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, stream, 0, &cfg);
  rtStreamSynchronize(stream);
  long long cycles = 0;
  rtMemcpy(&cycles, sizeof(cycles), dout, sizeof(cycles),
           RT_MEMCPY_DEVICE_TO_HOST);
  return cycles;
}

static long long minimum(const char *fn, rtStream_t stream, void *dout,
                         void *gm, int K, int nwarp, int iters, int mode,
                         int ops) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 5; ++rep) {
    long long value = runK(fn, stream, dout, gm, K, nwarp, iters, mode, ops);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            void *gm, int nwarp, int mode, int ops) {
  const int K = 4;
  const int I1 = 1024;
  const int I2 = 4096;
  long long c1 = minimum(fn, stream, dout, gm, K, nwarp, I1, mode, ops);
  long long c2 = minimum(fn, stream, dout, gm, K, nwarp, I2, mode, ops);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(nullptr);
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simt_scalar_gm_dep.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  void *gm = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);
  rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0);

  // Fill linked-list paths: for each warp start at warp+1 and wrap inside
  // the first 4096 entries.
  vector<int> pattern(GM_INTS, 0);
  for (size_t i = 0; i + 1 < 4096; ++i)
    pattern[i] = (int)(i + 1);
  pattern[4095] = 0;
  rtMemcpy(gm, GM_BYTES, pattern.data(), pattern.size() * sizeof(int),
           RT_MEMCPY_HOST_TO_DEVICE);

  runK(fn, stream, dout, gm, 1, 1, 16, 0, 1);
  printf("SIMT uniform scalar GM dependency probe\n");
  printf("warps,ops,independent_cycles,dependent_cycles,extra_per_edge_cycles\n");
  for (int warps : {1, 32}) {
    for (int ops : {1, 2, 4, 8}) {
      double indep = cyclesPerIter(fn, stream, dout, gm, warps, 0, ops);
      double dep = cyclesPerIter(fn, stream, dout, gm, warps, 1, ops);
      // For a per-warp uniform scalar chain, compare same number of warp ops.
      printf("%d,%d,%.6f,%.6f,%.6f\n", warps, ops, indep, dep,
             ops > 0 ? (dep - indep) / ops : 0.0);
    }
  }

  rtFree(gm);
  rtFree(dout);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  delete[] binary;
  return 0;
}
