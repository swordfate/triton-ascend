// Host driver for simt_predicate.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;

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
  int active_lanes;
  int mode;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, void *gm,
                      int K, int nwarp, int iters, int active_lanes, int mode) {
  Args args{dout, gm, K, 32, nwarp, iters, active_lanes, mode};
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
                         void *gm, int K, int nwarp, int iters,
                         int active_lanes, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 5; ++rep) {
    long long value = runK(fn, stream, dout, gm, K, nwarp, iters,
                           active_lanes, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            void *gm, int nwarp, int active_lanes, int mode) {
  const int K = 20;
  const int I1 = 256;
  const int I2 = 1024;
  long long c1 = minimum(fn, stream, dout, gm, K, nwarp, I1, active_lanes, mode);
  long long c2 = minimum(fn, stream, dout, gm, K, nwarp, I2, active_lanes, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  constexpr size_t GM_BYTES = 32ULL * 1024 * 1024;
  aclInit(nullptr);
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simt_predicate.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  void *gm = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);
  rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0);
  rtMemset(gm, GM_BYTES, 0, GM_BYTES);

  runK(fn, stream, dout, gm, 2, 4, 16, 32, 0);

  printf("SIMT masked/predicated execution sweep\n");
  printf("mode,active_lanes,warps,cycles_per_iter,effective_warps_per_cycle\n");
  const int modes[] = {0, 1, 2, 3};
  const int actives[] = {32, 24, 16, 8, 4, 1};
  const int warps_arr[] = {1, 2, 4, 8, 16, 32};
  for (int mode : modes) {
    for (int active : actives) {
      for (int warps : warps_arr) {
        double cpi = cyclesPerIter(fn, stream, dout, gm, warps, active, mode);
        double effective_warps_per_cycle = (double)warps / cpi;
        printf("%d,%d,%d,%.6f,%.6f\n", mode, active, warps, cpi,
               effective_warps_per_cycle);
      }
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
