// Host driver for simt_gm_memory_pattern.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;
constexpr size_t GM_BYTES = 128ULL * 1024 * 1024;

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
  int pattern;
  int stride;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, void *gm,
                      int K, int nwarp, int iters, int mode, int pattern,
                      int stride) {
  Args args{dout, gm, K, 32, nwarp, iters, mode, pattern, stride};
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
                         int pattern, int stride) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 5; ++rep) {
    long long value = runK(fn, stream, dout, gm, K, nwarp, iters, mode,
                           pattern, stride);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            void *gm, int nwarp, int mode, int pattern,
                            int stride) {
  const int K = 4;
  const int I1 = 1024;
  const int I2 = 4096;
  long long c1 = minimum(fn, stream, dout, gm, K, nwarp, I1, mode, pattern, stride);
  long long c2 = minimum(fn, stream, dout, gm, K, nwarp, I2, mode, pattern, stride);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(nullptr);
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simt_gm_memory_pattern.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  void *gm = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);
  rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0);
  rtMemset(gm, GM_BYTES, 0, GM_BYTES);

  runK(fn, stream, dout, gm, 1, 4, 16, 0, 0, 1);

  printf("SIMT GM memory pattern sweep\n");
  printf("mode,pattern,stride,warps,cycles_per_iter,bytes_per_cycle,warp_instructions_per_cycle\n");
  const int modes[] = {0, 1};
  const int patterns[] = {0, 1, 2};
  const int strides[] = {1, 2, 4, 8, 16};
  const int warps_arr[] = {1, 2, 4, 8, 16, 32};
  for (int mode : modes) {
    for (int pattern : patterns) {
      const int *stride_list = pattern == 1 ? strides : (const int[]){1};
      int stride_count = pattern == 1 ? 5 : 1;
      for (int si = 0; si < stride_count; ++si) {
        int stride = stride_list[si];
        for (int warps : warps_arr) {
          double cpi = cyclesPerIter(fn, stream, dout, gm, warps, mode,
                                     pattern, stride);
          double bytes = (double)warps * 32.0 * 8.0 * sizeof(float);
          double warp_instr = (double)warps * 8.0;
          printf("%d,%d,%d,%d,%.6f,%.6f,%.6f\n", mode, pattern, stride,
                 warps, cpi, bytes / cpi, warp_instr / cpi);
        }
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
