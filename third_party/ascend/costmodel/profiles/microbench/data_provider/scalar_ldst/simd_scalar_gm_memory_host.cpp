// Host driver for simd_scalar_gm_memory.cce.
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
  int iters;
  int mode;
  int ops;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, void *gm,
                      int K, int iters, int mode, int ops) {
  Args args{dout, gm, K, iters, mode, ops};
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
                         void *gm, int K, int iters, int mode, int ops) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 5; ++rep) {
    long long value = runK(fn, stream, dout, gm, K, iters, mode, ops);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            void *gm, int mode, int ops) {
  const int K = 4;
  const int I1 = 1024;
  const int I2 = 4096;
  long long c1 = minimum(fn, stream, dout, gm, K, I1, mode, ops);
  long long c2 = minimum(fn, stream, dout, gm, K, I2, mode, ops);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(nullptr);
  // rtSetDevice(0) selects the device exposed through
  // ASCEND_RT_VISIBLE_DEVICES, matching the other host drivers in this repo.
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simd_scalar_gm_memory.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  void *gm = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);
  rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0);
  rtMemset(gm, GM_BYTES, 0, GM_BYTES);

  runK(fn, stream, dout, gm, 1, 16, 0, 1);
  printf("SIMD MainScalar scalar GM memory probe\n");
  printf("ops,load_cycles,load_scalar_instr_per_cycle,store_cycles,store_scalar_instr_per_cycle\n");
  for (int ops : {1, 2, 4, 8}) {
    double loadCycles = cyclesPerIter(fn, stream, dout, gm, 0, ops);
    double storeCycles = cyclesPerIter(fn, stream, dout, gm, 1, ops);
    double loadRate = (double)ops / loadCycles;
    double storeRate = (double)ops / storeCycles;
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", ops, loadCycles, loadRate,
           storeCycles, storeRate);
  }

  rtFree(gm);
  rtFree(dout);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  delete[] binary;
  return 0;
}
