// Host driver for tput_ops.cce — measures additional FP32 SIMD/SIMT op throughput.
// Run after building tput_ops.o with the same toolchain as tput_host.cpp.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sys/time.h>

using namespace std;

static unsigned long us() {
  timeval t;
  gettimeofday(&t, 0);
  return t.tv_sec * 1000000UL + t.tv_usec;
}

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
  void *h = 0;
  rtDevBinaryRegister(&b, &h);
  return h;
}

struct Args {
  void *out;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

static long long runK(const char *fn, rtStream_t s, void *dout, int K,
                      int nlane, int nwarp, int iters, int mode) {
  Args a{dout, K, nlane, nwarp, iters, mode};
  rtArgsEx_t ai = {};
  ai.args = &a;
  ai.argsSize = sizeof(a);
  rtTaskCfgInfo_t c = {};
  c.localMemorySize = 192 * 1024;
  rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, s, 0, &c);
  rtStreamSynchronize(s);
  long long cyc = 0;
  rtMemcpy(&cyc, 8, dout, 8, RT_MEMCPY_DEVICE_TO_HOST);
  return cyc;
}

static long long mn(const char *fn, rtStream_t s, void *dout, int K, int nl,
                    int nw, int it, int md, int rep) {
  long long m = (long long)1e18;
  for (int i = 0; i < rep; i++) {
    long long c = runK(fn, s, dout, K, nl, nw, it, md);
    if (c < m)
      m = c;
  }
  return m;
}

static double cycPerIter(const char *fn, rtStream_t s, void *dout, int nl,
                         int nw, int md) {
  const int K = 20, I1 = 400, I2 = 1200;
  long long c1 = mn(fn, s, dout, K, nl, nw, I1, md, 7);
  long long c2 = mn(fn, s, dout, K, nl, nw, I2, md, 7);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

static double cycPerIterSmall(const char *fn, rtStream_t s, void *dout, int nl,
                              int nw, int md) {
  // exp/abs can be much slower; keep the same slope methodology but use
  // smaller iteration counts so the probe finishes quickly.
  const int K = 10, I1 = 20, I2 = 60;
  long long c1 = mn(fn, s, dout, K, nl, nw, I1, md, 5);
  long long c2 = mn(fn, s, dout, K, nl, nw, I2, md, 5);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(0);
  rtSetDevice(0);
  char *buf;
  void *h = reg("tput_ops.o", &buf);
  const char *fn = "measure";
  rtFunctionRegister(h, fn, fn, (void *)fn, 0);
  rtStream_t s;
  rtStreamCreate(&s, 0);
  void *dout;
  rtMalloc(&dout, 8, RT_MEMORY_HBM, 0);

  const char *simtNames[] = {"sub", "mul", "div", "exp", "abs"};
  const char *simdNames[] = {"sub", "mul", "div", "exp", "abs"};

  printf("== SIMT scalar op throughput (32 warps, ILP8) ==\n");
  for (int i = 0; i < 5; i++) {
    double cpi = (i >= 3) ? cycPerIterSmall(fn, s, dout, 32, 32, i)
                          : cycPerIter(fn, s, dout, 32, 32, i);
    double ops = 32.0 * 32 * 8;
    printf("simt.%-4s : %8.2f cyc/iter -> %8.1f scalar_ops/cyc\n",
           simtNames[i], cpi, ops / cpi);
  }

  printf("\n== SIMD vector op throughput (ILP8, full-width) ==\n");
  for (int i = 0; i < 5; i++) {
    double cpi = (i >= 3) ? cycPerIterSmall(fn, s, dout, 0, 8, 10 + i)
                          : cycPerIter(fn, s, dout, 0, 8, 10 + i);
    double ops = 8.0 * 64;
    printf("simd.%-4s : %8.2f cyc/iter -> %8.1f scalar_ops/cyc = %.3f full-width ops/cyc\n",
           simdNames[i], cpi, ops / cpi, (ops / cpi) / 64.0);
  }

  rtFree(dout);
  rtStreamDestroy(s);
  rtDeviceReset(0);
  aclFinalize();
  return 0;
}
