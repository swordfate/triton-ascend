# Triton-Ascend 构建流程详解: `TRITON_BUILD_WITH_CCAHE=true TRITON_BUILD_WITH_CLANG_LLD=true pip install -e .`

本文档顺着函数调用链，逐步解释 `pip install -e .` 在 Triton-Ascend 项目中背后发生的一切。

---

## 第 1 步：pip 读取 `pyproject.toml`

pip 首先读取 `pyproject.toml`，看到：

```toml
[build-system]
requires = ["setuptools>=40.8.0", "cmake>=3.20", "ninja>=1.11.1", "pybind11>=2.13.1", "nanobind>=2.4"]
build-backend = "setuptools.build_meta"
```

pip 会先在一个隔离环境中安装这些构建依赖，然后用 `setuptools` 作为构建后端。因为 `-e`（editable 模式），pip 会调用 `develop` 命令。

---

## 第 2 步：`setup.py` 模块级别代码执行（环境变量 + Backend 初始化）

Python 解释器加载 `setup.py` 时，**模块级别代码立即执行**。

### 2a. 环境变量默认值设置 (`setup.py:52-56`)

```python
os.environ.setdefault("TRITON_BUILD_WITH_CCACHE", "true")
os.environ.setdefault("TRITON_BUILD_WITH_CLANG_LLD", "true")
os.environ.setdefault("TRITON_BUILD_PROTON", "OFF")
os.environ.setdefault("TRITON_WHEEL_NAME", "triton-ascend")
os.environ.setdefault("TRITON_APPEND_CMAKE_ARGS", "-DTRITON_BUILD_UT=OFF")
```

因为用户已经在命令行显式设置了 `TRITON_BUILD_WITH_CCACHE=true` 和 `TRITON_BUILD_WITH_CLANG_LLD=true`，`setdefault` 不会覆盖它们。

### 2b. 初始化 Backend 列表 (`setup.py:764`)

```python
backends = [*BackendInstaller.copy(["ascend", "nvidia", "amd"]), *BackendInstaller.copy_externals()]
```

`BackendInstaller.prepare("ascend")` 的逻辑：

1. 确认 `third_party/ascend/` 目录存在
2. 如果是 git 仓库，执行 `git submodule update --init ascend`
3. 确认 `third_party/ascend/backend/` 下有 `compiler.py` 和 `driver.py`
4. 检查是否有 `language/` 和 `tools/` 子目录
5. 返回 `Backend` 对象：
   ```python
   Backend(name="ascend", src_dir="third_party/ascend",
           backend_dir="third_party/ascend/backend",
           language_dir="third_party/ascend/language",  # 如果有
           tools_dir=None,                               # 如果有
           install_dir="python/triton/backends/ascend")
   ```

### 2c. 执行 `setup()` 注册命令类 (`setup.py:1069-1121`)

关键的 `cmdclass` 映射：

| 命令 | 类 | 作用 |
|------|-----|------|
| `develop` | `plugin_develop` | `-e` 可编辑安装走这个 |
| `build_ext` | `CMakeBuild` | CMake 构建扩展 |
| `editable_wheel` | `plugin_editable_wheel` | 新版 pip editable wheel |
| `install` | `plugin_install` | 普通安装 |
| `build_py` | `CMakeBuildPy` | 先 build_ext 再 build_py |
| `bdist_wheel` | `BuildWheel` | wheel 打包 |

唯一的 C++ 扩展定义：

```python
ext_modules=[CMakeExtension("triton", "triton/_C/")]
```

---

## 第 3 步：`plugin_develop.run()` —— 建立符号链接 (`setup.py:862-866`)

```python
class plugin_develop(develop):
    def run(self):
        add_links(external_only=False)  # 先建链接
        super().run()                    # 再执行标准 develop 流程
```

### 3a. `add_links(False)` (`setup.py:849-852`)

遍历所有 backend，为每个 backend 创建符号链接：

```
python/triton/backends/ascend  →  third_party/ascend/backend/
python/triton/backends/nvidia  →  third_party/nvidia/backend/
python/triton/backends/amd     →  third_party/amd/backend/
```

如果 backend 有 `language/` 目录，链接到：
```
python/triton/language/extra/<name>  →  third_party/<name>/language/<name>/
```

如果 backend 有 `tools/` 目录，链接到：
```
python/triton/tools/extra/<name>  →  third_party/<name>/tools/<name>/
```

这样 Python 代码中 `import triton.backends.ascend` 就能找到对应的 backend Python 文件。

> 注意：因为 `TRITON_BUILD_PROTON=OFF`，Proton profiler 的链接被跳过。

### 3b. `super().run()` → `develop.run()`

`setuptools` 的 `develop` 命令内部会检查是否需要构建扩展，然后触发 `build_ext` 命令。

---

## 第 4 步：`CMakeBuild.run()` —— 下载依赖 + 触发构建 (`setup.py:493-523`)

```python
class CMakeBuild(build_ext):
    def run(self):
        download_and_copy_dependencies()  # ①

        # 检查 cmake 版本 >= 3.20
        out = subprocess.check_output(["cmake", "--version"])
        ...

        # ② 为每个 CMakeExtension 执行构建
        for ext in self.extensions:
            self.build_extension(ext)
```

### 4a. `download_and_copy_dependencies()` (`setup.py:678-761`)

从 NVIDIA 下载以下二进制文件到 `third_party/nvidia/backend/` 下：

| 文件 | 用途 | 路径 |
|------|------|------|
| `ptxas` | PTX 汇编器（Hopper） | `bin/ptxas` |
| `ptxas-blackwell` | PTX 汇编器（Blackwell） | `bin/ptxas-blackwell` |
| `cuobjdump` | CUDA 目标文件分析 | `bin/cuobjdump` |
| `nvdisasm` | GPU 反汇编 | `bin/nvdisasm` |
| CUDA CRT | C 运行时头文件 | `include/` |
| CUDA Runtime | CUDART 头文件 | `include/` |
| CUPTI | Profiling 接口（头文件 + 库） | `include/` + `lib/cupti/` |

版本号由 `cmake/nvidia-toolchain-version.json` 控制。

---

## 第 5 步：`build_extension(ext)` —— 核心 C++ 构建 (`setup.py:546-675`)

这是最关键的函数，分为以下阶段：

### 5a. 下载 LLVM 预编译包

```python
# setup.py:550
thirdparty_cmake_args = get_thirdparty_packages([get_llvm_package_info()])
```

`get_llvm_package_info()` (`setup.py:226-273`) 的工作：

1. 检测系统平台和架构（macOS-arm64 / ubuntu-x64 / almalinux-arm64 等）
2. 读取 `cmake/llvm-hash.txt` 获取 LLVM 提交 hash
3. 计算 `third_party/ascend/patch/llvm_patch_*.patch` 的 hash
4. 构造下载 URL：
   ```
   https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/llvm-<rev>-<patch_hash>-<system_suffix>.tar.gz
   ```
5. 下载到 `~/.triton/llvm/` 并解压

下载后，`get_thirdparty_packages()` 会设置这些 CMake 参数：

```
-DLLVM_INCLUDE_DIRS=<path>/include
-DLLVM_LIBRARY_DIR=<path>/lib
-DLLVM_SYSPATH=<path>
```

### 5b. 收集 CMake 构建参数 (`setup.py:552-632`)

```python
cmake_args = [
    "-G", "Ninja",                              # 使用 Ninja 构建系统（比 Make 快）
    "-DCMAKE_MAKE_PROGRAM=" + ninja_dir,         # 指定 ninja 二进制路径
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",        # 生成 compile_commands.json
    "-DLLVM_ENABLE_WERROR=ON",
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,  # .so 输出到 python/triton/_C/
    "-DTRITON_BUILD_PYTHON_MODULE=ON",            # 构建 Python 绑定模块
    "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
    "-DPython3_INCLUDE_DIR=" + python_include_dir,
    "-DTRITON_CODEGEN_BACKENDS=ascend;nvidia;amd",
    "-DTRITON_WHEEL_DIR=" + wheeldir,
    "-DLLVM_MAJOR_VERSION_22_COMPATIBLE=ON",
]
```

**`TRITON_BUILD_WITH_CLANG_LLD=true` 的作用**：
```python
# setup.py:586-594
if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
    cmake_args += [
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_LINKER=lld",
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
    ]
```
Clang + LLD 相比 GCC + ld 通常链接更快、内存占用更小。

**环境变量透传（Passthrough）**：
```python
# setup.py:611-616
passthrough_args = [
    "TRITON_BUILD_PROTON",
    "TRITON_BUILD_WITH_CCACHE",
    "TRITON_PARALLEL_LINK_JOBS",
]
cmake_args += [f"-D{option}={os.getenv(option)}" for option in passthrough_args]
```
`TRITON_BUILD_WITH_CCACHE=true` 和 `TRITON_BUILD_PROTON=OFF` 由此传入 CMake。

**追加参数**：
```
-DTRITON_BUILD_UT=OFF    # 来自 TRITON_APPEND_CMAKE_ARGS，不构建单元测试
```

### 5c. 执行 CMake 配置 + 构建 (`setup.py:636-639`)

```python
cmake_dir = get_cmake_dir()  # build/cmake.<plat>-<impl>-<pyver>/

# Step 1: CMake 配置（生成 Ninja build 文件）
subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=cmake_dir)

# Step 2: 编译所有目标
subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)
# build_args = ["--config", cfg, "-j" + max_jobs]

# Step 3: 构建 mlir-doc 目标
subprocess.check_call(["cmake", "--build", ".", "--target", "mlir-doc"], cwd=cmake_dir)
```

构建目录：`build/cmake.<plat>-<impl>-<pyver>/`（可通过 `TRITON_BUILD_DIR` 环境变量覆盖）。

编译日志和进度可以在该目录下看到。

### 5d. 拷贝 + Strip 二进制工具 (`setup.py:641-675`)

构建完成后，将生成的工具拷贝到 Python 包目录并 strip 减小体积：

```
build/cmake.<...>/bin/triton-mlir-opt  →  python/triton/_C/triton-mlir-opt  (strip)
build/cmake.<...>/bin/triton-opt       →  python/triton/_C/triton-opt       (strip)
```

> `triton-mlir-opt` 用于 MLIR → Bytecode 转换；`triton-opt` 用于 TTIR → TTAdapter 转换。

---

## 第 6 步：CostModel 单元测试可执行文件的完整生成链路

这是从命令行到最终 `CostModelPasses` / `CostModelHardwareConfig` / `CostModelPipelineScheduler` 三个可执行文件的**完整逐层追踪**。前提是把 `TRITON_BUILD_UT` 翻成 `ON`：

```bash
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=ON" \
pip install -e .
```

### 6.0 前置概念：MLIR 为什么需要 TableGen 生成这些代码？

在分析 TableGen 逐行代码之前，先回答一个更根本的问题：**如果不用 TableGen，纯手写一个 MLIR Dialect 要写什么？为什么这些是必须的？**

整个推导过程可以分成四步：

---

**第一步：MLIR 对一个新 Dialect 的硬性要求清单**

MLIR 不认识你的 "ascend" 方言。框架为了保证能加载、解析、打印、验证、优化你的 Op，要求你提供以下**全部**东西。如果纯手写：

```
┌──────────────────────────────────────────────────────────────┐
│ A. Dialect 类                  class AscendModelDialect      │
│    - 名字 "ascend" 的字符串声明  : public mlir::Dialect {}    │
│    - 类型 ID 注册               ~20 行                       │
│    - initialize() 里注册 Op/Types/Attributes                 │
│                                                              │
│ B. 每个 Op 的 C++ 类            class AddOp : public         │
│    - 继承链（CRTP + Traits）     mlir::Op<AddOp,             │
│    - operand accessor            OpTrait::Pure, ...> {}      │
│    - result accessor             每个 Op ~80 行              │
│    - build() 构造方法            25 个 Op = 2000 行           │
│    - parse() 文本→Op 解析        全是机械重复                  │
│    - print() Op→文本 输出                                    │
│    - verify() 语义校验                                       │
│                                                              │
│ C. 枚举的正反转换               stringifyHWUnit(Cube)→"cube" │
│    - 枚举 → 字符串              每个枚举 ~20 行                │
│    - 字符串 → 枚举              4 个枚举 = 80 行               │
│                                                              │
│ D. Interface 的 Trait(mixin)    EstimateCyclesOpInterface    │
│    - 提供默认实现的方法          ::Trait { ... }              │
│    - 标记纯虚函数               ~30 行                        │
│                                                              │
│ E. Pass 的注册                  "convert-triton-to-ascend"   │
│    - 名字 → C++ 类映射           → ConvertTritonToAscendPass │
│    - 命令行参数解析              每个 Pass ~10 行               │
│    - 依赖 Dialect 声明           7 个 Pass = 70 行             │
└──────────────────────────────────────────────────────────────┘

合计: 2000(Op) + 20(Dialect) + 80(枚举) + 30(Interface) + 70(Pass) ≈ 2200 行
其中 90% 是机械重复——每个 Op 的 build/parse/print/verify 长得几乎一样
```

---

**第二步：TableGen 帮我们做什么？——划分"机械"和"逻辑"**

2200 行里，哪些能自动生成、哪些必须手写？

```
MLIR 要求             纯手写?            TableGen 生成?       手写实现?
──────────────────────────────────────────────────────────────────────
Dialect 类声明         20行 机械          ✓ Dialect.h.inc       ✗
Dialect 注册           10行 机械          ✓ Dialect.cpp.inc     ✗
Op 类声明              每个Op ~40行 机械   ✓ Ops.h.inc          ✗
build()                每个Op ~10行 机械   ✓ Ops.cpp.inc        ✗
parse()                每个Op ~15行 机械   ✓ Ops.cpp.inc        ✗
print()                每个Op ~10行 机械   ✓ Ops.cpp.inc        ✗
verify() 骨架          每个Op ~10行 机械   ✓ Ops.cpp.inc        ✗
枚举字符串转换          每个枚举 ~20行 机械  ✓ Enums.h/cpp.inc   ✗
Interface Trait        ~30行 机械         ✓ Interfaces.h/cpp.inc ✗
Pass 注册               ~10行/pass 机械    ✓ Passes.h.inc       ✗
──────────────────────────────────────────────────────────────────────
estimateCycles()       每个Op ~5-15行      ✗                    ✓ 手写
getHWUnit()            每个Op ~1行         ✗                    ✓ 手写
initialize() 注册Op     ~5行               ✗                    ✓ 手写
Interface 辅助方法      ~15行              ✗                    ✓ 手写
Pass 内部逻辑           每个Pass ~100行     ✗                    ✓ 手写
硬件配置/调度            ~1000行            ✗                    ✓ 手写
──────────────────────────────────────────────────────────────────────
```

**分界线很清楚**：所有"从这个 Op 有 2 个 operand、1 个 result"能机械推导出来的 → TableGen。所有需要实际逻辑判断的 → 你手写。

---

**第三步：每个文件扮演什么角色**

基于上面的划分，文件组织如下：

```
手写的定义文件 (.td)           TableGen 生成的 (.inc)           手写的实现文件 (.h / .cpp)
────────────────────           ────────────────────            ──────────────────────────

AscendModelInterfaces.td ──→  Interfaces.h.inc   被 Dialect.h include ──→  (作为 Op 类声明的
  "定义接口的虚函数签名"        Interfaces.cpp.inc 被 Ops.cpp include       一部分被 include)

AscendModelBase.td ────────→  Dialect.h.inc      被 Dialect.h include
  "Dialect 名字和枚举定义"      Dialect.cpp.inc    被 Dialect.cpp include
                               OpsEnums.h.inc     被 Dialect.h include
                               OpsEnums.cpp.inc   被 Dialect.cpp include

AscendModelOps.td ─────────→  Ops.h.inc           被 Dialect.h include ──→  AscendModelOps.cpp
  "25 个 Op 的名字/输入/输出"   (Op 类声明 + accessor)                      每个 Op 的 estimateCycles()
                               Ops.cpp.inc         被 Dialect.cpp + Ops.cpp  getHWUnit() getFlops()
                               (build/parse/print    include                 每个 Op 的 5~15 行业务逻辑
                                /verify 骨架)
                               AttrDefs/Types .inc                           AscendModelDialect.cpp
                                                                              initialize() 里注册 Op

Passes.td ────────────────→  Passes.h.inc         被 Passes.h include ──→  PassRegistration.cpp
  "7 个 Pass 的名字/参数"      (Pass 声明 + Option)                          管线顺序(哪个 Pass 先跑)
                                                                            各 Pass 的 .cpp 文件
                                                                            每个 100+ 行业务逻辑
```

**手写的 `.h` 文件做什么？** 组织 include 顺序——Interface 先于 Dialect 先于枚举先于 Op 类，因为 Op 依赖前面全部：

```cpp
// AscendModelDialect.h —— 唯一职责：按正确顺序 include 所有 .inc
#include "AscendModel/IR/AscendModelInterfaces.h"     // ① Interface（Op 依赖）
#include "AscendModel/IR/AscendModelDialect.h.inc"     // ② Dialect 声明
#include "AscendModel/IR/AscendModelOpsEnums.h.inc"    // ③ 枚举
#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.h.inc"          // ④ Op 类（最后，依赖上面全部）
```

**手写的 `.cpp` 文件做什么？** 实现 TableGen 不能生成的逻辑：

```cpp
// AscendModelDialect.cpp —— 告诉 MLIR 我的 dialect 有这些 Op
void AscendModelDialect::initialize() {
  addOperations<
#define GET_OP_LIST                                    // X-Macro: 只要求展开 Op 名列表
#include "AscendModel/IR/AscendModelOps.cpp.inc"        // → AddOp, SubOp, MulOp, ...
  >();
}

// AscendModelOps.cpp —— 每个 Op 的 estimateCycles() 公式
int64_t AddOp::estimateCycles(const HardwareConfig &config) {
  return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());
}
int64_t MatmulOp::estimateCycles(const HardwareConfig &config) {
  return ceil(M/16) * ceil(N/16) * ceil(K/16) + config.getCubeStartupLatency();
}
// ... 25 个 Op 各写各的
```

---

**第四步：完整对应表**

```
手写定义                  TableGen 生成                 手写实现
────────                  ────────────                 ────────
.td 文件                  .inc 文件                    .h / .cpp 文件

AscendModelBase.td ──→ Dialect.h.inc            AscendModelDialect.h
                    → Dialect.cpp.inc              (组织 include 顺序)
                    → OpsEnums.h.inc             AscendModelDialect.cpp
                    → OpsEnums.cpp.inc              (initialize 注册)

AscendModelInter    → Interfaces.h.inc           AscendModelOps.cpp
faces.td            → Interfaces.cpp.inc           (辅助方法 + 所有 Op 的
                                                     业务逻辑)

AscendModelOps.td   → Ops.h.inc                  (所有 Op 声明 + accessor
                    → Ops.cpp.inc                   已随 Dialect.h include)
                    → AttrDefs.h.inc
                    → AttrDefs.cpp.inc
                    → Types.h.inc
                    → Types.cpp.inc

Passes.td           → Passes.h.inc               PassRegistration.cpp
                                                   (管线顺序注册)
                                                 各 Pass .cpp 文件
                                                   (实际 Pass 逻辑)
```

**核心结论**：TableGen 生成的是"**怎么做**"（how）的样板——类怎么声明、build 怎么写、parse/print 怎么映射参数。你手写的是"**是什么**"（what）的定义——这个 Op 叫什么、有几个输入输出、estimateCycles 的公式是什么。

---

以下是上述各要点的详细展开：

#### 6.0a. MLIR 对每个新 Dialect 的硬性要求

MLIR 是一个**插件化编译器框架**。它本身不包含任何具体 Op，所有 Op 都是外部插件（Dialect）。为了让框架能加载一个 Dialect，必须提供：

```
┌───────────────────────────────────────────────────────────┐
│ MLIR 框架运行时需要                    提供者              │
├───────────────────────────────────────────────────────────┤
│ ① HashMap: "ascend.add" → AddOp 工厂函数                  │
│    解析文本 IR 时，看到字符串 "ascend.add"               │
│    必须知道 new 哪个 C++ 类                              │
│                                                           │
│ ② HashMap: "cube" → HWUnit::Cube                          │
│    文本中的属性值 "cube" 要转成 C++ 枚举值               │
│    switch-case 逻辑只能靠查表                             │
│                                                           │
│ ③ 每个 Op: build()    — 构造 Op 插入 IR 树               │
│    每个 Op: parse()   — 文本 → C++ Op 对象               │
│    每个 Op: print()   — C++ Op 对象 → 文本               │
│    每个 Op: verify()  — 检查 Op 语义合法性               │
│                                                           │
│ ④ 每个 Op 的 C++ 类: operand accessor (getLhs/getRhs)     │
│    每个 Op 的 C++ 类: result accessor (getResult)         │
│                                                           │
│ ⑤ Interface: 跨不同 Op 类型的统一调用路径                  │
│    (MLIR 的 Op 用 CRTP 模板——不能做虚函数多态)             │
└───────────────────────────────────────────────────────────┘
```

**25 个 Op × 每个 Op 的 build/parse/print/verify/accessor ≈ 2000 行纯机械代码。这些代码 90% 可以从 "Op 有几个 operand、几个 result" 推导出来。TableGen 做的事就是用 3 行 .td 描述差异化信息，生成另外 97 行。**

---

#### 6.0b. 为什么 `initialize()` 里必须注册所有 Op/Type/Attribute？

MLIR 框架内部维护三个 HashMap：

```
Op 注册表:      "ascend.add"     → AddOp::create()
                "ascend.matmul"  → MatmulOp::create()
                "ascend.neg"     → NegOp::create()
                ...

Type 注册表:    "ascend.config"  → ConfigType::get()
                ...

Attr 注册表:    "ascend.tile"    → TileAttr::get()
                ...
```

当 MLIR 解析文本 `.mlir`：

```
ascend.add %0, %1 : tensor<4xf32>
```

解析器看到 `ascend.add` → 查 Op 注册表 → 找到 `AddOp::create()` → 调 `AddOp::parse()` → 生成 C++ 对象。

**没有这个注册表，MLIR 就是一个空框架，什么 Op 都不认识。`initialize()` 的唯一任务就是往这三个 HashMap 里塞条目。** 你手写 `addOperations<AddOp, SubOp, ...>()`，TableGen 在 `Ops.cpp.inc` 里帮你把 `AddOp::getOperationName()`、`AddOp::create()` 等工厂函数生成好。

---

#### 6.0c. 什么是 `mlir::Op + Traits`？为什么叫 Trait？

```cpp
class AddOp : public mlir::Op<AddOp,                    // CRTP: 子类类型
                               OpTrait::Pure,            // Trait: 无副作用
                               OpTrait::OneResult,       // Trait: 恰好1个结果
                               OpTrait::SameOperandsAndResultType, // 输入输出同类型
                               EstimateCyclesOpInterface::Trait>   // Interface Trait
```

**CRTP（奇异递归模板模式）**：`mlir::Op<AddOp, ...>` 是一个模板，模板参数是子类自己（`AddOp`）。编译器在模板展开时就知道 `Derived = AddOp`，所以基类里 `static_cast<AddOp*>(this)` 是零开销的编译期转换——不需要虚函数表。

**为什么 MLIR 必须用 CRTP？** MLIR 的 Op 是**值语义**——一个 `AddOp` 对象只存 8 字节（一个 `Operation*` 指针），不能在上面挂虚函数指针。CRTP 让所有方法调用在编译期确定，零开销，代价是失去 `Base*` 式的多态。

**Trait = 可以贴在 Op 上的行为标签**。它描述的不是"这个 Op 是什么"，而是"这个 Op 有什么特性"：

```cpp
// MLIR 框架代码（不关心具体 Op）:
if (op->hasTrait<OpTrait::Pure>()) {
    // 无副作用 → 如果结果没人用，安全删掉
    if (op->use_empty()) op->erase();
}

if (op->hasTrait<OpTrait::OneResult>() && op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
    // 结果类型 = 输入类型 → 类型推断直接传播
    resultType = op->getOperand(0).getType();
}
```

**Trait 是跨 Op 类型的可复用能力标记**，跟 Rust/Scala 的 Trait 相同语义。

---

#### 6.0d. 为什么需要 operand/result accessor？

```cpp
// 底层存储按编号索引:
this->getOperand(0);  // 第0个输入——是什么？不知道
this->getOperand(1);  // 第1个输入——是什么？不知道

// accessor 给编号起名字:
Value lhs = op.getLhs();    // 自文档化
Value rhs = op.getRhs();    // 编译器能检查
```

纯机械映射，TableGen 从 `.td` 的 `arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs)` 生成。

---

#### 6.0e. 为什么需要 `build()`？是生成 IR 节点吗？

**是的。** `build()` 的唯一职责是在内存中构造一个 Op 并插入 IR 树。你不能 `new AddOp(...)`——MLIR 的 Op 不是普通堆对象。

```cpp
// 正确方式: 通过 OpBuilder 创建，内部调 AddOp::build()
auto addOp = builder.create<AddOp>(loc, lhs, rhs);

// build() 做什么:
void AddOp::build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs, int64_t opId) {
  state.addOperands({lhs, rhs});          // ① 记录输入
  state.addTypes(lhs.getType());          // ② 推断输出类型
  if (opId != -1)
    state.addAttribute("op_id", ...);     // ③ 设置属性
}
// build() 返回后，MLIR 框架用 state 分配真正的 Operation 对象
```

---

#### 6.0f. 为什么需要 `parse()` / `print()` / `verify()`？

MLIR 的序列化三角：

```
文本 .mlir ──parse()──→ C++ Op 对象 ──Pass处理──→ 文本/二进制输出
                           │                      ↑
                           ↓ verify()             │ print()
                         校验                      │
```

- **`parse()`**：`ascend.add %0, %1 : tensor<4xf32>` → 内存 Op。没有它就没法读 `.mlir`。
- **`print()`**：内存 Op → 文本。没有它就没法 debug、没法输出优化结果。
- **`verify()`**：parse 之后、Pass 之前，检查 Op 是否合法（如类型一致、属性值合法）。

---

#### 6.0g. 为什么需要枚举 ↔ 字符串转换？

MLIR 文本格式里的属性值是**字符串**：

```
ascend.add {hw_unit = "cube"} %0, %1 : ...
                    ↑ 这是文本 "cube"，不是 C++ 的 HWUnit::Cube
```

运行时需要：
```cpp
// "cube" → HWUnit::Cube (解析)
HWUnit unit = symbolizeHWUnit("cube");  // → HWUnit::Cube

// HWUnit::Cube → "cube" (打印)
llvm::StringRef str = stringifyHWUnit(HWUnit::Cube);  // → "cube"
```

每个枚举值一行 switch-case，完全机械。TableGen 从 `.td` 的三行枚举定义生成全部转换代码。

---

#### 6.0h. 为什么需要 Op Interface？

**CRTP 失去了 `Base*` 式的多态**——`AddOp` 的基类是 `mlir::Op<AddOp, ...>`，`MatmulOp` 的基类是 `mlir::Op<MatmulOp, ...>`，它们是**两个不同的类**，不能互相转换。

但在 `EstimateCyclesPass` 里，你需要遍历所有 Op，不管具体是 AddOp 还是 MatmulOp，统一调 `estimateCycles(config)`：

```cpp
// ❌ 没有 Interface: 必须每个 Op 类型各写一个分支
if (auto add = dyn_cast<AddOp>(op))       add.estimateCycles(config);
else if (auto sub = dyn_cast<SubOp>(op)) sub.estimateCycles(config);
// ... 25 个 Op = 25 行。加一个 Op 就要改这里。

// ✅ 有 Interface: 一次判断覆盖所有 Op
if (auto iface = dyn_cast<EstimateCyclesOpInterface>(op)) {
    iface.estimateCycles(config);  // 虚函数调用 — 唯一的运行时开销
}
```

**Interface 是 CRTP 体系中补回多态能力的机制**。90% 走 CRTP（零开销），10% 走 Interface（有开销但灵活）。

Interface 里为什么有的方法是纯虚、有的有默认实现？

```cpp
InterfaceMethod<"estimateCycles",  ...>                  // 纯虚: 每个 Op 算法不同
InterfaceMethod<"getTransferBytes", ..., "return 0;">    // 默认: 非搬运 Op 都是 0
InterfaceMethod<"getFlops",        ..., "return 0;">    // 默认: 非计算 Op 都是 0
```

25 个 Op 里只有 5 个是搬运 Op——给另外 20 个各写 `return 0` 是噪音，所以给默认实现让那 5 个手动 override。`estimateCycles()` 每个 Op 都不一样，没法给有意义的默认值，所以纯虚。

---

#### 6.0i. 完整对应：.td → .inc → .cpp 各自做什么

```
手写定义 (.td)           TableGen 生成 (.inc)            手写实现 (.h/.cpp)
================        =====================           ======================

AscendModelBase.td ──→ Dialect.h.inc / .cpp.inc      (纯机械，无需手写补充)
                    ──→ OpsEnums.h.inc / .cpp.inc

AscendModelInter    ──→ Interfaces.h.inc              AscendModelOps.cpp:
faces.td               Interfaces.cpp.inc               getElementBits()
                                                        getNumElements()

AscendModelOps.td   ──→ Ops.h.inc                     AscendModelOps.cpp:
                         (Op 类声明，operand/result      每个 Op 的 estimateCycles()
                          accessor、虚函数声明)           每个 Op 的 getHWUnit()
                    ──→ Ops.cpp.inc                      每个 Op 的 getFlops()
                         (build/parse/print/verify     AscendModelDialect.cpp:
                          骨架、X-Macro:               initialize() 里注册所有 Op
                          GET_OP_LIST / GET_OP_CLASSES)  (业务逻辑，不是样板)
                    ──→ AttrDefs.h.inc/.cpp.inc
                    ──→ Types.h.inc/.cpp.inc

Passes.td           ──→ Passes.h.inc                   PassRegistration.cpp
                         (Pass 声明 + Option 结构体)       (管线顺序)
                                                       各 Pass 的 .cpp 文件
```

**总结**：TableGen 生成的是"**怎么做**"（how）——类怎么声明、build/parse/print 怎么写、枚举怎么转换。你手写的是"**是什么**"（what）——这个 Op 叫 ascend.neg、输入一个 tensor、输出一个 tensor、做取反操作、取反的 cycle 数是 ceil(N/128)。

---

---

### 6.0j. 四个 .td 文件的创建心路历程 —— 从零声明一个 Dialect

假如你是作者，打开空白编辑器，要从零定义 AscendModel 方言。**先写哪个 .td、后写哪个，由依赖关系决定。**

---

#### 文件 ①：`AscendModelBase.td` —— "先有家，才有家里的东西"

第一个念头：MLIR 连 "ascend" 这个方言名都不认识，必须先**登记 Dialect 的存在**：

```tablegen
def AscendModel_Dialect : Dialect {
  let name = "ascend";                              // MLIR 文本里用 "ascend" 前缀
  let cppNamespace = "::mlir::ascend";              // C++ 代码放这个命名空间
}
```

**为什么这是第一件事？** 后面所有 Op 的定义都要写 `AscendModel_Op<...>`——这个东西依赖 `AscendModel_Dialect` 已经声明好。

第二个念头：我的 Op 需要知道自己在哪个硬件单元上跑、数据在哪个内存空间。这些都表现为**枚举值**，在 MLIR 文本里写成 `"cube"` / `"hbm"` 这样的字符串。TableGen 用一种固定模式定义枚举：

```
I32EnumAttrCase<"C++ 枚举名", 数值, "MLIR 文本里的字符串">
    ↓               ↓            ↓          ↓
  C++: HWUnit::Cube   = 0        IR: "cube"
```

```tablegen
// 七个硬件单元
def HWUnit_Cube     : I32EnumAttrCase<"Cube", 0, "cube">;
def HWUnit_CubeMTE2 : I32EnumAttrCase<"CubeMTE2", 1, "cube_mte2">;
def HWUnit_FixPipe  : I32EnumAttrCase<"FixPipe", 2, "fixpipe">;
def HWUnit_Vector   : I32EnumAttrCase<"Vector", 3, "vector">;
def HWUnit_VecMTE2  : I32EnumAttrCase<"VecMTE2", 4, "vec_mte2">;
def HWUnit_MTE3     : I32EnumAttrCase<"MTE3", 5, "mte3">;
def HWUnit_Scalar   : I32EnumAttrCase<"Scalar", 6, "scalar">;

// 打包成完整枚举类型——生成 stringifyHWUnit() + symbolizeHWUnit()
def HWUnitAttr : I32EnumAttr<"HWUnit", "Hardware execution unit", [
  HWUnit_Cube, HWUnit_CubeMTE2, HWUnit_FixPipe,
  HWUnit_Vector, HWUnit_VecMTE2, HWUnit_MTE3, HWUnit_Scalar
]> { let cppNamespace = "::mlir::ascend"; }
```

`I32EnumAttr` 的作用是把散装的 case 打包，同时自动生成两个配套函数（后面会说为什么必须有它们）。

同样模式定义了另外三个枚举：

```tablegen
// 内存空间: HBM, L2, L1, UB
// 数据类型: FP32, FP16, BF16, INT8, INT32
// 向量操作种类: Add, Sub, Mul, Div, Exp, Log, Max, Min, Select, Cast
// 归约种类: Sum, Max, Min, Prod
```

最后，用一个 TableGen class 作为**所有 Op 的基类**——避免每个 Op 定义重复写 "属于 AscendModel 方言"：

```tablegen
class AscendModel_Op<string mnemonic, list<Trait> traits = []> :
    Op<AscendModel_Dialect, mnemonic, traits>;
```

没有它的话每个 Op 要写成 `def X : Op<AscendModel_Dialect, "name", [...traits...]>`，每个都要重复 `AscendModel_Dialect`。有了基类就变成 `def X : AscendModel_Op<"name", [...traits...]>`。

---

#### 文件 ②：`AscendModelInterfaces.td` —— "在写 Op 之前，先写好它们的共同契约"

**为什么在写 Op 之前就要定义 Interface？** 因为每个 Op 的 `.td` 里要写 `DeclareOpInterfaceMethods<EstimateCyclesOpInterface>`——你不先定义 Interface，Ops.td 就 include 不了它。

作者思路：**后续 EstimateCyclesPass 要遍历所有 Op、不管你是什么 Op 都调 `estimateCycles(config)`。为了实现这个统一的调用方式，需要一个 Interface。**

```tablegen
def EstimateCyclesOpInterface : OpInterface<"EstimateCyclesOpInterface"> {
  let cppNamespace = "::mlir::ascend";
```

然后逐个定义接口方法，**每个方法的"纯虚 vs 默认实现"取决于 25 个 Op 里有多少个需要不同的值**：

```tablegen
  let methods = [
    // 纯虚：25 个 Op 各有各的公式 → 没法给默认值
    InterfaceMethod<"estimateCycles", "int64_t", (ins "const HardwareConfig &":$config)>,
    // 纯虚：Add 在 Vector、Matmul 在 Cube → 每个都不一样
    InterfaceMethod<"getHWUnit", "HWUnit", (ins)>,

    // 有默认实现：只有 5 个搬运 Op 需要覆写
    InterfaceMethod<"getTransferBytes", "int64_t", (ins), "return 0;">,
    // 有默认实现：只有计算 Op 需要覆写
    InterfaceMethod<"getFlops", "int64_t", (ins), "return 0;">,
    // 有默认实现：大部分简单 Op = 1 cycle，只有 Div/Exp/Log 等覆写
    InterfaceMethod<"getCyclesPerVectorOp", "int", (ins), "return 1;">
  ];

  // 辅助方法——所有 Op 共享的工具，通过 getOperand/getResult 推断 element bits 和元素个数
  let extraClassDeclaration = [{
    int getElementBits();
    int64_t getNumElements();
  }];
}
```

---

#### 文件 ③：`AscendModelOps.td` —— "有家有契约了，写 Op 吧"

开篇 include 前面两个文件：

```tablegen
include "AscendModelBase.td"           // 需要 AscendModel_Op 基类 + 枚举定义
include "AscendModelInterfaces.td"     // 需要 EstimateCyclesOpInterface
```

**先定义最核心的——矩阵乘**：

```tablegen
def Ascend_MatmulOp : AscendModel_Op<"matmul", [
    Pure,                                               // Trait: 无副作用
    DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>
    // ↑ "我实现 Interface，并且要覆写 getFlops（默认返回 0）"
]> {
  let arguments = (ins
    AnyTensor:$lhs, AnyTensor:$rhs,      // 两个 tensor 输入
    I64Attr:$M, I64Attr:$N, I64Attr:$K,  // 矩阵维度
    OptionalAttr<I64Attr>:$estimated_cycles,  // 后续 Pass 写回
    OptionalAttr<I64Attr>:$op_id               // AssignOpIDs Pass 写
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  // 展开后文本格式: ascend.matmul %A, %B {op_id=3} : (tensor<128x64xf16>, tensor<64x128xf16>) -> tensor<128x128xf16>
}
```

参数含义解析：
- `I64Attr:$M` — 64 位整数属性，命名为 M，编译期已知。矩阵维度不写成 tensor shape 而是显式属性，因为 costmodel 需要直接读取 `getM()` 而不是从 shape 反推
- `OptionalAttr<I64Attr>:$estimated_cycles` — 可选的 64 位整数。初始 IR 里没有这个属性，EstimateCyclesPass 跑完之后才有
- `$lhs` `,` `$rhs` attr-dict — 文本格式：先打两个 operand 名，逗号分隔，然后打印属性字典（`{op_id=3, M=128, N=64, K=64}`）

**然后定义搬运 Op**——Cube 路径和 Vector 路径各有搬入和搬出：

```tablegen
// Cube 搬入: HBM →(MTE2)→ L1
def Ascend_CubeLoadOp : AscendModel_Op<"cube_load", [
    DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getTransferBytes"]>
]> { ... }   // ↑ 声明要覆写 getTransferBytes

// Cube 搬出: L0C →(FixPipe)→ HBM
def Ascend_CubeStoreOp : AscendModel_Op<"cube_store", [...]> {
  let results = (outs);              // ← 空！store 不产生新 MLIR Value
}

// Vector 搬入: HBM →(VecMTE2)→ UB
def Ascend_VectorLoadOp  : AscendModel_Op<"vector_load", [...]> { ... }

// Vector 搬出: UB →(MTE3)→ HBM
def Ascend_VectorStoreOp : AscendModel_Op<"vector_store", [...]> {
  let results = (outs);              // ← 也是空
}
```

**接下来用 TableGen class 继承做模板，批量定义结构相同的 Op**：

有 5 个二元向量运算（add/sub/mul/max/min），它们的输入输出结构完全相同。作者的做法是定义一个 TableGen class 作为模板，然后一行一个定义：

```tablegen
// 模板
class Ascend_VectorBinarySimple<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, OptionalAttr<I64Attr>:$estimated_cycles, OptionalAttr<I64Attr>:$op_id);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result) }];
}

// 五行，每行定义一个 Op
def Ascend_AddOp : Ascend_VectorBinarySimple<"add"> { let summary = "Element-wise addition on Vector Core"; }
def Ascend_SubOp : Ascend_VectorBinarySimple<"sub"> { ... }
def Ascend_MulOp : Ascend_VectorBinarySimple<"mul"> { ... }
def Ascend_MaxOp : Ascend_VectorBinarySimple<"max"> { ... }
def Ascend_MinOp : Ascend_VectorBinarySimple<"min"> { ... }
```

Div 不同——它不是 1 cycle/op，需要一个额外的 `cycles_per_op` 参数：

```tablegen
class Ascend_VectorBinaryComplex<string mnemonic, int cycles_per_op>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<
        EstimateCyclesOpInterface, ["getCyclesPerVectorOp", "getFlops"]>]> {
  int cyclesPerOp = cycles_per_op;        // ← 通过模板参数传入
}
def Ascend_DivOp : Ascend_VectorBinaryComplex<"div", 4> { ... }
```

同样套路定义了一元、比较、归约 Op：

```
Ascend_VectorUnarySimple  ─→ NegOp, AbsOp, ReluOp, CastOp (1 cycle)
Ascend_VectorUnaryComplex ─→ ExpOp(3), LogOp(4), TanhOp(6), SigmoidOp(5)
Ascend_VectorCmpOp        ─→ CmpEqOp, CmpNeOp, CmpLtOp, CmpLeOp, CmpGtOp, CmpGeOp
Ascend_ReduceOp           ─→ ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp
```

以及结构特殊、不用模板的：

```tablegen
def Ascend_BroadcastOp : AscendModel_Op<"broadcast", [...]> {    // 多一个 DenseI64ArrayAttr:$shape
  let arguments = (ins AnyTensor:$input, DenseI64ArrayAttr:$shape, ...);
}
def Ascend_SelectOp    : AscendModel_Op<"select", [...]> {       // 三个输入
  let arguments = (ins AnyTensor:$condition, AnyTensor:$true_value, AnyTensor:$false_value, ...);
}
def Ascend_SyncOp      : AscendModel_Op<"sync", []> {            // traits=[]，不实现 Interface，不算计算 Op
  let arguments = (ins StrAttr:$sync_type);
  let results = (outs);
}
```

**25 个 Op 的总览**：

| 类别 | 数量 | Op 列表 |
|------|------|--------|
| 矩阵乘 | 1 | MatmulOp |
| 搬运 | 4 | CubeLoadOp, CubeStoreOp, VectorLoadOp, VectorStoreOp |
| 简单二元 | 5 | AddOp, SubOp, MulOp, MaxOp, MinOp |
| 复杂二元 | 1 | DivOp |
| 比较 | 6 | CmpEqOp, CmpNeOp, CmpLtOp, CmpLeOp, CmpGtOp, CmpGeOp |
| 简单一元 | 4 | NegOp, AbsOp, ReluOp, CastOp |
| 复杂一元 | 4 | ExpOp, LogOp, TanhOp, SigmoidOp |
| 归约 | 4 | ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp |
| 特殊 | 3 | BroadcastOp, SelectOp, SyncOp |

---

#### 文件 ④：`Passes.td` —— "Op 都有了，定义怎么处理它们"

作者现在需要一系列 Pass 来转换 IR 和估算性能。每个 Pass 需要**一个名字让 MLIR 命令行能找到、声明依赖哪些 Dialect、定义可选的命令行参数**。

```tablegen
include "mlir/Pass/PassBase.td"
```

**7 个 Pass 按管线顺序定义**：

```tablegen
// ① TTIR → AscendModel IR。必须声明 dependentDialects，因为会创建这些 Dialect 的 Op
def ConvertTritonToAscendPass : Pass<"convert-triton-to-ascend", "ModuleOp"> {
  let dependentDialects = ["mlir::ascend::AscendModelDialect", "mlir::arith::ArithDialect", ...];
}

// ②③ 简单 Pass，没有额外参数
def InsertDataTransfersPass : Pass<"insert-data-transfers", "ModuleOp"> { ... }
def AssignOpIDsPass        : Pass<"assign-op-ids", "ModuleOp"> { ... }

// ④ EstimateCycles —— 需要参数，因为动态循环边界在编译期不知道值
def EstimateCyclesPass : Pass<"estimate-cycles", "ModuleOp"> {
  let options = [
    Option<"argBindingsStr", "arg-bindings", "std::string", "\"\"",
           "Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')">,
    // ↑ Option<C++字段名, 命令行参数名, C++类型, 默认值, 帮助文本>
    Option<"loopTripCountsStr", "loop-trip-counts", "std::string", "\"\"", "...">,
    Option<"hardwareConfigPath", "hardware-config", "std::string", "\"\"", "...">
  ];
}

// ⑤ PipelineAnalysis —— 同样需要动态边界绑定
def PipelineAnalysisPass : Pass<"analyze-pipeline", "ModuleOp"> {
  let options = [ /* 同 EstimateCycles 的三个参数 */ ];
}

// ⑥ HIVM 分析 —— 额外多了调度模式和 Perfetto trace 输出选项
def HIVMAnalysisPass : Pass<"analyze-hivm", "ModuleOp"> {
  let options = [
    Option<"schedulerMode", "scheduler", "std::string", "\"static\"", "...">,
    Option<"reportFile", "report-file", "std::string", "\"\"", "...">,
    Option<"perfettoTraceFile", "perfetto-trace-file", "std::string", "\"\"", "...">,
    /* + arg-bindings, hardware-config */
  ];
}

// ⑦ 最终汇报
def PerfReportPass : Pass<"perf-report", "ModuleOp"> { ... }
```

**关于 `"ModuleOp"` 操作范围**：

`Pass<"pass-name", "操作范围">` 的第二个参数告诉 MLIR：**每遇到一个什么类型的 IR 单元，就跑一次 `runOnOperation()`**。

MLIR 的 IR 是嵌套的：

```
ModuleOp                    ← 最顶层，整个 .mlir 文件
  ├── FuncOp @main          ← 函数
  │   ├── Block              ← 基本块
  │   │   ├── ascend.add     ← 具体 Op
  │   │   └── ...
  └── FuncOp @helper
```

| 操作范围 | 含义 | runOnOperation() 被调用几次 |
|---------|------|---------------------------|
| `"ModuleOp"` | 整个模块 | 1 次（整个 .mlir 文件算一个模块） |
| `"FuncOp"` | 每个函数 | N 次（有几个函数就跑几次） |
| `"Operation *"` | 任意 Op | 由 Pass 内部自己决定 |

AscendModel 的 Pass 全部用 `ModuleOp`，因为每个都需要**全局视角**——比如 `AssignOpIDsPass` 要给所有 Op 分配全局唯一序号，按 `FuncOp` 跑就会每个函数从 0 开始重新编号。

**关于 `arg-bindings` vs `loop-trip-counts` 的区别**：

两者都是解决同一个问题——编译器不知道动态循环边界的具体值——但方式不同。代码逻辑在 `EstimateCycles.cpp:137-168`：

```cpp
for (每个循环) {
    if (循环序号 < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[循环序号];  // ← loop-trip-counts 优先: 按序号直接填
    } else {
        auto result = getScfForTripCountWithBindings(
            forOp, argBindings, programIdBindings);     // ← arg-bindings: 解析循环上下界表达式
    }
}
```

- **`arg-bindings=arg2=128,pid_x=0`** — 符号绑定。IR 里循环是 `for i in range(arg2)`，你告诉 Pass 参数的值，Pass 自己去算 trip count
- **`loop-trip-counts=4,6588`** — 直接覆盖。你不管表达式是什么，直接告诉 Pass "第 0 个循环跑 4 次，第 1 个跑 6588 次"

| | arg-bindings | loop-trip-counts |
|---|---|---|
| 你对 IR 的了解 | 知道参数名，不知道循环顺序 | 知道每个循环跑几次，不关心为什么 |
| Pass 做的事 | 解析循环边界表达式，代入值算出 trip count | 直接按序号赋值 |
| 典型来源 | 手写 MLIR 测试时绑定参数 | JIT 编译框架从 Python 侧传入已计算好的值 |
| 优先级 | 低 | **高**（loop-trip-counts 优先于 arg-bindings） |

---

#### 四个文件的依赖关系总览

```
        ① AscendModelBase.td
        (Dialect 声明 + 枚举定义 + Op 基类)
                  │
                  │ include
                  ▼
        ② AscendModelInterfaces.td
        (Op Interface: estimateCycles/getHWUnit/...)
                  │
                  │ include (Base 和 Interfaces 两个)
                  ▼
        ③ AscendModelOps.td ────────────── ④ Passes.td
        (25 个 Op 定义)                    (7 个 Pass 声明，其中 6 个加入管线)
        依赖 ①②                           不依赖 ①②③（独立）
```

**Passes.td 不依赖前面三个**——它只定义 Pass 的元数据（名字、参数），不引用具体的 Op 类型。Pass 的真实逻辑在手写 `.cpp` 里才引用 Op。

---

#### 对应关系：Dialect 需要注册的 Op/Type/Attr 分别对应哪些 .td

回到你之前的问题：`initialize()` 里注册了三样东西——Op、Type、Attribute。对应的分别是：

| initialize() 注册 | 对应 .td 中的定义 | 说明 |
|-------------------|------------------|------|
| **Op** | `AscendModelOps.td` 中 25 个 `def Ascend_*Op` | 通过 `addOperations<AddOp, SubOp, ...>()` 注册到 Op 注册表 |
| **Type** | 无 | AscendModel 没有自定义类型，完全使用 MLIR 内置类型（`AnyTensor`、`RankedTensorType`）。`.td` 中的 `Types.h.inc`/`.cpp.inc` 是空壳——TableGen 按模板生成了文件框架但没有内容 |
| **Attr** | `AscendModelBase.td` 中 5 个 `I32EnumAttr` | `HWUnitAttr`、`MemSpaceAttr`、`DataTypeAttr`、`VecOpKindAttr`、`ReduceKindAttr`——这些是**枚举属性**。生成的 `Enums.cpp.inc` 里包含它们的 `stringifyXxx`/`symbolizeXxx` 函数，`Dialect.cpp.inc` 里通过 `addAttributes<>` 注册 |

**为什么 AscendModel 没有自定义 Type？** 因为这个 Dialect 的目的不是定义新的数据类型——它只描述"操作在哪个硬件单元上、消耗多少 cycle"。Op 的输入输出直接用 MLIR 内置的 `tensor<...>`，不需要自定义类型。

**为什么 AscendModel 的 Attr 全是枚举？** 因为硬件单元、内存空间、数据类型这些属性的合法值都是**有限集合**——不存在一个 Op 声明 `hw_unit = "某种不存在的单元"`。枚举属性天然保证了值的合法性，`verify()` 阶段自动拒绝非法值。

---

---

### 6.0k. 写完 .td 后的心路历程 —— 怎么组织 .h 和 .cpp

四个 .td 文件写完之后，TableGen 会生成 12 个 `.inc` 文件。但这些只是代码片段——它们需要被 include 进真正的 `.h` 和 `.cpp` 里，还需要大量手写的业务逻辑来填充。

作者接下来按四个阶段逐步搭建：

---

#### 阶段一：胶水文件 —— 组织 .inc 的 include 顺序（`include/IR/`）

`.inc` 文件不是独立的，它们必须按正确顺序 `#include` 进真正的 `.h`/`.cpp` 文件。作者需要两个胶水文件：

**① `AscendModelInterfaces.h`** —— 包装 Interface 的 .inc：

```cpp
// 先做 forward declaration——Op 类声明需要知道 HardwareConfig 和 HWUnit 的存在
namespace mlir::ascend {
class HardwareConfig;
enum class HWUnit : uint32_t;
}
// 然后 include TableGen 生成的 Interface 声明
#include "AscendModel/IR/AscendModelInterfaces.h.inc"
```

**为什么 `HardwareConfig` 只要 forward declare 而不是完整 include？** Interface 的方法签名里只用到 `const HardwareConfig &`——引用类型在 C++ 里不需要知道完整定义，只需要知道"这是一个类"。这样可以避免 `Interfaces.h` 依赖 `HardwareConfig.h`，打破循环依赖。

**② `AscendModelDialect.h`** —— 按依赖顺序组织所有生成的 .inc：

```
① #include "AscendModel/IR/AscendModelInterfaces.h"      ← Interface（Op 类依赖它）
② #include "AscendModel/IR/AscendModelDialect.h.inc"      ← Dialect 声明
③ #include "AscendModel/IR/AscendModelOpsEnums.h.inc"     ← 枚举声明
④ #include "AscendModel/IR/AscendModelOpsAttrDefs.h.inc"  ← 属性声明
⑤ #include "AscendModel/IR/AscendModelOpsTypes.h.inc"     ← 类型声明
⑥ #define GET_OP_CLASSES
   #include "AscendModel/IR/AscendModelOps.h.inc"           ← Op 类声明（最后！依赖上面全部）
```

**为什么 Op 类要放最后？** 因为 `Ops.h.inc` 展开的每个 Op 类使用了 `HWUnit` 枚举（从 ③ 来）、`EstimateCyclesOpInterface`（从 ① 来）、Dialect 类型（从 ② 来）。C++ 的 `#include` 顺序决定了编译时的可见性——不按这个顺序，编译直接报错 "unknown type"。

---

#### 阶段二：分析层头文件 —— 业务逻辑的数据结构（`include/Analysis/`）

有了 Op 定义，但 costmodel 还需要**非 Op 的数据结构和算法**。这些全是手写，没有 TableGen 参与。

**③ `Analysis/HardwareConfig.h`** —— 硬件参数的 C++ 表达

- `MemorySpace`、`ComputeUnit`、`DataMover`、`PipelinePath` 等结构体
- `HardwareConfig` 类：`loadFromFile()`、`parseJSON()`、`getClockFrequencyGHz()`、`estimateCubeCycles()` 等查询方法
- 全局配置管理：`getHardwareConfig()`、`setHardwareConfig()`

**为什么需要单独的 `HardwareConfig` 类而不是放在 `AscendModelBase.td` 的枚举里一起生成？** `.td` 文件只能生成 MLIR 相关的代码（Op、类型、属性、枚举、Pass）。`HardwareConfig` 是纯 C++ 业务逻辑——JSON 解析、带宽计算、校准参数管理——跟 MLIR 框架无关，TableGen 生成不了。

**④ `Analysis/PipelineAnalysis.h`** —— 调度器数据结构

- `PipelineOp`：opId、hwUnit、startCycle、duration、dependsOn
- `HWUnitPipeline`：一个硬件单元上的操作队列
- `DependencyGraph`：邻接表 + 拓扑排序 + 循环检测
- `PipelineScheduler`：ASAP 调度算法、getKernelCycles
- `RooflineAnalyzer`：算术强度、compute/memory bound 判断
- `PerformanceReport`：打印和 JSON 导出的报告结构

**⑤ `Analysis/Utils.h`** —— 通用工具（1000 行的庞然大物）

这是整个 costmodel 里最大的单文件。它的出现是因为 `EstimateCycles.cpp` 和 `PipelineAnalysisPass.cpp` 都需要：

- tensor 元素个数/字节数的提取 —— `getNumElements()`, `getByteSize()`, `getElementBitWidth()`
- 动态循环边界分析 —— `SymbolicBound`（表达 `%arg2 * 4 + C` 这样的表达式）, `analyzeScfForTripCount()`, `getScfForTripCountWithBindings()`
- 表达式求值 —— `evaluateValue()`（在给定 `arg2=128` 后计算表达式实际值），支持 arith 全套指令（add/sub/mul/div/mod/and/or/xor/shl/shr/min/max/select/cmp/ext/trunc/index_cast 等 20+ 种运算）
- `arg-bindings` 字符串解析 —— `parseBindings("arg2=128,pid_x=0")` → `DenseMap<unsigned, int64_t>`，兼容多种输入格式
- `loop-trip-counts` 字符串解析 —— `parseLoopTripCounts("4,6588")` → `SmallVector<int64_t>`
- 循环嵌套追踪 —— `getLoopMultiplier()`, `getEnclosingLoops()`, `isInsideLoop()`

**为什么 `Utils.h` 这么大？** 这些功能本可以拆成多个文件（类型工具、循环分析、绑定解析、表达式求值），但作者选择了单文件——因为所有 Pass 都需要它们，include 一个头文件比 include 五个方便。

**⑥ 顶层转发头文件**：

- `include/AscendModel/HardwareConfig.h` → `#include "AscendModel/Analysis/HardwareConfig.h"`
- `include/AscendModel/Utils.h` → `#include "AscendModel/Analysis/Utils.h"`

**为什么需要这一层？** 外部调用者（如单元测试、Python 绑定）写 `#include "AscendModel/HardwareConfig.h"` 比 `#include "AscendModel/Analysis/HardwareConfig.h"` 短。而且隐藏了内部目录结构——以后 Analysis 目录改名叫 Core 也不影响外部。

---

#### 阶段三：Pass 声明头文件（`include/Transforms/`）

**⑦ `Transforms/Passes.h`** —— 包装 `Passes.h.inc` + 声明管线注册函数：

```cpp
// ① TableGen 生成的 Pass 声明
#define GEN_PASS_DECL
#include "AscendModel/Transforms/Passes.h.inc"

// ② TableGen 生成的 Pass 注册宏
#define GEN_PASS_REGISTRATION
#include "AscendModel/Transforms/Passes.h.inc"

// ③ 手写：管线注册函数
void registerAscendModelPipeline();
```

**为什么 GEN_PASS_DECL 和 GEN_PASS_REGISTRATION 要分开两次 include 同一个 .inc？** 就跟 Ops 的 GET_OP_LIST vs GET_OP_CLASSES 一样——同一个 .inc 文件，用不同的 `#define` 展开出不同的内容。`GEN_PASS_DECL` 展开各个 Pass 创建函数的声明（`std::unique_ptr<Pass> createEstimateCyclesPass()`），`GEN_PASS_REGISTRATION` 展开 Pass 注册表的静态初始化代码。

---

#### 阶段四：.cpp 实现（`lib/`）

**⑧ `lib/IR/AscendModelDialect.cpp`** —— Dialect 初始化：`initialize()` 里 `#define GET_OP_LIST` → include `Ops.cpp.inc` → 展开 Op 名列表 → 注册到 MLIR。

**⑨ `lib/IR/AscendModelOps.cpp`** —— 25 个 Op 各自实现 `estimateCycles()`、`getHWUnit()`、`getFlops()`。用宏批量展开简单 Op，手写复杂 Op 的公式。

**⑩ `lib/Analysis/HardwareConfig.cpp`** —— JSON → C++ 配置对象。`getDefault910B()` 先搜 ASCEND_CONFIG_PATH 环境变量指定的路径，搜不到就用 hardcoded fallback。校准参数（`getVectorStartupLatency()=35`、`getAIVScalarOverheadFactor()=3.74`）也在这里。

**⑪ `lib/Analysis/PipelineAnalysis.cpp`** —— ASAP 调度算法。（已在 6.2d 节详细分析。）

**⑫ `lib/Analysis/RooflineAnalysis.cpp`** —— Roofline 瓶颈分析。（与 PipelineAnalysis 紧密相关，拆出来因为职责不同：前者调度，后者分析。）

**⑬ `lib/Transforms/ConvertTritonToAscend.cpp`** —— 最复杂的 Pass（850 行）。分两个阶段：Phase 1 标记 `tt.load` 是否被 `tt.dot` 使用（决定生成为 `cube_load` 还是 `vector_load`）；Phase 2 用 RewritePattern 做模式替换——`tt.dot → ascend.matmul`、`tt.load → ascend.cube_load/vector_load`、`arith.addf → ascend.add` 等 12 种转换模式。

**⑭ `lib/Transforms/InsertDataTransfers.cpp`** —— 插入显式搬运 Op。

**⑮ `lib/Transforms/AssignOpIDs.cpp`** —— 遍历所有 AscendModel Op，从 0 开始编号。

**⑯ `lib/Transforms/EstimateCycles.cpp`** —— 核心 Pass。三遍遍历：第一遍确定循环 trip count（arg-bindings 解析 + loop-trip-counts 覆盖）；第二遍遍历所有 Compute Op，调 `estimateCycles()` 写 attributes；第三遍标注循环体。同时收集 Roofline 统计。

**⑰ `lib/Transforms/PipelineAnalysisPass.cpp`** —— 包装 `PipelineScheduler` 为 Pass。

**⑱ `lib/Transforms/HIVMAnalysisPass.cpp`** —— HIVM IR 分析 Pass。

**⑲ `lib/Transforms/PerfReportPass.cpp`** —— 汇总所有 attributes → `PerformanceReport`。

**⑳ `lib/Transforms/PassRegistration.cpp`** —— 注册 `ascend-perf-model` 管线：6 个 Pass 的顺序在这里确定。

---

#### 完整创建顺序总览

```
┌─────────────────────────────────────────────────────────────┐
│  先 (仅依赖 MLIR 框架)                                       │
├─────────────────────────────────────────────────────────────┤
│  ① AscendModelBase.td        Dialect + 枚举 + Op 基类       │
│  ② AscendModelInterfaces.td  Op Interface 定义              │
│  ③ AscendModelOps.td         25 个 Op 定义                  │
│  ④ AscendModelInterfaces.h   胶水: 包装 Interfaces.h.inc     │
│  ⑤ AscendModelDialect.h      胶水: 按顺序 include 所有 .inc  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  中 (依赖 Op 定义 + 硬件知识)                                 │
├─────────────────────────────────────────────────────────────┤
│  ⑥ Analysis/HardwareConfig.h  硬件参数 C++ 表达              │
│  ⑦ Analysis/PipelineAnalysis.h 调度器 + Roofline 分析        │
│  ⑧ Analysis/Utils.h           1000 行通用工具                │
│  ⑨ HardwareConfig.h/Utils.h   顶层转发头文件                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  后 (依赖 Op + 分析层)                                       │
├─────────────────────────────────────────────────────────────┤
│  ⑩ Passes.td                  7 个 Pass 声明                │
│  ⑪ Transforms/Passes.h        胶水: 包装 Passes.h.inc        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  实现层 (全部依赖上面)                                        │
├─────────────────────────────────────────────────────────────┤
│  ⑫ lib/IR/AscendModelDialect.cpp     Dialect 初始化          │
│  ⑬ lib/IR/AscendModelOps.cpp         25 个 Op 的 estimateCycles │
│  ⑭ lib/Analysis/HardwareConfig.cpp   JSON → C++ 配置         │
│  ⑮ lib/Analysis/PipelineAnalysis.cpp 调度算法                 │
│  ⑯ lib/Analysis/RooflineAnalysis.cpp Roofline 分析            │
│  ⑰ lib/Transforms/ConvertTritonToAscend.cpp (第1个Pass,最复杂) │
│  ⑱ lib/Transforms/InsertDataTransfers.cpp                    │
│  ⑲ lib/Transforms/AssignOpIDs.cpp                            │
│  ⑳ lib/Transforms/EstimateCycles.cpp (核心Pass)               │
│  ○ lib/Transforms/PipelineAnalysisPass.cpp                   │
│  ○ lib/Transforms/HIVMAnalysisPass.cpp                       │
│  ○ lib/Transforms/PerfReportPass.cpp                         │
│  ○ lib/Transforms/PassRegistration.cpp (管线顺序注册)          │
└─────────────────────────────────────────────────────────────┘
```

---

### 6.0k. 写完 .td 后的心路历程 —— 从作者视角逐文件搭建 .h 和 .cpp

四个 .td 文件写完后，TableGen 自动产出 13 个 `.inc` 文件。但这些只是代码片段——必须嵌入到真正的 `.h`/`.cpp` 里。剩下的全部需要手写。

作者接下来面对的问题是：**从哪开始？按什么顺序？**

---

#### 阶段一：胶水文件 —— 让 .inc 有家可归

TableGen 生成了一堆 `.h.inc` 和 `.cpp.inc`，但它们不是独立的头文件——只是代码片段。作者需要一个"容器"来 include 它们。

**文件 1: `include/AscendModel/IR/AscendModelInterfaces.h`**

作者打开编辑器。第一个念头：Interface 的 .inc 需要一个头文件来包装。但它引用了 `HardwareConfig` 类——而 `HardwareConfig.h` 还在 `Analysis/` 目录下，还没写呢。怎么办？

**用 forward declaration 打破循环依赖**。C++ 里，如果只用到类型的引用（`const HardwareConfig &config`），不需要完整定义——只声明"有这么个类"就够了：

```cpp
#include "mlir/IR/OpDefinition.h"

namespace mlir::ascend {
  class HardwareConfig;            // ← forward declare: 只需要"这个类存在"
  enum class HWUnit : uint32_t;    // ← forward declare: 只需要"这个枚举存在"
}

#include "AscendModel/IR/AscendModelInterfaces.h.inc"   // ← 生成的 Interface 声明
```

**为什么加 include guard `ASCEND_MODEL_INTERFACES_INC_`？** `Interfaces.h.inc` 会被多个文件 include：`Interfaces.h` 一次（生成声明），`Ops.cpp` 一次（生成默认实现）。但声明只能出现一次，否则编译报 redefinition。所以用 `#ifndef` 确保第一次 include 时才展开声明，后面直接跳过。

---

**文件 2: `include/AscendModel/IR/AscendModelDialect.h`**

第二个胶水文件。作者需要把所有 Op 相关的 .inc 按正确的依赖顺序 include 进来。如果顺序错了，编译就报 "unknown type"——因为 C++ 不允许使用先于声明。

```cpp
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
// ... MLIR 基础头文件 ...

// === Forward declaration ===
namespace mlir::ascend {
class HardwareConfig;
}

// === ① 先 include Interface（Op 类依赖它） ===
#include "AscendModel/IR/AscendModelInterfaces.h"

// === ② Dialect 声明 ===
#include "AscendModel/IR/AscendModelDialect.h.inc"

// === ③ 枚举声明（HWUnit, MemSpace, ...） ===
#include "AscendModel/IR/AscendModelOpsEnums.h.inc"

// === ④ 自定义属性 ===
#define GET_ATTRDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsAttrDefs.h.inc"

// === ⑤ 自定义类型 ===
#define GET_TYPEDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsTypes.h.inc"

// === ⑥ Op 类声明（最后！） ===
#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.h.inc"
```

**作者思考 include 顺序的心路**：

- ① Interface 放前面——因为 `Ops.h.inc` 展开的每个 Op 类继承自 `EstimateCyclesOpInterface::Trait`，需要先用 `Interfaces.h.inc` 声明 `Trait` 类
- ② Dialect 放第二——`Ops.h.inc` 展开的 Op 类是 `AscendModelDialect` 的内部类型，Dialect 声明必须先存在
- ③ 枚举放第三——像 `AddOp::getHWUnit()` 返回 `HWUnit::Vector`，枚举不先声明，函数签名就编译不过
- ④⑤ 属性/类型放中间——如果有自定义 `#ascend.tile` 属性，Op 类可能用 `TileAttr` 做成员类型
- ⑥ Op 类最后——它需要前面所有声明都就绪

---

#### 阶段二：Analysis 头文件 —— 纯 C++ 业务逻辑

作者意识到：TableGen 能生成 MLIR Op 相关的代码，但 costmodel 需要的硬件参数加载、调度算法、循环边界分析这些**跟 MLIR 框架无关**的东西，必须纯手写。

**文件 3: `include/AscendModel/Analysis/HardwareConfig.h`**

作者设计 `HardwareConfig` 的出发点不是"硬件有哪些参数"，而是反向思维——**先看调用者需要什么，再决定放什么**。只有两类调用者：

```
调用者                                    调用了什么
────────                                  ──────────
每个 Op 的 estimateCycles():               getClockFrequencyGHz()
  (AscendModelOps.cpp)                    getHBMBandwidthGBs()
                                          getCubeFractalSize()
                                          getCubeStartupLatency()
                                          getMTE2StartupLatency()
                                          getFixPipeStartupLatency()
                                          getMTE3StartupLatency()
                                          getVectorStartupLatency()

PipelineScheduler + RooflineAnalyzer:     getClockFrequencyGHz()
  (PipelineAnalysis.cpp)                  getName()
                                          getPipeBarrierCyclesPerIter()
                                          getAIVScalarOverheadFactor()
                                          getNumAICCores()
                                          getNumAIVCores()
                                          getCubeTFlopsFP16()
                                          getMemoryBandwidthTBps()
                                          cyclesToMicroseconds()
```

这就决定了必须有哪些成员和方法。然后作者按"数据的自然归属"分组。

---

**设计推演：从调用需求倒推成员**

**需求 1：`getClockFrequencyGHz()`** ——每个 Op 估算 cycle 都需要。存一个 `double clockFreqGHz`。数据来源于 JSON 的 `"clock": {"frequency_ghz": 1.85}`，所以 `parseJSON()` 要解析这个字段。

**需求 2：Cube Op 估算——`getCubeFractalSize()` + `getCubeStartupLatency()` + `getCubeTFlopsFP16()`**。MatmulOp::estimateCycles 需要知道 Cube Core 的硬件粒度——910B 上 FP16 是 16×16×16 的 fractal，FP32 是 16×8×16。这些数据在 JSON 的 `compute_units.cube` 下，需要一个 `ComputeUnit` struct 承载：

```cpp
struct ComputeUnit {
  std::string name;
  ComputeUnitType type;       // MatrixEngine / SIMDEngine / ScalarEngine
  double tflopsFP16, tflopsFP32, tflopsINT8;
  int tileM, tileN, tileK;    // 默认 16x16x16
  StringMap<FractalSize> fractalSizes;  // {"fp16": {16,16,16}, "fp32": {16,8,16}}
  int widthElements;          // Vector 专用：一次处理 128 个元素
};
```

对应的 getter：

```cpp
void getCubeFractalSize(int elementBits, int &m, int &n, int &k) const {
  auto *cube = getComputeUnit("cube");
  StringRef dtypeKey = (elementBits == 8) ? "int8" : (elementBits == 32) ? "fp32" : "fp16";
  auto it = cube->fractalSizes.find(dtypeKey);
  if (it != cube->fractalSizes.end()) { m = it->second.m; n = it->second.n; k = it->second.k; }
}
```

`getCubeStartupLatency()` 不是 JSON 里读的，是 hand-tuned 校准参数——直接在函数体里 `return 20;`。

**需求 3：内存搬运 Op 估算——`getHBMBandwidthGBs()` + 四个 startup latency。** CubeLoadOp::estimateCycles 公式是 `bytes / bandwidth * clock`，需要知道 HBM 带宽。JSON 里写的是 `bandwidth_gbps`，解析时转换成 bytes/cycle：

```cpp
struct MemorySpace {
  std::string name;
  MemoryType type;           // OffChip(HBM) / OnChipShared(L2) / OnChipLocal(L1/UB) / RegisterFile(L0)
  size_t sizeBytes;
  double bandwidthBytesPerCycle; // ← 注意：parseJSON 时从 GB/s 一次性转换
  int latencyCycles;
};
```

**为什么 bandwidth 存 bytes/cycle 而不是 GB/s？** JSON 里填写时用 GB/s（人可读），但估算 cycle 时公式是 `ceil(bytes / bandwidthBytesPerCycle)`。如果存 GB/s，每次估算都要 `bytes * clockFreq / GBps` 多算一步，不如在 parseJSON 时一次性转换好。查询时如果需要 GB/s，再反向转换：

```cpp
double getHBMBandwidthGBs() const {
  auto *hbm = getMemorySpace("hbm");
  return hbm->bandwidthBytesPerCycle * clockFreqGHz * 1e9 / 1e9;  // bytes/cycle → GB/s
}
```

**需求 4：PipelineScheduler 的 kernel 级 cycle 估算——校准参数。** 这是最体现"costmodel 不是纯理论模型"的地方。理论上 kernel_cycles = total_cycles × waves，实际测下来完全不对——AIV 上 scalar 开销和 pipe_barrier 同步占了总时间的 79%：

```cpp
double getAIVScalarOverheadFactor() const { return 3.74; }
// 校准来源：FlashAttention 实测 —— AIV 墙钟时间中 vec_time 只占 21%
// 纯 vector 时间需要乘以 3.74 才能匹配实测

int getPipeBarrierCyclesPerIter() const { return 7500; }
// 校准来源：BM=64, 1-wave, 3 次内部迭代
// AIV wall = 59187 cycles, active = 61.1%, idle = 23044
// 23044 / 3 ≈ 7500 cycles/barrier

int getNumAICCores() const { return 20; }
int getNumAIVCores() const { return 40; }
// 来自 profiling 配置：Block Dim=20, Mix Block Dim=40
```

这些不是 JSON 里的硬件规格，而是**实测拟合出来的修正系数**。放在 HardwareConfig 还是 PipelineAnalysis？作者选择 HardwareConfig——因为它们是硬件行为特征的一部分，换一块不同型号的芯片就要重新 calibrate。

**需求 5：Roofline 分析——`getCubeTFlopsFP16()` + `getMemoryBandwidthTBps()`。** RooflineAnalyzer 算 ridge point：`peakTFLOPS / peakBWTBps`。数据已在 ComputeUnit 和 MemorySpace 里，只需要 getter。

**需求 6：DataMover——搬运路径建模。** JSON 里定义了 5 个 DataMover：

```cpp
struct DataMover {
  std::string name, srcSpace;
  std::vector<std::string> dstSpaces;
  double bandwidthBytesPerCycle;
  int maxBurstBytes, alignmentBytes;
  bool supportsAccumulate, supportsCast;
};
// cube_mte2: HBM→L1, mte1: L1→L0A/L0B, fixpipe: L0C→HBM
// vector_mte2: HBM→UB, mte3: UB→HBM
```

用途：为更精确的搬运建模预留接口。当前代码里各 Op 直接调 `getMTE2StartupLatency()` 而不是通过 DataMover 查，但结构体为精细化建模留了空间。

**需求 7：Pipeline 并行标记——`canRunInParallel()`。** 查 `parallelismFlags` Map——`"cube_and_vector"` → true。

**工厂方法。** 三条创建路径全部封装为静态工厂方法：

```cpp
static std::unique_ptr<HardwareConfig> loadFromFile(StringRef path);     // 读 JSON 文件
static std::unique_ptr<HardwareConfig> loadFromJSON(const json::Value &); // 从 JSON 对象
static std::unique_ptr<HardwareConfig> getDefault910B();                 // 搜文件或 hardcoded
```

**为什么用工厂方法而不是直接 `new`？** 工厂方法保证返回的是一个**完全初始化好、经过校验的对象**。如果让调用者直接 `new HardwareConfig()` 再手动调 `parseJSON()`，中间可能忘掉初始化步骤，拿到一个空壳。工厂方法把创建过程封装起来，外部只需要 `auto config = HardwareConfig::loadFromFile("ascend_910b.json");`，不可能拿到一个半成品。

---

**最终完整布局——按数据来源分类**

```
class HardwareConfig {
  // ========== 工厂方法 ==========
  static loadFromFile(path)           → 文件路径 → 读文件 → parseJSON
  static loadFromJSON(json)           → JSON 对象 → parseJSON
  static getDefault910B()             → 搜文件，找不到用 hardcoded fallback

  // ========== 基本信息 ==========
  getName()                           → "Ascend 910B"
  getClockFrequencyGHz()              → 1.85
  cyclesToMicroseconds(cycles)        → cycles / (GHz * 1000)

  // ========== 内存空间查询（MemorySpace） ==========
  getMemorySpace("hbm")               → {size=32GB, bw=865B/cycle, lat=200}
  getHBMBandwidthGBs()                → 1600 GB/s     (给 Op 估算)
  getMemoryBandwidthTBps("hbm")       → 1.6 TB/s      (给 Roofline)

  // ========== 计算单元查询（ComputeUnit） ==========
  getComputeUnit("cube")              → {320 TFLOPS FP16, 16x16x16}
  getCubeTFlopsFP16()                 → 320           (给 Roofline)
  getCubeFractalSize(bits, m,n,k)     → 16,16,16      (给 MatmulOp)
  getVectorTFlopsFP32()               → 10            (给 Roofline)
  getVectorWidthElements()            → 128           (给 estimateVectorCycles)

  // ========== 搬运单元查询（DataMover） ==========
  getDataMover("cube_mte2")           → {bw=108B/cycle, burst=64KB}

  // ========== 代价估算（高层封装） ==========
  estimateCubeCycles(M,N,K)           → ceil(M/16)*ceil(N/16)*ceil(K/16)
  estimateVectorCycles(N)             → ceil(N/128)
  estimateMemoryCycles(mover,bytes)   → ceil(bytes / bw)
  estimateMemoryCyclesWithLatency     → 同上 + latency

  // ========== 校准参数（硬件行为修正） ==========
  getMTE2StartupLatency() → 50   getCubeStartupLatency() → 20
  getMTE3StartupLatency() → 40   getVectorStartupLatency() → 35
  getFixPipeStartupLatency() → 30
  getAIVScalarOverheadFactor() → 3.74
  getPipeBarrierCyclesPerIter() → 7500
  getNumAICCores() → 20    getNumAIVCores() → 40

  // ========== 流水线并行查询 ==========
  getPipelinePath("cube_path")         → ["cube_mte2","mte1","cube","fixpipe"]
  canRunInParallel("cube", "vector")   → true

  // ========== 校验 ==========
  validate(error)                      → 检查必填字段、引用一致性

  // ========== 私有数据 ==========
  string name, vendor, version;
  double clockFreqGHz;
  StringMap<MemorySpace> memorySpaces;      // "hbm"/"l2"/"l1"/"ub"/"l0a"/"l0b"/"l0c"
  StringMap<ComputeUnit> computeUnits;      // "cube"/"vector"
  StringMap<DataMover> dataMovers;          // "cube_mte2"/"mte1"/"fixpipe"/"vector_mte2"/"mte3"
  StringMap<PipelinePath> pipelinePaths;    // "cube_path"/"vector_path"
  StringMap<bool> parallelismFlags;         // "cube_and_vector"→true
  StringMap<int> vectorOpCyclesPerInstruction; // "vexp"→4, "vdiv"→3, ...
};
```

**设计心法**：HardwareConfig 的本质是一个"硬件字典"——外部调用者用名字（`"cube"`、`"hbm"`、`"vector"`）来查参数。成员由 JSON schema 决定（`MemorySpace`/`ComputeUnit`/`DataMover` 对应 JSON 的三个顶层 object），方法由调用者的需求决定——`estimate` 系列给 Op 用、`get` 系列给 Roofline 用、校准系列给 PipelineScheduler 用。

---

**文件 4: `include/AscendModel/Analysis/PipelineAnalysis.h`**

作者写完 HardwareConfig 和 AscendModelOps.cpp 后，遇到一个根本问题：**每个 Op 的 `estimateCycles()` 只能算出单个 Op 的耗时。但一个 kernel 里有几十个 Op，它们的总耗时不是简单求和——不同硬件单元可以并行跑。需要调度。** 这就是这个文件存在的理由。

作者从头推演，逐步遇到问题、逐步加结构：

---

**第 1 步：先定义"被调度的最小单元"**

调度器要处理什么东西？每个 Op 有若干信息：它在哪个硬件单元上运行、它需要多少 cycle、它依赖哪些前置 Op。所有这些来自 EstimateCyclesPass 在 IR 上写的 attributes。

```cpp
struct PipelineOp {
  int64_t opId;                          // AssignOpIDs 给的唯一序号
  HWUnit hwUnit;                         // EstimateCyclesPass 写的 hw_unit
  int64_t startCycle;                    // 调度后由算法填充
  int64_t duration;                      // = estimated_cycles，来自 EstimateCyclesPass
  int64_t endCycle;                      // = startCycle + duration
  Operation *mlirOp;                     // 回指 MLIR IR 上的原始 Op
  std::string opName;                    // "ascend.add" 之类的可读名

  int64_t bytes;                         // 搬运量（搬运 Op 才有）
  int64_t flops;                         // 计算量（计算 Op 才有）
  int64_t loopMultiplier;                // 这个 Op 在循环里执行了几次

  SmallVector<int64_t, 4> dependsOn;     // 核心：依赖哪些 Op（必须等它们完成）
};
```

**为什么 `dependsOn` 直接存 opId 列表而不是用 MLIR 的 use-def chain？** MLIR 的 use-def 表示数据流——"Op B 用了 Op A 的输出"不等于"B 必须等 A 完成"。例如 `ascend.vector_load` 和 `ascend.add` 在不同硬件单元上，load 完成了 add 就可以开始（数据在 UB 里），不需要等 load 完全结束。但有些依赖是硬件强制性的——比如同一个 MTE2 上的两次加载必须串行。这些硬件层面的约束 MLIR use-def 表达不了，所以用**显式的 opId 列表**，由 PipelineAnalysisPass 写入。

---

**第 2 步：怎么表示"一个硬件单元的忙碌时间线"**

作者有 7 个硬件单元（Scalar、CubeMTE2、MTE1、Cube、FixPipe、Vector、VecMTE2、MTE3）。每个单元同一时刻只能执行一个 Op。需要一个数据结构来跟踪每个单元的"当前可用时间"。

```cpp
class HWUnitPipeline {
  HWUnit unit;                           // 这是哪个硬件单元
  int64_t currentCycle;                  // 这个单元下一次空闲是什么时候
  vector<PipelineOp *> scheduledOps;     // 已经排到这个单元上的所有 Op

  void scheduleOp(PipelineOp &op, int64_t earliestStart) {
    // start = max(这个管道什么时候空闲, 依赖什么时候满足)
    op.startCycle = max(currentCycle, earliestStart);
    op.endCycle = op.startCycle + op.duration;
    currentCycle = op.endCycle;          // 更新空闲时间
  }

  int64_t getTotalBusyCycles();          // 算利用率用
  double getUtilization(int64_t totalCycles);
};
```

**上面这段代码是整个调度器的物理基石**——一句话表达完了 ASAP 的全部规则：一个 Op 的起始时间 = "硬件管道不忙" AND "所有前置 Op 已完成" 中较晚的那个。

**为什么 `currentCycle` 就够了？** 假设 MTE2 上已经有 Op1（0→100 cycle），现在 Op2（duration=50）要排进来。`currentCycle=100`，`earliestStart=0`（无依赖），所以 Op2.start = max(100,0) = 100。硬件上 MTE2 只能串行——前一个搬运完，下一个才能开始。`currentCycle` 精确建模了这个约束。

---

**第 3 步：怎么表达"Op B 依赖 Op A"**

这是调度器中必须处理的最棘手问题——循环依赖导致调度死锁。作者用标准图论解决：

```cpp
class DependencyGraph {
  DenseMap<int64_t, Operation *> ops;                    // opId → MLIR Op
  DenseMap<int64_t, SmallVector<int64_t, 4>> edges;      // 正向: A → [B, C]（A 完成后 B、C 才能开始）
  DenseMap<int64_t, SmallVector<int64_t, 4>> reverseEdges; // 反向: B → [A]（B 依赖 A→查谁依赖我）

  void addOp(int64_t opId, Operation *op) { ops[opId] = op; }
  void addDependency(int64_t from, int64_t to) {
    edges[from].push_back(to);          // from → to
    reverseEdges[to].push_back(from);   // 同步更新反向
  }

  // Kahn 拓扑排序：BFS 减入度
  vector<int64_t> getTopologicalOrder() {
    // ① 算每个节点的入度
    // ② 入度为 0 的进队列
    // ③ BFS：出队 → 减后继入度 → 入度为 0 的入队
  }

  // 循环检测：如果拓扑序不包含全部节点，一定有环
  bool hasCycle() { return getTopologicalOrder().size() != ops.size(); }
};
```

**为什么同时维护 edges 和 reverseEdges？** 拓扑排序只需要 edges（正向）。但 `getEarliestStartTime()` 需要反向查"我的前置 Op 都结束了吗"——这需要 `reverseEdges`。作者选择用空间换时间：每个 `addDependency` 时多写一次，换来了 O(1) 的反向查询。

**为什么必须检测循环依赖？** 如果 PipelineAnalysisPass 的依赖分析有 bug，造出了 A→B→A 这样的环，调度器会死锁——每个 Op 都在等一个永远不会完成的前置 Op。`schedule()` 的第一步就是 `if (hasCycle()) return false;`，直接拒绝，避免无限循环。

---

**第 4 步：主角——PipelineScheduler**

现在有了被调度单元（`PipelineOp`）、硬件管道（`HWUnitPipeline`）、依赖关系（`DependencyGraph`），可以写出核心调度逻辑：

```cpp
class PipelineScheduler {
  const HardwareConfig *hwConfig;
  vector<PipelineOp> operations;                // 所有待调度的 Op
  DependencyGraph depGraph;                     // 依赖图
  map<HWUnit, HWUnitPipeline> pipelines;        // 7 个硬件单元各自的时间线
  int64_t totalCycles;                           // 调度结果的时钟周期总数

  bool schedule() {
    if (depGraph.hasCycle()) return false;       // ① 循环检测拒绝

    vector<int64_t> order = depGraph.getTopologicalOrder(); // ② 拓扑排序

    for (int64_t opId : order) {
      PipelineOp &op = operations[opIdToIndex[opId]];
      int64_t earliestStart = getEarliestStartTime(op);   // ③ 查所有前置 Op 的最大 endCycle

      auto &pipeline = pipelines[op.hwUnit];               // ④ 找到对应的硬件管道
      pipeline.scheduleOp(op, earliestStart);              // ⑤ 排上去！
      totalCycles = max(totalCycles, op.endCycle);         // ⑥ 更新全局时钟
    }
    return true;
  }

  // 最关键的时间计算：
  int64_t getEarliestStartTime(const PipelineOp &op) {
    int64_t earliest = 0;
    for (int64_t depId : op.dependsOn) {
      const auto &depOp = /* 找到 depId 对应的 Op */;
      earliest = max(earliest, depOp.endCycle);  // 等最慢的前置 Op 完成
    }
    return earliest;
  }
};
```

**拓扑排序为什么是必要的？** 如果 `schedule()` 直接按数组顺序遍历，可能遇到"先调度消费者、后调度生产者"——消费者排的时候生产者的 `endCycle` 还是 0，算出来的时间就错了。拓扑排序保证处理某个 Op 时，它依赖的所有前驱 Op 都已经排好了。

**调度过程实例**：假设 3 个 Op：Op0(Vector, dur=40)、Op1(Vector, dur=60, 依赖 Op0)、Op2(Cube, dur=100)

```
Step 1: 拓扑序 = [0, 1, 2]（0 不依赖任何人，1 依赖 0，2 不依赖任何人）
Step 2: 调度 Op0: start = max(Cube空闲0, 依赖完成0) = 0, end = 40
         Vector 管道: currentCycle = 40
Step 3: 调度 Op1: earliest = Op0.endCycle = 40
         start = max(Vector空闲40, 依赖完成40) = 40, end = 100
         Vector 管道: currentCycle = 100
Step 4: 调度 Op2: start = max(Cube空闲0, 依赖完成0) = 0, end = 100
         Cube 管道: currentCycle = 100
Step 5: totalCycles = max(40, 100, 100) = 100
         注意虽然 Op0+Op1 = 100 cycles，但因为 Op2 在 Cube 上并行跑 100 cycles，
         最终总时间 = max(100, 100) = 100
```

---

**第 5 步：从单程序 cycle → 完整 kernel cycle**

PipelineScheduler 算出来的是"一个程序实例跑一次的总时间"。但真实的 kernel 可能 launch 几千个程序实例，分布到几十个物理核上跑多轮（wave）。而且 AIV 上还有 scalar 开销和 barrier 同步。作者需要在 `getKernelCycles` 里做三种外推：

```cpp
int64_t getKernelCycles(int64_t numPrograms, int64_t numParallelUnits,
                        int64_t numInnerIters) const {
  // ① barrier 同步开销
  int64_t barrierCycles = numInnerIters * hwConfig->getPipeBarrierCyclesPerIter();
  // ② scalar overhead
  double scalarFactor = hwConfig->getAIVScalarOverheadFactor();
  int64_t perProgramCycles = (totalCycles + barrierCycles) * (1.0 + scalarFactor);
  // ③ wave 串行化
  int64_t numWaves = (numPrograms + numParallelUnits - 1) / numParallelUnits;
  return perProgramCycles * numWaves;
}
```

**为什么需要三步外推？** `totalCycles` 是**理想化的硬件计算时间**。现实中：(a) 每次循环迭代之间有 `pipe_barrier` 同步，(b) 循环控制、地址计算、分支标量指令消耗额外时间（scalar overhead），(c) 一个核同时只能跑一个程序，多个程序要排队（wave serialisation）。三步对应三种现实开销。

---

**第 6 步：还需要输出——PerformanceReport**

调度结果不能只打印到终端。Python 运行时需要通过文件读取 costmodel 估算的延迟，用于 `Estimated Time: 3.25 us` 这样的 regex 解析。所以需要结构化输出：

```cpp
struct PerformanceReport {
  string hardwareName;             // "Ascend 910B"
  double clockFreqGHz;             // 1.85

  int64_t totalCycles;             // 关键路径 cycle 数
  double totalTimeUs;              // 对应的微秒数
  int64_t kernelTotalCycles;       // 加上 scalar overhead + wave 后的总 cycle

  map<HWUnit, double> unitUtilization;    // 每个硬件单元的利用率百分比
  map<HWUnit, int64_t> unitBusyCycles;    // 每个硬件单元的忙碌 cycle 数
  HWUnit bottleneckUnit;                  // 瓶颈硬件单元

  double arithmeticIntensity;      // FLOP/Byte
  double achievedTFLOPS, peakTFLOPS;
  double achievedBandwidth, peakBandwidth;
  bool isComputeBound;             // 计算受限还是带宽受限

  void print(raw_ostream &os);     // 人类可读的表格输出
  string toJSON();                 // 机器可解析的 JSON
};
```

**为什么 `toJSON()` 和 `print()` 都需要？** 人类调试时看表格，自动化脚本解析 JSON。Python runtime 里用 regex 抓 `Estimated Time: 3.25 us` 这行文字。两种输出格式服务于两个不同的消费端。

---

**第 7 步：最后加 RooflineAnalyzer**

作者意识到：花了这么大功夫算出各硬件单元的 busy cycles 和利用率，不做瓶颈分析太浪费了。Roofline 模型是 HPC 领域判断"计算受限 vs 带宽受限"的经典方法：

```cpp
class RooflineAnalyzer {
  const PipelineScheduler &scheduler;
  const HardwareConfig &config;
  int64_t totalFLOPs;   // 所有计算 Op 的 flops 总和
  int64_t totalBytes;   // 所有搬运 Op 的 bytes 总和

  // 算术强度 = FLOP/Byte。AI 高 → 计算受限；AI 低 → 带宽受限
  double getArithmeticIntensity() { return totalFLOPs / totalBytes; }

  // 转折点 = peakTFLOPS / peakBWTBps
  // 910B: 320 TFLOPS / 1.6 TB/s = 200 FLOP/Byte
  bool isComputeBound() {
    return getArithmeticIntensity() >= config.getCubeTFlopsFP16() / config.getMemoryBandwidthTBps("hbm");
  }

  PerformanceReport analyze();  // 汇总所有数据生成报告
};
```

**为什么 RooflineAnalyzer 是独立的类而不是 PipelineScheduler 的方法？** PipelineScheduler 的核心职责是"怎么排"——调度算法本身。RooflineAnalyzer 的职责是"瓶颈在哪"——调度完之后的分析。分开后，想换一种瓶颈分析方法（比如不只看 Roofline，加一个利用率热力图）不需要改动调度器代码。

---

**完整结构——作者看着空文件逐步长出来的 7 个构件**

```
作者心路:

"我得算 kernel 总耗时"
  → 需要表示每个 Op 的耗时和归属  → ① PipelineOp

"不同硬件单元能并行，同一单元串行"
  → 需要跟踪每个单元的可用时间    → ② HWUnitPipeline

"Op 之间有数据依赖，必须等前置完成"
  → 需要依赖图 + 拓扑排序 + 循环检测 → ③ DependencyGraph

"把所有零件组合起来做调度"
  → 需要主调度器                → ④ PipelineScheduler + getKernelCycles

"把结果给别人看"
  → 需要结构化输出                → ⑤ PerformanceReport

"顺便做瓶颈分析"
  → 需要 Roofline 模型            → ⑥ RooflineAnalyzer
```

**这 6 个构件之间有清晰的依赖链**：① 是最小的数据单元，② 消费 ①，③ 消费 ①，④ 消费①②③ + HardwareConfig，⑤⑥ 消费 ④ 的输出。这个依赖顺序决定了它们在头文件中的定义顺序——`PipelineOp` 必须最先定义，因为 `HWUnitPipeline` 和 `DependencyGraph` 都引用它。

---

**文件 5: `include/AscendModel/Analysis/Utils.h`**

作者写到 EstimateCyclesPass 的设计时发现了问题：Triton kernel 里的循环边界经常是动态的——`for i in range(arg2 * 4, arg3, 2)`。编译器不知道 `arg2`、`arg3` 是多少，没法确定循环跑多少次。没有 trip count，连每个 Op 到底执行多少遍都不知道，costmodel 就没法给出有意义的 cycle 数。

解决方案分两层：**先分析表达式结构（是什么）→ 再代入绑定值求结果（算多少）**。

第一层 —— **符号分析** `SymbolicBound`：一个循环上下界表达式，分析它到底依赖什么。

```cpp
struct SymbolicBound {
  enum Kind { Constant,   // 0: 编译期常量，比如字面量 4
              Argument,   // 1: 函数参数，比如 %arg2
              ProgramId,  // 2: tt.get_program_id x/y/z
              Expression, // 3: 组合表达式，比如 arg2 * 4 + 8
              Unknown };  // 4: 分析不了的

  int64_t constantValue;                 // Constant 时有效
  unsigned argIndex;                     // Argument 时有效
  SmallVector<unsigned> dependentArgs;   // Expression 时：依赖哪些 %argN
  SmallVector<std::string> dependentProgramIds; // 依赖哪些 program_id
  std::string description;               // 人类可读的表达式文本
};
```

**用法**：`analyzeValue(forOp.getUpperBound())` → 返回 `SymbolicBound{kind=Expression, dependentArgs={2,3}, dependentProgramIds={"x"}, description="expr(%arg2, %arg3, program_id_x)"}`。

第二层 —— **求值** `evaluateValue()`：给定实际的绑定值（`arg2=128, arg3=256, pid_x=0`），算出表达式的结果。

```cpp
optional<int64_t> evaluateValue(Value v,
    const DenseMap<unsigned, int64_t> &argBindings,
    const StringMap<int64_t> &programIdBindings);
```

**这个函数之所以复杂（400 行），是因为它要支持 arith 的 20+ 种运算**。循环边界表达式可能是任意 arith Op 组合：`(arg2 * 4 + arg3) / 2`、`min(arg2, 1024)`、`arg2 << 2`、`select(cmp, a, b)`。作者必须递归求值，对每种 arith Op 写一个求值分支。

第三层 —— **字符串解析**。用户通过命令行传入的 `arg-bindings=arg2=128,pid_x=0` 是字符串，必须解析成 `DenseMap`。

```cpp
bool parseBindings(StringRef input,            // "arg2=128,pid_x=0"
    DenseMap<unsigned, int64_t> &argBindings,  // {2: 128}
    StringMap<int64_t> &programIdBindings,     // {"x": 0}
    std::string &error);
```

**作者的考虑：兼容多种输入格式**。用户可能写 `arg2=128`、也可能写 `2=128`、也可能写 `pid_x=0` 或 `pidx=0` 或 `program_id_x=0`。`parseBindings` 用 key 前缀自动识别并规范化。

**第四层 —— 循环嵌套**。内层循环里的 Op 会执行内层 trip count × 外层 trip count 次：

```cpp
int64_t getLoopMultiplier(Operation *op) {
  int64_t multiplier = 1;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (auto attr = forOp->getAttr("ascend.trip_count"))
        multiplier *= attr.getInt();       // 累积所有外层循环的 trip count
    }
    parent = parent->getParentOp();
  }
  return multiplier;
}
```

**为什么 Utils.h 这么大（1000 行）？** 这些功能本可以拆成 4 个文件（类型工具、符号分析、表达式求值、字符串解析），但作者选择了单文件——所有 Pass 都需要它们。include 一个头文件比 include 四个方便，尤其在快速开发阶段。

---

**文件 6-7: 顶层转发头文件**

```cpp
// include/AscendModel/HardwareConfig.h —— 直接转发
#include "AscendModel/Analysis/HardwareConfig.h"

// include/AscendModel/Utils.h —— 同上
#include "AscendModel/Analysis/Utils.h"
```

**为什么需要这两个文件？** 外部调用者写 `#include "AscendModel/HardwareConfig.h"` 比 `#include "AscendModel/Analysis/HardwareConfig.h"` 短。这是一个 C++ 约定——隐藏内部目录结构，以后 Analysis 改名也不影响外部代码。

---

#### 阶段三：Pass 声明头文件

**文件 8: `include/AscendModel/Transforms/Passes.h`**

```cpp
// ① GEN_PASS_DECL: 展开各 Pass 的创建函数声明
//    即: std::unique_ptr<Pass> createConvertTritonToAscendPass();
//        std::unique_ptr<Pass> createEstimateCyclesPass(...);
#define GEN_PASS_DECL
#include "AscendModel/Transforms/Passes.h.inc"

// ② GEN_PASS_REGISTRATION: 展开 Pass 注册表的静态初始化代码
//    即: static PassRegistration<ConvertTritonToAscendPass> X(...);
#define GEN_PASS_REGISTRATION
#include "AscendModel/Transforms/Passes.h.inc"

// ③ 手写: 管线注册
void registerAscendModelPipeline();
```

**同样一个 `Passes.h.inc`，用不同的 `#define` 前展开出不同的内容**——这就是 X-Macro 模式的再一次应用。`GEN_PASS_DECL` 展开函数声明，`GEN_PASS_REGISTRATION` 展开注册代码。两份代码物理上在同一个文件里，逻辑上互不干扰。

---

#### 阶段四：.cpp 实现 —— 从最底层开始

写完头文件，作者开始写实现。**从依赖链最底端开始**——被依赖的必须先写，依赖别人的后写。

**文件 9: `lib/IR/AscendModelDialect.cpp`**（已在 6.2c 节详细分析）

49 行。做两件事：① 注册 Dialect（include `Dialect.cpp.inc` 的构造函数），② `initialize()` 里用 X-Macro 注册所有 Op。

---

**文件 10: `lib/IR/AscendModelOps.cpp`**（已在 6.2c 节详细分析）

364 行。25 个 Op 各自的 `estimateCycles()` 实现。作者用宏批量展开结构相同的 Op（5 个简单二元、6 个比较、4 个简单一元），手写结构不同的（MatmulOp、搬运 Op、DivOp、ReduceOp、BroadcastOp）。

---

**文件 11: `lib/Analysis/HardwareConfig.cpp`**（已在 6.2d 节详细分析）

994 行。三个任务：① `parseJSON()` 把 JSON 转成 struct，② `getDefault910B()` 先搜文件，搜不到用 `createHardcodedDefault910B()` hardcode fallback，③ 校准参数（`getVectorStartupLatency()=35`、`getAIVScalarOverheadFactor()=3.74`）。

---

**文件 12-13: `lib/Analysis/PipelineAnalysis.cpp` + `RooflineAnalysis.cpp`**（已在 6.2d 节详细分析）

调度算法 + Roofline 瓶颈分析。约 600 行。分开是因为 PipelineScheduler 关注"怎么排"，RooflineAnalyzer 关注"瓶颈是什么"。

---

**文件 14: `lib/Transforms/ConvertTritonToAscend.cpp`**（Pass ①，最复杂 850 行）

作者面对的第一个真 Pass——把 Triton IR 转换成 AscendModel IR。设计决策：

**两阶段处理**：Phase 1 先遍历所有 `tt.dot`，回溯找到被使用的 `tt.load`，打上 `ascend.used_by_dot` 标记。Phase 2 用 RewritePattern 做模式替换。分两步是因为——在替换之前需要全局信息（哪个 load 是 cube load 哪个是 vector load），但 RewritePattern 是局部的。

**12 种转换模式**，按 benefit（优先级）排：

| benefit | 模式 | 做什么 |
|---------|------|--------|
| 10 | `ConvertTritonDot` | `tt.dot` → `ascend.matmul` + cube_store + (可选) vector_load。如果 dot 的输入来自 vector 路径，自动插入 vector_store + cube_load |
| 10 | `ConvertTritonLoad` | `tt.load` → `ascend.cube_load`（被 dot 使用）或 `ascend.vector_load`（一般使用） |
| 10 | `ConvertTritonStore` | `tt.store` → `ascend.vector_store`，丢弃指针 operand |
| 10 | `ConvertTritonReduce` | `tt.reduce` → `ascend.reduce_sum/max/min/prod`，检查归约体确定 kind |
| 10 | `ConvertTritonBroadcast` | `tt.broadcast` → `ascend.broadcast` |
| 10 | `ConvertTritonTrans` | `tt.trans` → 直接 pass-through（transpose 在 Ascend 上零开销） |
| 1 | `EraseDeadTritonAddrOps` | 删除死掉地址计算 Op（`tt.addptr`/`tt.splat`/`tt.make_range` 等）——在 load/store 被替换后变成孤立节点 |
| 1 | `ConvertArithBinaryOp` | `arith.addf/subf/mulf/divf/maxf/minf` → `ascend.add/sub/mul/div/max/min` |
| 1 | `ConvertArithCmpOp` | `arith.cmpf/cmpi` → `ascend.cmp_eq/ne/lt/le/gt/ge`，需映射 10+ 种 predicate |
| 1 | `ConvertMathUnaryOp` | `math.exp/log/sqrt/rsqrt/tanh` → `ascend.exp/log/sqrt/rsqrt/tanh` |
| 1 | `ConvertArithSelect` | `arith.select` → `ascend.select` |
| 1 | `ConvertArithCast` | `arith.ext/trunc/sitofp/fptosi/...` 全部 → `ascend.cast` |

**为什么 benefit=10 的先执行，benefit=1 的后执行？** 贪婪重写引擎按 benefit 降序应用规则——先做结构转换（Triton → AscendModel），再做 tail cleanup（删死代码、转换 arith/math）。

---

**文件 15: `lib/Transforms/InsertDataTransfers.cpp`**（Pass ②，214 行）

这一步的逻辑非常硬件特定。Ascend 910B 上 Cube 和 Vector 走不同的内存路径：

```
Vector path:  HBM →(MTE2)→ UB → Vector → UB →(MTE3)→ HBM
Cube path:    HBM →(MTE2)→ L1 →(MTE1)→ L0A/L0B → Cube → L0C →(FixPipe)→ HBM
```

当 Vector 的结果流向 Cube 时（比如 `ascend.mul` → `ascend.matmul`），数据从 UB 搬不进去 L1——必须**经过 HBM 中转**：`vector_store (UB→HBM)` → `cube_load (HBM→L1)`。

反向更简单：Cube 的结果 (`L0C`) → `cube_store (L0C→HBM)` → `vector_load (HBM→UB)`。

**作者怎么做**：遍历所有 `MatmulOp`，对每个 operand：
1. 回追溯到生产者 Op
2. 如果生产者是 Vector 路径 → 在中间插入 vector_store + cube_load
3. 检查 MatmulOp 的结果 users：如果有 Vector 路径的消费者 → 在 MatmulOp 之后插入 cube_store + vector_load，替换消费者的 operand

**用 `DenseMap<Value, Value>` 去重**——同一个值被多次使用，只插一次搬运链。

---

**文件 16: `lib/Transforms/AssignOpIDs.cpp`**（Pass ③，49 行）

最简单的 Pass。遍历所有 `"ascend"` namespace 的 Op，从 0 开始递增编号：

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();
  int64_t nextId = 0;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "ascend") {
      op->setAttr("op_id", IntegerAttr::get(..., nextId++));
    }
  });
}
```

**为什么需要这个 Pass？** 后续 PipelineAnalysisPass 需要引用 Op 的序号来表示依赖关系（`dependsOn = {0, 3, 5}`）。不分配全局唯一 ID 就没法表达"Op 3 依赖 Op 0"。

---

**文件 17: `lib/Transforms/EstimateCycles.cpp`**（Pass ④，316 行，已在 6.2c 节部分分析）

核心 Pass。三遍遍历：

1. **第一遍**：收集所有 `scf::ForOp`，按顺序解析 trip count——优先用 `loop-trip-counts` 覆盖，否则用 `arg-bindings` 解析循环边界表达式
2. **第二遍**：遍历所有 Compute Op（跳过 `scf::ForOp`/`YieldOp`/`IfOp`），调 `dyn_cast<EstimateCyclesOpInterface>(op)` 拿到 Interface，调 `estimateCycles(config)` 得到 cycle 数，乘以 `getLoopMultiplier(op)` 得到总 cycle 数，写入 `estimated_cycles`/`hw_unit`/`bytes`/`flops` attributes
3. **第三遍**：标注循环——计算每个循环体的 `body_cycles` 和 `total_cycles = body_cycles * trip_count`，写入 `ascend.body_cycles`/`ascend.total_cycles` attributes

**作者还要计算 Roofline 统计**——在第二遍遍历中按 HWUnit 分类累计 flops 和 bytes，最后用 `RooflineStats::calculateRooflineCycles()` 算出 Cube 路径和 Vector 路径各自的 `max(compute, memory)`，然后如果 Cube 和 Vector 可以重叠就 `max(cube, vector)`。

---

**文件 18-20: 剩余 Pass**（可结合 6.2e 节的 Pass 管线总览理解）

| 文件 | 做什么 | 行数 |
|------|--------|------|
| `PipelineAnalysisPass.cpp` | 读所有 Op 的 `estimated_cycles`/`hw_unit` → 建依赖图 → `PipelineScheduler::schedule()` → 写 module attributes（`scheduled_cycles_one_iter`/`roofline_cycles`/`simple_sum_cycles`） | ~200 |
| `PerfReportPass.cpp` | 读所有 attributes → 组装 `PerformanceReport` → `print()` / `toJSON()` | ~80 |
| `HIVMAnalysisPass.cpp` | 外部 IR（HIVM）的分析 Pass，可选择 static/des 调度模式 | ~120 |

---

**文件 21: `lib/Transforms/PassRegistration.cpp`**（已在 6.2e 节详细分析）

最后一站。注册完整的 `ascend-perf-model` 管线，**6 步顺序**：

```
① ConvertTritonToAscend → ② InsertDataTransfers → ③ AssignOpIDs
→ ④ EstimateCycles → ⑤ PipelineAnalysis → ⑥ PerfReport
```

---

#### 完整的文件创建顺序和依赖关系

```
                    ┌──────────────────────────────────┐
                    │  MLIR 框架 + TableGen 生成 13 个 .inc │
                    └──────────┬───────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │ 阶段一: 胶水文件          │                          │
    │ ① AscendModelInterfaces.h│                          │
    │    (forward declare +     │                          │
    │     include Interfaces    │                          │
    │     .h.inc)               │                          │
    │ ② AscendModelDialect.h   │                          │
    │    (按顺序 include 6 个   │                          │
    │     .inc 文件)            │                          │
    └──────────┬───────────────┘                          │
               │                                          │
    ┌──────────┴───────────────┐                          │
    │ 阶段二: Analysis 头文件    │                          │
    │ ③ HardwareConfig.h       │                          │
    │    (MemorySpace, Compute  │                          │
    │     Unit, 查询 getter)    │                          │
    │ ④ PipelineAnalysis.h     │                          │
    │    (PipelineOp, Scheduler,│                          │
    │     RooflineAnalyzer)     │                          │
    │ ⑤ Utils.h (1000 行)      │                          │
    │    (SymbolicBound,        │                          │
    │     evaluateValue,        │                          │
    │     parseBindings,        │                          │
    │     getLoopMultiplier)    │                          │
    │ ⑥⑦ 转发头文件             │                          │
    └──────────┬───────────────┘                          │
               │                                          │
    ┌──────────┴───────────────┐                          │
    │ 阶段三: Pass 声明          │                          │
    │ ⑧ Passes.h               │                          │
    │    (GEN_PASS_DECL +       │                          │
    │     GEN_PASS_REGISTRATION)│                          │
    └──────────┬───────────────┘                          │
               │                                          │
    ┌──────────┴──────────────────────────────────────┐  │
    │ 阶段四: .cpp 实现（按依赖链从底往上）               │  │
    │                                                  │  │
    │ ⑨ AscendModelDialect.cpp  (依赖 ②)               │  │
    │ ⑩ AscendModelOps.cpp      (依赖 ②③)              │  │
    │ ⑪ HardwareConfig.cpp      (依赖 ③)               │  │
    │ ⑫ PipelineAnalysis.cpp    (依赖 ④③)              │  │
    │ ⑬ RooflineAnalysis.cpp    (依赖 ④③)              │  │
    │                                                  │  │
    │ ── Pass 实现（依赖 ②⑤⑥⑧ 和 ⑨⑩⑪⑫⑬）────         │  │
    │ ⑭ ConvertTritonToAscend.cpp (最复杂, 850 行)       │  │
    │ ⑮ InsertDataTransfers.cpp   (硬件特定, 214 行)     │  │
    │ ⑯ AssignOpIDs.cpp           (最简单, 49 行)        │  │
    │ ⑰ EstimateCycles.cpp        (核心, 316 行)         │  │
    │ ⑱ PipelineAnalysisPass.cpp  (调度器包装, ~200 行)   │  │
    │ ⑲ PerfReportPass.cpp        (汇报, ~80 行)         │  │
    │ ⑳ HIVMAnalysisPass.cpp      (HIVM, ~120 行)        │  │
    │ ㉑ PassRegistration.cpp      (注册管线, ~100 行)     │  │
    └──────────────────────────────────────────────────┘  │
```

---

### 6.1 门槛问题：为什么默认 ctest 找不到测试？

**三层门槛，逐层分析：**

#### 门槛 1：`setup.py` 默认 `TRITON_BUILD_UT=OFF`

`setup.py:56`：
```python
os.environ.setdefault("TRITON_APPEND_CMAKE_ARGS", "-DTRITON_BUILD_UT=OFF")
```

你不显式设 `TRITON_APPEND_CMAKE_ARGS` 的话，CMake 参数里就是 `-DTRITON_BUILD_UT=OFF`。

注意：`TRITON_BUILD_UT` **不在 passthrough 列表里**（`setup.py:611-616` 只有 `TRITON_BUILD_PROTON`、`TRITON_BUILD_WITH_CCACHE`、`TRITON_PARALLEL_LINK_JOBS`），所以不能通过单独设置同名环境变量来覆盖，必须通过 `TRITON_APPEND_CMAKE_ARGS`：

```bash
# ❌ 不生效：setup.py 不 passthrough TRITON_BUILD_UT
TRITON_BUILD_UT=ON pip install -e .

# ✅ 生效：覆盖默认的 TRITON_APPEND_CMAKE_ARGS
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=ON" pip install -e .
```

#### 门槛 2：`third_party/ascend/CMakeLists.txt:112-114`

```cmake
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)    # ← TRITON_BUILD_UT=OFF 时这行不执行！
endif()
```

`TRITON_BUILD_UT=OFF` → `unittest/` 目录完全跳过 → `costmodel_ut/` 不可能被构建。

#### 门槛 3：`third_party/ascend/unittest/CMakeLists.txt:29-31`

```cmake
if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)
  add_subdirectory(costmodel_ut)
endif()
```

这层检测 costmodel 源码是否存在。`TRITON_ASCEND_HAS_INPROC_COSTMODEL` 由 `third_party/ascend/CMakeLists.txt:65` 设置——检查 `third_party/ascend/costmodel/` 下的 `include/AscendModel/CMakeLists.txt` 和 `lib/AscendModel/CMakeLists.txt` 是否都存在。你的环境满足这层。

### 6.2 CostModel C++ 库的编译

在进入单元测试之前，三个被测库必须先编译出来。调用链从 CMakeLists.txt 顶层开始：

```
CMakeLists.txt:220
  foreach(CODEGEN_BACKEND ascend;nvidia;amd)
    add_subdirectory(third_party/ascend)      ──→ third_party/ascend/CMakeLists.txt
```

#### 6.2a. 检测 + 配置 (`third_party/ascend/CMakeLists.txt:34-65`)

```cmake
option(TRITON_ASCEND_ENABLE_COSTMODEL_INPROC "Enable in-process costmodel C++ bridge" ON)

set(_costmodel_include_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel")
set(_costmodel_lib_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/lib/AscendModel")

if(EXISTS "${_costmodel_include_dir}/CMakeLists.txt" AND EXISTS "${_costmodel_lib_dir}/CMakeLists.txt")
  # ① 从模板生成 HardwareParams.h
  configure_file(
    "${_costmodel_include_dir}/HardwareParams.h.in"            # 模板
    "${_costmodel_gen_include}/AscendModel/HardwareParams.h"   # 输出
    @ONLY
  )
  # 模板中 @ASCEND_CUBE_TFLOPS@ 等被替换为 CMake 变量的值:
  #   ASCEND_CUBE_TFLOPS=320, ASCEND_VECTOR_TFLOPS=10, ASCEND_HBM_BW=1555, ...

  # ② TableGen 代码生成（.td → .h.inc / .cpp.inc）
  add_subdirectory(${_costmodel_include_dir}
    ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel)

  # ③ 编译三个 C++ 库
  add_subdirectory(${_costmodel_lib_dir}
    ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/lib_AscendModel)

  set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 1)   # ← 第 3 层门槛的关键标记
endif()
```

#### 6.2b. TableGen 代码生成 —— 逐行分析 (`costmodel/include/AscendModel/CMakeLists.txt`)

这个 CMakeLists.txt 负责把 `.td`（TableGen 定义文件）自动转换成 C++ 可 `#include` 的 `.inc` 文件。

**函数来源**：`mlir_tablegen()` 和 `add_public_tablegen_target()` 不在此仓库中，来自 LLVM 预编译包。调用链：

```
顶层 CMakeLists.txt:105-109
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(TableGen)   ← 提供 mlir_tablegen()
  include(AddMLIR)    ← 提供 add_public_tablegen_target()
```

**核心工具**：`llvm-tblgen`（LLVM TableGen），一个表格驱动的代码生成器，读 `.td` → 出 `.inc`。

---

**第 1 行：设置 .td 文件的 include 搜索路径**

```cmake
set(LLVM_TABLEGEN_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/IR")
```

- `LLVM_TABLEGEN_FLAGS` — LLVM 约定的变量名。`mlir_tablegen()` 内部调用 `llvm-tblgen` 时，把这个变量的值作为命令行参数传入。
- `-I...IR` — 等价于 `gcc -I`，告诉 tblgen **去哪里找被 include 的其他 .td 文件**。因为 `AscendModelOps.td` 中有 `include "AscendModelInterfaces.td"`，没有这行就找不到。

---

**第 3-6 行：从 Interfaces.td 生成 2 个 .inc**

```cmake
set(LLVM_TARGET_DEFINITIONS IR/AscendModelInterfaces.td)
mlir_tablegen(AscendModel/IR/AscendModelInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AscendModel/IR/AscendModelInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AscendModelInterfacesIncGen)
```

- `LLVM_TARGET_DEFINITIONS` — 也是 LLVM 约定的变量，指定源 `.td` 文件路径
- `mlir_tablegen(输出 -gen-xxx)` — 跑 `llvm-tblgen -gen-xxx 源.td -o 输出`
- `-gen-op-interface-decls` → 从 `.td` 里的 `OpInterface` 定义生成 C++ 纯虚类**声明**
- `-gen-op-interface-defs` → 生成**实现代码**（默认空壳，由各 Op 覆写）
- `add_public_tablegen_target(名字)` — 把上面生成的 .inc 打包成 CMake 构建目标，后续 C++ 编译可以通过 `DEPENDS` 等这个目标先完成

---

**第 8-23 行：从 Ops.td 生成 10 个 .inc**

```cmake
set(LLVM_TARGET_DEFINITIONS IR/AscendModelOps.td)
mlir_tablegen(AscendModel/IR/AscendModelOps.h.inc        -gen-op-decls)
mlir_tablegen(AscendModel/IR/AscendModelOps.cpp.inc       -gen-op-defs)
mlir_tablegen(AscendModel/IR/AscendModelOpsEnums.h.inc    -gen-enum-decls)
mlir_tablegen(AscendModel/IR/AscendModelOpsEnums.cpp.inc  -gen-enum-defs)
mlir_tablegen(AscendModel/IR/AscendModelDialect.h.inc     -gen-dialect-decls)
mlir_tablegen(AscendModel/IR/AscendModelDialect.cpp.inc   -gen-dialect-defs)
mlir_tablegen(AscendModel/IR/AscendModelOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(AscendModel/IR/AscendModelOpsAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(AscendModel/IR/AscendModelOpsTypes.h.inc    -gen-typedef-decls)
mlir_tablegen(AscendModel/IR/AscendModelOpsTypes.cpp.inc  -gen-typedef-defs)

add_public_tablegen_target(AscendModelOpsIncGen)
add_dependencies(AscendModelOpsIncGen AscendModelInterfacesIncGen)
```

同一个 `.td` 源，**不同 `-gen-*` flag 生成不同内容**：

| flag | 生成的 .inc | 内容 | 被哪个 C++ include |
|------|-----------|------|--------------------|
| `-gen-op-decls` | `Ops.h.inc` | 每个 Op 的 C++ 类声明 | `AscendModelDialect.h:42` |
| `-gen-op-defs` | `Ops.cpp.inc` | Op 的 `build()`/`verify()`/`parse()`/`print()` 骨架实现 | `AscendModelOps.cpp:87` |
| `-gen-enum-decls` | `OpsEnums.h.inc` | HWUnit 等枚举类型声明 | `AscendModelDialect.h:33` |
| `-gen-enum-defs` | `OpsEnums.cpp.inc` | 枚举 → 字符串 转换函数 | `AscendModelDialect.cpp:34` |
| `-gen-dialect-decls` | `Dialect.h.inc` | Dialect 类声明 | `AscendModelDialect.h:30` |
| `-gen-dialect-defs` | `Dialect.cpp.inc` | Dialect 注册代码 | `AscendModelDialect.cpp:21` |
| `-gen-attrdef-decls` | `OpsAttrDefs.h.inc` | 自定义属性声明 | `AscendModelDialect.h:35` |
| `-gen-attrdef-defs` | `OpsAttrDefs.cpp.inc` | 属性解析/打印实现 | `AscendModelDialect.cpp:40` |
| `-gen-typedef-decls` | `OpsTypes.h.inc` | 自定义类型声明 | `AscendModelDialect.h:38` |
| `-gen-typedef-defs` | `OpsTypes.cpp.inc` | 类型实现 | `AscendModelDialect.cpp:47` |

`add_dependencies(AscendModelOpsIncGen AscendModelInterfacesIncGen)` — **关键**：因为 `AscendModelOps.td` 中有 `include "AscendModelInterfaces.td"`，tblgen 处理 Ops.td 时依赖 Interfaces.td 的内容，所以必须先等 Interfaces 的 TableGen 完成。

**一个具体例子**：`.td` 里一句话变成 `.inc` 里 30+ 行 C++

```tablegen
// AscendModelOps.td
def Ascend_AddOp : Ascend_VectorBinarySimple<"add"> {
  let summary = "Element-wise addition on Vector Core";
}
```

`-gen-op-decls`（`Ops.h.inc`）生成（简化示意）：

```cpp
class AddOp : public Op<AddOp, OpTrait::ZeroRegions, ...> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "ascend.add"; }
  Value getLhs();     // 自动生成的 operand accessor
  Value getRhs();
  Value getResult();  // 自动生成的 result accessor
  static void build(OpBuilder &, OperationState &, Value lhs, Value rhs);
  static ParseResult parse(OpAsmParser &, OperationState &);
  void print(OpAsmPrinter &);
  LogicalResult verify();
};
```

`-gen-op-defs`（`Ops.cpp.inc`）生成：

```cpp
void AddOp::build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType());  // 结果类型 = 输入类型
}
ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) { ... }
void AddOp::print(OpAsmPrinter &p) { ... }
LogicalResult AddOp::verify() { ... }  // 空壳，需手写覆盖
```

**程序员只写 .td 里的 Op 定义和 `estimateCycles()` 的实现**，其他 build/parse/print/verify/getLhs/getRhs/getResult 全部自动生成。

---

**第 25-28 行：从 Passes.td 生成 Pass 声明**

```cmake
set(LLVM_TARGET_DEFINITIONS Transforms/Passes.td)
mlir_tablegen(AscendModel/Transforms/Passes.h.inc -gen-pass-decls -name AscendModel)
add_public_tablegen_target(AscendModelTransformsPassIncGen)
```

`-gen-pass-decls -name AscendModel` → 在 Pass 类前加 `AscendModel` 命名空间前缀。

---

**第 30-31 行：清理全局变量**

```cmake
set(LLVM_TABLEGEN_FLAGS "")
set(LLVM_TARGET_DEFINITIONS "")
```

CMake 变量是全局的——不清的话后面其他目录的 `mlir_tablegen()` 会读到错误的源 .td 路径和 include flags。

#### 6.2c. `AscendModelIR` — Dialect + Op 实现 (`costmodel/lib/AscendModel/IR/`)

**CMakeLists.txt 分析** (`costmodel/lib/AscendModel/IR/CMakeLists.txt`)：

```cmake
add_mlir_library(AscendModelIR
  AscendModelDialect.cpp       # Dialect 注册 + 初始化
  AscendModelOps.cpp           # 所有 Op 的 estimateCycles() + 宏展开

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/IR

  DEPENDS
  AscendModelOpsIncGen         # 等 TableGen 完成
  AscendModelInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR                       # MLIR 核心 IR 基础设施
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)
```

**`add_mlir_library`**：来自 LLVM 预编译包的 `AddMLIR.cmake` 模块（不是此仓库的函数）。它做四件事：

1. **`add_library(NAME OBJECT ...)`** — 编译 `.cpp` 为 `.o` 目标文件
2. **`add_dependencies`** — 确保 `DEPENDS` 里的 TableGen 目标先完成
3. **`target_include_directories`** — 加入 `ADDITIONAL_HEADER_DIRS`
4. **`target_link_libraries`** — 链接 `LINK_LIBS` 中的库

**产物**：`libAscendModelIR.a`。

---

**源码详解：`AscendModelDialect.cpp`**

这个文件只有 49 行，但每一行都有 MLIR 的约定在背后。先看总体结构：

```cpp
// AscendModelDialect.cpp
#include "AscendModel/IR/AscendModelDialect.h"

// ── ① 包含 TableGen 生成的 Dialect 实现 ──
#include "AscendModel/IR/AscendModelDialect.cpp.inc"

// ── ② Dialect 初始化：注册所有 Op ──
void AscendModelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AscendModel/IR/AscendModelOps.cpp.inc"   // ← X-Macro 魔法
      >();
}

// ── ③ 包含生成的枚举/属性/类型实现 ──
#include "AscendModel/IR/AscendModelOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsAttrDefs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsTypes.cpp.inc"
```

**关键语法解析**：

**`#include "xxx.cpp.inc"`**：这些 `.inc` 是 TableGen 自动生成的 C++ 代码片段。MLIR 惯例是 source 里直接 `#include` 进来。`Dialect.cpp.inc` 展开后生成类似：

```cpp
// AscendModelDialect typeID 定义、静态注册等
AscendModelDialect::AscendModelDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<AscendModelDialect>()) {
  initialize();
}
```

**X-Macro 模式**：`initialize()` 里：

```cpp
addOperations<
#define GET_OP_LIST
#include "AscendModel/IR/AscendModelOps.cpp.inc"
>();
```

`GET_OP_LIST` 这个宏告诉 `Ops.cpp.inc` **这次只展开 Op 类名列表**（AddOp, SubOp, MulOp, ...），而不是展开完整实现。TableGen 生成的 `Ops.cpp.inc` 里有两个分支：

```cpp
#ifdef GET_OP_LIST
  AddOp, SubOp, MulOp, MaxOp, MinOp, DivOp,           // 简单二元
  CmpEqOp, CmpNeOp, ...,                               // 比较
  NegOp, AbsOp, ReluOp, CastOp,                        // 简单一元
  SqrtOp, RsqrtOp, ExpOp, LogOp, TanhOp, SigmoidOp,   // 复杂一元
  ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp, // 归约
  MatmulOp,                                            // 矩阵乘
  CubeLoadOp, CubeStoreOp, VectorLoadOp, VectorStoreOp,// 搬运
  BroadcastOp, SelectOp
#endif
```

C++ 模板 `addOperations<AddOp, SubOp, MulOp, ...>()` 把这些类型注册到 MLIR 的 Dialect 类型系统中。

---

**源码详解：`AscendModelOps.cpp`**

这是 **costmodel 最核心的文件**（364 行）。它实现了 IR 层定义的每种 Op 的 `estimateCycles()` 方法——即"这个 Op 需要多少 cycles"的计算逻辑。

**`estimateCycles()` 接口**：在 `AscendModelInterfaces.td` 里定义，TableGen 生成纯虚函数声明：

```cpp
// 生成在 AscendModelInterfaces.h.inc 中
class EstimateCyclesOpInterface {
  virtual int64_t estimateCycles(const HardwareConfig &config) = 0;
  virtual HWUnit getHWUnit() = 0;
  // ...
};
```

每个 Op 必须覆写这两个方法——告诉外界"我在哪个硬件单元上跑"和"跑多久"。

**① 辅助函数**：

```cpp
// 从 MLIR RankedTensorType 中提取元素个数
static int64_t getNumElementsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)         // 动态 shape → 用默认值 1024
        return 1024;
      count *= dim;
    }
    return count;                              // 静态 shape → 连乘
  }
  return 1;
}
```

- `dyn_cast<T>` — LLVM 的 RTTI 替代方案，比 `dynamic_cast` 快。如果 type 是 `RankedTensorType` 就返回指针，否则返回 null。
- `getShape()` 返回 `ArrayRef<int64_t>`——tensor 各维度大小。`tensor<2x3x4xf32>` → `[2, 3, 4]`。

```cpp
// 从 tensor 元素类型中提取 bit 宽度
static int getElementBitsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elemType = tensorType.getElementType();
    if (elemType.isF16() || elemType.isBF16()) return 16;
    else if (elemType.isF32()) return 32;
    else if (auto intType = dyn_cast<IntegerType>(elemType))
      return intType.getWidth();               // int8 → 8, int32 → 32
  }
  return 32;
}
```

**② 核心估算公式**：

```cpp
// Vector 运算的通用 cycle 公式
// Vector unit = 2048 bits wide
static int64_t estimateVectorCycles(int64_t numElements, int cyclesPerVectorOp,
                                    int elementBits, int startupLatency) {
  int64_t vectorWidth = 2048 / elementBits;          // 一次 Vector 指令处理几个元素
                                                     // FP16: 2048/16=128, FP32: 2048/32=64
  int64_t numVectorOps = (numElements + vectorWidth - 1) / vectorWidth;  // 向上取整
  return numVectorOps * cyclesPerVectorOp + startupLatency;
}

// 内存搬运的通用 cycle 公式
// cycles = bytes / (bandwidth * 1e9) * (clock * 1e9) = bytes * clock / bandwidth_GBps
static int64_t estimateMemoryCycles(int64_t bytes, const HardwareConfig &config,
                                    int startupLatency) {
  double bandwidth_gbs = config.getHBMBandwidthGBs();
  double time_seconds = static_cast<double>(bytes) / (bandwidth_gbs * 1e9);
  double cycles = time_seconds * config.getClockFrequencyGHz() * 1e9;
  return static_cast<int64_t>(cycles) + startupLatency;
}
```

**③ 对 .inc 的包含条**：

```cpp
// TableGen 生成的 Interface 默认实现
#include "AscendModel/IR/AscendModelInterfaces.cpp.inc"

// TableGen 生成的 Op 类骨架（build/parse/print/verify 等）
#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.cpp.inc"
```

`GET_OP_CLASSES` 告诉 Ops.cpp.inc 这次展开**完整 Op 类定义**（不是 Op 名列表）。

**④ 各种 Op 的 `estimateCycles()` 实现**：

**矩阵乘（MatmulOp）**：

```cpp
int64_t MatmulOp::estimateCycles(const HardwareConfig &config) {
  int64_t m = getM(), n = getN(), k = getK();  // .td 定义的属性
  config.getCubeFractalSize(elemBits, fracM, fracN, fracK);  // 获取 Cube 粒度
  int64_t numFracM = (m + fracM - 1) / fracM;   // ceil(m/16)
  int64_t numFracN = (n + fracN - 1) / fracN;   // ceil(n/16)
  int64_t numFracK = (k + fracK - 1) / fracK;   // ceil(k/16)
  int64_t totalFractals = numFracM * numFracN * numFracK;
  return totalFractals + config.getCubeStartupLatency();
}
HWUnit MatmulOp::getHWUnit() { return HWUnit::Cube; }
int64_t MatmulOp::getFlops() { return 2 * getM() * getN() * getK(); }  // 每个 MAC = 2 FLOP
```

**内存搬运（CubeLoadOp/VectorLoadOp 等）**——都调同一个通用公式，区别只是 HWUnit 归属不同：

```cpp
int64_t CubeLoadOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getMTE2StartupLatency());
}
HWUnit CubeLoadOp::getHWUnit() { return HWUnit::CubeMTE2; }

int64_t VectorStoreOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getMTE3StartupLatency());
}
HWUnit VectorStoreOp::getHWUnit() { return HWUnit::MTE3; }
```

**向量宏展开**——大量相似的 Op 用宏避免重复：

```cpp
#define IMPL_SIMPLE_VECTOR_BINARY(OpClass)
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {
    int64_t n = getNumElementsFromType(getLhs().getType());
    int bits = getElementBitsFromType(getLhs().getType());
    return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency()); // 1 cycle/op
  }
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }

IMPL_SIMPLE_VECTOR_BINARY(AddOp)   // → 展开为 AddOp::estimateCycles() 完整实现
IMPL_SIMPLE_VECTOR_BINARY(SubOp)   // → 展开为 SubOp::estimateCycles() 完整实现
IMPL_SIMPLE_VECTOR_BINARY(MulOp)
IMPL_SIMPLE_VECTOR_BINARY(MaxOp)
IMPL_SIMPLE_VECTOR_BINARY(MinOp)
#undef IMPL_SIMPLE_VECTOR_BINARY  // 用完立即清理，避免污染后续代码
```

Add/Sub/Mul/Max/Min 都是 1 cycle/vector-op。而 Div 不同：

```cpp
int64_t DivOp::estimateCycles(const HardwareConfig &config) {
  return estimateVectorCycles(n, 12, bits, ...);  // 不是 1，是 12 cycles
}
```

transcendental 函数更贵（`.td` 里的默认值 vs `.cpp` 里的校准后值）：

| Op | .td 默认 | .cpp 校准值 | 原因（代码注释） |
|----|---------|-----------|-----------------|
| Exp | 3 | 9 | pipeline chain RAW stall |
| Log | 4 | 12 | 迭代近似 |
| Tanh | 6 | 18 | 内部多次 exp |
| Sigmoid | 5 | 15 | 1/(1+exp(-x)) |
| Sqrt/Rsqrt | 2 | 6 | 延迟限制 |
| Div | 4 | 12 | 延迟限制 |

`.td` 是"理论值"，`.cpp` 是 **FlashAttention 实测校准后的值**。这就是 costmodel 优化的核心——根据实际 profiling 数据调整 `cyclesPerVectorOp`。

**Reduce 运算**——公式最复杂的一个：

```cpp
// Reduce 分三步:
// ① 每 vector 一个 cycle (向量内计算)
// ② 向量内 tree-reduce: log2(vector_width) cycles
// ③ 跨向量 tree-reduce: log2(num_vectors) cycles

int64_t vectorWidth = 2048 / bits;
int64_t numVectors = (numElems + vectorWidth - 1) / vectorWidth;

int vectorReduceCycles = 0;
for (int64_t w = vectorWidth; w > 1; w /= 2)   // ② 每个向量内部 log2 级归约
  vectorReduceCycles++;

int crossVectorCycles = 0;
for (int64_t v = numVectors; v > 1; v /= 2)     // ③ 跨向量的归约
  crossVectorCycles++;

return numVectors + vectorReduceCycles + crossVectorCycles + startupLatency;
```

例：8192 个 FP32 元素做 ReduceSum
- vectorWidth = 2048/32 = 64
- numVectors = ceil(8192/64) = 128
- vectorReduceCycles = log2(64) = 6
- crossVectorCycles = log2(128) = 7
- total = 128 + 6 + 7 + startup = 141 + startup cycles

**Broadcast**——几乎是零开销：

```cpp
int64_t BroadcastOp::estimateCycles(const HardwareConfig &config) {
  return 1 + config.getVectorStartupLatency();  // 只是一个地址重映射操作
}
```

**总结：每种 Op 的 estimateCycles 公式**

| Op 类型 | 公式 |
|---------|------|
| Matmul | `ceil(M/16) * ceil(N/16) * ceil(K/16) + startup` |
| 搬运(MTE2/MTE3/FixPipe) | `bytes * clock / bandwidth + startup` |
| 简单向量(Add/Sub/Mul...) | `ceil(N/(2048/bit)) * 1 + startup` |
| Div | `ceil(N/(2048/bit)) * 12 + startup` |
| Exp/Log/Tanh/Sigmoid | `ceil(N/(2048/bit)) * 9~18 + startup` |
| Reduce | `numVectors + log2(vecWidth) + log2(numVectors) + startup` |
| Broadcast | `1 + startup` |

**产物**：`libAscendModelIR.a`，链接了 MLIRIR、MLIRSupport 等。

#### 6.2d. `AscendModelAnalysis` — 硬件配置 + 流水线分析 (`costmodel/lib/AscendModel/Analysis/`)

**CMakeLists.txt**：

```cmake
add_mlir_library(AscendModelAnalysis
  HardwareConfig.cpp           # JSON 解析 → HardwareConfig 对象 + 查询 API
  HIVMAnalysis.cpp             # HIVM IR 分析
  PipelineAnalysis.cpp         # PipelineScheduler + RooflineAnalyzer + PerformanceReport

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/Analysis

  DEPENDS
  AscendModelOpsIncGen         # 等 TableGen 完成

  LINK_LIBS PUBLIC
  AscendModelIR                # 依赖 Op 定义
  MLIRIR MLIRSupport ...
)
```

**产物**：`libAscendModelAnalysis.a`

---

源码详解（三个文件，共 1000+ 行，下面分块分析）：

##### HardwareConfig.cpp — 硬件参数加载与查询

这个文件有三个核心职责：

**① 全局配置管理**：

```cpp
static std::unique_ptr<HardwareConfig> globalConfig;   // 全局单例

HardwareConfig &getHardwareConfig() {
  if (!globalConfig)
    globalConfig = HardwareConfig::getDefault910B();  // 懒加载
  return *globalConfig;
}

void setHardwareConfig(std::unique_ptr<HardwareConfig> config) {
  globalConfig = std::move(config);                   // 替换全局配置
}
```

这提供了全局的"当前硬件是什么"——后续所有 costmodel 代码通过 `getHardwareConfig()` 拿到硬件参数。

**② JSON → C++ 转换**（`loadFromFile` + `parseJSON`）：

`loadFromFile()` 一条流水线：文件路径 → `MemoryBuffer::getFile` 读取 → `json::parse` 解析 → `loadFromJSON` → `parseJSON`。

`parseJSON()` 的结构：

```
JSON 根对象
├── "name"/"vendor"/"version"  → 字符串直接读
├── "clock": {"frequency_ghz"} → clockFreqGHz
├── "memory_spaces": {         → 遍历每个 kv:
│     "hbm": {                 → MemorySpace 结构体
│       "type": "off_chip"     → parseMemoryType()
│       "size_gb": 32          → sizeBytes = gb * 1024³
│       "bandwidth_gbps": N    → bandwidthBytesPerCycle = gbps*1e9/(GHz*1e9)
│       "latency_cycles": 200  → latencyCycles
│     }
│   }
├── "compute_units": {...}     → ComputeUnit 结构体（Cube/Vector）
├── "data_movers": {...}       → DataMover 结构体（MTE1/MTE2/MTE3/FixPipe）
├── "pipeline": {              → PipelinePath（cube_path/vector_path）
│     "parallelism": {...}     → parallelismFlags（哪些路径可以并行）
│   }
└── "calibration": {           → vectorOpCyclesPerInstruction 表
      "vector_op_cycles_per_vec_instruction": {
        "simple_ops_add_sub_mul_etc": 2,    → vadd=2, vsub=2, vmul=2
        "exp": 7                             → vexp=7
      }
    }
```

关键转换：**带宽从 GB/s 转成 bytes/cycle**：

```cpp
// JSON 里写的是 GB/s，代码里存的是 bytes/cycle
space.bandwidthBytesPerCycle = (gbps * 1e9) / (clockFreqGHz * 1e9);
// 例：HBM 1600 GB/s @ 1.85 GHz:
//   = 1600e9 / 1.85e9 = 865 bytes/cycle
//   → 每 0.54ns 搬 865 字节
```

**③ Hardcoded fallback**（`createHardcodedDefault910B`）：当 JSON 文件找不到时，硬编码了完整的 910B 配置：7 级存储空间（HBM→L2→L1→L0A/L0B/L0C→UB）、Cube（320 TFLOPS FP16）/Vector（128 元素/cycle）、5 个 DataMover（cube_mte2/mte1/fixpipe/vector_mte2/mte3）、2 条 pipeline 路径、并行标记。

**④ 校准参数**——这些是 costmodel 优化的**调参入口**：

```cpp
int getVectorStartupLatency() const { return 35; }
// 从 10 → 35：反映 dependent vector 指令之间的 UB RAW 惩罚

double getAIVScalarOverheadFactor() const { return 3.74; }
// vec_ratio=0.211 → 纯 vector 时间只占 21%，79% 是 scalar+barrier+idle
// factor = 0.789/0.211 = 3.74

int getPipeBarrierCyclesPerIter() const { return 7500; }
// BM=64, 1-wave: AIV wall 59187 cycles, idle=39%=23044, /3 iters = 7500/iter
```

**⑤ 查询方法**——给外部用的 getter：

```cpp
double getHBMBandwidthGBs() const {
  // 反向转换: bytes/cycle → GB/s
  return hbm->bandwidthBytesPerCycle * clockFreqGHz * 1e9 / 1e9;
}

int64_t estimateCubeCycles(int64_t M, N, K) const {
  return ceil(M/16) * ceil(N/16) * ceil(K/16);
}

int64_t estimateVectorCycles(int64_t numElements) const {
  return ceil(numElements / getVectorWidthElements());     // ceil(N/128)
}

int64_t estimateMemoryCycles(llvm::StringRef moverName, int64_t bytes) const {
  return ceil(bytes / mover->bandwidthBytesPerCycle);      // ceil(B/bandwidth)
}
```

---

##### PipelineAnalysis.cpp — 调度算法

这个文件实现了 costmodel 的核心算法：怎么把一组 Op 排列到各硬件单元上，算出总耗时。

**数据结构**：

```cpp
struct PipelineOp { int64_t opId; HWUnit hwUnit; int64_t startCycle/endCycle/duration;
                    Operation *mlirOp; int64_t bytes/flops; SmallVector<int64_t,4> dependsOn; };

class HWUnitPipeline { void scheduleOp(op, earliestStart);  // 核心：在一根流水线上调度一个 Op
                        int64_t currentCycle; };             // 这个单元下一次可用的时间

class DependencyGraph { void addOp/addDependency;
                        vector<int64_t> getTopologicalOrder();   // Kahn 拓扑排序
                        bool hasCycle(); };                       // 循环检测

class PipelineScheduler {   // 主调度器
  bool schedule();          // ASAP 算法：每步把 Op 放到对应单元，start = max(依赖完成, 单元空闲)
  int64_t getTotalCycles(); // 单程序关键路径
  int64_t getKernelCycles(numPrograms, numParallelUnits, numInnerIters);
      // = (totalCycles + barrierCycles) * (1 + scalar_factor) * ceil(programs/units)
};
```

**ASAP 调度核心逻辑**（`PipelineScheduler::schedule()`）：

```cpp
bool PipelineScheduler::schedule() {
  // ① 拓扑排序（保证生产者先于消费者）
  vector<int64_t> order = depGraph.getTopologicalOrder();

  // ② 按拓扑序逐个调度
  for (int64_t opId : order) {
    PipelineOp &op = operations[...];
    int64_t earliestStart = getEarliestStartTime(op);   // 依赖完成的最晚时间

    auto &pipeline = pipelines[op.hwUnit];
    pipeline.scheduleOp(op, earliestStart);              // start = max(单元空闲, 依赖完成)
    totalCycles = max(totalCycles, op.endCycle);
  }
  return true;
}
```

**Kernel 级 cycle 公式**（从单次程序外推到完整 kernel）：

```cpp
int64_t getKernelCycles(numPrograms, numParallelUnits, numInnerIters) {
  int64_t barrierCycles = numInnerIters * getPipeBarrierCyclesPerIter();  // ① barrier 同步
  double scalarFactor = getAIVScalarOverheadFactor();                      // ② scalar overhead
  int64_t perProgramCycles = (totalCycles + barrierCycles) * (1.0 + scalarFactor);
  int64_t numWaves = ceil(numPrograms / numParallelUnits);                 // ③ wave 串行化
  return perProgramCycles * numWaves;
}

// 例：totalCycles=1000, innerIters=3, barrier=7500, scalar_factor=3.74,
//     numPrograms=80, numParallelUnits=40
//   perProgram = (1000 + 3*7500) * (1 + 3.74) = 23500 * 4.74 = 111,390
//   numWaves   = ceil(80/40) = 2
//   kernelCycles = 111,390 * 2 = 222,780
```

**Roofline 分析**：

```cpp
bool RooflineAnalyzer::isComputeBound() const {
  double ai = totalFLOPs / totalBytes;          // 算术强度
  double ridgePoint = peakTFLOPS / peakBWTBps;  // Roofline 转折点 = 算力/带宽
  return ai >= ridgePoint;                      // AI 高于转折点 → 计算受限
}
// 例：910B peak=320 TFLOPS, peakBw=1.6 TB/s
//   ridge = 320/1.6 = 200 FLOP/Byte
//   如果 kernel 的 AI=50 FLOP/Byte → 内存受限
//   如果 kernel 的 AI=500 FLOP/Byte → 计算受限
```

**依赖关系图**（用拓扑排序检测循环 + 保证调度顺序）：
- `addOp()` → edges / reverseEdges（邻接表）
- `getTopologicalOrder()` → Kahn 算法（入度为零入队列 → BFS）
- `hasCycle()` → `order.size() != ops.size()`（循环检测）

---

#### 6.2e. `AscendModelTransforms` — 6+1 个 MLIR Pass（6 个管线内 + 1 个独立，`costmodel/lib/AscendModel/Transforms/`）

**CMakeLists.txt**：

```cmake
add_mlir_library(AscendModelTransforms
  ConvertTritonToAscend.cpp    # ① TTIR → AscendModel IR
  InsertDataTransfers.cpp      # ② 补全数据搬运 Op
  AssignOpIDs.cpp              # ③ 给每个 Op 分配唯一 op_id (0,1,2...)
  EstimateCycles.cpp           # ④ 遍历 Op，调 estimateCycles(config)，写 attributes
  HIVMAnalysisPass.cpp         # (独立) HIVM IR 分析包装，不加入 ascend-perf-model 管线
  PipelineAnalysisPass.cpp     # ⑤ 调 PipelineScheduler，写 module-level attributes
  PerfReportPass.cpp           # ⑥ 读所有 attributes，汇编成 PerformanceReport
  PassRegistration.cpp         # 注册 ascend-perf-model 管线

  DEPENDS
  AscendModelOpsIncGen
  AscendModelTransformsPassIncGen  # Passes.td → Passes.h.inc

  LINK_LIBS PUBLIC
  AscendModelIR             # 依赖 Op 类型（AddOp/CubeLoadOp/MatmulOp...）
  AscendModelAnalysis       # 依赖分析工具（HardwareConfig/PipelineScheduler/RooflineAnalyzer）
  MLIRIR MLIRPass MLIRSupport ...
)
```

**产物**：`libAscendModelTransforms.a`

**Pass 的职责和顺序**——`ascend-perf-model` 管线包含 6 个 Pass（`PassRegistration.cpp:59-95`），HIVMAnalysisPass 是独立入口（`--analyze-hivm`），不参与这个管线：

```
管线内 (ascend-perf-model):
  ① ConvertTritonToAscend:   TTIR Op → AscendModel Op (tt.add → ascend.add)
  ② InsertDataTransfers:     补充 ascend.cube_load/vector_load/cube_store/vector_store
  ③ AssignOpIDs:             给每个 Op 写 op_id = 0, 1, 2, ... (用于依赖跟踪)
  ④ EstimateCycles:          调 Op::estimateCycles(config) → 写 estimated_cycles, bytes, flops, hw_unit
  ⑤ PipelineAnalysisPass:    读所有 Op 的 estimated_cycles → 建依赖图 → PipelineScheduler::schedule()
                              → 写 module 属性: scheduled_cycles_one_iter / roofline_cycles / simple_sum_cycles
  ⑥ PerfReportPass:          读所有 attributes → 组装 PerformanceReport → 打印/导出 JSON

独立入口 (单独调用):
  HIVMAnalysisPass:         分析 HIVM 原生 IR 的调度和同步 (--analyze-hivm)，不参与 ascend-perf-model 管线
```

**Pass 间数据流（通过 IR 上的 attributes 传递）**：

```
Step ①: IR 从 Triton dialect → AscendModel dialect
  %0 = tt.add %a, %b          →  %0 = ascend.add %a, %b

Step ②: 自动插入数据搬运 Op
  ascend.add %a, %b           →  ascend.vector_load %a ...
                                  ascend.vector_load %b ...
                                  ascend.add ...
                                  ascend.vector_store ...

Step ③: 分配 ID
  ascend.add %a, %b           →  ascend.add {op_id = 5} %a, %b

Step ④: 估算 cycles
  ascend.add {op_id = 5}      →  ascend.add {op_id = 5, estimated_cycles = 57,
                                             flops = 1024, hw_unit = "Vector"}

Step ⑤: HIVM 分析（外部 IR 依赖，非核心路径）

Step ⑥: 调度分析
  (读每个 Op 的 estimated_cycles + hw_unit + dependsOn)
       ↓
  PipelineScheduler::schedule()
       ↓
  → module attributes: {scheduled_cycles_one_iter = 60320,
                        roofline_cycles = 45200,
                        simple_sum_cycles = 88000}

Step ⑦: 汇总报告
  (读所有 attributes → PerformanceReport → print/toJSON)
```

### 6.3 进入单元测试：两级 `if` 大门

回到 `third_party/ascend/CMakeLists.txt:112-114`：
```cmake
if(TRITON_BUILD_UT)           # ← 第一级：TRITON_BUILD_UT 必须为 ON
  add_subdirectory(unittest)  # 进入 third_party/ascend/unittest/
endif()
```

进入 `third_party/ascend/unittest/CMakeLists.txt:29-31`：
```cmake
if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)  # ← 第二级：costmodel 源必须存在
  add_subdirectory(costmodel_ut)         # 进入 unittest/costmodel_ut/
endif()
```

### 6.4 `costmodel_ut/CMakeLists.txt` — 声明三个测试可执行文件

`third_party/ascend/unittest/costmodel_ut/CMakeLists.txt:1-32`：

```cmake
include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME CostModelHardwareConfig              # 可执行文件名
  SRCS HardwareConfigTest.cpp               # 测试源文件
  LIBS AscendModelAnalysis                  # 需要链接的被测库
)

add_triton_ut(
  NAME CostModelPipelineScheduler
  SRCS PipelineSchedulerTest.cpp
  LIBS AscendModelAnalysis AscendModelIR
)

add_triton_ut(
  NAME CostModelPasses
  SRCS PassesTest.cpp
  LIBS AscendModelTransforms AscendModelAnalysis AscendModelIR
    MLIRArithDialect MLIRFuncDialect MLIRIR MLIRParser
    MLIRPass MLIRSCFDialect MLIRSupport
)
```

三个源文件及其包含的 GTest 测试（`TEST(ClassName, TestName)`）：

| 源文件 | GTest 测试 | 被测模块 |
|--------|-----------|---------|
| `HardwareConfigTest.cpp` | `Default910BHasExpectedBasics` | 硬件配置 |
| | `ParsesCustomJson` | JSON 解析 |
| | `RejectsInvalid` | 配置校验 |
| | `LoadFromCommitConfigFile` | 配置文件加载 |
| | `VectorWidthValidation` | Vector 宽度约束 |
| `PassesTest.cpp` | `AssignOpIDs` | Pass: 分配 Op ID |
| | `EstimateCycles` | Pass: Cycle 估算 |
| | `PipelineAnalysis` | Pass: 流水线分析 |
| | `PerfReport` | Pass: 性能报告 |
| | `InvalidArgBindings` | 错误处理 |
| `PipelineSchedulerTest.cpp` | `DifferentHardwareUnits` | 不同硬件单元并行 |
| | `SameHardwareUnitSerializes` | 同一单元串行 |
| | `DependenciesDelayConsumers` | 数据依赖 |
| | `RejectsCyclic` | 循环依赖检测 |
| | `KernelCyclesApplyBarrier` | Kernel 级估算 |
| | `KernelCyclesWaves` | 多 Wave 估算 |
| | `RooflineAnalyzer` | Roofline 瓶颈分析 |

### 6.5 `add_triton_ut()` 展开 — 5 件事

`cmake/AddTritonUnitTest.cmake:1-37`：

```cmake
include(GoogleTest)          # CMake 内置，提供 gtest_discover_tests()
enable_testing()             # 启用 CTest，给 build 目录写 CTestTestfile.cmake

function(add_triton_ut)
  # ── ① 注册到 CTest ──
  add_test(NAME ${__NAME} COMMAND ${__NAME})
  # 效果：CTestTestfile.cmake 里多一条:
  #   add_test(CostModelPasses "CostModelPasses")

  # ── ② 编译可执行文件 ──
  add_executable(${__NAME} ${__SRCS})
  # 例如: add_executable(CostModelPasses PassesTest.cpp)
  # 默认输出到 CMAKE_CURRENT_BINARY_DIR:
  #   build/<cmake_dir>/third_party/ascend/unittest/costmodel_ut/CostModelPasses

  # ── ③ 链接库 ──
  target_link_libraries(${__NAME} PRIVATE
    GTest::gtest_main          # GoogleTest 框架（含 main 函数入口）
    gmock                      # GoogleMock
    ${__LIBS}                  # AscendModelTransforms, AscendModelAnalysis, MLIR*...
  )
  # 以 CostModelPasses 为例，最终的链接依赖图:
  #   CostModelPasses
  #     ├── libgtest_main.a     (GoogleTest)
  #     ├── libgmock.a          (GoogleMock)
  #     ├── libAscendModelTransforms.a
  #     │     ├── libAscendModelIR.a
  #     │     └── libAscendModelAnalysis.a
  #     │           └── libAscendModelIR.a
  #     ├── libAscendModelAnalysis.a
  #     ├── libAscendModelIR.a
  #     ├── libMLIRArithDialect.a  (LLVM 预编译)
  #     ├── libMLIRIR.a            (LLVM 预编译)
  #     ├── libMLIRParser.a        (LLVM 预编译)
  #     ├── libMLIRPass.a          (LLVM 预编译)
  #     └── ...

  target_compile_options(${__NAME} PRIVATE -fno-rtti)

  # ── ④ 发现每个 GTest case ──
  gtest_discover_tests(${__NAME} DISCOVERY_TIMEOUT 60)
  # CMake 在 build 阶段运行 CostModelPasses --gtest_list_tests，解析输出:
  #   CostModelPassesTest.
  #     AssignOpIDs
  #     EstimateCycles
  #     PipelineAnalysis
  #     PerfReport
  #     InvalidArgBindings
  # 然后为每个 case 生成独立的 ctest 条目，类似:
  #   add_test(CostModelPassesTest.AssignOpIDs CostModelPasses --gtest_filter=...)

  # ── ⑤ 加入聚合目标 ──
  add_dependencies(TritonUnitTests ${__NAME})
  # TritonUnitTests 是 CMakeLists.txt:76 定义的自定义目标
endfunction()
```

### 6.6 GoogleTest 下载

`cmake/AddTritonUnitTest.cmake:1` → `unittest/googletest.cmake:1-22`：

```cmake
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)
```

CMake 在 **configure 阶段**（即 `cmake -G Ninja ...` 这一步）从 GitHub 下载 GoogleTest v1.17.0 源码，编译出 `libgtest.a`、`libgtest_main.a`、`libgmock.a`。

### 6.7 最终产物：可执行文件位置 + CTest 注册

```
build/<cmake_dir>/third_party/ascend/unittest/costmodel_ut/
├── CostModelHardwareConfig     ← g++ HardwareConfigTest.cpp + libAscendModelAnalysis.a + GTest
├── CostModelPipelineScheduler  ← g++ PipelineSchedulerTest.cpp + libAscendModelAnalysis.a + libAscendModelIR.a + GTest
└── CostModelPasses             ← g++ PassesTest.cpp + libAscendModelTransforms.a + libAscendModelAnalysis.a + libAscendModelIR.a + MLIR* + GTest
```

`build/<cmake_dir>/CTestTestfile.cmake` 中注册的内容（`add_test` + `gtest_discover_tests` 的结果）：

```
# add_triton_ut → add_test() 生成的条目：
add_test(CostModelHardwareConfig "CostModelHardwareConfig")
add_test(CostModelPipelineScheduler "CostModelPipelineScheduler")
add_test(CostModelPasses "CostModelPasses")

# gtest_discover_tests() 生成的条目（每个 GTest case 一个）：
add_test(CostModelHardwareConfigTest.Default910BHasExpectedBasics ...)
add_test(CostModelHardwareConfigTest.ParsesCustomJson ...)
add_test(CostModelPassesTest.AssignOpIDs ...)
add_test(CostModelPassesTest.EstimateCycles ...)
...
```

### 6.8 运行方式：ctest vs 直接执行

```bash
# ctest 方式（必须在 build 目录下运行）
cd build/<cmake_dir>
ctest -R CostModel                     # 正则匹配所有含 CostModel 的测试
ctest -R CostModelPasses               # 只跑 PassesTest 的
ctest -R "CostModelPassesTest.EstimateCycles"  # 精确到单个 GTest case

# 直接执行可执行文件方式（可以在任意目录运行）
./build/<cmake_dir>/third_party/ascend/unittest/costmodel_ut/CostModelPasses
./build/<cmake_dir>/third_party/ascend/unittest/costmodel_ut/CostModelPasses \
    --gtest_filter='*EstimateCycles*'
./build/<cmake_dir>/third_party/ascend/unittest/costmodel_ut/CostModelPasses \
    --gtest_list_tests                 # 列出所有 GTest case
```

> **注意**：这个项目有两个测试体系：
> - **CTest** → C++ GTest 单元测试（`TRITON_BUILD_UT=ON` 时才启用，即 `costmodel_ut/*` 和 `unittest/*`）
> - **LIT** → MLIR 集成测试（始终构建，通过 `ninja check-triton-lit-tests` 运行，测试 `.mlir`/`.ll` 文件）
>
> CTest 和 LIT 是两套独立的系统：CTest 运行 C++ GTest 可执行文件，LIT 运行 shell 风格的 MLIR 测试脚本。

---

## 第 7 步：CMake 内部流程 (`CMakeLists.txt`)

### 7a. ccache 配置 (`CMakeLists.txt:28-41`)

```cmake
if(TRITON_BUILD_WITH_CCACHE)          # true（从命令行传入）
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()
```

`ccache` 通过缓存之前的编译结果，在重复编译时大幅加速（缓存命中时几乎零开销）。

### 7b. 查找 MLIR/LLVM

```cmake
# CMakeLists.txt:93-109
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

include(TableGen)    # 引入 add_mlir_tablegen 等
include(AddLLVM)
include(AddMLIR)
```

MLIR_DIR 指向下载的 LLVM 预编译包中的 `lib/cmake/mlir/`，包含了 `TableGen`、`AddMLIR`、`AddLLVM` 等 CMake 模块。

### 7c. 添加子目录（构建顺序）

```
include/           → TableGen 生成 + 头文件
lib/               → 库代码
  lib/Dialect/     → Triton/TritonGPU/TritonAscend 等 Dialect 定义
  lib/Conversion/  → Dialect 之间的转换 Pass
  lib/Target/      → 目标代码生成
bin/               → triton-opt、triton-mlir-opt 工具
test/              → 测试
unittest/          → C++ 单元测试（TRITON_BUILD_UT=OFF 时跳过）
```

### 7d. Python 模块构建 (`CMakeLists.txt:204-354`)

当 `TRITON_BUILD_PYTHON_MODULE=ON` 时：

#### 添加各后端子目录

```cmake
foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})  # ascend;nvidia;amd
  add_subdirectory(third_party/${CODEGEN_BACKEND})
endforeach()
```

这会进入 `third_party/ascend/CMakeLists.txt`、`third_party/nvidia/CMakeLists.txt`、`third_party/amd/CMakeLists.txt`，编译各后端各自的 C++ Dialect、Pass 和 Conversion。

#### 编译 `libtriton.so` —— Python 绑定

```cmake
add_library(triton SHARED
    ${PYTHON_SRC_PATH}/main.cc          # pybind11 模块入口，注册子模块
    ${PYTHON_SRC_PATH}/ir.cc            # MLIR IR 操作绑定
    ${PYTHON_SRC_PATH}/gluon_ir.cc      # Gluon IR 绑定
    ${PYTHON_SRC_PATH}/linear_layout.cc # LinearLayout 绑定
    ${PYTHON_SRC_PATH}/passes.cc        # Pass 绑定
    ${PYTHON_SRC_PATH}/interpreter.cc   # Triton Interpreter 绑定
    ${PYTHON_SRC_PATH}/llvm.cc          # LLVM 相关绑定
    ${PYTHON_SRC_PATH}/specialize.cc    # 特化相关
    ${PYTHON_SRC_PATH}/../triton/extension/buffer/src/buffer_ir.cc
)
```

链接依赖：

```cmake
target_link_libraries(triton PRIVATE ${TRITON_LIBRARIES})
```

`TRITON_LIBRARIES` 包含：
- 所有 `triton_libs`（通过 `add_triton_library` 注册的库）
- 所有 `triton_plugins`（通过 `add_triton_plugin` 注册的插件）
- MLIR 库：`MLIRLLVMDialect`、`MLIRPass`、`MLIRTransforms`、`MLIRTargetLLVMIRExport` 等
- LLVM 库：`LLVMPasses`、`LLVMNVPTXCodeGen`、`LLVMAMDGPUCodeGen` + 架构相关 CodeGen
- `pybind11::headers`、`Python3::Module`

#### 生成 entryC 辅助库

```cmake
add_library(entryC SHARED ${PROJECT_SOURCE_DIR}/lib/runtime/libentry/libentry.cpp)
```

这是 Triton kernel 启动时的 C 入口点辅助库。

#### 拷贝 FileCheck

```cmake
configure_file("${LLVM_SYSPATH}/bin/FileCheck" "${TRITON_WHEEL_DIR}/FileCheck" COPYONLY)
```

`FileCheck` 用于运行 LIT 测试时的输出验证。

---

## 第 8 步：`get_cmake_dir()`——构建目录位置 (`python/build_helpers.py`)

```python
def _get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return Path(get_base_dir()) / "build" / dir_name
```

例如 macOS arm64 + CPython 3.10 的构建目录：
```
build/cmake.macosx-14.0-arm64-cpython-310/
```

可通过 `TRITON_BUILD_DIR` 环境变量覆盖。

---

## 第 9 步：构建目录所有产物详解

`build/cmake.<plat>-cpython-<ver>/` 目录下，`cmake --build .` 之后的完整产物：

### 9a. 目录总览

```
build/cmake.<plat>-cpython-<ver>/
├── bin/                              # ── ① 可执行文件 ──
│   ├── triton-opt
│   ├── triton-mlir-opt
│   ├── triton-llvm-opt
│   ├── triton-reduce
│   ├── triton-lsp
│   └── triton-tensor-layout
│
├── third_party/ascend/unittest/costmodel_ut/  # ── ② CostModel 测试可执行文件 ──
│   ├── CostModelHardwareConfig
│   ├── CostModelPipelineScheduler
│   └── CostModelPasses
│
├── third_party/ascend/costmodel_build/        # ── ③ CostModel 静态库 + .inc ──
│   ├── include_AscendModel/AscendModel/       # TableGen 生成的 .inc 文件
│   └── lib_AscendModel/                       # libAscendModel*.a
│
├── lib/                              # ── ④ Triton 核心库的中间 .o / .a ──
│   ├── Analysis/
│   ├── Conversion/
│   ├── Dialect/
│   ├── Target/
│   ├── Tools/
│   └── Instrumentation/
│
├── test/                             # ── ⑤ LIT 测试配置 ──
│   ├── lit.site.cfg.py              # 由 lit.site.cfg.py.in 生成
│   └── lib/                          # 测试辅助库 (TritonTestAnalysis 等)
│
├── python/                           # ── ⑥ libtriton.so 构建链表 ──
│   └── (中间 .o 文件)
│
├── python_packages/                  # ── ⑦ MLIR Python 绑定 ──
│   └── triton/
│
├── CTestTestfile.cmake               # ── ⑧ CTest 入口（TRITON_BUILD_UT=ON 时才有内容）──
├── compile_commands.json             # ── ⑨ IDE 编译数据库 ──
├── build.ninja                       # Ninja 构建规则
└── CMakeFiles/                       # CMake 内部文件
```

### 9b. ① `bin/` — 可执行文件

**代码依据：`bin/CMakeLists.txt`**

使用 LLVM 的 `add_llvm_executable` 宏。LLVM CMake 模块默认将可执行文件输出到 `${CMAKE_BINARY_DIR}/bin/`。

| 可执行文件 | 来源 | 用途 | 引用 |
|-----------|------|------|------|
| `triton-opt` | `bin/triton-opt.cpp` | MLIR 主力优化/转换工具 | `bin/CMakeLists.txt:8` |
| `triton-mlir-opt` | `bin/triton-mlir-opt.cpp` | Ascend 编译器前端 | `bin/CMakeLists.txt:181` |
| `triton-llvm-opt` | `bin/triton-llvm-opt.cpp` | LLVM IR 优化工具 | `bin/CMakeLists.txt:128` |
| `triton-reduce` | `bin/triton-reduce.cpp` | 测试用例缩减 | `bin/CMakeLists.txt:48` |
| `triton-lsp` | `bin/triton-lsp.cpp` | 语言服务器 | `bin/CMakeLists.txt:88` |
| `triton-tensor-layout` | `bin/triton-tensor-layout.cpp` | Layout 可视化 | `bin/CMakeLists.txt:148` |

这些二进制如何被 LIT 测试找到——`test/lit.cfg.py:45-46`：

```python
config.triton_tools_dir = os.path.join(config.triton_obj_root, 'bin')
tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir, config.filecheck_dir]
tools = ['triton-opt', 'triton-llvm-opt', ...]
llvm_config.add_tool_substitutions(tools, tool_dirs)
```

### 9c. ② `third_party/ascend/unittest/costmodel_ut/` — CostModel C++ 测试

详见第 6 步完整链路。由 `add_triton_ut()`→`add_executable()` 生成，输出到 `${CMAKE_CURRENT_BINARY_DIR}`。

### 9d. ③ CostModel 静态库 + .inc

TableGen 产物和三个 `.a` 库，位置由 `third_party/ascend/CMakeLists.txt:63-64` 的 `add_subdirectory` 第二个参数指定。

### 9e. ④ `lib/` — Triton 核心 C++ 库（中间产物）

`lib/CMakeLists.txt:1-6`：
```cmake
add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Tools)
add_subdirectory(Instrumentation)
```

这些子目录中的代码用 `add_triton_library`（实际是 `add_triton_object`→`add_library(... OBJECT)`）编译为 OBJECT 库，最终链接进 `triton-opt` 和 `libtriton.so`。在 build 目录中留下 `.o` 和 `.a` 文件。

### 9f. ⑤ `test/` — LIT 测试配置（不是 CTest!）

`test/CMakeLists.txt:7-11`：
```cmake
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in    # 模板
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py        # 输出
)
```

`test/CMakeLists.txt:23-26`：
```cmake
add_lit_testsuite(check-triton-lit-tests
  ${CMAKE_CURRENT_BINARY_DIR}     # build/test/
  DEPENDS triton-opt triton-tensor-layout triton-llvm-opt
)
```

运行方式是 `ninja -C build check-triton-lit-tests`（通过 `Makefile:25-26`）。不是 ctest。

### 9g. ⑧ `CTestTestfile.cmake` — CTest 入口

由 `CMakeLists.txt:14` 的 `include(CTest)` 无条件生成。但当 `TRITON_BUILD_UT=OFF` 时，`add_test()` 和 `add_subdirectory(unittest)` 都不执行，所以这个文件实际上是**空壳**——ctest 找不到任何测试。

当 `TRITON_BUILD_UT=ON` 时，`add_triton_ut()`→`add_test()` 和 `gtest_discover_tests()` 才会往里面写入测试条目。

### 9h. ⑨ `compile_commands.json`

由 `setup.py:562` 的 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 生成。`setup.py:637` 在构建完成后做顶层 symlink：

```python
update_symlink(Path(self.base_dir) / "compile_commands.json",
               cmake_dir / "compile_commands.json")
```

产物：
```
./compile_commands.json → build/<cmake_dir>/compile_commands.json
```
IDE（clangd、VSCode 等）通过这个文件做代码补全和跳转。

---

## 最终产物

构建完成后，`python/triton/_C/` 目录下会包含：

| 文件 | 作用 |
|------|------|
| `libtriton.so` | 核心 C++ 编译器的 Python pybind11 绑定 |
| `triton-opt` | MLIR 优化工具（TTIR → TTAdapter 等转换） |
| `triton-mlir-opt` | MLIR 操作工具（支持 MLIR → Bytecode 格式） |

以及通过符号链接安装的 Python 包：

| 包 | 来源 |
|----|------|
| `triton.backends.ascend` | `third_party/ascend/backend/` |
| `triton.backends.nvidia` | `third_party/nvidia/backend/` |
| `triton.backends.amd` | `third_party/amd/backend/` |
| `triton.language.extra.*` | 各 backend 的 `language/` 子目录 |

---

## 完整调用链总结

```
pip install -e .
  └─ pyproject.toml → 安装构建依赖 → 调用 setuptools develop
  └─ setup.py 模块加载
       ├─ setdefault 环境变量（不覆盖用户已设置的）
       ├─ BackendInstaller.copy(["ascend","nvidia","amd"])  # 初始化 backends
       └─ setup(cmdclass={"develop": plugin_develop, "build_ext": CMakeBuild, ...})
  └─ plugin_develop.run()
       ├─ add_links(False)
       │    ├─ python/triton/backends/ascend → third_party/ascend/backend/
       │    ├─ python/triton/backends/nvidia → third_party/nvidia/backend/
       │    ├─ python/triton/backends/amd    → third_party/amd/backend/
       │    └─ language/tools extra 链接
       └─ develop.run() → 触发 build_ext
            └─ CMakeBuild.run()
                 ├─ download_and_copy_dependencies()
                 │    └─ 下载 ptxas, cuobjdump, cudart, cupti 等 NVIDIA 工具
                 └─ for ext in extensions:
                      └─ CMakeBuild.build_extension(ext)
                           ├─ get_thirdparty_packages([get_llvm_package_info()])
                           │    └─ 从华为云 OBS 下载 LLVM 预编译包 → ~/.triton/llvm/
                           ├─ 收集 cmake_args:
                           │    ├─ TRITON_BUILD_WITH_CCACHE=true → -DTRITON_BUILD_WITH_CCACHE=true
                           │    ├─ TRITON_BUILD_WITH_CLANG_LLD=true → clang/clang++/lld
                           │    ├─ TRITON_CODEGEN_BACKENDS=ascend;nvidia;amd
                           │    └─ PYTHON paths, LLVM paths, build type, ...
                           ├─ cmake configure (生成 Ninja build 文件)
                           ├─ cmake --build . (编译 C++ 代码 + Python 绑定)
                           ├─ cmake --build . --target mlir-doc
                           └─ 拷贝 + strip triton-opt, triton-mlir-opt → _C/

CMakeLists.txt 内部：
  ├─ find_program(ccache) → CMAKE_C_COMPILER_LAUNCHER       ← TRITON_BUILD_WITH_CCACHE
  ├─ clang/clang++/lld 作为编译器/链接器                      ← TRITON_BUILD_WITH_CLANG_LLD
  ├─ find_package(MLIR)
  ├─ add_subdirectory(include)      # TableGen + 头文件
  ├─ add_subdirectory(lib)          # Dialect/Conversion/Target C++ 库
  ├─ add_subdirectory(third_party/ascend)   # Ascend 后端 C++
  │    ├─ AscendNPU-IR (BiShengIR) submodule
  │    ├─ costmodel 检测 + 构建
  │    │    ├─ configure_file(HardwareParams.h.in → HardwareParams.h)
  │    │    ├─ TableGen: .td → .h.inc/.cpp.inc (Ops, Dialect, Passes...)
  │    │    ├─ AscendModelIR.a        (AscendModelDialect.cpp, AscendModelOps.cpp)
  │    │    ├─ AscendModelAnalysis.a  (HardwareConfig.cpp, PipelineAnalysis.cpp...)
  │    │    └─ AscendModelTransforms.a (AssignOpIDs.cpp, EstimateCycles.cpp...)
  │    ├─ add_triton_plugin(TritonAscend ...)  # pybind11 插件
  │    └─ if(TRITON_BUILD_UT):
  │         └─ add_subdirectory(unittest)
  │              └─ if(TRITON_ASCEND_HAS_INPROC_COSTMODEL):
  │                   └─ costmodel_ut/
  │                        ├─ add_triton_ut(CostModelHardwareConfig)
  │                        ├─ add_triton_ut(CostModelPipelineScheduler)
  │                        └─ add_triton_ut(CostModelPasses)
  │                             └─ add_triton_ut():
  │                                  ├─ add_test() → CTest 注册
  │                                  ├─ add_executable() → 编译 .cpp
  │                                  ├─ target_link_libraries(PRIVATE GTest::gtest_main gmock ${LIBS})
  │                                  ├─ gtest_discover_tests() → 发现 GTest case
  │                                  └─ add_dependencies(TritonUnitTests)
  ├─ add_subdirectory(third_party/nvidia)   # NVIDIA 后端 C++
  ├─ add_subdirectory(third_party/amd)      # AMD 后端 C++
  ├─ add_library(triton SHARED python/src/*.cc)  → libtriton.so
  │    └─ link: TRITON_LIBRARIES + MLIR + LLVM + pybind11 + Python3
  ├─ add_library(entryC SHARED lib/runtime/libentry/libentry.cpp)
  ├─ add_subdirectory(bin)              # triton-opt, triton-mlir-opt
  ├─ add_subdirectory(test)             # LIT tests (add_lit_testsuite)
  ├─ add_subdirectory(unittest)         # C++ 核心 unit tests (当 TRITON_BUILD_UT=ON)
  └─  FetchContent: 下载 GoogleTest v1.17.0 → libgtest.a + libgmock.a
```

---

## 环境变量速查

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `TRITON_BUILD_WITH_CCACHE` | `true` | 使用 ccache 加速编译 |
| `TRITON_BUILD_WITH_CLANG_LLD` | `true` | 使用 clang 编译器 + lld 链接器 |
| `TRITON_BUILD_PROTON` | `OFF` | 是否构建 Proton profiler |
| `TRITON_BUILD_UT` | (append) `OFF` | 是否构建 C++ 单元测试 |
| `TRITON_BUILD_DIR` | `build/cmake.<plat>-<impl>-<pyver>/` | CMake 构建输出目录 |
| `TRITON_CODEGEN_BACKENDS` | 自动检测 | 启用的后端（ascend;nvidia;amd） |
| `TRITON_PLUGIN_DIRS` | 空 | 外部插件目录（分号分隔） |
| `MAX_JOBS` | `2 * cpu_count()` | 并行编译任务数 |
| `TRITON_LLVM_SYSTEM_SUFFIX` | 自动检测 | LLVM 预编译包的系统后缀 |
| `LLVM_SYSPATH` | 自动下载 | 指向 LLVM 预编译包的路径 |
| `TRITON_APPEND_CMAKE_ARGS` | `-DTRITON_BUILD_UT=OFF` | 追加的 CMake 参数 |
