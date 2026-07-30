# costmodel 接入 autotune：从目标逆推的实现全记录

## 文档定位

面向需要理解或二次开发 costmodel-autotune 集成的人。以**目标逆推**方式展开，每段代码标注参考来源。

## 代码阅读地图

| 文件 | 阅读目的 |
|------|---------|
| `python/triton/runtime/jit.py` `KernelInterface.__getitem__` (L364) | `kernel[grid]` 如何变成 `self.run(grid=grid, ...)`，确认 `grid`/`warmup` 来源 |
| `python/triton/runtime/jit.py` `JITFunction.run()` (L695) | binder → `_pack_args` → `_do_compile` 完整链 |
| `python/triton/runtime/jit.py` `JITFunction._pack_args()` (L671) | options/signature/constexprs/attrs 分拣逻辑 |
| `python/triton/runtime/jit.py` `JITFunction.create_binder()` (L658) | `device_caches[device]` 5 元组结构 |
| `python/triton/runtime/autotuner.py` `Autotuner.run()` (L212) | 父类 autotune 流程 |
| `python/triton/runtime/autotuner.py` `Autotuner.prune_configs()` (L260) | `perf_model`/`early_config_prune` 剪枝机制 |
| `third_party/ascend/backend/runtime/autotuner.py` `AutoTilingTuner.__init__` (L210) | ascend tuner 入口 |
| `third_party/ascend/backend/runtime/autotuner.py` `AutoTilingTuner.run()` (L2192) | **主战场** — `prune_configs()` 和 `_batch_bench()` 之间 |
| `third_party/ascend/backend/runtime/costmodel_runtime.py` `costmodel_bench()` (L200) | costmodel 现有接口 |
| `python/triton/compiler/compiler.py` `triton.compile()` (L228) | `ast_to_ttir()` + stages 完整流程 |
| `python/triton/compiler/code_generator.py` `ast_to_ttir()` (L1659) | AST → raw TTIR |

---

## 目标：在 `run()` 中插入一行

`AutoTilingTuner.run()` 中 `prune_configs()` 和 `_batch_bench()` 之间是天然空隙：

```python
if cache_miss:
    pruned_configs = self.prune_configs(kwargs)

    # ======== costmodel pre-filter (inserted) ========
    if self.enable_costmodel_prune and len(pruned_configs) > self.costmodel_top_k:
        pruned_configs = self._prune_by_costmodel(pruned_configs, *args, **kwargs)
    # ================================================

    if self.enable_ubtuner or len(pruned_configs) > 1:
        timings = self._batch_bench(*args, configs=pruned_configs, **kwargs)
```

---

## 倒推第 1 层：`_prune_by_costmodel()` → `costmodel_prune()`

需求：接收 `List[Config]`，返回 costmodel 预测最快的 `top_k` 个。

`costmodel_bench()` 需要 `{config, ttir_text}` dict 列表，但 autotune 下 configs 动态生成，不能预编译。所以需要惰性 TTIR 回调 + 批量评估包装。

### `costmodel_prune()` — `costmodel_runtime.py`

```python
def costmodel_prune(configs, ttir_for_config, top_k=10, arg_bindings=None, hardware_config=None):
    # 惰性收集 TTIR（并行化，ThreadPoolExecutor）
    items = []
    for cfg in configs:
        ttir = ttir_for_config(cfg)   # 惰性回调, 只编译需要的
    # 批量调 costmodel_bench（内置多线程）
    latencies = costmodel_bench(items)
    # 按预测延迟升序, 取 top_k
    return sorted(latencies, key=lambda c: latencies[c])[:top_k], latencies
```

**并行化**：TTIR 收集和 costmodel 评估都用了 `ThreadPoolExecutor`，worker 数由 `TRITON_COSTMODEL_WORKER_NUM` 环境变量或 `os.cpu_count()` 决定。

### `_prune_by_costmodel()` — `autotuner.py`

```python
def _prune_by_costmodel(self, configs, *args, **kwargs):
    arg_bindings = self._build_costmodel_arg_bindings(**kwargs)
    pruned, predictions = costmodel_prune(
        configs,
        ttir_for_config=lambda cfg: self._costmodel_compile_ttir(cfg, *args, **kwargs),
        top_k=self.costmodel_top_k,
        arg_bindings=arg_bindings,
        hardware_config=self.costmodel_hardware_config or None,
    )
    if self.costmodel_save_predictions:
        self._costmodel_predictions.update(predictions)  # 留作精度分析
    return pruned if pruned else list(configs)
```

---

## 倒推第 2 层：`_costmodel_compile_ttir()` → `generate_ttir_for_costmodel()`

需求：给定一个 `Config`，返回优化后 TTIR 文本。

### `generate_ttir_for_costmodel()` — `compiler.py`

参考两个上游函数，各取其一部分：

**参数装配部分 → `JITFunction._do_compile()` (jit.py:826)**

```python
# 上游：
kernel_cache, _, target, backend, _ = self.device_caches[device]
src = self.ASTSource(self, signature, constexprs, attrs)

# 我们的对应：
_, _, _, backend, binder = jit_fn.device_caches[device]
_, specialization, base_options = binder(**bound_args)
options, signature, constexprs, attrs = jit_fn._pack_args(
    backend, runtime_kwargs, bound_args, specialization, base_options)
src = ASTSource(jit_fn, signature, constexprs, attrs)
```

**TTIR 生成部分 → `triton.compile()` (compiler.py:228)**

```python
# 上游：
context = ir.context(); ir.load_dialects(context); ...
module = src.make_ir(target, options, ...)
for ext, compile_ir in stages: next_module = compile_ir(module, ...)

# 我们的对应：
ctx = triton_ir.context(); triton_ir.load_dialects(ctx); ...
mod = ast_to_ttir(jit_fn, src, ...)
mod = make_ttir(mod, metadata, options)  # stages 的第一个 stage
return str(mod)  # ← 在这里停住, 不继续 linalg/npu
```

### `_costmodel_compile_ttir()` — `autotuner.py`

上层的缓存包装：

```python
def _costmodel_compile_ttir(self, config, *args, **kwargs):
    # cache key: 只用 config.kwargs（tl.constexpr），不含 backend 参数
    cache_raw = str(sorted(config.kwargs.items()))
    cache_key = sha256(f"{self.fn.cache_key}-{cache_raw}")

    if cache_key in self._costmodel_ttir_cache:
        cached = self._costmodel_ttir_cache[cache_key]
        return cached[0] if isinstance(cached, tuple) else cached  # tuple=有temp path, str=无

    # 过滤 grid/warmup（__getitem__ 注入, _pack_args 不认识）
    bound_args = dict(self.nargs or {})
    ttir_kwargs = {k: v for k, v in kwargs.items() if k not in ("grid", "warmup")}
    current_kwargs = dict(config.all_kwargs(), **ttir_kwargs)
    bound_args.update(current_kwargs)

    ttir_text = generate_ttir_for_costmodel(self.fn, current_kwargs, bound_args)

    # 可选：写 temp file 供硬件阶段 ir_override 复用
    if self.costmodel_reuse_ttir:
        tmp = NamedTemporaryFile(suffix=".ttir", delete=False)
        tmp.write(ttir_text); tmp.close()
        self._costmodel_ttir_cache[cache_key] = (ttir_text, tmp.name)
    else:
        self._costmodel_ttir_cache[cache_key] = ttir_text
    return ttir_text
```

---

## 倒推第 3 层：开关与配置

```python
# __init__ — 双入口: 环境变量或 prune_configs_by
costmodel_cfg = (prune_configs_by or {}).get("costmodel", {})
self.enable_costmodel_prune = (
    os.getenv("TRITON_ENABLE_COSTMODEL_PRUNE", "0") == "1"
    or bool(costmodel_cfg)
)
self.costmodel_top_k = int(os.getenv("TRITON_COSTMODEL_TOP_K", costmodel_cfg.get("top_k", 10)))
self.costmodel_reuse_ttir = (os.getenv("TRITON_COSTMODEL_REUSE_TTIR", "1") == "1")
self.costmodel_hardware_config = costmodel_cfg.get("hardware_config") or ""
self.costmodel_verbose = (
    self.print_autotuning
    or os.getenv("TRITON_COSTMODEL_VERBOSE", "0") == "1"
)
```

### `_build_costmodel_arg_bindings()` — `autotuner.py`

```python
def _build_costmodel_arg_bindings(self, **kwargs):
    parts = []
    # 1. 标量参数 → argN=value
    for i, name in enumerate(self.arg_names):
        val = self.nargs.get(name)
        if isinstance(val, int):
            parts.append(f"arg{i}={val}")
    # 2. 评估 grid(lambda) → num_programs（wave 乘数）
    grid = kwargs.get("grid")
    if grid is not None and callable(grid) and self.nargs and self.configs:
        meta = {**self.nargs, **self.configs[0].kwargs}
        dims = grid(meta)
        if dims:
            np_val = 1
            for d in dims: np_val *= d
            parts.append(f"num_programs={np_val}")
    return ",".join(parts)
```

传给 C++ 的格式：`arg3=98432,num_programs=97`。`num_programs` 是自定义 key，C++ 侧通过 `parsed->get<int64_t>("num_programs")` 读取。

---

## 倒推第 4 层：TTIR 复用（ir_override）—— 两层缓存模型

costmodel 和硬件 benchmark 共享同一份 TTIR 编译产物，通过 `ir_override` 机制避免重复 `ast_to_ttir()` + `make_ttir()`。

### 第一层缓存：`_costmodel_ttir_cache`（我们自己维护的）

key 为 `sha256(fn.cache_key + sorted(config.kwargs))`，value 为 `(ttir_text, temp_path)`。
**不含** `ir_override`，因为它是用来查 TTIR 文件的，不需要。

### 第二层缓存：JITFunction 的编译缓存（Triton 内置）

JIT 的 `compute_cache_key()` 把 **所有 kwargs**（包括 `ir_override`）纳入 key：

```python
# jit.py: compute_cache_key
cache_key = str(specialization) + str(options)  # options 含 ir_override
```

### ir_override 必须在两处注入

| 注入点 | 代码位置 | 作用 |
|--------|---------|------|
| `_make_kernel_call()` | autotuner.py L2495 | HW bench 时复用 costmodel 的 TTIR，跳过 `ast_to_ttir` |
| `run()` | autotuner.py L2304 | **最终执行时复用** HW bench 编译好的 binary |

两处不注入的后果：

```
_batch_bench (HW bench):              run() (最终执行):
─────────────────────────             ─────────────────────
current["ir_override"] = "/tmp/..."   final_kwargs 里没有 ir_override
        │                                     │
        ▼                                     ▼
JIT key = hash({...,                      JIT key = hash({...,
  ir_override:"/tmp/XXX"})                   ir_override:None})   ← 不同!
        │                                     │
        ▼                                     ▼
   存入 JIT 缓存                           查 JIT 缓存 → MISS → 重编译
```

**启用 `reuse_ttir` 后，两处注入保证最终执行直接命中 `_batch_bench` 编译好的 binary，零冗余编译。**

由 `TRITON_COSTMODEL_REUSE_TTIR` 控制（默认 `0` 关闭，需要显式开启）：

- **启用时**：costmodel 阶段将 TTIR 写入 `/tmp`（tmpfs），`_make_kernel_call` 和 `run()` 都注入 `ir_override`。TTIR 编译仅在 costmodel 阶段发生一次，HW bench 和最终执行都是复用。
- **关闭时**：缓存值仅存 `ttir_text`（不含 temp path），各阶段独立编译。用于对比端到端延迟差异。

### 代码索引

```
costmodel 阶段: _costmodel_compile_ttir()          ← 写 TTIR 文件，存 (ttir_text, temp_path)
HW bench 阶段:  _make_kernel_call()                ← current["ir_override"] = cached[1]
最终执行阶段:   run()                               ← final_kwargs["ir_override"] = cached[1]
```

---

## 倒推第 5 层：pid_x 与 num_programsx — C++ 循环分析

costmodel 的 C++ PipelineAnalysisPass 需要对 kernel 中的 stride-loop（`for row in range(pid, n_tasks, num_programs)`）做静态 trip count 估算。这些循环依赖运行时值：

- **Lower bound**: `tl.program_id(0)` → C++ 需要 `pid_x` 绑定
- **Step**: `tl.num_programs(0)` → C++ 需要 `num_programsx` 绑定
- **Upper bound**: 从 kernel 参数推导（如 `n_rows`） → C++ 需要 `argN` 绑定

Python 端的 `_build_costmodel_arg_bindings()` 生成逗号分隔的绑定字符串：

```
arg2=4096,arg3=4096,arg4=4096,num_programsx=48,pid_x=0
```

C++ 端解析链路（`Utils.h`）：

```
parseBindings("pid_x=0,...")
  → 归一化: "x"=0, "pid_x"=0, "program_id_x"=0  （三个别名）
  → exportProgramIdBindings → programIdBindings["x"]=0

parseBindings("num_programsx=48,...")
  → 归一化: "num_programsx"=48, "num_programs_x"=48  （两个别名）
  → exportProgramIdBindings → programIdBindings["num_programsx"]=48

evaluateValue(lower) → find("x") → 0
evaluateValue(step)  → find("num_programsx") → 48
evaluateValue(upper) → find("arg3") → 4096

tripCount = ceil((4096 - 0) / 48) = 86
```

同时 `num_programs` 也用于 wave 计算：

```cpp
pythonNumPrograms = parsed->get<int64_t>("num_programsx");  // 优先 num_programsx
// fallback: parsed->get<int64_t>("num_programs")           // 向后兼容
numWaves = ceil(numPrograms / numParallelUnits);
```

**注意**：关键词用 `num_programsx`（无下划线），因为 C++ `evaluateValue` 里查找的是 `"num_programs" + dim` = `"num_programsx"`，不是 `"num_programs_x"`。不过 `parseBindings` 也接受 `num_programsx` 格式并写入两个别名，所以两个格式都能命中。

### 数据流总览

```
Python: grid tuple (48,) → "num_programsx=48,pid_x=0"
C++:    parseBindings → programIdBindings["x"]=0, ["num_programsx"]=48
        getKernelCycles(48, ...) → perProgramCycles × numWaves

---

## 调试踩坑

### 坑 1：autotune 补丁时序

`_patch_autotune()` 在 `triton.backends.ascend.runtime` 首次 import 时执行，但 `@triton.autotune` 装饰器在模块加载时就求值——可能补丁还没打。**修复**：装饰器之前显式 `import triton.backends.ascend.runtime`。

### 坑 2：import 路径

symlink `python/triton/backends/ascend/` → `third_party/ascend/backend/`，正确路径是 `triton.backends.ascend.compiler`，不是 `triton.backends.ascend.backend.compiler`。

### 坑 3：`grid`/`warmup` 泄露

`__getitem__` 注入的 `grid`/`warmup` 透传到 `_pack_args`，后者校验所有 key 报错。**修复**：在 `_costmodel_compile_ttir` 中过滤。

### 坑 4：arg-bindings 格式

C++ `parseBindings` 期望逗号分隔的 `argN=value`，最初用了分号分隔的 `name=value` 格式，解析失败后静默忽略导致预测全 0.1 us。

### 坑 5：`config.kwargs` vs `config.all_kwargs()` 做缓存 key

`all_kwargs()` 含 `num_warps`/`num_stages` 等后端参数，不同 `num_stages` 的同 BLOCK_SIZE config 会有**不同** cache key，缓存不能复用。**修复**：改用 `config.kwargs()`——只含 `tl.constexpr` 值和用户显式传入的编译选项。

### 坑 6：NPUOptions frozen dataclass 和 KernelMetadata 缺字段

这两个坑出现在尝试走 `triton.compile()` 路径时（设置 `compile_ttir_only` 标志、构建 `CompiledKernel`）。最终方案：放弃 `triton.compile()` 路径，直接调用 `ast_to_ttir` + `make_ttir`，不产生 `CompiledKernel`，也不需要修改 `NPUOptions`。

---

## 最终改动清单

| 文件 | 改动 |
|------|------|
| `compiler.py` | 新增 `generate_ttir_for_costmodel()` — 直接调 `ast_to_ttir` + `make_ttir` |
| `costmodel_runtime.py` | 新增 `costmodel_prune()` — 惰性 TTIR + 批量评估；并行化 TTIR 编译和 costmodel 评估 |
| `autotuner.py` | `__init__` 新增开关配置；`_costmodel_compile_ttir()`(TTIR 编译+缓存+可选的 ir_override)；`_prune_by_costmodel()`(批量剪枝)；`_build_costmodel_arg_bindings()`(参数绑定+num_programs)；`run()` 插入点；`TRITON_COSTMODEL_VERBOSE` 日志 |
| `PipelineAnalysisPass.cpp` | 调用 `getKernelCycles()`；从 bindings 读取 `num_programs`；设置 `ascend.scheduled_cycles` |
| `EstimateCycles.cpp` | 同上 |
| `bench/` | `example_costmodel_autotune.py`(端到端示例)、`bench_costmodel.py`(A/B 对比) |

---

## 运行方式

```bash
# 开发调试
TRITON_COSTMODEL_VERBOSE=1 TRITON_ENABLE_COSTMODEL_PRUNE=1 \
TRITON_COSTMODEL_TOP_K=5 python bench/example_costmodel_autotune.py

# A/B 对比 (不复用 TTIR 可加 TRITON_COSTMODEL_REUSE_TTIR=0)
TRITON_ENABLE_COSTMODEL_PRUNE=1 TRITON_COSTMODEL_TOP_K=5 \
    python bench/bench_costmodel.py

# 指定硬件配置
@triton.autotune(..., prune_configs_by={
    "costmodel": {"top_k": 5, "hardware_config": "/path/to/config.json"}
})
```
