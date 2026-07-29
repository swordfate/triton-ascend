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

## 倒推第 4 层：TTIR 复用（ir_override）

costmodel 和硬件 benchmark 会重复 `ast_to_ttir()` + `make_ttir()`。通过 `Config.ir_override` 机制复用。

由 `TRITON_COSTMODEL_REUSE_TTIR` 控制（默认 `1` 启用）：

- **启用时**：costmodel 阶段将 TTIR 写入 `/tmp`（tmpfs），缓存值存 `(ttir_text, temp_path)`。硬件阶段 `_make_kernel_call` 注入 `current["ir_override"] = temp_path`，`triton.compile()` stages 循环在 `.ttir` 匹配时 `parse()` 直接读文件，跳过 `make_ttir`。
- **关闭时**（`TRITON_COSTMODEL_REUSE_TTIR=0`）：缓存值仅存 `ttir_text`，硬件阶段正常重编译。用于对比端到端延迟差异。

---

## 下游修复：C++ pipeline 的 kernel 级 cycle 估算

### 问题一：`getKernelCycles` 未被调用，`ascend.scheduled_cycles` 未设置

`PipelineAnalysisPass` 只设了 `scheduled_cycles_one_iter`。`extractEstimatedTimeUs` 找不到 `ascend.scheduled_cycles`，fallback 到逐 op 求和——无 wave 乘数、无 scalar overhead。

**修复**：调用 `scheduler.getKernelCycles()`，写入 `ascend.scheduled_cycles`。

### 问题二：`numPrograms` 无法从 IR 推导

costmodel 管线不含 auto-blockify（wave loop 在 `bishengir-compile` 创建），IR 里无 wave loop。从循环 trip count 推导的方案永远返回 1。

**方案**：Python 层评估 `grid(lambda)` → `num_programs` → C++ `parseBindings` 保留自定义 key → `parsed->get<int64_t>("num_programs")` 读取。

```cpp
// PipelineAnalysisPass.cpp
auto parsed = parseBindings(argBindingsStr);       // 保留所有 key(不只是 argN)
parsed->exportArgBindings(argBindings);
auto pythonNumPrograms = parsed->get<int64_t>("num_programs");  // 自定义 key
int64_t numPrograms = pythonNumPrograms.value_or(1);
int64_t scheduledCycles =
    scheduler.getKernelCycles(numPrograms, numParallelUnits=40, numInnerIters=0);
```

`getKernelCycles` 内部：
```
barrierCycles = numInnerIters × pipeBarrierCycles
perProgram = (oneIterCycles + barrier) × (1 + scalarFactor)    // 1+3.74
numWaves   = ceil(numPrograms / numParallelUnits)               // ceil(97/40)=3
return perProgram × numWaves
```

### 数据流总览

```
Python: grid(meta) → (97,) → "arg3=98432,num_programs=97"
C++:    parseBindings → argBindings[3]=98432, numPrograms=97
        getKernelCycles(97, 40, 0) = oneIterCycles × 4.74 × 3
```

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
