# SIMT Auto-Scope Costmodel 交接说明

## 给下一个模型的任务背景

我们在 triton-ascend 仓库上优化 Ascend A5 的 SIMT auto-scope costmodel。
目标是：给定一个 Triton kernel，自动判断它应该走 all-SIMD、all-SIMT-only，
还是 mixed SIMD/SIMT，并且判断要准。

当前分支：`kx_simt_costmodel`

## 先读这些文档

按顺序读：

1. `docs/simt-costmodel-dataset-plan.md`
   - 完整优化计划；
   - 第 0 节：路线图；
   - 第 8 节：有效优化存档；
   - 第 8.5 节：每个优化的设计考量和备选方案；
   - 第 9 节：剩余优化点。

2. `docs/simt-costmodel-cpp-refactor-plan.md`
   - 当前 v6 C++ 公式基线；
   - C++ 结构重构计划。

3. `bench/simt_autoscope/README.md`
   - benchmark 和残差分析脚本用法。

## 再读这些代码

- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimdSimtCostModel.cpp`
  核心评分公式。
- `third_party/ascend/costmodel/include/AscendModel/RouteModel/SimdSimtCostModel.h`
  特征和报告结构。
- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimtAnchorAnalysis.cpp`
  SIMT anchor 识别。
- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/Transforms/SelectSimdSimtCostModel.cpp`
  selector pass 和 materialization。
- `third_party/ascend/backend/compiler.py`
  Python 集成，包括 `_run_cpp_simd_simt_costmodel`。
- `python/triton/runtime/jit.py`
  launch grid 传入 compile 的位置。
- `third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json`
  当前 profile，包含所有 rate 参数。
- `third_party/ascend/costmodel/profiles/microbench/data_provider/`
  cce probe 和 host。

## 数据在哪

`ascend_results/` 目录下：
- `simt_autoscope_bench_v6.jsonl`（当前 5-case 验证数据）
- 更早版本 v1-v5
- `simd_memory_microbench_v2.jsonl`
- `simd_components_microbench_v2.jsonl`
- `simt_components_microbench_v2.jsonl`
- `simt_predicate_host.log`
- `simt_gm_memory_pattern_host.log`
- `scan_simd.jsonl` / `scan_simt.jsonl`

## 本来是怎样的

原 costmodel 的问题：
- 固定单点 rate，例如 SIMD memory 202.25 B/cycle、SIMT GM 0.176/0.129、
  SIMT predicate 0.038；
- 所有惩罚通过 `(1 + Pstruct)` 整体乘在 SIMD 分数上；
- domain multiplier 掩盖公式误差；
- SIMT dot 按 scalar FMA 141 flops/cycle 计算；
- scan 没有专用成本，且被标 unsupported；
- TTIR 特征是 block-local 的，但没有 grid，无法换算 whole-program 工作量。

## 现在怎样了

已经完成：
- 搭好 benchmark / 微基准 / 残差分析工具；
- 惩罚按组件归位；
- SIMT dot 改成 cube；
- scan 专用模型；
- launch grid 传入 C++；
- `program_issue_scale=1.0`，切换实际 cycle 口径；
- domain multiplier 暂时全部 1.0；
- SIMD memory 使用 size-dependent contiguous rate + gather rate；
- SIMD memory 拆分 contiguous/gather load bytes；
- 5 个内置 case 的 raw ratio 基本对齐。

## 关键约定，不要破坏

1. `program_issue_scale` 现在必须是 1.0，所有组件都是实际 cycle 口径。
2. TTIR 特征是 per-CTA 的。whole-kernel 测的 rate（SIMD memory、dot）要乘 grid；
   CTA 级 cce 测的 rate（SIMT GM、shuffle、predicate、ALU、scan）不要乘 grid。
3. 旧的 domain multiplier 已置 1.0，目的是验证公式。改公式期间不要恢复旧值。
4. C++ 和 pybind 改完必须在 A5 上重新编译 native 扩展再跑 5-case。
5. 5-case 验证命令在 `bench/simt_autoscope/README.md` 里。

## 建议的下一步

按 `docs/simt-costmodel-dataset-plan.md` 第 9 节做：

1. 接入外部 25 个真实算子做泛化验证；
2. 细化 dot 模型（更多 shape、SIMT dot 数据）；
3. SIMT GM rate 按 num_warps 查表；
4. SIMD strided rate 接入 C++；
5. predicate / shuffle 指令数细化；
6. mixed transition 成本实测；
7. domain coverage / multiplier 重校；
8. auto 模式端到端验证；
9. 建立多 shape/grid 回归集。

## 注意事项

- TTIR 中拿不到 launch grid，grid 是从 Python JIT 传入的，不要试图从 TTIR 解析。
- 微基准脚本有些会因特定 shape 在 SIMT 路由下 NPU 507035 崩溃，脚本已容错并
  记录 error，属正常现象。
- 修改 profile 后要同步更新 `docs/simt-costmodel-dataset-plan.md` 的存档。
