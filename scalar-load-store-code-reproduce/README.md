# scalar load/store 白盒 + 板级实测代码汇总

本目录把之前 scalar load/store costmodel 白盒工作最终使用的 CCE 代码、以及当前板级实测用的 Triton/Python 代码放在一起，方便本地查看和后续复用。

> 复现步骤见 [`REPRODUCE.md`](REPRODUCE.md)。

## 目录

```text
13-scalar-load-store-code/
├── cce/                                      # 白盒 CAModel / 真卡 SYS_CNT 的 CCE 探针
│   ├── load/scalar_o1/                       # 单条 SIMD MainScalar / SIMT warp-uniform load
│   ├── load/scalar_o4/                       # same-line / diff-line 4-op load
│   ├── store/scalar_o1/                      # 单条 SIMT SIMT_STG (CCE 对照)
│   ├── store/scalar_o4/                      # same-line / diff-line 4-op store (CCE 对照)
│   ├── syscnt/                               # 真卡 SYS_CNT marginal / CAModel 对照 + board_marginal
│   ├── compute/scalar_add/                   # syscnt runner 依赖的 compute 对照
│   ├── load_vec_compute/scalar_o1_o4/        # syscnt runner 依赖的 load→vec compute 对照
│   ├── camodel_o8_uniform_load_20260921/     # 后续 O8 uniform load CAModel 探针
│   ├── camodel_simd_ld_diff_o8_nc_20260921/  # SIMD diff O8 非连续地址变体
│   ├── camodel_simd_ld_diff_o8_sched_20260921/
│   └── camodel_simt_ld_uniform_diff_o16_o32_20260921/
├── triton/
│   ├── measurement/                          # 当前板级 K-sweep / store variant 实测代码
│   │   ├── profiler_pair_probe.py            # 2026-09-27 multi-ptr K=1/2/4/8 load/store
│   │   ├── store_probe_variants.py           # 2026-09-28 matched-sink / readback / barrier 变体
│   │   ├── summarize_board_k_sweep.py        # 汇总 pred_cycle / measured / err
│   │   ├── run_store_variants.sh
│   │   └── run_store_sync_recheck.sh
│   └── whitebox_store/                       # Triton scalar store MTE3 路径白盒 demo
│       ├── triton_scalar_store_demo.py
│       └── run_triton_scalar_store_camodel.sh
└── docs/
    ├── README-targeted-v1.md                 # 目标 6 kernel 的最终白盒建模/公式/验证说明
    ├── README-diff-k-sweep.md                # diff-line load/store K-sweep 公式与实测
    └── pr-2305-des.md                        # PR #2305 描述
```

## 来源

- CCE 探针与 `triton_scalar_store` demo：来自本地 git 分支 `origin/scalar-load-whitebox` @ `6704e35fa` 的
  `third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst_whitebox/`。
- 当前板级实测 Triton 代码：来自本地 `63-workspace/12-scalar-load-store-pr/triton_probe/`。
- 本目录只放代码和文档，不包含 CAModel dump / OPPROF / kernel_details_csv 等大产物；
  这些数据仍在原工作区或服务器路径中。

## 快速使用

CCE 探针（需要服务器 CANN 环境，当前服务器不可用）：

```bash
cd cce/load/scalar_o1
bash build_scalar_o1.sh
bash run_scalar_o1_camodel.sh
python3 parse_load_scalar_o1.py camodel_results/simd camodel_results/simt32 camodel_results/simt1
```

板级实测（需要 Ascend NPU + torch_npu）：

```bash
cd triton/measurement
python3 profiler_pair_probe.py --modes simd simt_only --kinds load store --ks 1 2 4 8
python3 store_probe_variants.py --modes simd simt_only --ks 1 2 4 8 --variants 0 1 2
python3 summarize_board_k_sweep.py
```

## 公式口径（当前结论）

- SIMD load：`T = prep + fill + max(0,K-outstanding)*extra + (K-1)*issue`
- SIMT load：`T = prep + fill + max(0,K-threshold)*extra + (K-1)*issue`（issue=1）
- SIMD store：用 `aiv_mte3_time` 的资源占用口径（K=1 约 6.5 ns），**不把 readback/barrier 的可见性成本并入 store resource**；readback/barrier 变体只作为同步/可见性边界证据。
- SIMT store：matched-sink `Duration` 口径，`T = base + (K-1)*subsequent`（subsequent=12.348 raw cycle）。
- K=16 store 外推复核见 `docs/README-diff-k-sweep.md` §3：SIMT 确实比 SIMD 慢，方向与公式一致；K=16 不参与 target 系数重拟合。

更完整的公式、系数和误差见 `docs/README-targeted-v1.md`、`docs/README-diff-k-sweep.md`、`docs/pr-2305-des.md`。
