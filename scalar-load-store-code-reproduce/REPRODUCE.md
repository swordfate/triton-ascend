# 复现实测结果：从 CCE 白盒到板级 Triton K-sweep

本文面向拿到这个文件夹、想在 Ascend950PR 服务器上复现 scalar load/store 实测结果的人。
当前服务器不可用；以下命令是历史成功流程整理，脚本本身来自最终工作分支。

## 0. 前置环境

### 0.1 服务器 / CANN

```bash
ssh <your-ascend-server>
# 仓库和脚本默认路径是 /home/c00946898，其他用户请按需修改脚本里的绝对路径
source ~/env_ascend.sh          # CANN set_env.sh + bishengir PATH + conda libs
ulimit -n 1048576
```

CCE 探针还需要：

- `ccec`（CANN tools/bisheng_compiler/bin 或 PATH 中）
- `g++` + CANN `include/` `lib64/`（wrapper 已用 `$ASCEND_TOOLKIT_HOME`）
- `msopprof`（`$ASCEND_HOME_PATH/bin/msopprof`）
- fork 内 NPUIR 模板头：默认
  `$HOME/AscendNPU-IR-triton/bishengir/lib/Template/include`；
  如果不在该路径，给 `build_*` 脚本设置 `INC=/path/to/Template/include`。

### 0.2 板级 Triton 实测

需要：

- Ascend950PR 物理卡（或至少真卡环境）
- conda env `wj_autoscope`，已安装 `torch`、`torch_npu`、`triton-ascend`
- `source ~/env_ascend.sh`
- `source /data/miniconda3/etc/profile.d/conda.sh && conda activate wj_autoscope`
- `export ASCEND_RT_VISIBLE_DEVICES=0`
- 其他用户的 msprof / 仿真任务会干扰 profiler，跑之前先确认卡和进程空闲。

---

## 1. CCE 白盒：load / store CAModel 探针

最终公式用到的 CCE 探针：

| 目录 | 用途 |
|---|---|
| `cce/load/scalar_o1/` | SIMD MainScalar / SIMT warp-uniform 单条 load |
| `cce/load/scalar_o4/` | same-line / diff-line 4-op load |
| `cce/store/scalar_o1/` | 单条 SIMT `SIMT_STG` / SIMD MainScalar store（CCE 对照） |
| `cce/store/scalar_o4/` | same-line / diff-line 4-op store（CCE 对照） |
| `cce/syscnt/board_marginal/` | 真卡 SYS_CNT marginal / clock 探针 |

以 `load/scalar_o1` 为例：

```bash
cd cce/load/scalar_o1

bash build_scalar_o1.sh
# 产出 load_scalar_o1.o + load_scalar_o1_host

bash run_scalar_o1_camodel.sh
# 对 simd_main_ld_o1 / simt_ld_uniform_o1 / simt_ld_uniform_o1_t1
# 各跑一次 msopprof，产出 camodel_simd.log / camodel_simt32.log / camodel_simt1.log
# 以及 OPPROF_* 目录
```

`msopprof` 的 OPPROF 默认结构是：

```text
OPPROF_<timestamp>_<rand>/<kernel_name>/0/dump/
```

每次 msopprof 只跑一个 kernel，因此 OPPROF 是分开的。把需要的 dump 收进 parser 期望的目录：

```bash
collect_one () {
  tag="$1"; kernel="$2"
  for d in $(ls -dt OPPROF_* 2>/dev/null); do
    if [ -d "$d/$kernel/0/dump" ]; then
      mkdir -p "camodel_results/$tag"
      cp -r "$d/$kernel/0/dump/." "camodel_results/$tag/"
      echo "collected $tag <- $d/$kernel"
      return 0
    fi
  done
  echo "OPPROF dump not found for $kernel" >&2
  return 1
}

collect_one simd  simd_main_ld_o1
collect_one simt32 simt_ld_uniform_o1
collect_one simt1  simt_ld_uniform_o1_t1

python3 parse_load_scalar_o1.py camodel_results/simd camodel_results/simt32 camodel_results/simt1
```

`load/scalar_o4`、`store/scalar_o1`、`store/scalar_o4` 同理，先按各自 `run_*_camodel.sh` 里的
`run_one <kernel> <tag>` 列表逐个 `collect_one <tag> <kernel>`，再调用对应 parser。

`load/scalar_o4`、`store/scalar_o1`、`store/scalar_o4` 的 build / run / parse 用法完全相同，
只是 kernel 名和目录名不同，见各目录里 `run_*_camodel.sh`。

### 1.1 真卡 SYS_CNT marginal

`cce/syscnt/` 里是 load/store 的板级 `get_sys_cnt()` 探针和 CAModel 对照：

```bash
cd cce/syscnt

bash build_syscnt.sh
# 编译 load/scalar_o1、load/scalar_o4、store/scalar_o1、store/scalar_o4 的 *_syscnt.o
# 以及 syscnt_host / syscnt_rep_host

export ASCEND_RT_VISIBLE_DEVICES=0
bash run_board_syscnt.sh       # 真卡跑 board marginal，结果在 syscnt/results/board/
bash run_camodel_syscnt.sh     # CAModel 对照，结果在 syscnt/results/camodel_ind_all/
python3 parse_syscnt_results.py --root "$PWD"
# 产出 syscnt/syscnt_compare.csv/.md
```

注意：`board_marginal/board_vs_model_agg.csv` 是历史多次 run 的聚合结果，直接用来对照；
重新跑 `run_board_syscnt.sh` 会生成单次原始 log，不会自动覆盖这个聚合文件。

---

## 2. 板级 Triton K-sweep（当前 store/load 系数来源）

代码在 `triton/measurement/`：

| 文件 | 用途 |
|---|---|
| `profiler_pair_probe.py` | 2026-09-27 多 ptr K=1/2/4/8 load/store，输出 `profiler_pair_results.json/csv` |
| `store_probe_variants.py` | 2026-09-28 store matched-sink baseline + readback / barrier 变体 |
| `summarize_board_k_sweep.py` | 用结果 JSON 计算 `measured_cycle / pred_cycle / err`，生成汇总 CSV/MD |
| `run_store_variants.sh` / `run_store_sync_recheck.sh` | 上述脚本的服务器包装脚本 |

### 2.1 load 四组：simd/simt × load

```bash
cd triton/measurement
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0

# summarize_board_k_sweep.py 读 results/simd_load.json / simt_load.json
run_pair () {
  mode="$1"; kind="$2"; flat="$3"
  python3 profiler_pair_probe.py --modes "$mode" --kinds "$kind" \
    --ks 1 2 4 8 --rounds 2 --reps 40 --warmup 5 --device 0 --out "results/${mode}_${kind}"
  cp "results/${mode}_${kind}/profiler_pair_results.json" "results/${flat}.json"
  cp "results/${mode}_${kind}/profiler_pair_results.csv"  "results/${flat}.csv"
}
run_pair simd      load simd_load
run_pair simt_only load simt_load
# 如需保留旧 store sweep 原始数据，可再跑：
# run_pair simd      store simd_store
# run_pair simt_only store simt_store
```

当前最终 store 模型来自 2.2，不依赖旧 store sweep；summarize 会跳过旧 store 行，只使用 2.2 的结果。

### 2.2 store variants：matched sink + readback / barrier

```bash
# 主结果：variant 0 = matched sink baseline，variant 1/2 = readback/barrier 边界证据
python3 store_probe_variants.py --modes simd simt_only \
  --ks 1 2 4 8 --variants 0 1 2 --rounds 3 --reps 80 --warmup 5 \
  --device 0 --out store_sync_results

# fresh cross-run 复核（summarize 用它们算跨 run 中位数；原 K=8 229 ns 异常 run 会被排除）
recheck () {
  mode="$1"; out="$2"
  python3 store_probe_variants.py --modes "$mode" --ks 1 2 4 8 --variants 0 \
    --rounds 6 --reps 80 --warmup 5 --device 0 --out "store_recheck_results/$out"
  cp "store_recheck_results/$out/store_variant_results.json" \
     "store_recheck_results/$out.json"
}
recheck simt_only simt_store_first_recheck1
recheck simt_only simt_store_first_recheck2
recheck simd      simd_store_recheck1
recheck simd      simd_store_recheck2

# 单独复核 SIMT K=8（原 run 的 229 ns 已判定为不可复现异常）
python3 store_probe_variants.py --modes simt_only --ks 8 --variants 0 \
  --rounds 6 --reps 80 --warmup 5 --device 0 \
  --out store_recheck_results/simt_store_matched_k8
cp store_recheck_results/simt_store_matched_k8/store_variant_results.json \
   store_recheck_results/simt_store_matched_k8.json

# 可选：readback/barrier 边界证据（不进入 store resource 公式）
python3 store_probe_variants.py --modes simd simt_only \
  --ks 1 2 4 8 --variants 1 2 --rounds 6 --reps 80 --warmup 5 \
  --device 0 --out store_recheck_results/sync_recheck_variant12
cp store_recheck_results/sync_recheck_variant12/store_variant_results.json \
   store_recheck_results/sync_recheck_variant12.json
```

指标约定（与 `docs/README-diff-k-sweep.md` 一致）：

- load：`Duration(us)_delta`
- SIMT store：matched-sink `Duration(us)_delta`
- SIMD store：`aiv_mte3_time(us)_delta`（MTE3 pipe active 资源占用）
- readback/barrier 变体只作为可见性/同步的边界证据，不并入 `ScalarStore` 公式

### 2.2.1 K=16 store 外推复核（可选）

`store_probe_variants.py` 默认只带 8 个独立 `p0..p7`，最大 K=8。复核 K=16 时把
`_arith_k` / `_load_k` / `_store_probe` 扩展到 `p0..p15`、增加 `if K > 8..15` 分支、
`ptrs = [... for _ in range(16)]`，然后：

```bash
python3 store_probe_variants_k16.py --modes simd simt_only --ks 1 2 4 8 16 \
  --variants 0 --rounds 3 --reps 80 --warmup 5 --device 0 --out results_k16
```

结果见 `triton/measurement/k16_results/k16_*.json` 与 `docs/README-diff-k-sweep.md` §3：
三组 run 中位数 SIMD `aiv_mte3_time` 0.122 us、SIMT matched `Duration` 0.157 us；
K=16 时 SIMT 确实比 SIMD 慢（crossing ≈ K=12–14），与公式方向一致；
两者仍是不同资源窗口，K=16 超出 K≤8 拟合范围，不重拟合 target-used 系数。

### 2.3 生成汇总表

`summarize_board_k_sweep.py` 默认读取同目录下的：

```text
results/{simd,simt_only}_{load,store}.json
store_sync_results/store_variant_results.json
store_recheck_results/*.json
```

跑：

```bash
python3 summarize_board_k_sweep.py
```

会生成：

```text
results/scalar_k_board_summary.csv
results/scalar_k_board_summary.md
```

预期关键结果（2026-09-27/28 版本）：

```text
SIMD load      K=1/2/4/8 误差最大 ≈ −9.9%
SIMT load      K=1/2/4/8 误差最大 ≈ −12.6%
SIMD store     K=1/2/4/8 误差最大 ≈ −6.2%（aiv_mte3_time 窗口）
SIMT store     K=1/2/4/8 误差最大 ≈ ±5.3%（matched-sink Duration）
```

## 3. 复现时容易踩的坑

1. **必须 `ulimit -n 1048576`**：msopprof 写 dump 时默认 fd 限制会直接 abort。
2. **CCE build 脚本默认路径是 `/home/c00946898`**：换用户后改 `INC`/`env_ascend.sh` 路径。
3. **板级 profiler 需要独占 NPU**：脚本用 `grid=(1,)`、单 program；其他任务会污染 `min`/median。
4. **store 旧 K=8 229 ns 是异常 run**：final 表用 fresh K=8 中位数，不要直接用原 run 的 K=8。
5. **readback/barrier 不是 store resource**：若未来 kernel 真的消费 store 结果，应单独建模 dependency/synchronization，而不是把 ~96 ns 直接加到 store 系数上。
6. **本目录不含 OPPROF / kernel_details_csv 大产物**：执行时自行生成；原始数据在服务器对应目录。
