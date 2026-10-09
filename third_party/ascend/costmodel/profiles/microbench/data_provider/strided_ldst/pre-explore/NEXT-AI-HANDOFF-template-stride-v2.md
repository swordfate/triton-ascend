# NEXT AI HANDOFF — SIMT local-scope strided template v2 in-sample calibration

> 生成时间：2026-10-09 17:46 CST
> 当前数据**尚未补齐**。这份文件是给下一个 AI 的接力提示词。

## 0. 任务目标

对 triton-ascend 的 **SIMT local scope strided load/store 模板路径**
（`triton_stride_load` / `triton_stride_store`）做 **in-sample 标定**，
产出 v2 半白盒公式和数据集。要求：

- 只做 in-sample；不做 W control sweep；不做 LOBO/LOSO/holdout。
- 基线：`origin/feature/strided-load-store-costmodel`，commit `64e2782ae`。
- 交付目录：
  `third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/`
- 产出：
  `results/model_template_stride_v2/`
  - `model_template_stride_load_v2.json`
  - `model_template_stride_store_v2.json`
  - `dataset.csv`
  - `errors_load.csv` / `errors_store.csv` / `errors_all.csv`
  - `metrics_summary.csv` / `metrics_summary.json`
  - `plots/parity_loglog.png` / `error_hist.png` / `error_vs_stride.png` / `error_vs_block.png`
  - `raw/*.json` + `ir_evidence/template_stride_path_check_v2.json`
  - `ir_evidence/measure_asm/load_b*_s*_w32.ttadapter`
  - `ir_evidence/measure_asm/store_b*_s*_w32.ttadapter`
- 拟合要求：`T = intercept + Σ c_i*f_i`，`c_i>=0`，NNLS/Lawson-Hanson；
  无 log；alpha 至少尝试 1.0/1.5/2.0；只报 in-sample 指标
  （n、MAPE、p50、p90、p95、max、bias、RMSE）。
- 最后更新 README §5，并说明：
  - 是否替换 profile 中 `template_strided_memory` 当前 5+5 个字段；
  - 如果字段变化，给出 commit message 和需要重跑的 costmodel UT。

## 1. 本地代码位置与分支

本地 Mac：

```text
/Users/weijianchen/Documents/2026/triton-ascend-worktrees/strided-template-calibration-v2
```

分支：

```text
feature/strided-template-calibration-v2
base = origin/feature/strided-load-store-costmodel @ 64e2782ae
```

本文件所在目录：

```text
third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/
```

本轮已修改/新增（尚未提交时，代码就在上述 worktree）：

- 修改 `scripts/template_stride_path_check.py`
  - 默认矩阵扩到 BLOCK `4..2048`、模板命中 stride `3..31`、bail stride `1,2,4,...,256`；
  - 默认 `num_warps=32`；
  - 输出路径默认 v2 `ir_evidence/template_stride_path_check_v2.json`。
- 修改 `scripts/template_stride_load_probe.py` / `scripts/template_stride_store_probe.py`
  - 新增 `--min-freq-mhz`（默认 1500）；
  - 每次 attempt 后读 `npu-smi info -t common -i 0` 的 `Aicore curFreq(MHZ)`，
    写入 `post_freq_mhz` / `frequency_ok`；
  - `valid=True` 现在要求 `witness_ok && stable_ok && frequency_ok`；
  - payload 增加 `launch_manifest`。
- 修改 `scripts/run_template_stride_measure_load.sh` / `_store.sh`
  - 默认 `NUM_WARPS=32`、完整 BLOCK/STRIDE v2 矩阵；
  - 默认输出到 `results/model_template_stride_v2/raw/`；
  - 默认 asm 目录 `results/model_template_stride_v2/ir_evidence/measure_asm/`。
- 修改 `scripts/build_template_stride_dataset.py`
  - 默认读取 v2 的 pass1/pass2 raw JSON，输出 v2 `dataset.csv`。
- 新增 `scripts/run_template_stride_calibration_v2.sh`
  - `PASS=path|load|store|all` 的串联驱动。
- 新增 `scripts/fit_template_stride_v2.py`
  - 前向选择 + 0.98/0.95 绝对相关 guard + NNLS；
  - alpha `1.0/1.5/2.0`；
  - 输出 v2 四个误差文件 + metrics_summary；
  - 已在 v1 dataset 上冒烟通过（load MAPE≈8.35%，store≈9.03%）。
- 新增 `scripts/predict_template_stride_v2.py`
  - 支持 `--verify`，逐点复算 `errors_{load,store}.csv`。
- 尚未验证：predictor v2 的 `--verify` 要在真正 v2 数据拟合后跑。

## 2. 服务器环境与代码位置

服务器：

```bash
ssh ascend-950pr-63
```

服务器工作目录：

```text
/home/c00946898/strided-template-calibration-v2/
  scripts/          # 已 rsync 本轮脚本
  results/model_template_stride_v2/
    raw/            # 板卡 raw JSON + log
    ir_evidence/    # path check + measure_asm + logs
  triton_cache_template_v2_measure/
```

环境必须这样进（**不要** `export PYTHONPATH=$HOME/sb64_latest_env`）：

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -n 1048576
```

原因：必须用可 launch 的 `wj_autoscope` 安装包，且保留 CANN 导出的
`PYTHONPATH`；否则 acl / `is_compile_on_910_95` 检测会失败。实际运行统一用
`scripts/board_env.sh`。

模板路径触发必须同时传：

```python
compile_mode = "simd_simt_template"
parallel_mode = "mix_simd_simt"
compile_on_910_95 = True
auto_simt_scope_mode = "off"
enable_auto_blockify = False
superblock_factor = 1
```

否则 TTAdapter 会回退 `memref.copy`。测量前 `assert_template_call()` 会失败并停止。

## 3. 当前实际进度（截至 2026-10-09 17:46 CST）

服务器上有两个 tmux 在跑：

```text
stride_v2_path   # compile-only path check
stride_v2_load   # load pass1，跑完后 driver 会自动接着跑 load pass2
```

当前日志：

```text
~/strided-template-calibration-v2/v2_path_outer.log
~/strided-template-calibration-v2/v2_load_outer.log

~/strided-template-calibration-v2/results/model_template_stride_v2/ir_evidence/path_check_v2.log
~/strided-template-calibration-v2/results/model_template_stride_v2/raw/load_pass1.log
```

进度：

- path check：约 460 / 700 个 case；`template_stride_path_check_v2.json`
  只在全部结束后写盘，所以现在 missing 是正常的。
- load pass1：约 119 / 370 个 case；invalid 约 8 个（多为 spread>0.40）。
- load pass2：尚未开始。
- store pass1 / pass2：尚未开始。
- 没有开始拟合 v2（用户要求本阶段先不拟合）。

已发现的现象：

- 环境正常；Aicore 频率 1650 MHz；`--min-freq-mhz 1500` 正常通过。
- W=32 下 path check 的模板命中判定正确：
  - stride 非 2 次幂且 ≥3：TTAdapter 含 `triton_stride_load/store`；
  - stride 1/2/pow2：不含模板调用。
- 共享板卡噪声明显；少数 case 的 spread > 0.40 被标 invalid。
  例如 b64 s255 W32 的 2000 vs 8000 iteration 斜率发散（~0.5 spread）。
  继续按协议：多 pass，同 key 取所有 valid+correct 的最小 target。

## 4. 运行矩阵与协议

主矩阵（load/store 分开跑）：

```text
num_warps = 32
BLOCK  = 4 8 16 32 64 128 256 512 1024 2048
STRIDE = 3 5 6 7 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
         40 48 63 65 80 96 127 129 160 192 255
```

测量协议（probe 已实现）：

- rotate-loop：每次 iteration 一个 shaped strided load/store；
- `iters=2000,8000`；每点重复 reps 次，取 **min Event**；
- target = 对 `(iters, min_event_ns)` 线性斜率，单位 ns/iteration；
- 每次 measured launch 前跑 ALU busy burst（`busy_iters=2,000,000`）；
- attempt 后 ALU witness + `npu-smi` Aicore 频率；
- `witness_ok && stable_ok && frequency_ok` 才算 valid；
- load 正确性对 torch reference；store 对 reference buffer；
- pass1：`REPS=2 MAX_ATTEMPTS=2`；pass2：`REPS=3 MAX_ATTEMPTS=1`；
- dataset builder 对同一 `(path, block, stride, num_warps)` 取所有
  valid+correct pass 的最小 target。

## 5. 后续步骤（下一个 AI 直接照做）

### Step 1 — 让 path check 跑完并验收

```bash
ssh ascend-950pr-63
tmux ls
tail -f ~/strided-template-calibration-v2/results/model_template_stride_v2/ir_evidence/path_check_v2.log
# 直到看到: wrote ...; mismatches=0
ls -l ~/strided-template-calibration-v2/results/model_template_stride_v2/ir_evidence/template_stride_path_check_v2.json
```

验收条件：

- `template_stride_path_check_v2.json` 存在；
- 所有 `match == true`；`mismatches=0`；
- 命中 stride 包含 `call @triton_stride_load/store`，bail stride 不含。

### Step 2 — 让 load pass1/pass2 跑完

当前 `stride_v2_load` 驱动会先跑 pass1 再跑 pass2。

```bash
tail -f ~/strided-template-calibration-v2/v2_load_outer.log
ls -l ~/strided-template-calibration-v2/results/model_template_stride_v2/raw/board_template_stride_load_pass*.json
```

预期文件：

```text
raw/board_template_stride_load_pass1.json
raw/board_template_stride_load_pass2.json
raw/load_pass1.log
raw/load_pass2.log
```

### Step 3 — 跑 store pass1/pass2

等 load 驱动结束后再启动 store，避免同一 NPU 同时测量。

```bash
cd ~/strided-template-calibration-v2
tmux new-session -d -s stride_v2_store -c "$HOME/strided-template-calibration-v2" \
  "TRITON_CACHE_DIR=$HOME/strided-template-calibration-v2/triton_cache_template_v2_measure \
   PASS=store bash scripts/run_template_stride_calibration_v2.sh > v2_store_outer.log 2>&1"
tmux ls
tail -f ~/strided-template-calibration-v2/v2_store_outer.log
```

预期文件：

```text
raw/board_template_stride_store_pass1.json
raw/board_template_stride_store_pass2.json
raw/store_pass1.log
raw/store_pass2.log
```

### Step 4 — build v2 dataset

`build_template_stride_dataset.py` 默认已指向 v2 raw/pass1/pass2。

```bash
cd ~/strided-template-calibration-v2
python3 scripts/build_template_stride_dataset.py
# 输出 results/model_template_stride_v2/dataset.csv
```

检查：

- rows = 最少 2 × 370 = 740（每个 path 370 个 key；
  无效 key 会在 builder 中标记 valid=0，不应直接丢掉后再凑数，按脚本默认逻辑处理）；
- 打印各 path valid rows、invalid 数、分布。

### Step 5 — 本阶段之后如果用户恢复“拟合”指令，再执行

> 用户当前说“先暂时不拟合”。以下保留为完成数据后的下一步命令。

```bash
cd ~/strided-template-calibration-v2
python3 scripts/fit_template_stride_v2.py \
  --dataset results/model_template_stride_v2/dataset.csv \
  --out-dir results/model_template_stride_v2 \
  --alphas 1.0 1.5 2.0 \
  --max-terms 6 \
  --corr-threshold 0.95

python3 scripts/make_template_stride_plots.py \
  --dir results/model_template_stride_v2

python3 scripts/predict_template_stride_v2.py --verify
```

拟合通过标准：

- MAPE < 10%~15%；
- 若 max 较大，列出最差点（block/stride/target/pred/error）并判断是
  模板真实行为还是板卡噪声；
- 不报告任何 LOBO/LOSO/holdout/CV。

### Step 6 — 本地 copy 数据并 push

用户明确要求：数据补齐后本地 copy 一份，然后远程 push 到仓库
`feature/strided-load-store-costmodel` 中。

推荐做法（不要直接在服务器 dirty worktree push）：

1) 服务器打包：

```bash
cd ~/strided-template-calibration-v2
tar -czf /tmp/template_stride_v2_results.tgz \
  results/model_template_stride_v2 \
  v2_path_outer.log v2_load_outer.log v2_store_outer.log
scp ascend-950pr-63:/tmp/template_stride_v2_results.tgz /tmp/
```

2) 本地解包到 worktree：

```bash
cd /Users/weijianchen/Documents/2026/triton-ascend-worktrees/strided-template-calibration-v2
tar -xzf /tmp/template_stride_v2_results.tgz
```

3) 本文件与脚本先提交到 `feature/strided-template-calibration-v2`：

```bash
git add third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore
git commit -m "feat(costmodel): calibrate SIMT strided template load/store v2"
git push origin feature/strided-template-calibration-v2
```

4) 如果用户要求最终必须落在 `origin/feature/strided-load-store-costmodel`：
   - 在该分支上 cherry-pick 上述 v2 数据 commit（或直接合并 v2 分支）；
   - 不要覆盖该分支上已有的 v4 纯 SIMT 数据；
   - `git push origin feature/strided-load-store-costmodel`。

## 6. 参考命令（监控）

```bash
# 当前所有相关进程
pgrep -af "template_stride|run_template_stride_calibration_v2"

# path check 当前条数
grep -c "expected=" \
  ~/strided-template-calibration-v2/results/model_template_stride_v2/ir_evidence/path_check_v2.log

# load pass1 当前条数 / valid / invalid
f=~/strided-template-calibration-v2/results/model_template_stride_v2/raw/load_pass1.log
grep -c "template load b" "$f"
grep -c "valid=True" "$f"
grep -c "valid=False" "$f"

# 查看 invalid case 的原因
python3 - <<'PY'
import json
p="/home/c00946898/strided-template-calibration-v2/results/model_template_stride_v2/raw/board_template_stride_load_pass1.json"
# 注意：pass1 跑完才有文件
d=json.load(open(p))
for c in d["cases"]:
    if not c["valid"]:
        print(c["block"], c["stride"], c["target_ns"], c["correctness_ok"])
        for a in c["attempts"]:
            print("  ", a["witness_ok"], a.get("post_freq_mhz"),
                  a["frequency_ok"], round(a["spread"], 3), a["stable_ok"])
PY
```

## 7. 已知坑与注意事项

1. `path_check_v2.json` 只在进程结束时写盘；不要看到 missing 就以为失败。
2. `board_env.sh` 不能替换成 `board_env_latest.sh`，也不能设置
   `PYTHONPATH=$HOME/sb64_latest_env`；用户明确要求保留 CANN PYTHONPATH。
3. 共享板卡很忙（load average 50+），Event 波动大；
   `valid` 只认 witness/frequency/spread 三项都过的 attempt。
4. `b64 s255` 这类 case 可能在低 iters 点斜率发散被标 invalid，
   不要手工改 max_spread 去“保数据”；可以让 pass2/同 key 其它 valid pass 覆盖。
5. 不要跑 W sweep；`num_warps` 固定 32，模板内部固定 1024 threads。
6. 不要测 stride 1/2/pow2；它们只用于 path check bail。
7. 不要把 `results/model_template_stride_v1` 覆盖掉；v2 用新目录。
8. 当前分支脚本改动已在本地 worktree；服务器 `~/strided-template-calibration-v2/scripts`
   已 rsync 同一版本。若后续修改脚本，记得重新 rsync。
9. 如果测量驱动中途被 kill：已写出的 raw JSON 只在进程正常结束时落盘；
   日志中有每条 case 结果，但完整 attempts 只在最终 JSON。优先让进程自然结束。
10. 如果 path check 被 kill：`path_check_v2.json` 不完整，需重跑。

## 8. 一句话交接

**当前数据没补齐。** 先让 `stride_v2_path` 和 `stride_v2_load` 跑完，
再启动 `PASS=store`；四个 raw pass JSON 齐了之后 build v2 dataset，
本地 copy + commit + push 到 `feature/strided-load-store-costmodel`。
拟合执行前先等用户新指令。
