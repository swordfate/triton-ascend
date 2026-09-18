# costmodel_eval — 白盒公式调整与 6-kernel 打分验证

本目录是 `README.md` §11 的可复现脚本与结果。

## 文件

| 文件 | 作用 |
|---|---|
| `run_scalar_dominate_one.py` | 6 个 scalar-dominated kernel 的固定 shape/config runner（data + 单 config + launch） |
| `run_costmodel_one.sh` | 用 tuned profile 生成单 kernel costmodel report（`simd_simt` + `report`） |
| `run_camodel_one.sh` | 单 kernel / 单 mode（`simd` 或 `simt_only`）CAModel 运行 |
| `run_all_costmodel.sh` / `run_all_camodel.sh` | 6-kernel 批跑（旧 baseline profile 脚本已删） |
| `make_profile.py` | 从服务器已安装 profile 生成 `profile.json`：load 用 CAModel 测试点；SIMD store 用 MTE3 白盒 `prep/fill/serial`；SIMT store 用白盒窗口；`microbenchmark_profile` 写成绝对路径 |
| `compare_scalar_loads.py` | 从 costmodel report + CAModel dump 提取 scalar load 预测/实测 |
| `summarize_direct_scalar.py` | 旧 direct-only 汇总，生成 `results/direct_scalar_eval.csv`（保留参考） |
| `run_scalar_dominate_seeded.py` | 固定 `--seed` 的 runner：padded seed=12 走 indirect 分支，binned seed=0 走 `index_*` |
| `run_camodel_seeded_one.sh` | 单 kernel / 单 mode 的 seeded CAModel |
| `run_seeded_camodel.sh` | 6 kernel seeded CAModel 一键跑（seed 12 / 0，在 20260918 目录里另存为 20260918_seeded 结果） |
| `summarize_all_scalar.py` | matched-only + per-stage-union 全量汇总；SIMT 按 DC `size<128` 排除 vector tile load |
| `results/all_scalar_eval.csv` / `.md` | 6-kernel 全量打分表（tuned profile，seeded CAModel） |
| `results/` | 验证结果 CSV |

## 运行环境

服务器历史 eval：`~/scalar_dominate_eval_20260918/`；本轮 seeded eval：`~/scalar_dominate_eval_seeded/`。
kernel 源码在 `/home/c00946898/scalar_dominate_autotune_latest/`（与分支 `test_cases/scalar_dominate_kernels/` 一致）。

```bash
cd <此目录>
# 1) 生成 tuned profile
python3 make_profile.py
# 2) costmodel report（tuned）；6 kernel
for k in padded_copy_gather padded_copy_scatter padded_copy_wgrad \
         binned_copy_gather binned_copy_scatter binned_copy_wgrad; do
  bash run_costmodel_one.sh "$k"
done
# 3) seeded CAModel：padded seed=12，binned seed=0，单独 launch-count=1
bash run_seeded_camodel.sh
# 4) matched-only + stage-union 全量对比（report 用新代码生成的 round4）
python3 summarize_all_scalar.py --base ~/scalar_dominate_eval_seeded \
  --report-dir ~/scalar_dominate_eval_round4/out \
  --csv results/all_scalar_eval.csv
```

固定 shape/config：`(sl,hs,ne,top_k)=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1`。

> 本目录只归档脚本和结果 CSV；原始 `OPPROF_*`、report JSON、编译 cache 保留在服务器
> `~/scalar_dominate_eval_20260918/`、`~/scalar_dominate_eval_seeded/`，不入 git。

## 全量 scalar load/store 验证（matched-only + per-stage union，2026-09-18 代码修订后）

代码/profile 修订：

- SIMD scalar store 改为 MTE3 白盒 `T = 20 + 450 + (K-1)*480`，删除 `main_store_*` 字段/公式/schema/UT；
- SIMT scalar store 保持白盒 `555+(K-1)*480`（same-line）/ `450+(K-1)*20`（first-store/diff-line）；
- indirect dependency latency 改为 `max(0, exposure - 1) * latency`，首条 shallow edge 不再重复收费；
- 6 kernel 在 fixed shape 下 tuned route 全部变为 `all_simt_only`。

`summarize_all_scalar.py` 口径：

- route：取 `decision_kind`（当前全部 `all_simt_only` → CAModel `simt_only`）；
- **costmodel matched**：只累加真正执行的 stage；`stage_5`（`if expert_idx>0`）在 block0 不执行，因此不计；
- **CAModel stage-union**：每个 matched stage 取其 instructions 的 `max(retire)-min(issue)`，再对 matched stage 求和；
  同一 stage 内重叠 instruction 只算一次，不同 stage 不合并成一个大 union；
- SIMT 按 DC request `size<128` 过滤 scalar load，`size=128` 的 vector tile load 不计入 scalar；
- matched 实测量按地址族匹配（padded: indices/bin_ids vs bins/padded_bins/weights，binned: bins vs indices）。

结果 `results/all_scalar_eval.{csv,md}`：direct 最大误差 +19.9%/-4.9%，indirect +30.5%/+4.1%，
store -18.3%/-19.5%，全部 <50%。残余误差来自 SIMT 单条 window 的核间 BIU 仲裁波动（436–558 cycle 量级）、
`stage_5` 控制流未执行（matched 口径已排除）以及 CAModel 冷链路与公式拟合差；
不再列 static worst-case。详细表和分析见 `README.md` §11.4/§11.5。
