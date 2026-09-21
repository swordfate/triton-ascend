# ascend-950pr-63 上 triton-ascend 环境搭建与复现指南
> ⚠️ **本文件以及整个 `63-workspace/` 目录都是本地 Mac 工作笔记，不要 commit / push 到 GitHub。**
> 如误提交，请撤回并保持该目录仅存在于本地。


> 日期：2026-09-03（全程打通，NPU 冒烟测试通过）；2026-09-12 更新：新增 §10.1 cce 探针 CAModel 配方（实测通过）；2026-09-17 更新：补充 GitHub 推送重试 / heredoc 卡住 / scp 中断改 tar 单包 / macOS AppleDouble / CAModel 结果归档（§3.2、§10.4）
> 服务器：`ascend-950pr-63`（hostname `tbe`，Ubuntu 20.04.6 x86_64，Ascend950PR ×2，**docker 容器**）
> 本文档用于让另一个人/agent 在此服务器上复现或理解现状。服务器上的权威脚本与本文一致。

**配套实体文件**：
- 编译/环境/测试脚本 → 同目录 [`scripts/`](scripts/README.md)（本文内联的脚本与之一一对应）
- CAModel 模拟器探索（cce 探针 2026-09-12 打通 / triton sim 打通 / tiny SIMT 配方）→ 同目录 [`02-camodel-simulator/`](02-camodel-simulator/01-CAModel-探索与复现记录.md)；日常使用路径摘要见本文 §10

---

## 1. 最终状态（先看这个）

| 组件 | 版本/内容 | 位置（服务器） | 说明 |
|---|---|---|---|
| conda env | `wj_autoscope`，Python 3.11 | `/home/c00946898/.conda/envs/wj_autoscope` | 主运行环境 |
| torch | `2.10.0+cpu` | 同上 | 从 pytorch.org cpu index 安装 |
| torch-npu | `2.10.0.post2` | 同上 | 从 TUNA 镜像安装 |
| triton-ascend | `3.6.0.dev0+git25c270cd` | 同上 | fork wheel，见版本配对 |
| LLVM | `22.0.0git`（OBS 预编译） | `/home/c00946898/llvm-install` | **不是**源码编译 |
| bishengir 工具链 | NPUIR @ 9756b309 构建 | `/home/c00946898/bishengir-install` | 含 bishengir-compile/opt + meta_op.*.bc |
| NPUIR 源码（外部） | `simd_simd_compiling` @ `9756b309` | `/home/c00946898/AscendNPU-IR` | 编 bishengir-install 用 |
| NPUIR 源码（fork 内部 pin） | @ `572a94bda` | `/home/c00946898/AscendNPU-IR-triton` | **只供头文件/tablegen**，不编译 |
| triton-ascend fork 源码 | @ `50519ad23` | `/home/c00946898/triton-ascend` | kaixin1976/feature/simd-simt-compile-mode |
| 工具 env | clang **19.1.7** / ninja / (ccache 单独) | `/home/c00946898/.conda/envs/bisheng_build` | clang 23 不可用（见坑 5） |
| ccache 4.14 | 他人安装的独立二进制 | `/home/s00653124/Softwares/ccache/ccache` | 只读借用，缓存目录自建 |

**验证结果**（冒烟测试）：`add_kernel`（triton @jit，BLOCK=1024，N=4096）在 `npu:0` 编译执行通过，结果与 `x+1` allclose。

**运行时配方**（日常使用）：

```bash
ssh ascend-950pr-63          # 实际用户 c00946898
source ~/env_ascend.sh       # CANN env + bishengir PATH + conda libstdc++ LD_LIBRARY_PATH
source /data/miniconda3/etc/profile.d/conda.sh   # conda 不在 env_ascend.sh 中自动加载
conda activate wj_autoscope
```

---

## 2. 版本配对矩阵（⚠️ 最重要，混用必挂）

本套件是 4 个 repo 的互锁组合，由仓库 gitlink / llvm-hash 决定，**互相不能随意替换**：

| 组件 | 取值 | 判定来源 | 备注 |
|---|---|---|---|
| triton-ascend fork | `50519ad23` | github `kaixin1976/triton-ascend` 分支 `feature/simd-simt-compile-mode` | = QuarkUp（用户公司机）构建成功版 |
| fork 内部 AscendNPU-IR pin | `572a94bda`（legacy-interface 线，FixpipeOp 无 `unit_flag_group_id`） | fork 的 `.gitmodules` gitlink | 见坑 1、2 |
| 外部 AscendNPU-IR | gitcode `yangkaixin/AscendNPU-IR` 分支 `simd_simd_compiling` @ `9756b309`（submodule：llvm `97f59c7c`、shmem `4cb0d6e`、torch-mlir `5e9c503`） | 用户 QuarkUp 同款 | 编 bishengir-compile/hivmc-a5 |
| LLVM | `f6ded0be897e2878612dd903f7e8bb85448269e5` | fork 的 `cmake/llvm-hash.txt` | 预编译产物版本号 22.0.0git |

⚠️ **分支 tip 陷阱**：本流程完成后 fork 分支被 push 到 `9b9edc1`（amend），其 gitlink pin 更新为 `aea934a`（unit-flag 合并后），**与 fork 自身代码不兼容**（FixpipeOp 16 参数 vs 调用方 15 参数，编不过）。若需更新 fork，需作者对齐 pin 或自行验证。

---

## 3. 网络拓扑（决定了安装路径）

| 目标 | 服务器直连 | 处理 |
|---|---|---|
| pypi.org | ✅ 200 | 直连 |
| files.pythonhosted.org（下载） | ❌ 极慢 ~12KB/s | 走 TUNA/阿里云镜像 |
| download.pytorch.org | ✅ GET 可用（HEAD 403 属正常） | pip 直装 |
| github.com | ⚠️ 时通时断（2026-09-17 一次 `ls-remote` 成功，随后同命令 `Connection timed out`） | 默认按不可靠处理：clone/下载 → Mac 中转；push → Mac 干净工作树（见 §3.2） |
| gitcode.com | ✅ 200 | 直连（NPUIR 等子模块都在此） |
| triton-ascend-artifacts.obs.myhuaweicloud.com | ✅ 快（1.2G ~2 分钟） | 直连 |
| pypi.tuna.tsinghua.edu.cn | ✅ | torch-npu/pyyaml/numpy 等依赖 |

经验：**github 上小文件走 raw/codeload 本机下载后 scp；服务器端一律优先 gitcode/OBS/国内镜像**。本机 github git 协议可用（ls-remote 秒回）但大 tarball 传输会中途卡死，需 `curl -C -` 断点续传 + 重试循环。

### 3.1 用本机（Mac）做外网跳板的完整姿势

服务器 GitHub **时通时断**（读路径也不稳定，默认按不通处理），但**本机 Mac 稳定**。中继原则：**能在服务器直连的绝不中继**（gitcode / OBS / TUNA / pypi 均直连），只有 github 系资源走 Mac。按文件大小分三种姿势：

**① git 协议小请求（秒级，用于查 ref/commit/gitlink）**

```bash
# 在本机 Mac 上执行
git ls-remote https://github.com/kaixin1976/triton-ascend feature/simd-simt-compile-mode

# 查 fork 分支 pin 的 submodule commit（blobless 浅克隆，只拉树对象，不拉文件内容）
git clone -q --depth 1 --filter=blob:none --no-checkout --single-branch \
    --branch feature/simd-simt-compile-mode https://github.com/kaixin1976/triton-ascend.git fork-git
git -C fork-git ls-tree HEAD third_party/ascend/AscendNPU-IR
# → 得到 pin 的 commit SHA，之后到服务器上 gitcode 直接 fetch 该 SHA
```

**② 小文件（<几 MB，如 patch）：raw.githubusercontent.com 可直接 curl**

```bash
curl -sL -o llvm_patch_f6ded0b.patch \
  https://raw.githubusercontent.com/triton-lang/triton-ascend/refs/heads/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch
```

**③ 大文件（tarball 等）：codeload 直连 + 断点续传重试循环（重要：不要用 `github.com/.../archive/` 重定向 URL，会卡死；用 codeload 域名直连）**

```bash
# 本机 Mac 执行；下载完成再 scp 到服务器
cd /tmp/ascend-relay
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sL -C - --retry 3 --retry-delay 5 --speed-time 90 --speed-limit 1024 -m 900 \
    -o triton-ascend-fork.tar.gz \
    "https://codeload.github.com/kaixin1976/triton-ascend/tar.gz/refs/heads/feature/simd-simt-compile-mode" \
    && break
  echo "attempt $i failed, retrying"; sleep 5
done

# 传服务器
scp /tmp/ascend-relay/triton-ascend-fork.tar.gz ascend-950pr-63:/home/c00946898/
# 服务器侧解压使用
```

经验要点：
- codeload HEAD 探测秒回（200）≠ 传输稳定；中途会静默卡死 → 必须 `-C -`（续传）+ `--speed-limit/--speed-time`（卡死即断）+ 外层 for 循环重试。llvm 180MB 源码包当时反复重试才成功，而 fork 11MB 一次成功——**没耐心的活交给循环**。
- 服务器能直接 `git fetch --depth 1 origin <sha>` gitcode 的任意 commit（gitcode 支持按 SHA fetch），所以拿到 pin 后完全不需要把仓库整个搬过去。
- 中继的终点形态是"文件到了服务器"，后续安装/构建全在服务器本地进行。

### 3.2 push GitHub：从 Mac 干净工作树推，失败就重试（2026-09-17 实测）

2026-09-17 实测：

- 服务器 `git ls-remote origin HEAD` 曾成功一次（返回 `dc7cdf0445...`），随后同命令又报
  `Failed to connect to github.com port 443: Connection timed out`。**服务器 GitHub 读写都不可靠**，不要当主路径。
- 服务器 `triton-ascend` 工作树有 **36 modified / 15 deleted / 45 untracked**，且 remote-tracking refs 可能 stale；
  **不要从服务器工作树直接 `git push`**。
- 本机 Mac 配了代理：`git config --get http.proxy` → `http://127.0.0.1:7897`；`curl -x` 和直连 GitHub 都能 200，
  但 `git push` 仍可能先报一次
  `LibreSSL SSL_connect: SSL_ERROR_SYSCALL in connection to github.com:443`；**重试即可**。

验证过的 push 姿势：

```bash
cd /Users/weijianchen/Documents/2026/triton-ascend
git checkout scalar-load-whitebox
git add <只 add 需要提交的目录，例如 .../data_provider/scalar_ldst_whitebox>
git commit -m "docs(costmodel): ..."
for i in 1 2 3; do
  git push origin scalar-load-whitebox && break
  echo "push retry $i"; sleep 5
done
```

2026-09-17 结果：`256eb7ab2..5c1683f24`（load）之后又推了 `4f4cc84f7`（4-op load）和 `244b7eebe`（scalar store o1/o4）；`origin/scalar-load-whitebox` 当前到 `244b7eebe`。

CAModel 大产物策略：只提交 README / 脚本 / 解析 JSON + 少量关键 dump；`OPPROF_*`、`*.o`、host 二进制用目录内
`.gitignore` 排除（如 `scalar_ldst_whitebox/load/scalar_o1/.gitignore`）。本次提交 31 files / 2693 insertions，约 290 KB；
GitHub 单文件上限 100 MB，远无压力。

### 3.3 push 大文件 / CAModel 原始 OPPROF 到 GitHub（2026-09-21 实测）

本轮需要把 padded_gather F16/F32 的完整 CAModel `OPPROF_*` 传给远端，
走了一遍“大文件 + GitHub 网络”的完整流程，记录如下。

#### 3.3.1 GitHub 的文件大小限制

- **单文件硬限制 100 MB**：超过会被 GitHub 直接拒绝（不是 warning）。
- **建议单文件 ≤ 50 MB**：超过会收到 warning：
  `File ... is 81.02 MB; larger than GitHub's recommended maximum file size of 50.00 MB`。
- 大二进制 **不要提交到代码分支**（如 `debug/sb64-pr2305`），否则以后每次 clone/fetch 都会变重。
- 本次的合并包 `padded_camodel_tiny_full.tgz` 是 **102 MB**，超过单文件硬限制；
  拆成两个包后才成功：
  - `padded_camodel_tiny_force16.tgz`：29 MB
  - `padded_camodel_tiny_force32.tgz`：81 MB

#### 3.3.2 用独立 artifact 分支，不污染代码分支

推荐做法：

1. 在**干净的 worktree** 里从目标代码分支拉一个 artifact 分支：

   ```bash
   cd /private/tmp/triton-sb64-pr2305     # 干净、无 modified/untracked 的 worktree
   git switch -c debug/sb64-pr2305-camodel-tiny-artifacts
   ```

2. 大产物统一放到 `debug_artifacts/<topic>/`，并写一个 `README.md`：

   ```text
   debug_artifacts/camodel_tiny/
   ├── README.md
   ├── padded_camodel_tiny_force16.tgz
   └── padded_camodel_tiny_force32.tgz
   ```

   README 至少写：
   - 实验配置（shape、num_warps、factor、soc-version）；
   - `sha256`/`md5` 校验值；
   - 解压命令；
   - 关键文件位置（例如直接 grep `SIMT_STK` / `SIMT_LDK`）。

3. 正常 `git add` / `git commit`：

   ```bash
   git add debug_artifacts/camodel_tiny
   git commit -m "debug(camodel): add padded_gather F16/F32 tiny full OPPROF outputs"
   ```

4. 远程也推成一个**新分支**，不要推到原来的 `debug/sb64-pr2305`：

   ```text
   debug/sb64-pr2305-camodel-tiny-artifacts
   ```

#### 3.3.3 HTTPS / 代理不通时改用 SSH

本轮实测：

- `origin` 是 HTTPS：`https://github.com/swordfate/triton-ascend.git`
- Mac 上 `git config --get http.proxy` 仍是 `http://127.0.0.1:7897`，
  但当天本机代理没有真正监听。
- 表现：
  - `git push origin ...` 卡约 75 s 后报
    `Failed to connect to github.com port 443`；
  - `git -c http.proxy= ls-remote origin HEAD` 有时能通，说明是**不稳定**而不是被封。
- **稳定路径是 SSH**：

  ```bash
  ssh -T git@github.com
  # Hi swordfate! You've successfully authenticated...
  ```

  大文件 push 显式走 SSH URL，并加 keepalive：

  ```bash
  cd /private/tmp/triton-sb64-pr2305
  GIT_TERMINAL_PROMPT=0 \
  GIT_SSH_COMMAND='ssh -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=15' \
  git push git@github.com:swordfate/triton-ascend.git \
    debug/sb64-pr2305-camodel-tiny-artifacts:refs/heads/debug/sb64-pr2305-camodel-tiny-artifacts
  ```

  本次 push 结果：

  ```text
  remote: warning: File debug_artifacts/camodel_tiny/padded_camodel_tiny_force32.tgz
  remote: warning: ... 81.02 MB; larger than GitHub's recommended maximum file size of 50.00 MB
  To github.com:swordfate/triton-ascend.git
   * [new branch] debug/sb64-pr2305-camodel-tiny-artifacts -> ... 
  ```

#### 3.3.4 long-running push 要放到 screen/tmux 里

大文件 push 可能几十秒到几分钟。直接在 agent 的持久 bash 里跑，
一旦工具调用超时/重置，后台子进程也可能被一起清掉。

- Linux 服务器用 `tmux`（见 §10.2）。
- macOS 默认没有 tmux，但有 `screen`：

  ```bash
  screen -dmS camopush /bin/bash -c '
    cd /private/tmp/triton-sb64-pr2305 &&
    GIT_TERMINAL_PROMPT=0 \
    GIT_SSH_COMMAND="ssh -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=15" \
    git push git@github.com:swordfate/triton-ascend.git \
      debug/sb64-pr2305-camodel-tiny-artifacts:refs/heads/debug/sb64-pr2305-camodel-tiny-artifacts \
      > /tmp/push_camodel_ssh.log 2>&1
  '

  screen -ls
  tail -f /tmp/push_camodel_ssh.log
  ```

  看到 `[new branch]` 后可以 `screen -S camopush -X quit` 结束会话。

#### 3.3.5 push 后验证

不要只看本地 `git log`，必须查远端 ref：

```bash
GIT_SSH_COMMAND='ssh -o ConnectTimeout=15 -o BatchMode=yes' \
git ls-remote git@github.com:swordfate/triton-ascend.git \
  refs/heads/debug/sb64-pr2305-camodel-tiny-artifacts
```

本次返回：

```text
48392de880ee9bf8875b45b5ac90d60fb82b3038  refs/heads/debug/sb64-pr2305-camodel-tiny-artifacts
```

GitHub 访问页面：

```text
https://github.com/swordfate/triton-ascend/tree/debug/sb64-pr2305-camodel-tiny-artifacts/debug_artifacts/camodel_tiny
```

#### 3.3.6 经验总结

1. **>50 MB 的单个文件，先拆包**；>100 MB 一定推不上去。
2. **大产物用独立 artifact 分支**，不要混进代码分支。
3. **HTTPS/代理不稳定时直接 SSH**；大文件 push 加 `ServerAliveInterval`。
4. **长时间 push 放到 screen/tmux**，日志重定向，轮询而不是阻塞等待。
5. README 里写清 `sha256`、shape、factor、解压和 grep 命令。
6. 如果产物很多/很大，优先用 GitHub Release asset（单资产 2 GB 级），
   这样不会把 git 仓库体积撑大；本次用户要求“push 分支”，所以走了 artifact 分支方案。

---

## 4. 分步安装（可复现）

### Step 0 — 前置

```bash
# 服务器系统
# Ubuntu 20.04.6, glibc 2.31（决定 LLVM 必须用 almalinux 预编译版，见坑 9）
# CANN: /usr/local/Ascend/cann-9.1.0（set_env.sh 在此），ccec 在 tools/bisheng_compiler/bin
# NPU: Ascend950PR ×2（共享机器，其他人也在跑 pytest）
# docker 容器：nproc=256 是假象，编译并行度最大设 8（见坑 8）

# 工具 env（clang 19.1.7 对齐 llvmorg-19.1.7 源码；ninja）
conda create -y -n bisheng_build -c conda-forge clang clangxx ninja
```

网络：服务器 pypi 直连装依赖很慢时全部换 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### Step 1 — torch + torch-npu（进 wj_autoscope）

```bash
# torch 2.10.0+cpu（只有 pytorch index 有）
conda run -n wj_autoscope pip install --trusted-host download-r2.pytorch.org \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# torch-npu（pypi 直连慢 → TUNA）
conda run -n wj_autoscope pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-npu==2.10.0.post2

# 运行依赖（torch-npu import 链上需要的）
conda run -n wj_autoscope pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml numpy decorator

# 验证（必须 source CANN env 才能 import torch_npu 成功）
source /usr/local/Ascend/cann-9.1.0/set_env.sh
conda run -n wj_autoscope python -c "import torch, torch_npu; print(torch.__version__, torch_npu.npu.is_available())"
# → 2.10.0+cpu True
```

### Step 2 — LLVM（OBS 预编译，免源码编译）

```bash
# fork 的 cmake/llvm-hash.txt = f6ded0be；patch 哈希 = sha256(llvm_patch_f6ded0b.patch) 前 8 位
# 本套件 = 4ca23101 → 工件名 llvm-f6ded0be-4ca23101-<suffix>.tar.gz
# suffix: ubuntu-x64(需 GLIBC≥2.33, 本机跑不了) → 用 almalinux-x64(基于 glibc 2.28, 兼容)

cd /home/c00946898
curl -sL -o llvm-prebuilt-alma.tar.gz \
  https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/llvm-f6ded0be-4ca23101-almalinux-x64.tar.gz
mkdir -p llvm-install && tar -xzf llvm-prebuilt-alma.tar.gz -C llvm-install --strip-components=1
llvm-install/bin/llvm-config --version   # → 22.0.0git
```

> 工件名 hash 重算方法：`shasum -a 256 third_party/ascend/patch/llvm_patch_f6ded0b.patch | cut -c1-8`（patch 目录下所有 `llvm_patch_*` 文件内容拼接后 sha256 前 8 位，见 fork `setup.py::get_llvm_patch_hash`）。

### Step 3 — NPUIR 构建 → bishengir-install

源码（gitcode 直连，含 3 个 submodule）：

```bash
cd /home/c00946898
git clone -b simd_simd_compiling https://gitcode.com/yangkaixin/AscendNPU-IR.git
cd AscendNPU-IR
git submodule update --init --depth 1     # 失败重试即可，gitcode 偶发 "Couldn't find remote ref HEAD"
```

构建脚本（服务器 `/home/c00946898/build_npuir.sh`，与用户 QuarkUp 参考脚本一致并适配本机）：

```bash
#!/usr/bin/env bash
set -euo pipefail
# NPUIR (bishengir) build for triton-ascend backend
source /usr/local/Ascend/cann-9.1.0/set_env.sh || true

export PATH="/home/c00946898/.conda/envs/bisheng_build/bin:/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:${PATH}"
# conda toolchain libs (libstdc++ newer than system's on Ubuntu 20.04) must win at runtime
export LD_LIBRARY_PATH="/home/c00946898/.conda/envs/bisheng_build/lib:${LD_LIBRARY_PATH:-}"

CCEC_PATH="$(command -v ccec)"
if [ -z "${CCEC_PATH}" ]; then
  echo "ccec not found after sourcing CANN set_env.sh" >&2
  exit 1
fi
BISHENG_INSTALL_PATH="$(dirname "$(readlink -f "${CCEC_PATH}")")"

cd /home/c00946898/AscendNPU-IR
mkdir -p build_bishengir

export BISHENGIR_BUILD_TARGET="bishengir-compile;bishengir_template_bitcode"

./build-tools/build.sh \
  --c-compiler clang --cxx-compiler clang++ \
  '--add-cmake-options=-DLLVM_ENABLE_LLD=ON' \
  --build-type Release \
  --enable-assertion \
  --disable-werror --disable-bishengir-werror \
  --build-triton -t -j 96 \
  --bisheng-compiler "${BISHENG_INSTALL_PATH}" \
  --build "/home/c00946898/AscendNPU-IR/build_bishengir" \
  --install-prefix /home/c00946898/bishengir-install

# meta libs 复制（模板 bitcode，运行时需要）
cd /home/c00946898/AscendNPU-IR/build_bishengir
mkdir -p /home/c00946898/bishengir-install/lib
cp lib/meta* /home/c00946898/bishengir-install/lib/
echo "NPUIR_BUILD_ALL_DONE"
```

运行：`nohup /home/c00946898/build_npuir.sh > logs/npuir_build.log 2>&1 &`

> 本机实测：clang19 + -j96 下两阶段（~6000 + ~5340 targets）约 13-15 分钟。⚠️ 该脚本的 `-t`（`--build-bishengir-template`）**必须带**——meta_op.*.bc 由此生成（见坑 6）。

### Step 4 — triton-ascend fork 源码 + 内部 NPUIR

```bash
cd /home/c00946898
# github 时通时断：默认仍走本机 Mac 下载分支 tarball 后 scp（或 git clone 本机打包）；
# 2026-09-17 实测服务器 `git ls-remote` 偶发可通，但下一分钟又 timeout，不能当稳定通路
# tar 包解压改名
mv triton-ascend-feature-simd-simt-compile-mode triton-ascend

# tarball 无 .git —— setup.py 的 apply_triton_ascend_patch 需要 git checkout，先 git init（见坑 3）
cd triton-ascend && git init -q && git config user.email build@local && git config user.name build \
  && git add -A && git commit -qm init

# fork 内部 AscendNPU-IR submodule（tarball 不含内容；tarball 也无 gitlink → 用本机 blobless clone 查 pin）
# pin = 572a94bda9c13f8a8bf9c0ad65f4341d16ad1d8b（查询方法：git ls-tree HEAD third_party/ascend/AscendNPU-IR）
mkdir -p /home/c00946898/AscendNPU-IR-triton && cd /home/c00946898/AscendNPU-IR-triton
git init -q && git remote add origin https://gitcode.com/Ascend/AscendNPU-IR.git
git fetch --depth 1 origin 572a94bda9c13f8a8bf9c0ad65f4341d16ad1d8b && git checkout -q FETCH_HEAD

# 用真实 checkout 替换空目录（符号链接亦可，cmake 能跟随）
rmdir /home/c00946898/triton-ascend/third_party/ascend/AscendNPU-IR
ln -s /home/c00946898/AscendNPU-IR-triton /home/c00946898/triton-ascend/third_party/ascend/AscendNPU-IR
```

> Triton-distributed-ascend submodule 缺失无妨：构建时 `TRITON_BUILD_DISTRIBUTED=OFF`。

**2026-09-03 更新：服务器源码已换成真 clone（不再用 tarball 方式）**

```text
origin     = https://github.com/swordfate/triton-ascend.git   （用户自己的源）
upstream-kx = https://github.com/kaixin1976/triton-ascend.git （kaixin 上游）
默认分支    = generic-stage-partition-kx @ 80c2d7d（浅克隆 depth=100，含 origin 各分支 ref + upstream-kx/feature/simd-simt-compile-mode）
```

做法：本机 Mac 浅克隆（`git clone --depth 100 --no-single-branch --branch generic-stage-partition-kx https://github.com/swordfate/triton-ascend.git`）→ 把 kaixin 分支 fetch 到 `refs/remotes/upstream-kx/` → tar/scp 上服务器 → 替换原目录。注意点：
- 原 tarball 版树保留在 `/home/c00946898/triton-ascend-tarball-old/`（含旧 build/，已把 `build/`、`dist/` 移入新树，wheel 增量重编不受影响）。
- macOS tar 会带 AppleDouble `._*` 垃圾进 .git → 解压后 `find . -name "._*" -delete`。
- `third_party/ascend/AscendNPU-IR` 仍是指向 `/home/c00946898/AscendNPU-IR-triton`（572a94bda）的符号链接 → `git status` 恒显示 submodule modified（与用户 Mac 上的状态一致，**这是有意的**——pin aea934a 编不过 wheel）。
- 服务器→github 时通时断（默认按不通处理）：fetch origin/upstream-kx 仍建议经本机 Mac 中继（3.1 节方法）；push 实测见 §3.2。

### Step 5 — wheel 构建与安装

服务器 `/home/c00946898/build_triton_wheel6.sh`（最终可用版；配方参考了本机另一位用户 h00951476 的构建命令）：

```bash
#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh || true

export PATH="/home/s00653124/Softwares/ccache:/home/c00946898/.conda/envs/bisheng_build/bin:/home/c00946898/.conda/envs/wj_autoscope/bin:${PATH}"
export LD_LIBRARY_PATH="/home/c00946898/.conda/envs/bisheng_build/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"   # conda clang 找不到 -lz 的修复（见坑 10）
export CCACHE_DIR=/home/c00946898/.ccache
export MAX_JOBS="${MAX_JOBS:-64}"

cd /home/c00946898/triton-ascend
pip uninstall triton-ascend -y 2>/dev/null || true
rm -rf dist/

export LLVM_SYSPATH=/home/c00946898/llvm-install
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_PROTON=OFF
export TRITON_WHEEL_NAME="triton-ascend"
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF"

python3 setup.py sdist bdist_wheel 2>&1 | tail -30

pip install dist/triton_ascend*.whl --force-reinstall --no-deps
pip show triton-ascend | head -3
echo "TRITON_WHEEL_ALL_DONE"
```

产出：`dist/triton_ascend-3.6.0.dev0+git<commit>-cp311-cp311-linux_x86_64.whl`，自动 pip 安装进 wj_autoscope。

> 增量重编技巧：保留 `build/` 目录再跑即可（ninja 断点续编）。**不要**在 setup.py 的日志管道里截断错误——`| tail -30` 会吞掉 ninja 报错，排查时直接在 `build/cmake.linux-x86_64-cpython-3.11/` 下手动 `ninja -j64`。

### Step 6 — 运行时环境 + 冒烟测试

```bash
# 服务器 /home/c00946898/env_ascend.sh
#!/usr/bin/env bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh || true
export PATH="/home/c00946898/bishengir-install/bin:${PATH}"
export PATH="/home/s00653124/Softwares/ccache:/home/c00946898/.conda/envs/bisheng_build/bin:${PATH}"
export LD_LIBRARY_PATH="/home/c00946898/bishengir-install/lib:/home/c00946898/.conda/envs/bisheng_build/lib:${LD_LIBRARY_PATH}"
# export TRITON_NPU_COMPILER_PATH=/home/c00946898/bishengir-install/bin   # 可选覆盖
```

triton-ascend 运行时找 bishengir-compile 的顺序：wheel 内 `backends/ascend/bishengir/bin/` → `PATH` → `TRITON_NPU_COMPILER_PATH`。

冒烟测试（服务器 `/home/c00946898/smoke_test.py`）：triton add_kernel 在 `npu:0` 上执行并与 `x+1` 比对。通过即代表全链路 OK。

### Step 7 — 运行 simd/simt costmodel 样例测试（test_simd_simt_costmodel_cases.py）

2026-09-03 实跑记录：**10 passed in 210.36s**（三个测试函数，solve_tril 参数化 8 组）。

**测试内容**（文件在 fork 源码树，本地与服务器 md5 一致）：
- `test_costmodel_gather_dot_min` — 路由断言 all_simt_only + 性能 ≤ 5.478us×1.35
- `test_costmodel_solve_tril`（8 组 B/T/H/BT）— 路由断言 mixed_simd_simt、superblock factor=4、tail=0、数值 + 性能 ≤ documented_us
- `test_costmodel_fbgemm_rowwise_quant` — 路由断言 all_simt_only、row_coalescing_factor=2、FP8 1 ULP、性能 ≤ 8.904us×1.25

前置（一次性）：
```bash
conda run -n wj_autoscope pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-xdist
# pytest-xdist 必须：unittest/pytest_ut/conftest.py 的 assign_npu fixture 用 worker_id（xdist 提供），缺它所有用例 setup ERROR
```

运行脚本（服务器 `/home/c00946898/run_costmodel_tests.sh`）：
```bash
#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh || true
export PATH="/home/c00946898/bishengir-install/bin:/home/c00946898/.conda/envs/bisheng_build/bin:${PATH}"
export LD_LIBRARY_PATH="/home/c00946898/bishengir-install/lib:/home/c00946898/.conda/envs/bisheng_build/lib:${LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
cd /home/c00946898/triton-ascend

conda run -n wj_autoscope python -m pytest -v \
  third_party/ascend/unittest/pytest_ut/test_simd_simt_costmodel_cases.py \
  2>&1 | tee /home/c00946898/logs/pytest_costmodel.log | tail -60
```

注意事项：
- 测试带 `@simd_simt_910_95_only`（非 910_95 soc 直接 skip）：Ascend950PR 的 soc 名含 "ascend950" → 正常执行。
- 耗时大头是每个 kernel 的 triton 编译（首次跑 210s 全花这）；性能断言用 torch_npu profiler 计时（kernel_details.csv 中位数）对比 documented_us，通过条件 ≤ documented×tolerance。
- 共享 NPU：`ASCEND_RT_VISIBLE_DEVICES=0` 固定用 0 号卡（conftest 默认也给 npu 0）。

---

## 5. 目录布局（服务器 /home/c00946898/）

```
AscendNPU-IR/            # 外部 NPUIR 源码 @9756b309（+ build_bishengir/ 构建目录）
AscendNPU-IR-triton/     # fork 内部 pin @572a94bda（wheel 构建用头文件）
bishengir-install/       # bin/{bishengir-compile,bishengir-opt,...} lib/meta_op.*.bc lib/*.a
llvm-install/            # LLVM 22.0.0git（OBS almalinux 预编译）
triton-ascend/           # fork 源码 @50519ad23（build/ 可增量，dist/ wheel）
env_ascend.sh            # 运行时环境
build_npuir.sh           # NPUIR 构建脚本
build_triton_wheel6.sh   # wheel 构建脚本
rerun_ninja.sh           # 手动续编 ninja（-j64）
smoke_test.py            # 冒烟测试
run_costmodel_tests.sh   # 跑 simd/simt costmodel 三个样例测试（见 Step 7）
camodel-cce-test/        # cce 探针 + msopprof CAModel 配方工作目录（见 §10.1）
microbench-run/          # data_provider 副本 + camodel/ 解析脚本（见 §10.2）
logs/                    # 各次构建/测试日志（pytest_costmodel.log 等）
.conda/envs/wj_autoscope     # 运行 env
.conda/envs/bisheng_build    # 编译 env（clang19/ninja）
.ccache/                 # ccache 缓存（自建）
```

---

## 6. 踩坑与修改全记录

### 6.1 版本/源码类

1. **fork 分支 tip 的 NPUIR pin 与代码不兼容**
   - 现象：wheel 编译 DynamicCVPipeline/InterCoreTransferAndSync.cpp 报 `FixpipeOp` `no matching function for call to 'build'`（生成的头文件要求 16 参，调用方传 15）。
   - 排查：对比 fork 各调用点与 NPUIR master（aea934a / 9756b309）的 `HIVMDMAOps.td`——**两者 FixpipeOp 定义相同**（都带 `unit_flag_cond`），而 fork 代码是旧签名。即最新 fork + 最新 NPUIR master 本来就编不过。
   - 解决：向用户确认 QuarkUp 成功组合 → fork `50519ad23` + 内部 NPUIR `572a94bda`（legacy-interface 线，无 `unit_flag_group_id`）→ 换上后 0 error。
   - 理由：wheel 构建**不编译**内部 NPUIR 的 .cpp（实测 764 个编译进程 0 个在编它），只用其 .td tablegen 出头文件，因此内部 pin 只需与 fork 代码 **API 一致**即可；运行时编译真正用的是外部 bishengir-install。

2. **外部 NPUIR 与内部 NPUIR 是两回事，别混用**（用户强调）：triton-ascend 内部的 AscendNPU-IR 只是"头文件供应商"，不代表 bishengir-compile/hivmc-a5 的构建源。

3. **tarball 无 git → setup.py patch 失败**
   - 现象：`RuntimeError: init code failed, list:['python/triton/runtime/autotuner.py']`。
   - 原因：`setup.py::apply_triton_ascend_patch` 先 `git checkout -- <file>` 还原再打补丁，无 .git 必挂。
   - 解决：`git init && git add -A && git commit -qm init`。

4. **查 fork 的 submodule pin 用 blobless 浅克隆**：`git clone --depth 1 --filter=blob:none --no-checkout --single-branch` 后 `git ls-tree HEAD <path>`，秒级完成；gitcode 支持任意 SHA 直接 `git fetch --depth 1 origin <sha>`。

### 6.2 工具链类

5. **conda clang 23 太新**
   - 现象：google benchmark 头文件报 `'__COUNTER__' is a C2y extension [-Werror,-Wc2y-extensions]`。
   - 解决：`conda install -n bisheng_build -c conda-forge clang=19 clangxx=19`（与 NPUIR 的 llvmorg-19.1.7 对齐）。
   - 理由：LLVM 自带的 -Werror（来自 third_party benchmark）无法被 `--disable-werror` 关掉；降级比加 `-Wno-c2y-extensions` 更稳（避免 23 的其他新警告）。

6. **`-t/--build-bishengir-template` 必须带**
   - 现象：用户脚本最后 `cp lib/meta* bishengir-install/lib/` 找不到文件。
   - 原因：meta_op.*.bc（算子模板 bitcode，运行 bishengir-compile 时需要）由 lib/Template 生成，受 cmake 变量 `BISHENGIR_BUILD_TEMPLATE` 控制；build.sh 只在 `-t` flag 时置 ON。第一次构建漏了该 flag。
   - 解决：脚本补 `-t`（增量重配：`cmake -S .../llvm -B build_bishengir -DBISHENGIR_BUILD_TEMPLATE=ON && ninja` 亦可）。产物：`lib/meta_op.{aic,aiv,mix}.*.bc`。

7. **A5 宏守卫包着 `<cstdint>`（对 fork 源码的小补丁）**
   - 现象：host clang++ 编译 MLIR 报 `ValueBoundsOpInterfaceImpl.h:34: error: unknown type name 'int64_t'`。
   - 原因：fork（Ascend）的 llvm-project 头文件里 `#ifndef BSPUB_DAVINCI_BISHENGIR_A5 / #include <cstdint> / #endif`——该宏下不包含 cstdint（A5 编译器 ccec 把 int64_t 当内建类型），但 cmake 把 `-DBSPUB_DAVINCI_BISHENGIR_A5` 也传给了所有 host 编译。
   - 修改（**仅 2 个文件**，其余文件的 BSPUB 守卫未动）：
     - `mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h`
     - `mlir/include/mlir/Target/SPIRV/Deserialization.h`
     - 处理：删除守卫行，`<cstdint>` 无条件包含（纯类型头，ccec 也有，无副作用）。
   - ⚠️ 若以后重 clone 该 llvm submodule 需重打。

8. **编译并行度上限 8（用户多次强调，最终遵从）**
   - 现象：wheel 构建默认 `MAX_JOBS = 2×os.cpu_count() = 512`，直接把机器打瘫：负载 400+、72% iowait、编译实际停滞（几分钟 0 个 .o）。
   - 教训：docker 里 `nproc=256`、cgroup `cpu.max=-1`、affinity 0-255 全是假象/无效指标；**实际可用算力用户说了算（≤8）**。NPUIR 用 -j96 能过不代表合理。
   - 处理：所有构建默认 `export MAX_JOBS=64`；手动 ninja 默认 `-j 64`，可按机器负载调整。

9. **Ubuntu 20.04 glibc 2.31 与预编译 LLVM**
   - ubuntu-x64 预编译版需要 GLIBC_2.33（`llvm-config` 直接报错）→ 换 **almalinux-x64** 版（glibc 2.28 基线，向后兼容）✓。h00951476 的 llvm-install 也验证了这点（他要 GLIBC 2.34+，在这台机器跑不起来）。
   - 结论：**本机一律用 almalinux-x64 工件**。

10. **conda clang 链接找不到 `-lz`**
    - 现象：链接 libtriton.so 报 `ld.lld: error: unable to find library -lz`，尽管 `/lib/x86_64-linux-gnu/libz.so` 存在。
    - 原因：conda clang 的库搜索路径不含 Debian multiarch 目录。
    - 解决：`export LIBRARY_PATH=/lib/x86_64-linux-gnu`（clang 的 -l 搜索会读 LIBRARY_PATH）。

11. **运行时 libstdc++ 版本不匹配（GLIBCXX_3.4.30）**
    - 现象：NPUIR 构建产出的 llvm-min-tblgen 无法运行：`libstdc++.so.6: version GLIBCXX_3.4.30 not found`。
    - 原因：clang 19 链接了 conda 的较新 libstdc++，运行时却解析到系统旧版。
    - 解决：构建/运行脚本统一 `export LD_LIBRARY_PATH=/home/c00946898/.conda/envs/bisheng_build/lib`（conda 的 libstdc++ 向后兼容系统二进制，安全）。

12. **CANN set_env.sh 在 `set -u` 下报 unbound variable**
    - 现象：`LD_LIBRARY_PATH/PYTHONPATH/CMAKE_PREFIX_PATH: unbound variable`。
    - 解决：source 它之前不要 `set -u`（脚本用 `set -eo pipefail`），或 `|| true` 容忍。

### 6.3 网络类

13. **pip 直连 files.pythonhosted.org 只有 ~12KB/s**：杀进程换 TUNA 镜像秒下（torch-npu 36MB wheel）。pypi.org 索引 API 通 ≠ 文件下载快，先测速再决定。

14. **本机（Mac）下载 github 大文件会中途卡死**：`curl -sL -C - --retry 3 --speed-time 90 --speed-limit 1024 -m 900` 循环重试；codeload 直连 URL（`https://codeload.github.com/<owner>/<repo>/tar.gz/<ref>`）比 github.com/archive 重定向稳定。

15. **`--trusted-host download-r2.pytorch.org` 保留**：torch 索引页在 download.pytorch.org，wheel 实际落在 R2 CDN；HEAD 请求 403/404 均正常，GET 可用。

---

## 7. 失败尝试简表（供参考，不复述细节）

| # | 尝试 | 失败原因 | 一句话教训 |
|---|---|---|---|
| 1 | clang 23 全量编 NPUIR | C2y 警告被 benchmark 的 -Werror 升级 | 工具链对齐源码年代的 LLVM 主版本 |
| 2 | 漏 `-t` 编 NPUIR | 无 meta 产物 | flag 照抄用户原命令，别自以为是删减 |
| 3 | 用 9756b309 当 fork 内部 NPUIR | FixpipeOp API 不匹配 | 内部 pin 查 gitlink，别猜 |
| 4 | 用 fork 最新 tip(9b9edc1) 的 pin(aea934a) | 该组合本身断的 | 以 QuarkUp 实证组合为准 |
| 5 | `MAX_JOBS=512/-j96` 编 wheel | I/O 打爆（负载 400+） | ≤8 |
| 6 | 手动 `cmake -B .`（不带 -S） | 报 source 目录无 CMakeLists | cmake 重配要带 `-S` |
| 7 | 构建日志管道 `\| tail -25` | 吞掉 ninja 真错 | 排查期直接手动 ninja |
| 8 | pkill -f "ninja -j 512" | 模式匹配到自身 ssh shell（exit 255） | pkill 模式避免包含自己命令行 |
| 9 | `ssh host 'cat > f <<EOF ...'` 嵌套 heredoc 写脚本 | 外层 shell/agent 一直等 EOF，300s 超时 | 本地写好文件再 `scp`；别在 ssh 引号里塞多行 heredoc |
| 10 | Mac `git push` 首次 `SSL_ERROR_SYSCALL` | 代理/链路瞬时失败，不是仓库或大小问题 | push 放 for 循环里重试 |
| 11 | 对 raw CAModel dump 跑 `git diff --check` | dump 本身带行尾空格，检查全红 | 原始证据不做 whitespace 校验；必要时 `.gitattributes` 标 `-diff` |

---

## 8. 后续维护提醒

1. **重编 wheel**：`bash ~/build_triton_wheel6.sh`（ccache 已热，增量很快；保留 build/ 目录）。
2. **升级 fork/NPUIR 前**：先确认三处配对（fork 代码 ↔ 内部 NPUIR pin ↔ LLVM hash），并小范围编译验证 DynamicCVPipeline 等敏感文件。
3. **NPUIR 重 clone 后**：重打 6.2-7 的 2 个 cstdint 头文件补丁。
4. **他人共享**：ccache 二进制在 `/home/s00653124/Softwares/ccache/`（只读借用，别写它的缓存目录）；NPU 与其他用户共享（npu-smi 会看到别人的 pytest）。
5. **fork tip 已前移**：当前可用组合是 50519ad23+572a94bda；需要 9b9edc1 的新功能时需作者修复 pin 或自行适配代码。
6. **CAModel 结果入库**：完整 `OPPROF_*` 不进 git；仓库只放 `scalar_ldst_whitebox/...` 的 README、脚本、解析 JSON 和少量关键 dump；完整原始输出本地存到 `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/`。2026-09-17 已从 Mac 的 `scalar-load-whitebox` 分支 push 单 op load、4-op load、scalar store o1/o4 白盒报告及 BIU/GSU 数据通路修正，远程分支当前为 `244b7eebe`。

---

## 9. CostModel for Autoscope 开发笔记（代码架构 / Scalar Load/Store StageKind 添加）
> 本节整理自 [`05-add-scalarldst/costmodel-autoscope-development-scalarldst-example.md`](05-add-scalarldst/costmodel-autoscope-development-scalarldst-example.md)，聚焦 CostModel 代码架构与 StageKind 扩展流程。

---

### 9.1 代码路径

#### 9.1.1 本地 Mac

```text
/Users/weijianchen/Documents/2026/triton-ascend
```

#### 9.1.2 服务器

```text
/home/c00946898/triton-ascend
```

#### 9.1.3 主要 cost model 路径

```text
third_party/ascend/costmodel/
├── include/AscendModel/
│   ├── RouteModel/
│   │   ├── StageCostModels.h
│   │   └── StageRouteCostModel.h
│   └── ...
├── lib/AscendModel/
│   ├── Analysis/
│   │   └── StagePartitioner.cpp
│   └── RouteModel/
│       ├── SimdSimtCostModel.cpp
│       ├── StageCostModels.cpp
│       └── StageRouteCostModel.cpp
├── profiles/
│   ├── microbench/
│   │   ├── ascend_davidv100_v1.json
│   │   └── data_provider/
│   │       ├── scalar_ldst/
│   │       └── ...
│   └── simd_simt/
│       ├── david_v100_simd_simt_v1.json
│       └── simd_simt_profile_schema.json
└── unittest/
    └── pytest_ut/
        └── test_scalar_dominate_costmodel_cases.py
```

---

### 9.2 CostModel for Autoscope 整体架构

#### 9.2.1 目标

对 Triton kernel 的 TTIR 做 stage 切分，估算 SIMD / SIMT / mixed 三种候选的 cycle 成本，选择最优 route。

#### 9.2.2 调用链

```text
Triton 编译
  -> SelectSimdSimtCostModelPass
    -> StagePartitioner：切分 logical stages
    -> StageFeatureAnalysis：填 StageModelFeatures
    -> StageWorkloadAnalysis：填 StageWorkload
    -> StageCostEvaluator：对每个 stage 计算资源 cycles
      -> mapWorkload()
      -> estimateStage()
      -> applySuperBlock()
    -> 汇总 candidate cost
    -> 输出 route report JSON
```

#### 9.2.3 重要文件与函数

| 文件 | 重要函数/概念 | 作用 |
|---|---|---|
| `StagePartitioner.cpp` | `classifySemanticRoot()` | 把 TTIR 操作归类到 StageCostModelKind |
| `StagePartitioner.cpp` | `accumulateOneOperation()` | 累加 scalar load/store / vector load/store / dot 等 workload |
| `StagePartitioner.cpp` | `isScalarIndirectLoadOperation()` | 判断 scalar load 地址是否依赖另一个 scalar load |
| `StageRouteCostModel.h` | `StageWorkload` | 保存 count/bytes/instructions 等 workload |
| `StageRouteCostModel.h` | `StageModelFeatures` | 保存 stage 结构特征 |
| `StageRouteCostModel.h` | `StageResourceCycles` | 保存计算出的 resource cycles |
| `StageCostModels.cpp` | `mapWorkload()` | 将 workload 映射为 resource cycles |
| `StageCostModels.cpp` | `estimateStage()` | 按 StageCostModelKind/mode 选择公式 |
| `SimdSimtCostModel.cpp` | `resolveNumberOrMeasurement()` | 解析 profile 数值或 measurement 引用 |
| `ascend_davidv100_v1.json` | measurements | 存放可复用的硬件/CAModel 实测 measurement |
| `david_v100_simd_simt_v1.json` | stage_resources | 存放 model-specific resource 参数 |

---

### 9.3 添加 Scalar Load/Store StageKind 的步骤

#### 9.3.1 增加 StageCostModelKind 枚举

文件：

```text
include/AscendModel/RouteModel/StageCostModels.h
```

新增：

```cpp
ScalarLoad,
ScalarStore,
```

#### 9.3.2 增加 workload / feature 字段

文件：

```text
include/AscendModel/RouteModel/StageRouteCostModel.h
```

`StageWorkload` 增加：

```cpp
double scalarLoadCount = 0.0;
double directScalarLoadCount = 0.0;
double indirectScalarLoadCount = 0.0;
double scalarStoreCount = 0.0;
```

`StageModelFeatures` 增加：

```cpp
bool hasScalarIndirectMemory = false;
bool hasScalarIndirectLoad = false;
bool hasScalarIndirectStore = false;
```

#### 9.3.3 在 StagePartitioner 中识别 scalar load/store

`accumulateOneOperation()` 中：

- scalar `tt.load` 且结果不是 shaped type：计入 `scalarLoadCount`；
- 若 load 地址依赖另一个 scalar load：计入 `indirectScalarLoadCount`；
- 否则计入 `directScalarLoadCount`；
- scalar `tt.store` 且 value 不是 shaped type：计入 `scalarStoreCount`；
- 若 store 地址依赖另一个 scalar load：计入 `indirectScalarStoreCount`。

#### 9.3.4 在 classifySemanticRoot 中分类

若 operation tree 只有 scalar load / store，则返回：

```cpp
StageCostModelKind::ScalarLoad
StageCostModelKind::ScalarStore
```

#### 9.3.5 增加公式

`mapWorkload()` 中新增 scalar memory 计算：

```text
scalarMemory =
    scalar_load_count / scalar_load_throughput
  + scalar_load_latency
  + indirect_scalar_load_count * scalar_indirect_dependency_latency
  + scalar_store_count / scalar_store_throughput
  + scalar_store_latency
  + indirect_scalar_store_count * scalar_indirect_dependency_latency
```

> 注意：`scalar_indirect_dependency_latency` 表示“每个 indirect edge 的额外延迟”，不是“每个 dependent load/store 的总边际成本”。  
> 如果直接按 dependent chain 的边际成本取值，会重复计入普通 scalar load/store 的 throughput 成本，即 double counting。

#### 9.3.6 增加 profile 参数

在：

```text
profiles/microbench/ascend_davidv100_v1.json
```

新增/使用 measurement：

```text
simd.scalar.load.throughput
simd.scalar.load.direct_latency
simd.scalar.store.throughput
simd.scalar.store.direct_latency
simt.scalar_gm.load.throughput
simt.scalar_gm.store.throughput
```

在：

```text
profiles/simd_simt/david_v100_simd_simt_v1.json
```

的 scalar_memory 中引用这些 measurement。

> 这些 measurement 的 CAModel 标定数据怎么跑：cce 探针 → §10.1；triton 载体 → §10.2。

#### 9.3.7 增加验证测试

新增 `third_party/ascend/unittest/pytest_ut/test_scalar_dominate_costmodel_cases.py`，包含 6 个 scalar-heavy Megablocks kernel，验证 route 选择仍为 `all_simt_only` 且数值正确。

```bash
cd ~/triton-ascend
python -m pytest -v \
  third_party/ascend/unittest/pytest_ut/test_scalar_dominate_costmodel_cases.py
```

预期：`6 passed, 18 warnings`。

---

### 9.4 构建/运行补充

- **wheel 构建的 cmake 版本**：系统 `/usr/bin/cmake` 是 3.16，不满足 `CMake >= 3.20`；构建前把服务器已有的 `/home/h00967832/.conda/envs/hyj/bin`（cmake 4.2.3）放到 `PATH` 最前（`~/build_triton_wheel6.sh` 里相应调整）。
- **手动 cmake/ninja 前先打 patch**：`cd ~/triton-ascend && git apply third_party/ascend/patch/triton-ascend-3.6.0.patch`；否则根 `CMakeLists.txt` 缺 LLVM 22 兼容宏，HIVM 编译大量报错。
- **远程 push GitHub（2026-09-17 修订）**：服务器 GitHub 时通时断（一次 `ls-remote` 成功，随后 timeout），反向隧道脚本只作 fallback；可靠路径是从 Mac 干净工作树 push（`scalar-load-whitebox` 分支，代理 `127.0.0.1:7897`，首次可能 `SSL_ERROR_SYSCALL`，重试即可）。服务器工作树 dirty 且 remote-tracking ref stale，不要直接 push。完整姿势见 §3.2。

---

### 9.5 可复用经验

1. 添加新 StageKind 时，按“枚举 → feature/workload → partitioner → formula → profile → test”顺序修改；
2. 修改 profile JSON 后，如果安装了 wheel，需要同步到 `~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/`。

---

## 10. CAModel 跑法

> 不需要物理卡；仿真吃满 CPU，跑前确认没有别人的 msprof/msopprof。
> 公共前提：
>
> ```bash
> source ~/env_ascend.sh
> ulimit -n 1048576        # 必需；默认 1024 会 Too many open files
> ```
>
> 产物/文件索引见 §10.3；执行/提交踩坑见 §10.4；结论与证据见 `02-camodel-simulator/`。
> 长任务不要用裸 `nohup ... &` 等 ssh 返回；用 tmux 挂后台并轮询日志（见 §10.2）。

### 10.1 CCE 探针（2026-09-12 实测）

文件：`02-camodel-simulator/scripts/cce_scalar_memory/`（服务器 `~/camodel-cce-test/`）：`.cce` + `host.cpp` + wrapper `.sh`。

```bash
cd ~/camodel-cce-test
source ~/env_ascend.sh
ulimit -n 1048576

INC=~/AscendNPU-IR-triton/bishengir/lib/Template/include   # fork 内 submodule 的 Template 头
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" simt_scalar_memory.cce -o simt_scalar_memory.o

g++ -O2 simt_scalar_memory_host.cpp -o simt_scalar_memory_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=2 --timeout=3 ./simt_scalar_memory_profiler.sh 2>&1 | tee run.log
```

跑对的样子：`load,...,9915` / `store,...,8203`，`All task success`，产出 `OPPROF_*/measure/{0,1}/`（`0`=load，`1`=store）。完整日志见 `02-camodel-simulator/artifacts/cce_scalar_memory_run_20260912.log`。

只记 3 个坑：

1. `ulimit` 1024 会在 host 初始化时 `Too many open files` abort。
2. wrapper 必须原样用（HAL/runtime 指到 `dav_3510/*_camodel.so`）；host 必须是新 aclrt API（`aclrtBinaryLoad/GetFunction` + `aclrtLaunchKernelWithHostArgs`），legacy `rtDevBinaryRegister` 拿不到 dump。
3. 当前 probe 实际只有 lane0 干活；`active_threads=32` 是打印标签，别当 32-lane 吞吐。

### 10.2 triton kernel（2026-09-03 实测）

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576
export SIMLIB="$ASCEND_HOME_PATH/tools/simulator/dav_3510/lib"
export BISHLIB=/home/c00946898/.conda/envs/bisheng_build/lib
export BISHLIB2=/home/c00946898/bishengir-install/lib
export PATH=/home/c00946898/bishengir-install/bin:/home/c00946898/.conda/envs/wj_autoscope/bin:$PATH

# 真卡预检（期望 all_simt_only）
cd ~ && conda run -n wj_autoscope python3 gather_tiny.py 2 8

# 仿真（~7 min）
LD_LIBRARY_PATH=$SIMLIB:$BISHLIB:$BISHLIB2:$LD_LIBRARY_PATH \
  msprof op simulator --kernel-name=gather_dot_min python3 gather_tiny.py 2 8

# 解析
cd ~/microbench-run
python3 camodel/parse_camodel_counts.py <OPPROF根目录> -o parsed_gather.json
python3 camodel/extract_camodel_system_cycle_profile.py parsed_gather.json \
  --simulator-clock-mhz 1650.0 --sys-cnt-mhz 988.9 --scope gather_tiny > parsed_gather_rates.json
```

**后台跑（重要）**：共享服务器上直接 `ssh ... 'nohup msprof op simulator ... &'` 时，ssh 会话可能一直不释放，本地连续超时；这不是仿真失败。改用 tmux 挂后台，再轮询日志：

```bash
# 先准备 run_xxx.sh：内部 source ~/env_ascend.sh、conda activate、ulimit、msprof ...
ssh ascend-950pr-63 'tmux new-session -d -s camodel -c "$HOME/<workdir>" "bash run_xxx.sh > run.log 2>&1"'

# 轮询：tmux 会话在 + 有 msprof 进程 + run.log 持续输出 = 正常
ssh ascend-950pr-63 'tmux ls; pgrep -af "[m]sprof op simulator"; tail -5 $HOME/<workdir>/run.log'
# run.log 出现 Start profiling on kernel -> All task success -> Profiling results saved 即完成
# 完成后清理会话：ssh ascend-950pr-63 'tmux kill-session -t camodel'
```

坑：`LD_LIBRARY_PATH` 少了 `$BISHLIB` 会假报 `No profiling data matched`；输入要 CPU 构造后 `.to("npu")`；`--kernel-name` 是前缀匹配；SIMT 要 hint（vector core=56）；常规尺寸 SIMT 太慢，只用 tiny。

### 10.3 文件与产物

- 本地 `02-camodel-simulator/`：`scripts/cce_scalar_memory/`、`scripts/gather_tiny.py`、`artifacts/cce_scalar_memory_run_20260912.log`、`artifacts/parsed_gather*.json`
- 本地 `10-scalar-camodel-bench/`：scalar load/store round-1 benchmark 源码、CAModel 结果包、`ANALYSIS.md`
- 本地 `10-scalar-camodel-bench/camodel_full_outputs/`：完整原始 `OPPROF_*`（cce 路线、load scalar_o1/o4、store_scalar_o1/o4 等；**不入 git**）
- 服务器：`~/camodel-cce-test/`（cce 成功产物 `OPPROF_20260912165738_*`）、`~/microbench-run/`、`~/OPPROF_*`
- scalar load/store 白盒报告（2026-09-17 已推，远程分支当前 `244b7eebe`）：分支 `scalar-load-whitebox` 的
  `third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst_whitebox/README.md`；
  结论：SIMD MainScalar 447 cycles（4+440+3），SIMT 32T 530（21+477+32），SIMT 1T 483（21+430+32）；
  BIU fill 差是 run/双核仲裁的观察值，**不要归因于 `GSU_I2_OUT target_size`**（见 §10.4 第 8 条）。

### 10.4 执行/提交踩坑（2026-09-17 实测）

1. **ssh 嵌套 heredoc 会假死**：在 `ssh host '...'` 的单引号命令里再写多行 `<<'EOF'` heredoc 时，
   EOF 可能被外层 shell/agent 工具吞掉，表现为一直等待、300s 超时——不是远端命令本身慢。
   正确做法：在 Mac 本地用编辑器写文件、再 `scp` 过去；少量内容用 `printf '%s\n' ... | ssh host 'cat > f'` 或 `tee`。
   在 agent 的持久 shell 里也别一次写超大 heredoc，超时后 shell 会被 reset，后续命令状态易乱。
2. **`--launch-count=1` 的 OPPROF 结构不同**：只跑一个 kernel 时产物是 `OPPROF_*/{dump,simulator}`（扁平结构），
   不是多 kernel 的 `OPPROF_*/<kernel>/0/{dump,simulator}`。解析脚本要同时兼容两种。
3. **SIMT mix kernel 会同时在两个 AIV subcore 上执行**：`core0.veccore0` 与 `core0.veccore1` 共享 BIU，
   绝对 fill 周期受仲裁顺序影响。做单 op 标定时固定看 `core0.veccore0`，并保留 `veccore1` 做 control；
   32T/1T 的微指令结构差异稳定，但 fill 绝对值跨 run 可能波动几十~上百 cycle；
   **`GSU_I2_OUT` 发生在 BIU 数据回来之后，不能用它解释 fill 差**（路径见同节第 8 条）。
4. **macOS `tar` 传文件会带 `._*` AppleDouble**：从 Mac 打 tar/scp 到服务器后，`find <dir> -name '._*' -delete` 清理；或者 `COPYFILE_DISABLE=1 tar ...`。
5. **批量/大文件 `scp` 中断（2026-09-17 实测）**：一次 `scp -r` 传多个 OPPROF 目录时，传到一半会话中断、本地工具 300s 超时，
   目标端只留下不完整目录；继续逐个 `scp -r` 重试仍不稳。
   可靠姿势：**先在服务器把要传的东西 `tar -czf /tmp/xxx.tgz <dirs...>` 打成单个归档**（本次 6 个 OPPROF ~40MB → 压缩后 1.7MB），
   再 `scp host:/tmp/xxx.tgz /tmp/` 一次传回 Mac，最后本地解压；小归档通常秒级，单个大文件也可断点/重试。
   解压后必须核对目录层级和目标路径：tar 里的相对路径 `scalar_o1/...` 会解到当前目录下，本次曾误留顶层 `scalar_o1/`、`scalar_o4/`
   两个 staging 目录，确认后再移动到 `store/scalar_o1`、`store/scalar_o4` 并删除 staging。
6. **raw dump 带行尾空格**：对 `camodel_results/*.dump`、`camodel_simt1.log` 跑 `git diff --cached --check` 会全红。
   不要为了过 check 去改原始证据；提交原始 dump 时跳过 whitespace check，或在 `.gitattributes` 里给 `*.dump` / `*.log`
   标 `-diff`。
7. **push 别从服务器工作树发**：服务器 `triton-ascend` 工作树有 36 modified / 15 deleted / 45 untracked，remote ref 也可能 stale；
   从 Mac 的干净 `scalar-load-whitebox` 分支 push，失败重试（见 §3.2）。
8. **`GSU_I2_OUT` 不是 BIU/DCache**：它是 GSU 向 UB 接口发出的访问。SIMT_LDG 的路径是
   `BIU send_rd_cmd/recv_biu_data`（GM→128B line 数据返回）→ `DC_BHU_WR` 把 128B line 写入 UB staging
   （本 case `addr=0x32000`）→ `DC_MROB_RD` / `GSU_I2_OUT` 从 UB 读出、经 RWDB 走 SIMT 寄存器回写路径。
   所以 32T/1T 的 `GSU_I2_OUT target_size`（128 vs 4）是 BIU 之后的事，不能用来解释 BIU fill 周期变化；
   `biu.brif.log.dump` 的 `send_rd_cmd` / `recv_biu_data` 才是 GM→BIU 的 line 数据返回。

---

## 11. CostModel 调试经验：stage 划分与 SuperBlock 候选打分观测（2026-09-20）

> 目标：不只看最终 JSON 的 `selected_superblock_factor`，而是能看到**实际 stage 划分**、**每个 SIMT factor 候选（含落选）的 per-stage / route total**。
> 本节记录本次实测用过的 cherry-pick、构建、运行方式与坑，配套脚本在服务器 `/home/c00946898/`。

### 11.1 需要 cherry-pick 的提交

基准：`upstream/feature/simd-simt-compile-mode` 最新 tip（本次 `1f2666afb`，含 #2149/#2112），再 cherry-pick PR2305 `66744e10b`。

从 `generic-stage-partition-kx` 系列带来 3 类东西：

| 提交/内容 | 作用 |
|---|---|
| `ed4bfc7856 feat(costmodel): add COSTMODEL trace logging and file redirect on generic-stage-partition-kx` | 新增 `AscendModel/CostModelTrace.h`，支持 `COSTMODEL_LOG_LEVEL` / `COSTMODEL_LOG_FILE`，在 partitioner / route solver / evaluator 等位置加 trace |
| `6620e26c25 feat(costmodel): log per-stage root operation mapping` | 在 `StageBoundaryAnalysis::analyze` 后打印每个 stage 的 `kind / iter / ops=[root[i] tt.xxx]`，用于看 stage 划分变化 |
| `dd8d80ee3 debug(costmodel): dump all all-SIMT SuperBlock factor candidates` | 本工作 debug 提交：`StageCostModelSummary` 保存所有 all-SIMT F 的 `StageRoutePlan`，`toJSON()` 输出 `stage_model.routes.all_simt_only_by_factor` |

⚠️ `ed4bfc7856` 基于旧 pass API，带 `preLayoutModulePath` / `postLayoutModulePath` 两个 option；最新 feature 分支的 `Passes.td` / `triton_ascend.cc` 已删掉它们，直接整提交 cherry-pick 会在 `SelectSimdSimtCostModel.cpp` 报 undeclared identifier。处理办法：

1. 保留最新分支的 pass signature/options，只移植 `CostModelTrace.h` + `COSTMODEL_TRACE/costModelLog/costModelDebug` 调用；
2. 单独一个修复提交 `772403ace fix(costmodel): drop pre/post-layout pass options after logging port`。

本次工作分支（可对照）：

```text
本地 worktree：/private/tmp/triton-sb64-latest   分支 debug/sb64-latest
本地 worktree：/private/tmp/triton-sb64-pr2305   分支 debug/sb64-pr2305（含 PR2305 + logging + 候选 dump + cap fix）
远程：origin/debug/sb64-pr2305                   tip 1af05641b
服务器：~/triton-ascend-sb64-latest              运行 env ~/sb64_latest_env
```

### 11.2 构建/装配可运行 triton（重点：避免卡 linker）

完整 wheel：

```bash
cd ~/triton-ascend-sb64-latest
source ~/env_ascend.sh
export MAX_JOBS=4~8
python3 setup.py bdist_wheel
```

共享机器上链接 `libtriton.so` / `triton-opt` 时 ninja 可能长时间 0% CPU；等超过 3–5 分钟建议换增量法：

```bash
cd ~/triton-ascend-sb64-latest
source ~/env_ascend.sh
export PATH=/home/h00967832/.conda/envs/hyj/bin:/home/s00653124/Softwares/ccache:/home/c00946898/.conda/envs/bisheng_build/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/.conda/envs/bisheng_build/lib:$LD_LIBRARY_PATH
export CCACHE_DIR=/home/c00946898/.ccache
ninja -C build/cmake.linux-x86_64-cpython-3.11 -j6 libtriton.so
cp build/lib.linux-x86_64-cpython-311/triton/_C/libtriton.so ~/sb64_latest_env/triton/_C/
chmod +x ~/sb64_latest_env/triton/_C/libtriton.so
```

手工拼 env 时必须把最新源码映射补齐，否则 `import triton` 会 fallback 到 site-packages
（旧 whitebox 包 → stage 变 `scalar_load`、cap 只剩 F1/2/4，跑出假的 F4 结论）：

```text
third_party/ascend/backend/*        -> sb64_latest_env/triton/backends/ascend/
third_party/ascend/language/cann/*  -> sb64_latest_env/triton/language/extra/cann/
third_party/ascend/backend/lib/libdevice.10.bc -> .../triton/backends/ascend/lib/
```

校验实际 import：

```bash
PYTHONPATH=~/sb64_latest_env python3 -c \
  'import triton, triton.backends.ascend.utils as u; print(triton.__file__); print(u._get_modeled_superblock_factors())'
# 预期：~/sb64_latest_env/triton/__init__.py 和 (1,2,4,8,16,32,64)
```

### 11.3 用 report 模式跑单个编译配置

核心环境变量（跑前设置；`run_sb64_six_case.py` 内即这套）：

```python
os.environ["TRITON_ASCEND_COMPILE_MODE"] = "simd_simt"
os.environ["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "report"
os.environ["TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"] = str(REPORT)   # report JSON
os.environ["TRITON_CACHE_DIR"] = str(CACHE)
os.environ["TRITON_CACHE_AUTOTUNING"] = "0"
os.environ["COSTMODEL_LOG_LEVEL"] = "2"      # 2 才有 costModelDebug 逐项细节
os.environ["COSTMODEL_LOG_FILE"] = str(LOG)  # [COSTMODEL] trace
```

然后把 autotuner 截成一个 config，确保目标 BX / NW 被编译：

```python
at = getattr(mod, autotuner_name)
at.configs = [triton.Config({"BLOCK_X": bx, "superblock_factor": 1}, num_warps=nw)]
at.cache.clear()
```

并通过包装 `JITFunction.run` 注入 `physical_vector_core_count_hint=56`。服务器现成脚本：

```text
~/run_sb64_six_case.py    # --kernel/--block-x/--num-warps/--outdir，六 kernel 通用
~/run_sb64_candidates.py  # wgrad 单 case
~/sb64_six_run_cap.sh     # 批量跑 6 个 case
```

`report` mode 只打分/报告，`effective_decision_kind=backend_default`，不会真正应用选中的 factor。

### 11.4 看 stage 划分

```bash
grep -E 'stage ".*" kind=' costmodel.log
```

会得到：

```text
[COSTMODEL] stage "stage_0_auto_blockify_dispatch" kind=auto_blockify_dispatch iter=1 ops=[root[0] tt.get_num_programs{schedule}, ...]
[COSTMODEL] stage "stage_12_loop_carried_recurrence" kind=loop_carried_recurrence iter=3 ops=[root[43] scf.for]
[COSTMODEL] stage "stage_16_scalar_store" kind=scalar_store iter=1 ops=[root[47] tt.store]
```

同一份信息也在 `report.json` 的：

```text
stage_model.logical_stages[].id / model / iteration_count / workload / source_locations
```

### 11.5 看不同 factor 的打分

1）每个 implementation 的 per-factor 分：

```bash
grep -E 'impl (SIMD|SIMT) F=' costmodel.log
# [COSTMODEL] impl SIMT F=16 local=no cycles=1920
```

2）每条 route 在某个 F 下的总 cycles：

```bash
grep -E 'route=all_simt_only F=' costmodel.log
# [COSTMODEL] route=all_simt_only F=16 totalCycles=1.785848e+04 legal=true
```

3）落选 factor 的完整逐 stage plan：

```text
report.json -> stage_model.routes.all_simt_only_by_factor[]
  .route_superblock_factor
  .runtime_wave_count
  .stages[].logical_stage_system_cycles
```

该字段只有 `dd8d80ee3` debug 提交后才有；没有时 `routes` 只有 selected 的 `all_simd / all_simt_only / mixed_simd_simt` 三条。

### 11.6 坑

1. **`COSTMODEL_LOG_FILE` 懒创建**：只有真正跑到 costmodel pass 才会创建文件；编译失败/命中缓存都不会有新日志。
2. **`TRITON_ASCEND_AUTO_SIMT_PROFILE` 不要指向 /tmp 下的 profile**：profile 里的 `microbenchmark_profile` 是相对路径，复制到 /tmp 会报 `failed to read microbenchmark profile`；要改就复制到原 profile 同目录再改。
3. **每次换配置先删 `TRITON_CACHE_DIR`**，否则 report dump 会复用旧编译缓存。
4. **确认 import 路径**：`triton.__file__` 必须是自己的 build/env，不是 site-packages；否则 stage 划分和 factor 候选都会是错的包。
5. **日志 level**：默认 level=1 只有调用图；逐 stage feature、impl cost、maxFactor 需要 `COSTMODEL_LOG_LEVEL=2`。
6. **不要把 `ed4bfc7856` 的 pass option 改动一起 cherry-pick 进来**：最新 feature 分支没有 pre/post-layout option，会把 C++ 编译搞挂。
7. **`report` mode 不真正应用 factor**：想看运行效果/性能必须用 `auto` 或手写 config，不能把 report JSON 的 selected 当执行结果。

