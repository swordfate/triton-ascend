#!/bin/bash

CUR_PATH=$(cd $(dirname $0);pwd)
source ${CUR_PATH}/../common/common.sh
work_dir=${CUR_PATH}

cd ${work_dir}/Liger_Ascend/
source /data/set_proxy.bash
python setup.py install
cd ${work_dir}
cp ${work_dir}/perf_extra/* ${work_dir}/Liger_Ascend/benchmark/scripts/

# ==================== 路径配置 ====================
BASE_DIR="${work_dir}"
SCRIPT_DIR="${BASE_DIR}/Liger_Ascend/benchmark/scripts"
OUTPUT_ROOT="${BASE_DIR}/perf_result"
PERF_LIST="${BASE_DIR}/perf_list.txt"
LOG_FILE="${BASE_DIR}/ligerkernel_perf/msprof_run_$(date +%Y%m%d).log"
# =================================================

# 创建日志目录（如果不存在）
mkdir -p "$(dirname "$LOG_FILE")"

# 清空或创建日志文件
> "$LOG_FILE"

# 检查 msprof 命令是否存在
if ! command -v msprof &> /dev/null; then
    echo "错误: 未找到 msprof 命令，请确保昇腾 AI 处理器软件包已安装并正确配置环境。" | tee -a "$LOG_FILE"
    exit 1
fi

# 检查 perf_list.txt 是否存在
if [ ! -f "$PERF_LIST" ]; then
    echo "错误: 未找到 $PERF_LIST 文件。" | tee -a "$LOG_FILE"
    exit 1
fi

# 创建输出根目录（如果不存在）
mkdir -p "$OUTPUT_ROOT"

echo "开始批量执行 msprof profile ..." | tee -a "$LOG_FILE"
echo "日志将写入: $LOG_FILE" | tee -a "$LOG_FILE"
echo "性能数据将保存在: $OUTPUT_ROOT 下各子目录中" | tee -a "$LOG_FILE"
echo "将从 $PERF_LIST 读取待执行的脚本列表" | tee -a "$LOG_FILE"

# 逐行读取 perf_list.txt 中的脚本文件名，去除回车符和首尾空白
while IFS= read -r line || [ -n "$line" ]; do
    # 去除行首行尾的空白字符（包括空格、制表符），并删除Windows换行符 \r
    pyfile_name=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/\r$//')
    # 跳过空行
    if [ -z "$pyfile_name" ]; then
        continue
    fi

    # 脚本的完整路径
    pyfile="${SCRIPT_DIR}/${pyfile_name}"

    # 检查文件是否存在
    if [ ! -f "$pyfile" ]; then
        echo "警告: 脚本文件 $pyfile 不存在，跳过。" | tee -a "$LOG_FILE"
        continue
    fi

    # 提取文件名（不含路径和 .py 扩展名）作为输出子目录名
    base_name=$(basename "$pyfile_name" .py)
    output_dir="${OUTPUT_ROOT}/${base_name}"

    echo "========================================" | tee -a "$LOG_FILE"
    echo "正在处理: $pyfile_name" | tee -a "$LOG_FILE"
    echo "输出目录: $output_dir" | tee -a "$LOG_FILE"
    echo "执行命令: msprof --output=\"$output_dir\" python \"$pyfile\"" | tee -a "$LOG_FILE"

    # 执行 msprof，将 stdout 和 stderr 都追加到日志文件
    msprof --output="$output_dir" python "$pyfile" >> "$LOG_FILE" 2>&1

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "完成: $pyfile_name" | tee -a "$LOG_FILE"
    else
        echo "错误: $pyfile_name 执行失败，请查看日志。" | tee -a "$LOG_FILE"
    fi
done < "$PERF_LIST"

echo "========================================" | tee -a "$LOG_FILE"
echo "所有任务执行完毕。" | tee -a "$LOG_FILE"

# 执行对比脚本（假设 compare_perf.py 位于 SCRIPT_DIR）
COMPARE_PY="${BASE_DIR}/compare_perf.py"
if [ -f "$COMPARE_PY" ]; then
    echo "开始执行性能对比脚本: $COMPARE_PY" | tee -a "$LOG_FILE"
    python "$COMPARE_PY" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "性能对比完成。" | tee -a "$LOG_FILE"
    else
        echo "性能对比脚本执行失败，请查看日志。" | tee -a "$LOG_FILE"
    fi
else
    echo "警告: 未找到对比脚本 $COMPARE_PY，跳过。" | tee -a "$LOG_FILE"
fi

echo "========================================" | tee -a "$LOG_FILE"
echo "脚本执行完毕。" | tee -a "$LOG_FILE"
