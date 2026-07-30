#!/usr/bin/env python3
"""
自动收集昇腾性能数据，与 GPU 数据进行对比。
优化版 V6：
- benchmark.csv 增加 ratio 列，作为通过阈值
- 输出文件带日期，放在 ligerkernel_perf 目录
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# ========== 配置路径 ==========
#BASE_DIR = Path("/home/CI/20260320043149/triton_ascend/test/ligerkernel_operator_cases")
BASE_DIR = Path.cwd()
PERF_RESULT_DIR = BASE_DIR / "perf_result"
BENCHMARK_FILE = BASE_DIR / "benchmark.csv"
OUTPUT_DIR = BASE_DIR / "ligerkernel_perf"

# 生成带日期的输出文件名
date_suffix = datetime.now().strftime("%Y%m%d")
OUTPUT_FILE = OUTPUT_DIR / f"LigerKernel_benchmak_result_{date_suffix}.csv"
SUMMARY_FILE = OUTPUT_DIR / f"LigerKernel_op_statistic_{date_suffix}.csv"
# =================================

def detect_delimiter(file_path):
    """自动检测 CSV 文件的分隔符（制表符或逗号）"""
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            raise ValueError(f"无法识别 {file_path} 的分隔符")

def clean_str(s):
    """去除字符串两端的空白和引号"""
    return s.strip().strip('"\'')

def main():
    # 创建输出目录（如果不存在）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查输入文件
    if not BENCHMARK_FILE.exists():
        print(f"错误：未找到 {BENCHMARK_FILE}")
        sys.exit(1)
    if not PERF_RESULT_DIR.exists() or not PERF_RESULT_DIR.is_dir():
        print(f"错误：性能结果目录 {PERF_RESULT_DIR} 不存在")
        sys.exit(1)

    # 1. 读取 benchmark.csv，构建记录列表（增加 ratio 列）
    delim = detect_delimiter(BENCHMARK_FILE)
    print(f"benchmark.csv 分隔符为：{repr(delim)}")

    benchmark_records = []          # 存储所有 benchmark 行
    benchmark_file_to_script = {}   # 键：benchmark_file，值：脚本名（加 .py）

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delim)
        required_cols = ['op_kernel', 'gpu_perf', 'benchmark_file', 'mapping_op', 'ratio']
        if not all(col in reader.fieldnames for col in required_cols):
            print(f"错误：benchmark.csv 必须包含列：{required_cols}")
            sys.exit(1)

        for row in reader:
            clean_row = {k: clean_str(v) for k, v in row.items()}
            benchmark_records.append(clean_row)
            bf = clean_row['benchmark_file']
            if bf:
                benchmark_file_to_script[bf] = bf + '.py'

    print(f"从 benchmark.csv 读取到 {len(benchmark_records)} 条记录。")

    # 2. 汇总所有 op_statistic*.csv 文件，提取 benchmark_dir
    print(f"正在从 {PERF_RESULT_DIR} 汇总 op_statistic*.csv 文件...")
    all_op_files = list(PERF_RESULT_DIR.rglob("op_statistic*.csv"))
    if not all_op_files:
        print("错误：未找到任何 op_statistic*.csv 文件")
        sys.exit(1)
    print(f"找到 {len(all_op_files)} 个文件。")

    all_records = []  # 存储 (op_type, avg_time, source_file, benchmark_dir)
    for csv_file in all_op_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) < 7:
                        continue
                    op_type = clean_str(row[1])      # 第2列：OP Type
                    avg_time = clean_str(row[6])     # 第7列：Avg Time(us)
                    if op_type and avg_time:
                        rel_path = csv_file.relative_to(PERF_RESULT_DIR)
                        benchmark_dir = rel_path.parts[0]  # 第一级目录名
                        all_records.append((op_type, avg_time, str(rel_path), benchmark_dir))
        except Exception as e:
            print(f"警告：处理文件 {csv_file} 时出错：{e}")

    print(f"共收集到 {len(all_records)} 条算子记录。")

    # 3. 写入汇总文件 LigerKernel_op_statistic.csv，新增 op_kernel_scripts 列
    all_records.sort(key=lambda x: (x[3], x[0]))
    with open(SUMMARY_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["OP Type", "Avg Time(us)", "Source File", "benchmark_dir", "op_kernel_scripts"])
        for op, avg, src, bdir in all_records:
            script_name = benchmark_file_to_script.get(bdir, bdir + '.py')  # 默认回退
            writer.writerow([op, avg, src, bdir, script_name])
    print(f"汇总数据已保存至 {SUMMARY_FILE}")

    # 4. 构建查找字典：(benchmark_dir, op_type) -> avg_time (取第一个出现)
    op_avg_dict = {}
    for op, avg, _, bdir in all_records:
        key = (bdir, op)
        if key not in op_avg_dict:
            op_avg_dict[key] = avg

    # 5. 对比生成结果（使用 benchmark.csv 中的 ratio 作为阈值）
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["op_kernel", "gpu_perf", "mapping_op", "op_kernel_scripts", "npu_perf", "ratio", "result"])

        for row in benchmark_records:
            op_kernel = row['op_kernel']
            gpu_perf = row['gpu_perf']
            benchmark_file = row['benchmark_file']
            mapping_op = row['mapping_op']
            ratio_threshold = row['ratio']   # 从 CSV 读取阈值
            op_kernel_scripts = benchmark_file + '.py'

            # 查找 NPU 耗时
            key = (benchmark_file, mapping_op)
            npu_perf = op_avg_dict.get(key, "N/A")

            if npu_perf == "N/A":
                ratio = "N/A"
                result = "faild"
            else:
                try:
                    gpu_val = float(gpu_perf)
                    npu_val = float(npu_perf)
                    ratio_val = gpu_val / npu_val
                    ratio = f"{ratio_val:.6f}"
                    # 使用阈值判断
                    threshold = float(ratio_threshold)
                    result = "pass" if ratio_val > threshold else "faild"
                except ValueError:
                    ratio = "N/A"
                    result = "faild"

            writer.writerow([op_kernel, gpu_perf, mapping_op, op_kernel_scripts, npu_perf, ratio, result])

    print(f"对比完成，结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
