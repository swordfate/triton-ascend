#!/bin/bash

source ${WORKSPACE}/triton_ascend/test/script/setenv.bash
source /data/set_proxy.bash
script=$(readlink -f "$0")
script_dir=$(dirname "$script")

CUR_PATH=$(cd $(dirname $0);pwd)
cd ${CUR_PATH}
rm -rf mojo_opset_950perf
source /data/set_proxy.bash
git config --global http.sslverify false
git clone -b hw/950-perf https://zhiliangtang0727:hpyNCursTuTNMY1sq4bNxNTw@gitcode.com/TritonAscendTest/mojo_opset mojo_opset_950perf

#cp run_mojo_accuracy_test.sh ./mojo_opset/
#cp testcase_a3.txt ./mojo_opset/
cd ./mojo_opset_950perf
cp -r /data/PytorchFile/Mojo_opset/dllm ./mojo_opset/tests/specific_shape_perf_acc/test_data/
pip install -e .
pip install transformers==5.3.0
pip install einops

LIST_FILE=${CUR_PATH}/testcase_hw_950_perf.txt
# 黄区 WORKSPACE ：/home/CI
TEST_CODE="${WORKSPACE}/triton_ascend"
TEST_DIR="${WORKSPACE}/test"
# define summary file path
SUMMARY_FILE="${TEST_CODE}/examples/summary.txt"
TESTSUITEDIR_NAME=$(ls -d ${CUR_PATH} | xargs -I {} basename {})
#OP_LOG_DIR="${TEST_DIR}/logs/${TESTSUITEDIR_NAME}"
OP_LOG_DIR="${TEST_DIR}/logs/mojo_opset_950perf"

# 清理旧日志
rm -rf ${OP_LOG_DIR} && mkdir -p ${OP_LOG_DIR}

function run_case_by_multi_card() {
    NPU_DEVICES=$(ls /dev/davinci? 2>/dev/null | wc -l)
    [ $NPU_DEVICES -eq 0 ] && {
        echo "No Ascend devices found!"
        exit 1
    }

    echo "Detected $NPU_DEVICES Ascend devices"

    test_dir=$1
    cd ${test_dir}

    # 清理旧日志
    rm -rf logs && mkdir logs

    export PYTHONPATH=./:$PYTHONPATH

    # 记录测试开始时间
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "===== 测试开始时间: ${start_time} ====="

    fifo="/tmp/$$.fifo"
    mkfifo $fifo
    exec 9<>$fifo
    rm -f $fifo

    if [ -n "$DEVICE_ID" ]; then
        IFS=',' read -ra ids <<< "$DEVICE_ID"
        for id in "${ids[@]}"; do
          echo "$id" >&9
        done
    else

        npu_number=5
#        for ((id=0; id<$npu_number; id++)); do
#            echo "$id" >&9
#        done
        echo "4" >&9
        echo "2" >&9
        echo "3" >&9
    fi

    while IFS= read -r -d '' line; do
        # 去前后空格、跳过空行/注释
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z $line || $line == \#* ]] && continue

        base_id="${line%.py}"          # 去掉 .py
        safe_name="${base_id//::/_}"  # :: → _
        safe_name="${base_id////_}"  # :: → _
        # 2. 按旧目录、旧后缀落盘
        test_log="${OP_LOG_DIR}/${safe_name}.log"
        test_xml="${OP_LOG_DIR}/${safe_name}_results.xml"

        read -u 9 npu_id
        {
#            echo "[DEBUG] $$(date +%F_%T) NPU$npu_id  PWD=$(pwd)  CMD=pytest -sv $line -n 8 --junitxml=$test_xml"
            \cp ${CUR_PATH}/../script/template.xml $test_xml
            TRITON_CACHE_DIR=$(pwd)/cache/${line} ASCEND_RT_VISIBLE_DEVICES=$npu_id MOJO_Backend=ttx pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" -sv "$line" --junitxml="$test_xml" > "$test_log" 2>&1
            echo "$npu_id" >&9
        } &

        echo "[INFO] Activated '$line' on NPU $npu_id, PID=$!, logging into $test_log."

    done < <(tr '\n' '\0' < "$LIST_FILE")
    wait
    exec 9>&-

    echo "[INFO] All test processes completed"
    find $(pwd)/cache/ -type f ! -name "*.ttadapter" ! -name "*.ttir" -delete

    # 新增：解析测试结果统计
    total_tests=0
    passed_tests=0
    failed_tests=0
    skipped_tests=0
    error_tests=0

    # 使用Python解析JUnit XML报告
    python3 -c "
import xml.etree.ElementTree as ET
import os
import glob
import sys

# 查找所有XML报告文件
xml_files = glob.glob(os.path.join('${OP_LOG_DIR}', '*.xml'))
print(f'Found {len(xml_files)} XML report files', file=sys.stderr)

if not xml_files:
    print('No JUnitXML reports found', file=sys.stderr)
    print('total_tests=0')
    print('passed_tests=0')
    print('failed_tests=0')
    print('skipped_tests=0')
    print('error_tests=0')
    exit(1)

total = 0
passed = 0
failed = 0
skipped = 0
errors = 0

# 遍历所有XML文件并统计
for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for testsuite in root.findall('testsuite'):
            total += int(testsuite.get('tests', 0))
            passed += int(testsuite.get('tests', 0)) - int(testsuite.get('errors', 0)) - int(testsuite.get('failures', 0)) - int(testsuite.get('skipped', 0))
            failed += int(testsuite.get('failures', 0))
            skipped += int(testsuite.get('skipped', 0))
            errors += int(testsuite.get('errors', 0))
    except Exception as e:
        print(f'Error parsing {xml_file}: {e}', file=sys.stderr)
        continue

print(f'total_tests={total}')
print(f'passed_tests={passed}')
print(f'failed_tests={failed}')
print(f'skipped_tests={skipped}')
print(f'error_tests={errors}')
" > logs/stats.tmp

    # 记录测试结束时间
    source logs/stats.tmp
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "end_time:${end_time}"
    duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
    duration_str=$(printf "%02dh %02dm %02ds" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)))
    echo "duration:${duration_str}"
# 构造 JSON 字符串
    json_output=$(cat <<EOF
Test Report:
{
    "start_time": "${start_time}",
    "duration": "${duration_str}",
    "log_dir": "${OP_LOG_DIR}",
    "total_op": "${total_tests}",
    "passed_op": "${passed_tests}",
    "failed_op": "$((failed_tests + error_tests))"
}
EOF
)

    # 加载统计结果
    source logs/stats.tmp
    rm logs/stats.tmp
    # 去掉换行，变成单行输出
    echo "$json_output" | tr '\n' ' '
    echo ""
    # 新增：生成统计摘要
    stats_summary="
===== generalization_cases测试统计摘要 =====
测试目录:       $(basename ${test_dir})
测试开始时间:   ${start_time}
测试结束时间:   ${end_time}
总耗时:         ${duration_str}
------------------------
总用例数:       ${total_tests}
成功用例:       ${passed_tests}
失败用例:       ${failed_tests}
跳过用例:       ${skipped_tests}
错误用例:       ${error_tests}
成功率:         $(( passed_tests * 100 / (passed_tests + failed_tests + error_tests) ))% (成功/(成功+失败+错误))
设备数量:       ${NPU_DEVICES}
========================
"

    # 输出统计信息到控制台
    echo "${stats_summary}"

    # 追加统计信息到summary.txt
    echo "${stats_summary}" >> ${SUMMARY_FILE}

    echo "========================================"
    echo "All tests completed!"
    echo "JUnit Reports: ${OP_LOG_DIR}/*.xml"
    echo "Merged JUnit Report: logs/generalization_results.xml"
    echo "Combined Log: logs/combined.log"
    echo "统计摘要已追加到: ${SUMMARY_FILE}"
    echo "========================================"

    cd ${test_dir}/logs
    if [[ -d "${test_dir}/logs" ]]; then
        cp -r ${test_dir}/logs/* ${OP_LOG_DIR}
    fi

    # 返回pytest的退出状态
    return $pytest_exit
}

# build in torch 2.6.0
source ${WORKSPACE}/triton_ascend/test/script/setenv.bash
sleep 10

env
pip list
echo "========================================================="

cd ${WORKSPACE}

# 初始化统计文件
echo "生成时间: $(date +"%Y-%m-%d %H:%M:%S")" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# run gene case
TEST_generalization="${CUR_PATH}/mojo_opset_950perf"
run_case_by_multi_card ${TEST_generalization}

\cp *operator_perf.csv ${OP_LOG_DIR}/

cd ${CUR_PATH}
python ${CUR_PATH}/../script/perf_statistics.py
\cp npu_performance_summary.csv ${OP_LOG_DIR}/mojo_opperf.csv

echo "========================================" >> ${SUMMARY_FILE}
