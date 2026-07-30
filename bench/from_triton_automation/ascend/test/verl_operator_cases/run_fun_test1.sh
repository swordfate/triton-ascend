#!/bin/bash
#export WORKSPACE=/home/t30059325
CUR_PATH=$(cd $(dirname $0);pwd)
cd ${CUR_PATH} #/home/CI/20260227000019/triton_ascend/test/test_sglang_cases
rm -rf triton-ascend-kernels
source /data/set_proxy.bash
git config --global http.sslverify false
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
\cp  newtest_cases/test_verl_linear_cross_entropy.py triton-ascend-kernels/tests/loss/test_verl_linear_cross_entropy.py
#\cp  newtest_cases/verl_cross_entropy_kernels.py triton-ascend-kernels/src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py

TESTCASE_DIR=${CUR_PATH}/triton-ascend-kernels/

script=$(readlink -f "$0")
script_dir=$(dirname "$script")

#CUR_PATH=$(cd $(dirname $0);pwd)
# 黄区 WORKSPACE ：/home/CI
TEST_CODE="${WORKSPACE}/triton_ascend"
TEST_DIR="${WORKSPACE}/test"
# define summary file path
TESTSUITEDIR_NAME=$(ls -d ${CUR_PATH} | xargs -I {} basename {})
OP_LOG_DIR="${TEST_DIR}/logs/${TESTSUITEDIR_NAME}"
SUMMARY_FILE="${OP_LOG_DIR}/summary.txt"

SGLANG_PT_DIR="/data/PytorchFile/Vllm_pt/vllmptFile"
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

    # 记录测试开始时间
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "===== 测试开始时间: ${start_time} ====="

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
        id=3
    fi
    npu_id=$id

#    log_name="${OP_LOG_DIR}/test_verl_linear_cross_entropy"
#    #_backward: BackwardEnum = BackwardEnum._Total_Separate
#    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_kernel_general_mainloop_epilogue"
    #_backward: BackwardEnum = BackwardEnum._Total_Separate
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_backward_kernel_d_weight"
    #_backward: BackwardEnum = BackwardEnum._Total_Separate
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_efficient_entropy_backward_kernel_d_weight -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_backward_kernel_d_hidden"
    #_backward: BackwardEnum = BackwardEnum._Total_Separate
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_efficient_entropy_backward_kernel_d_hidden -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_triton_kernel_epilogue_tp"
    #_backward: BackwardEnum = BackwardEnum._Total_Separate
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_epilogue_tp -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_triton_epilogue_tp_update"
    #_backward: BackwardEnum = BackwardEnum._Total_Separate
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory" tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_epilogue_tp_update -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1


#    log_name="${OP_LOG_DIR}/test_verl_linear_cross_entropy_Total_Fuse_MN"
#    backward='_Total_Fuse_MN'
#    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
#    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_backward_kernel_general_mainloop_MN"
    backward='_Total_Fuse_MN'
    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1

#    log_name="${OP_LOG_DIR}/test_verl_linear_cross_entropy_Total_Separate"
#    backward='_Total_Separate'
#    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
#    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_backward_kernel_general_d_logits"
    backward='_Total_Separate'
    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1

#    log_name="${OP_LOG_DIR}/test_verl_linear_cross_entropy__Split_Dlogits_N"
#    backward='_Split_Dlogits_N'
#    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
#    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1
    log_name="${OP_LOG_DIR}/test_efficient_entropy_backward_kernel_general_d_logits_split_N"
    backward='_Split_Dlogits_N'
    sed -i "s/_backward: BackwardEnum = BackwardEnum.*$/_backward: BackwardEnum = BackwardEnum.${backward}/" src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py
    TRITON_DEBUG=1 TRITON_CACHE_DIR=$(pwd)/cache/${file} ASCEND_RT_VISIBLE_DEVICES=$npu_id  pytest -sv  --reruns 1 --reruns-delay 60 --only-rerun "Failed to start the device|out of memory"  tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd -v --junitxml="${log_name}_results.xml" > ${log_name}.log 2>&1

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
            total_t = int(testsuite.get('tests', 0))
            failed_t = int(testsuite.get('failures', 0))
            skipped_t = int(testsuite.get('skipped', 0))
            errors_t = int(testsuite.get('errors', 0))
            passed_t = total_t - errors_t - failed_t - skipped_t
            total += total_t
            passed += passed_t
            failed += failed_t
            skipped += skipped_t
            errors += errors_t
            print(f'已解析: {xml_file} - 总用例: {total_t}, 通过: {passed_t}, 失败: {failed_t}, 跳过: {skipped_t}, 错误: {errors_t}')
    except Exception as e:
        print(f'Error parsing {xml_file}: {e}', file=sys.stderr)
        continue

print(f'total_tests={total}')
print(f'passed_tests={passed}')
print(f'failed_tests={failed}')
print(f'skipped_tests={skipped}')
print(f'error_tests={errors}')
" > logs/stats.tmp1
    cat logs/stats.tmp1 |tail -n 5 > logs/stats.tmp

    # 记录测试结束时间
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
    "log_dir": "${OP_LOG_DIR}"
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
===== 测试统计摘要 =====
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
#cd /home/t30059325/triton-ascend_with8d/ascend/examples/generalization_cases
#
#source /home/Ascend/1030/latest/x86_64-linux/bin/setenv.bash
#export PATH=/usr/local/Ascend/8.2.RC1.alpha002/compiler/bishengir/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/Ascend/8.2.RC1.alpha002/compiler/bishengir/lib:$LD_LIBRARY_PATH
#export TRITON_ALWAYS_COMPILE=1
#export TRITON_DISABLE_FFTS=1
#export TRITON_ASCEND_ARCH=Ascend910_9589
sleep 10

env
pip install pandas
pip list
echo "========================================================="

cd ${WORKSPACE}

# 初始化统计文件
echo "生成时间: $(date +"%Y-%m-%d %H:%M:%S")" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# run gene case
#TEST_generalization="${WORKSPACE}/triton_ascend/test/generalization_cases"
#run_case_by_multi_card /home/t30059325/triton-ascend_with8d/ascend/examples/generalization_cases
run_case_by_multi_card ${TESTCASE_DIR}

echo "========================================" >> ${SUMMARY_FILE}
