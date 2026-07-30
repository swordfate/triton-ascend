import os
import re
from openpyxl import Workbook

def analyze_logs():
    # 创建Excel工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "测试结果分析"
    
    # 写入表头
    headers = ["序号", "测试", "pass用例数", "fail用例数", "测试结果", "info"]
    ws.append(headers)
    
    # 获取logs目录下的所有文件
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        print("logs目录不存在！")
        return
    
    files = os.listdir(logs_dir)
    files.sort()  # 按文件名排序
    
    for index, file in enumerate(files, 1):
        file_path = os.path.join(logs_dir, file)
        if os.path.isfile(file_path):
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 提取文件名（不包含后缀）
                    test_name = os.path.splitext(file)[0]
                    lines = f.readlines()
                    if not lines:
                        ws.append([index, test_name, 0, 0, False, "empty file"])
                        continue  # 跳过空文件
                    
                    # 遍历所有行，寻找包含测试结果的行
                    test_result_line = None
                    for line in reversed(lines):  # 从最后一行开始向上查找
                        if "[warning]" in line.lower():
                            continue  # 跳过警告行
                        if re.search(r'\d+ (passed|failed|warnings)', line):
                            test_result_line = line.strip()
                            break
                    
                    if not test_result_line:
                        ws.append([index, test_name, 0, 0, False, "No result summary, check logs please"])
                        continue  # 未找到测试结果行，跳过该文件
                    
                    # 初始化计数器
                    passed = 0
                    failed = 0
                    warnings = 0
                    
                    # 使用更灵活的正则表达式匹配
                    # 匹配 "X passed" 或 "X failures" 等格式
                    matches = re.findall(r'(\d+)(\s+)(passed|failed|warnings|failures|errors)', test_result_line, re.IGNORECASE)
                    
                    for match in matches:
                        num = int(match[0])
                        keyword = match[2].lower()
                        if keyword in ['passed']:
                            passed = num
                        elif keyword in ['failed', 'failures']:
                            failed = num
                        elif keyword in ['warnings', 'errors']:
                            warnings = num
                    
                    # 去除标示的"="
                    info = test_result_line.replace('=', '').strip()
                    ret = passed > 0 and failed == 0
                    
                    # 写入数据行
                    ws.append([index, test_name, passed, failed, ret, info])

            except Exception as e:
                print(f"处理文件 {file} 时出错：{str(e)}")
                continue
    
    # 保存Excel文件
    output_file = "Liger_output.xlsx"
    wb.save(output_file)
    print(f"分析结果已保存到 {output_file}")

if __name__ == "__main__":
    analyze_logs()