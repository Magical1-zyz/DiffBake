import os
import sys
import subprocess
import time
import glob

# ================= 配置区域 =================

# 模式 A: 手动指定要运行的 Config 文件列表 (相对路径)
# 优先级最高。如果这里有内容 (不为空)，脚本将只运行列表里的文件，忽略模式 B。
# 示例: ["configs/SM_RedHouse.json", "configs/SM_Building.json"]
MANUAL_CONFIG_LIST = [

]

# 模式 B: 自动扫描文件夹
# 如果 MANUAL_CONFIG_LIST 为空，脚本会自动扫描该目录下所有 .json 文件
AUTO_SCAN_DIR = "configs"

# 排除列表 (文件名): 在自动扫描模式下，如果你想跳过某些特定的 JSON 文件，写在这里
# 示例: ["template.json", "base_config.json"]
EXCLUDE_FILES = [

]

# Python解释器路径
PYTHON_EXECUTABLE = sys.executable


# =========================================

def run_command(command):
    """执行系统命令并实时输出日志"""
    print(f"\n[Batch] Executing: {command}")
    print("-" * 60)

    try:
        start_time = time.time()
        # Windows下通常需要 shell=True
        result = subprocess.run(command, shell=True, check=True)
        duration = time.time() - start_time
        return True, duration
    except subprocess.CalledProcessError as e:
        print(f"\n[Batch] Error: Task failed with return code {e.returncode}")
        return False, 0.0
    except KeyboardInterrupt:
        print("\n[Batch] Interrupted by user.")
        sys.exit(1)


def main():
    # 1. 确定任务列表
    tasks = []

    if len(MANUAL_CONFIG_LIST) > 0:
        print(f"[Batch] Mode: Manual List ({len(MANUAL_CONFIG_LIST)} items)")
        tasks = MANUAL_CONFIG_LIST
    else:
        print(f"[Batch] Mode: Auto Scan directory '{AUTO_SCAN_DIR}'")
        # 直接匹配所有 .json 文件
        search_pattern = os.path.join(AUTO_SCAN_DIR, "*.json")
        files = glob.glob(search_pattern)

        # 过滤排除列表
        files = [f for f in files if os.path.basename(f) not in EXCLUDE_FILES]

        # 按文件名排序，保证执行顺序一致
        tasks = sorted(files)
        print(f"[Batch] Found {len(tasks)} configs.")

    if not tasks:
        print("[Batch] No config files found to run.")
        print(f"        Please check directory: {AUTO_SCAN_DIR}")
        return

    print(f"\n[Batch] Task Queue:")
    for idx, t in enumerate(tasks):
        print(f"  {idx + 1}. {t}")

    print("\n[Batch] Start processing...")
    print("=" * 60)

    # 2. 循环执行
    success_count = 0
    fail_list = []
    report = []

    total_start = time.time()

    for i, config_path in enumerate(tasks):
        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"[Batch] Skipping {config_path}: File not found.")
            fail_list.append(config_path)
            report.append((config_path, "0.0s", "MISSING"))
            continue

        print(f"\n>>> Task {i + 1}/{len(tasks)}: {config_path}")

        # 构造命令
        cmd = f'"{PYTHON_EXECUTABLE}" train.py --config "{config_path}"'

        # 执行
        success, duration = run_command(cmd)

        if success:
            success_count += 1
            report.append((config_path, f"{duration:.1f}s", "OK"))
            print(f">>> Task {i + 1} Finished in {duration:.1f}s")
        else:
            fail_list.append(config_path)
            report.append((config_path, "0.0s", "FAILED"))
            print(f">>> Task {i + 1} FAILED. Continuing to next task...")

        # (可选) 任务间隔休息，防止显卡过热
        if i < len(tasks) - 1:
            time.sleep(2)

    # 3. 最终总结
    total_duration = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Batch Processing Complete.")
    print(f"Total Time: {total_duration / 60:.1f} minutes")
    print(f"Success: {success_count} | Failed: {len(fail_list)}")
    print("-" * 60)

    # 打印格式化的报告表
    # 动态调整列宽以适应长文件名
    max_name_len = max([len(t[0]) for t in report]) if report else 40
    max_name_len = max(max_name_len, 20)  # 最小宽度

    header = f"{'Config File':<{max_name_len}} | {'Time':<10} | {'Status'}"
    print(header)
    print("-" * len(header))

    for name, time_str, status in report:
        print(f"{name:<{max_name_len}} | {time_str:<10} | {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()