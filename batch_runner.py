import os
import sys
import subprocess
import time
import glob

# =================配置区域=================

# 模式 A: 手动指定要运行的 Config 文件列表 (相对路径)
# 优先级最高。如果这里有内容，就会忽略模式 B。
MANUAL_CONFIG_LIST = [
    "configs/SM_Bp_Building01_C_1.json",
    "configs/SM_Bp_Building02_C_1.json",
    "configs/SM_Bp_Building03_C_1.json",
    "configs/SM_Bp_Building04_C_1.json",
    "configs/SM_Bp_Building05_C_1.json",
    "configs/SM_Bp_Building06_C_1.json",
    "configs/SM_Bp_Building07_C_1.json",
    "configs/SM_Bp_Building08_C_1.json",
    "configs/SM_Bp_Building09_C_1.json",
    "configs/SM_Bp_Building10_C_1.json"
]

# 模式 B: 自动扫描文件夹
# 如果 MANUAL_CONFIG_LIST 为空，脚本会自动扫描该目录下所有符合条件的文件
AUTO_SCAN_DIR = "configs"
AUTO_SCAN_PREFIX = "bake_"  # 只运行以此开头的json，防止运行错误的配置
AUTO_SCAN_SUFFIX = ".json"

# Python解释器路径 (通常直接用 'python' 即可，如果是特定环境可指定绝对路径)
PYTHON_EXECUTABLE = sys.executable


# =========================================

def run_command(command):
    """执行系统命令并实时输出日志"""
    print(f"\n[Batch] Executing: {command}")
    print("-" * 60)

    # 使用 subprocess.run 调用，check=True 会在失败时抛出异常
    try:
        # shell=True 在 Windows 下通常是必须的
        start_time = time.time()
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
        search_pattern = os.path.join(AUTO_SCAN_DIR, f"{AUTO_SCAN_PREFIX}*{AUTO_SCAN_SUFFIX}")
        files = glob.glob(search_pattern)
        # 按文件名排序，保证顺序一致
        tasks = sorted(files)
        print(f"[Batch] Found {len(tasks)} configs: {[os.path.basename(f) for f in tasks]}")

    if not tasks:
        print("[Batch] No config files found to run.")
        return

    print(f"\n[Batch] Start processing {len(tasks)} tasks...")
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
            continue

        print(f"\n>>> Task {i + 1}/{len(tasks)}: {config_path}")

        # 构造命令： python train.py --config xxx.json
        # 这里的 train.py 对应你实际的主程序文件名
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

        # (可选) 可以在这里加一个简短的 sleep 让显卡喘口气
        time.sleep(2)

    # 3. 最终总结
    total_duration = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Batch Processing Complete.")
    print(f"Total Time: {total_duration / 60:.1f} minutes")
    print(f"Success: {success_count} | Failed: {len(fail_list)}")
    print("-" * 60)

    print(f"{'Config File':<40} | {'Time':<10} | {'Status'}")
    print("-" * 60)
    for name, time_str, status in report:
        print(f"{name:<40} | {time_str:<10} | {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()