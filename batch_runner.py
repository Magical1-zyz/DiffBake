import os
import sys
import subprocess
import time
import glob
import gc
import csv

# ================= 配置区域 =================

# 模式 A: 手动指定要运行的 Config 文件列表 (相对路径)
# 优先级最高。如果这里有内容 (不为空)，脚本将只运行列表里的文件，忽略模式 B。
MANUAL_CONFIG_LIST = [

]

# 模式 B: 自动扫描文件夹
# 如果 MANUAL_CONFIG_LIST 为空，脚本会自动扫描该目录下所有 .json 文件
AUTO_SCAN_DIR = "configs/ablation"

# 排除列表 (文件名): 在自动扫描模式下，如果你想跳过某些特定的 JSON 文件，写在这里
EXCLUDE_FILES = [
    # "bagonghouse.json",
    # "Build_corner.json",
    # "Build_double_corner.json",
    # "Build_entrance.json",
    # "Build_entrance_big.json",
    # "Build_middle.json",
    # "building-buildify-nyc.json",
    # "City_Building_Downtown.json",
    # "dock_pier.json",
    # "doreamonhouse.json",
    # "psx_abandoned_house.json",
    # "realistic_wooden_cottage_garden_house.json",
    # "SM_BagongHouse2.json",
    # "SM_KRHistoricalGovernmentOffice.json",
    # "test.json",
    # "SM_BagongHouse2.json",
    # "wooden_house.json",
    # "SM_Bp_Building01_C_1.json",
    # "SM_Bp_Building02_C_1.json",
    # "SM_Bp_Building03_C_1.json",
    # "SM_Bp_Building04_C_1.json",
    # "SM_Bp_Building05_C_1.json",
    # "SM_Bp_Building06_C_1.json",
    # "SM_Bp_Building07_C_1.json",
    # "SM_Bp_Building08_C_1.json",
    # "SM_Bp_Building09_C_1.json",
    # "SM_Bp_Building10_C_1.json",
    # "SM_RedHouse1.json",
    # "SM_RedHouse2.json",
    # "SM_RedHouse3.json",
    # "SM_RedHouse4.json",
    # "SM_RedHouse5.json",
    # "SM_RedHouse6.json",
    # "Free_Small_Old_House.json",
    # "SM_Group_117.json",
    # "SM_Group_16_rec_9.json ",
    # "SM_Group_38_16.json",
    # "SM_Group_38_9.json",
    # "SM_Group_76_0_0.json",
    # "SM_Group_8_10_left.json",
    # "SM_Group_8_11.json",
    # "SM_Group_8_3.json"
]

# 任务间隔冷却时间 (秒)
COOLDOWN_TIME = 15

# Python解释器路径
PYTHON_EXECUTABLE = sys.executable

# 结果导出的目标文件夹 (支持多级目录，例如 "out/reports/ablation")
CSV_OUTPUT_DIR = "reports/ablation_results"

# 结果导出文件名
CSV_FILENAME = "batch_report_ablation.csv"


# =========================================

class BatchRunner:
    def __init__(self, manual_list, auto_dir, exclude_files, cooldown, python_exe, csv_dir, csv_filename):
        """初始化 Runner"""
        self.manual_list = manual_list
        self.auto_dir = auto_dir
        self.exclude_files = exclude_files
        self.cooldown = cooldown
        self.python_exe = python_exe

        # 保存 CSV 路径信息
        self.csv_dir = csv_dir
        self.csv_filename = csv_filename

        # 存储结果数据: [(Config, Time, Status, PSNR, VRAM, RAM), ...]
        self.report_data = []
        self.success_count = 0
        self.fail_list = []

    def get_tasks(self):
        """获取任务列表"""
        tasks = []
        if len(self.manual_list) > 0:
            print(f"[Batch] Mode: Manual List ({len(self.manual_list)} items)")
            tasks = self.manual_list
        else:
            print(f"[Batch] Mode: Auto Scan directory '{self.auto_dir}'")
            search_pattern = os.path.join(self.auto_dir, "*.json")
            files = glob.glob(search_pattern)
            # 过滤排除列表
            files = [f for f in files if os.path.basename(f) not in self.exclude_files]
            # 按文件名排序
            tasks = sorted(files)
            print(f"[Batch] Found {len(tasks)} configs.")
        return tasks

    def run_single_task(self, command):
        """执行单个任务并捕获输出"""
        print(f"\n[Batch] Executing: {command}")
        print("-" * 60)

        start_time = time.time()
        captured_info = {"PSNR": "N/A", "VRAM": "N/A", "RAM": "N/A"}
        success = False

        try:
            # 启动子进程
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            # 实时读取输出
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.rstrip())  # 实时打印
                    # 抓取 [BATCH_INFO]
                    if "[BATCH_INFO]" in line:
                        try:
                            parts = line.split()
                            for p in parts:
                                if "PSNR:" in p: captured_info["PSNR"] = p.split(":")[1]
                                if "VRAM:" in p: captured_info["VRAM"] = p.split(":")[1]
                                if "RAM:" in p:  captured_info["RAM"] = p.split(":")[1]
                        except:
                            pass

            return_code = process.poll()
            duration = time.time() - start_time
            success = (return_code == 0)

            if not success:
                print(f"\n[Batch] Error: Task failed with return code {return_code}")

        except KeyboardInterrupt:
            print("\n[Batch] Interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[Batch] Execution Exception: {e}")
            duration = time.time() - start_time
            success = False

        return success, duration, captured_info

    def cleanup(self):
        """清理资源"""
        print(f"[Batch] Cooling down for {self.cooldown} seconds to release VRAM...")
        gc.collect()
        time.sleep(self.cooldown)

    def print_summary_table(self):
        """打印控制台汇总表"""
        if not self.report_data:
            return

        headers = ["Config File", "Time", "Status", "PSNR", "VRAM", "RAM"]

        # 自动计算列宽
        col_widths = [len(h) for h in headers]
        for row in self.report_data:
            for j, val in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(val)))

        col_widths = [w + 2 for w in col_widths]  # Padding
        row_fmt = "".join([f"{{:<{w}}}" for w in col_widths])

        print("-" * 90)
        print(row_fmt.format(*headers))
        print("-" * 90)
        for row in self.report_data:
            print(row_fmt.format(*row))
        print("=" * 90)

    def export_to_csv(self):
        """导出结果到CSV文件"""
        if not self.report_data:
            print("[Batch] No data to export.")
            return

        # 检查并创建目录
        if self.csv_dir:
            try:
                os.makedirs(self.csv_dir, exist_ok=True)
            except Exception as e:
                print(f"[Batch] Warning: Failed to create directory '{self.csv_dir}': {e}")

            full_path = os.path.join(self.csv_dir, self.csv_filename)
        else:
            full_path = self.csv_filename

        try:
            # utf-8-sig 确保 Excel 打开中文不乱码
            with open(full_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(["Config File", "Time", "Status", "PSNR", "VRAM", "RAM"])
                # 写入数据
                writer.writerows(self.report_data)
            print(f"[Batch] Report exported successfully to: {os.path.abspath(full_path)}")
        except Exception as e:
            print(f"[Batch] Failed to export CSV: {e}")

    def run(self):
        """主运行逻辑"""
        tasks = self.get_tasks()
        if not tasks:
            print("[Batch] No config files found.")
            return

        print(f"\n[Batch] Task Queue:")
        for idx, t in enumerate(tasks):
            print(f"  {idx + 1}. {t}")

        print("\n[Batch] Start processing...")
        print("=" * 60)

        total_start = time.time()

        for i, config_path in enumerate(tasks):
            if not os.path.exists(config_path):
                print(f"[Batch] Skipping {config_path}: File not found.")
                self.fail_list.append(config_path)
                self.report_data.append((os.path.basename(config_path), "0.0s", "MISSING", "-", "-", "-"))
                continue

            print(f"\n>>> Task {i + 1}/{len(tasks)}: {config_path}")

            # 构造命令
            cmd = f'"{self.python_exe}" train.py --config "{config_path}"'

            # 执行
            success, duration, info = self.run_single_task(cmd)

            duration_str = f"{duration:.1f}s"
            config_name = os.path.basename(config_path)

            if success:
                self.success_count += 1
                status = "OK"
                print(f">>> Task {i + 1} Finished in {duration_str}")
            else:
                self.fail_list.append(config_path)
                status = "FAILED"
                print(f">>> Task {i + 1} FAILED.")

            # 记录数据
            self.report_data.append((
                config_name, duration_str, status,
                info["PSNR"], info["VRAM"], info["RAM"]
            ))

            # 执行间隔清理 (最后一个任务后不需要)
            if i < len(tasks) - 1:
                self.cleanup()

        # 结束总结
        total_duration = time.time() - total_start
        print("\n" + "=" * 90)
        print(f"Batch Processing Complete.")
        print(f"Total Time: {total_duration / 60:.1f} minutes")
        print(f"Success: {self.success_count} | Failed: {len(self.fail_list)}")

        # 打印表格
        self.print_summary_table()

        # 导出CSV
        self.export_to_csv()


if __name__ == "__main__":
    # 实例化并运行
    runner = BatchRunner(
        manual_list=MANUAL_CONFIG_LIST,
        auto_dir=AUTO_SCAN_DIR,
        exclude_files=EXCLUDE_FILES,
        cooldown=COOLDOWN_TIME,
        python_exe=PYTHON_EXECUTABLE,
        csv_dir=CSV_OUTPUT_DIR,
        csv_filename=CSV_FILENAME
    )
    runner.run()