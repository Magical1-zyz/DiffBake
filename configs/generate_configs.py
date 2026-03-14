import os
import json

# ================= 配置区域 =================

# 基础模板 (基于您提供的内容)
TEMPLATE_CONFIG = {
    "base_mesh": "",  # 将被动态替换
    "ref_mesh": "",  # 将被动态替换
    "out_dir": "",  # 将被动态替换

    "multi_materials": False,
    "use_custom_uv": False,
    "texture_res": [1024, 1024],

    "use_opt_pbr": True,
    "bake_mode": "color_only",

    "iter": 3000,
    "batch": 8,
    "learning_rate": 0.03,
    "train_res": [1024, 1024],
    "spp": 2,
    "loss_type": "logl1",
    "smooth_weight": 0.005,
    "coarse_to_fine": True,
    "amp": True,
    "enable_batch_split": True,
    "use_gt_cache": True,

    "envmap": "data/irrmaps/docklands_01_4k.hdr",
    "env_scale": 1.0,
    "background_rgb": [1.0, 1.0, 1.0],

    "cam_radius_scale": 2.2,
    "cam_near_far": [0.1, 1000.0],
    "cam_views_top": 64,
    "cam_views_bottom": 16,

    "save_interval": 200,
    "display_interval": 0,
    "render_env_bg": False
}

# 支持的模型后缀优先级
SUPPORTED_EXTENSIONS = ['.gltf', '.glb', '.obj']

# 你的模型存放根目录 (相对于本项目根目录)
MODELS_ROOT_DIR = "data/models"

# 生成的配置文件存放的目录
OUTPUT_CONFIG_DIR = "configs"


# ================= 脚本逻辑 =================

def main():
    # 1. 确定路径
    # 获取当前脚本所在目录 (configs/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (configs 的上一级)
    project_root = os.path.dirname(current_dir)
    # data 目录路径
    models_root_dir = os.path.join(project_root, MODELS_ROOT_DIR)
    config_out_path = os.path.join(project_root, OUTPUT_CONFIG_DIR)

    if not os.path.exists(models_root_dir):
        print(f"错误: 找不到模型目录: {models_root_dir}")
        print("请确认您的目录结构是否为: DiffBake/data/models/模型名/模型文件")
        return

    # 确保输出配置的文件夹存在
    os.makedirs(config_out_path, exist_ok=True)

    print(f"正在扫描 data 目录: {models_root_dir} ...")

    # 2. 遍历 data/models 目录下的所有文件夹
    count = 0
    for folder_name in os.listdir(models_root_dir):
        folder_path = os.path.join(models_root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # 3. 定义目标 JSON 文件名 (保存在 configs/ 下)
        json_filename = f"{folder_name}.json"
        json_path = os.path.join(config_out_path, json_filename)

        # 4. 检查是否存在
        if os.path.exists(json_path):
            print(f"[跳过] {json_filename} 已存在")
            continue

        # 5. 寻找模型文件
        model_filename = None
        for ext in SUPPORTED_EXTENSIONS:
            potential_name = f"{folder_name}{ext}"
            if os.path.exists(os.path.join(folder_path, potential_name)):
                model_filename = potential_name
                break

        if model_filename is None:
            for f in os.listdir(folder_path):
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    model_filename = f
                    break

        if model_filename is None:
            print(f"[警告] 在 {folder_name} 中未找到支持的模型文件，已跳过。")
            continue

        # 6. 生成配置内容
        model_rel_path = f"data/simModels/{folder_name}/{model_filename}"

        # 输出路径
        out_dir_path = f"out/ours/{folder_name}"

        new_config = TEMPLATE_CONFIG.copy()
        new_config["base_mesh"] = model_rel_path
        new_config["ref_mesh"] = model_rel_path
        new_config["out_dir"] = out_dir_path

        # 7. 写入 JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=4)
            print(f"[生成] {json_filename}")
            count += 1
        except Exception as e:
            print(f"[错误] 写入 {json_filename} 失败: {e}")

    print(f"\n完成！共生成了 {count} 个新的配置文件。")
    print(f"配置文件保存在: {os.path.abspath(config_out_path)}")


if __name__ == "__main__":
    main()