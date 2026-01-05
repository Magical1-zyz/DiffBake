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
    "texture_res": [2048, 2048],

    "use_opt_pbr": True,
    "bake_mode": "full_pbr",

    "iter": 3000,
    "batch": 4,
    "learning_rate": 0.03,
    "train_res": [1024, 1024],
    "spp": 4,
    "loss_type": "logl1",
    "smooth_weight": 0.005,
    "coarse_to_fine": True,
    "amp": True,

    "envmap": "data/irrmaps/docklands_01_4k.hdr",
    "env_scale": 1.0,
    "background_rgb": [1.0, 1.0, 1.0],

    "cam_radius_scale": 2.0,
    "cam_near_far": [0.1, 1000.0],
    "cam_views_top": 96,
    "cam_views_bottom": 32,

    "save_interval": 100,
    "display_interval": 50,
    "render_env_bg": False
}

# 支持的模型后缀优先级
SUPPORTED_EXTENSIONS = ['.gltf', '.glb', '.obj']


# ================= 脚本逻辑 =================

def main():
    # 1. 确定路径
    # 获取当前脚本所在目录 (configs/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (configs 的上一级)
    project_root = os.path.dirname(current_dir)
    # data 目录路径
    data_dir = os.path.join(project_root, 'data')

    if not os.path.exists(data_dir):
        print(f"错误: 找不到 data 目录: {data_dir}")
        return

    print(f"正在扫描 data 目录: {data_dir} ...")

    # 2. 遍历 data 目录下的所有文件夹
    count = 0
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        # 确保是文件夹而非文件
        if not os.path.isdir(folder_path):
            continue

        # 3. 定义目标 JSON 文件名
        json_filename = f"{folder_name}.json"
        json_path = os.path.join(current_dir, json_filename)

        # 4. 检查是否已存在，存在则跳过
        if os.path.exists(json_path):
            print(f"[跳过] {json_filename} 已存在")
            continue

        # 5. 寻找模型文件 (优先寻找与文件夹同名的文件)
        model_filename = None
        for ext in SUPPORTED_EXTENSIONS:
            potential_name = f"{folder_name}{ext}"
            if os.path.exists(os.path.join(folder_path, potential_name)):
                model_filename = potential_name
                break

        # 如果找不到同名文件，尝试找任何支持的格式 (可选逻辑)
        if model_filename is None:
            for f in os.listdir(folder_path):
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    model_filename = f
                    break

        if model_filename is None:
            print(f"[警告] 在 {folder_name} 中未找到支持的模型文件 (.gltf/.glb/.obj)，已跳过。")
            continue

        # 6. 生成配置内容
        # 使用 forward slashes (/) 以兼容 JSON 格式和跨平台
        model_rel_path = f"data/{folder_name}/{model_filename}"
        out_dir_path = f"out/baking_opt_pbr/{folder_name}"

        new_config = TEMPLATE_CONFIG.copy()
        new_config["base_mesh"] = model_rel_path
        new_config["ref_mesh"] = model_rel_path
        new_config["out_dir"] = out_dir_path

        # 7. 写入 JSON 文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=4)
            print(f"[生成] {json_filename}")
            count += 1
        except Exception as e:
            print(f"[错误] 写入 {json_filename} 失败: {e}")

    print(f"\n完成！共生成了 {count} 个新的配置文件。")


if __name__ == "__main__":
    main()