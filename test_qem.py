import pymeshlab
import os


def batch_simplify_to_custom_folder(input_root_folder, output_root_folder, percentage=0.5, target_face_count=None):
    """
    批量简化模型并保存到指定目录 (修复保存格式问题)
    """

    # 检查输入目录
    if not os.path.exists(input_root_folder):
        print(f"错误: 输入目录不存在 -> {input_root_folder}")
        return


    # 获取所有模型文件夹
    model_dirs = [d for d in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, d))]

    if not model_dirs:
        print(f"在 {input_root_folder} 下未找到任何模型文件夹。")
        return

    print(f"找到 {len(model_dirs)} 个模型，开始处理...\n")

    ms = pymeshlab.MeshSet()

    success_count = 0
    fail_count = 0

    for model_name in model_dirs:
        # 1. 构建输入路径: Root/ModelName/baked_mesh/ModelName.gltf
        input_model_dir = os.path.join(input_root_folder, model_name)
        input_gltf_path = os.path.join(input_model_dir, "baked_mesh", "baked_mesh.gltf")
        print(input_gltf_path)

        if not os.path.exists(input_gltf_path):
            # 尝试找找有没有 .glb 结尾的，以防万一
            input_gltf_path = os.path.join(input_model_dir, "baked_mesh", "baked_mesh.glb")
            if not os.path.exists(input_gltf_path):
                print(f"[跳过] 未找到文件: {model_name}")
                print(input_gltf_path)
                continue

        try:
            print(f"正在处理: {model_name} ...")

            # 2. 加载模型
            ms.load_new_mesh(input_gltf_path)
            current_face_count = ms.current_mesh().face_number()

            # 计算目标面数
            if target_face_count:
                target_faces = target_face_count
                if current_face_count <= target_faces:
                    target_faces = current_face_count
            else:
                target_faces = int(current_face_count * percentage)

            print(f"  - 简化: {current_face_count} -> {target_faces}")

            # 3. 执行简化
            if target_faces < current_face_count:
                ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                                targetfacenum=target_faces,
                                preserveboundary=True,
                                preservenormal=True,
                                preservetopology=True)

            # 4. 构建输出路径
            target_model_dir = os.path.join(output_root_folder, model_name)
            if not os.path.exists(target_model_dir):
                os.makedirs(target_model_dir)
            output_file_path = os.path.join(target_model_dir, f"{model_name}.gltf")

            # 5. 保存
            try:
                # 尝试保存
                ms.save_current_mesh(output_file_path)
                print(f"  - 保存成功: {output_file_path}")
                success_count += 1
            except Exception as save_err:
                # 如果 GLTF 失败，尝试回退到 OBJ (至少能保留几何体)
                print(f"  - [警告] 保存 GLTF 失败 ({save_err})，尝试保存为 OBJ...")
                fallback_path = os.path.join(target_model_dir, f"{model_name}.obj")
                ms.save_current_mesh(fallback_path)
                print(f"  - 已回退保存为: {fallback_path}")
                success_count += 1

        except Exception as e:
            print(f"  - [错误] 处理 {model_name} 失败: {e}")
            fail_count += 1

        finally:
            ms.clear()
            print("-" * 30)

    print(f"\n任务完成! 成功: {success_count}, 失败: {fail_count}")


# ================= 配置区域 =================
if __name__ == "__main__":
    # 输入：存放所有原始模型文件夹的地方
    input_dir = "./out/baking_opt_color"

    # 输出：你想把简化后的模型存到哪里
    output_dir = "./out/qem_10%"

    # 运行：简化到 50%
    batch_simplify_to_custom_folder(input_dir, output_dir, percentage=0.1)