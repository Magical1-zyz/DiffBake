from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


# ====== 你需要改这里：Blender 可执行文件路径（Windows）======
BLENDER_EXE = r"F:\Software\Science\Blender3.6\blender.exe"
# ======================================================


@dataclass
class DecimationConfig:
    """
    减面配置：
    - ratio: 保留比例 (0,1]，例如 0.1 表示保留 10% 三角形
    - apply_to_all_meshes: 是否对场景内所有 mesh 物体应用 decimate
    """
    ratio: float = 0.5
    apply_to_all_meshes: bool = True

    def validate(self) -> None:
        if not (0.0 < self.ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1].")


@dataclass
class PipelinePaths:
    """
    路径规则：
    输入： modelname/baked_mesh/baked_mesh.gltf
    输出： modelname 目录下（或 output_root 镜像结构下）指定文件名
    """
    input_root: Path
    output_root: Optional[Path] = None
    input_relative: Path = Path("baked_mesh") / "baked_mesh.gltf"
    output_filename: str = "baked_mesh.gltf"  # 你现在想输出成 baked_mesh.gltf

    def resolve_input_file(self, model_dir: Path) -> Path:
        return model_dir / self.input_relative

    def resolve_output_dir(self, model_dir: Path) -> Path:
        root = self.input_root if self.output_root is None else self.output_root
        rel = model_dir.relative_to(self.input_root)
        return root / rel

    def resolve_output_file(self, model_dir: Path) -> Path:
        return self.resolve_output_dir(model_dir) / self.output_filename


class BlenderDecimator:
    """
    使用 Blender headless 执行：
    import glTF -> Decimate(Collapse) -> export glTF
    重点：材质/贴图/UV 由 Blender 管线保留（比 Open3D 方案靠谱得多）
    """

    def __init__(self, blender_exe: str):
        self.blender_exe = blender_exe

    def _ensure_blender(self) -> None:
        if not Path(self.blender_exe).exists():
            raise FileNotFoundError(f"Blender not found: {self.blender_exe}")

    def decimate_gltf(self, in_gltf: Path, out_gltf: Path, config: DecimationConfig) -> None:
        self._ensure_blender()
        config.validate()

        out_gltf.parent.mkdir(parents=True, exist_ok=True)

        # Blender 内部执行脚本（写到临时文件，避免命令行引号地狱）
        blender_script = self._build_blender_script(
            in_gltf=in_gltf,
            out_gltf=out_gltf,
            ratio=config.ratio,
            apply_all=config.apply_to_all_meshes,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            script_path = Path(f.name)
            f.write(blender_script)

        try:
            # --background: 无界面
            # --factory-startup: 干净启动，避免用户偏好设置影响
            cmd = [
                self.blender_exe,
                "--background",
                "--factory-startup",
                "--python",
                str(script_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                raise RuntimeError(
                    "Blender failed.\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n"
                )
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _build_blender_script(in_gltf: Path, out_gltf: Path, ratio: float, apply_all: bool) -> str:
        # 注意：这里的脚本运行在 Blender 自带 Python 里
        # export 选项：export_format='GLTF_SEPARATE' 会生成 .gltf + .bin + textures
        # 如果你想生成单文件 .glb，把 out_gltf 后缀改成 .glb 并设 export_format='GLB'
        in_path = in_gltf.as_posix()
        out_path = out_gltf.as_posix()

        return f"""
import bpy

# 清空默认场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 导入 glTF
bpy.ops.import_scene.gltf(filepath=r"{in_path}")

# 对 mesh 物体加 Decimate(Collapse)
for obj in list(bpy.data.objects):
    if obj.type != 'MESH':
        continue
    if (not {str(apply_all)}) and (obj.name.lower() != "mesh"):
        continue

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = {ratio}

    # 应用 modifier
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.select_set(False)

# 导出 glTF
# 关键：export_materials='EXPORT' 会把材质导出
# export_format 可选：'GLTF_SEPARATE' / 'GLTF_EMBEDDED' / 'GLB'
bpy.ops.export_scene.gltf(
    filepath=r"{out_path}",
    export_format='GLTF_SEPARATE',
    export_materials='EXPORT',
    export_texcoords=True,
    export_normals=True,
    export_colors=True,
    export_yup=True
)
"""


class BatchGltfDecimationApp:
    """
    批处理：
    遍历 input_root 下所有一级目录视为 modelname
    """

    def __init__(self, paths: PipelinePaths, config: DecimationConfig, decimator: BlenderDecimator):
        self.paths = paths
        self.config = config
        self.decimator = decimator

    def find_model_dirs(self) -> List[Path]:
        root = self.paths.input_root
        if not root.exists():
            raise FileNotFoundError(f"input_root not found: {root}")
        return [p for p in root.iterdir() if p.is_dir()]

    def run(self) -> None:
        model_dirs = self.find_model_dirs()
        if not model_dirs:
            print(f"[WARN] No model directories found under: {self.paths.input_root}")
            return

        total = 0
        ok = 0

        for model_dir in model_dirs:
            total += 1
            in_file = self.paths.resolve_input_file(model_dir)
            out_file = self.paths.resolve_output_file(model_dir)

            if not in_file.exists():
                print(f"[SKIP] Missing input: {in_file}")
                continue

            try:
                self.decimator.decimate_gltf(in_file, out_file, self.config)
                ok += 1
                print(f"[OK] {model_dir.name}: out={out_file}")
            except Exception as e:
                print(f"[FAIL] {model_dir.name}: {e}")

        print(f"\\nDone. success={ok}/{total}")


def main():
    # 你现在的路径
    input_root = Path("./out/baking_opt_color")
    output_root = Path("./out/qem_blender")

    paths = PipelinePaths(
        input_root=input_root,
        output_root=output_root,
        output_filename="baked_mesh.gltf",  # 输出到 ./out/qem/{modelname}/baked_mesh.gltf
    )

    config = DecimationConfig(
        ratio=0.1,
        apply_to_all_meshes=True,
    )

    app = BatchGltfDecimationApp(paths, config, BlenderDecimator(BLENDER_EXE))
    app.run()


if __name__ == "__main__":
    main()