import os
import json
import numpy as np
import torch
from pygltflib import GLTF2

from . import mesh
from . import material
from . import texture
from . import util


# ==============================================================================================
#  Load glTF / GLB
# ==============================================================================================

@torch.no_grad()
def load_gltf(filename, mtl_override=None, merge_materials=False):
    """
    加载 .gltf 或 .glb 文件。
    支持读取外部 .bin 缓冲区或 GLB 内部嵌入的二进制块。
    返回一个包含材质列表的 Mesh 对象。
    """
    gltf_path = os.path.dirname(filename)

    # 1. 使用 pygltflib 加载 (自动处理 gltf/glb 差异)
    try:
        # 使用类方法加载，确保兼容性
        doc = GLTF2.load(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to load GLTF/GLB file {filename}: {e}")

    # 将 doc 转换回 python dict 结构以复用解析逻辑
    gltf = json.loads(doc.to_json())

    # 2. 提取二进制 Buffers (修复 Method/Bytes 问题)
    buffers = []
    for i, buf in enumerate(doc.buffers):
        # 情况 A: GLB 内部嵌入的二进制块
        if buf.uri is None:
            # 某些版本的库或异常情况下 binary_blob 可能是方法，这里做防御性编程
            blob = doc.binary_blob
            if callable(blob):
                blob = blob()
            # 确保是 bytes，如果是 None 则由 b'' 替代
            buffers.append(blob if blob is not None else b'')

        # 情况 B: 外部 .bin 文件
        elif buf.uri is not None:
            try:
                bin_path = os.path.join(gltf_path, buf.uri)
                with open(bin_path, 'rb') as bf:
                    # 关键修复：确保调用 read()
                    data = bf.read()
                    buffers.append(data)
            except Exception as e:
                print(f"Warning: Failed to load buffer uri {buf.uri}: {e}")
                buffers.append(b'')
        else:
            buffers.append(b'')

    # 3. 辅助函数：读取 Accessor 数据
    def read_accessor(acc_idx):
        if acc_idx is None or acc_idx < 0: return None
        # 安全检查访问器索引
        if acc_idx >= len(gltf['accessors']): return None

        acc = gltf['accessors'][acc_idx]
        buf_view = gltf['bufferViews'][acc['bufferView']]
        buffer_idx = buf_view.get('buffer', 0)

        # 获取对应 buffer 数据
        if buffer_idx >= len(buffers): return None
        data = buffers[buffer_idx]

        # 再次检查 data 类型，防止 method 混入
        if callable(data):
            try:
                data = data()
            except:
                pass
        if not isinstance(data, (bytes, bytearray)):
            # 如果依然不是 bytes，打印警告并跳过，防止 crash
            # print(f"Warning: Buffer {buffer_idx} is not bytes but {type(data)}")
            return None

        # 计算偏移和步长
        base_offset = buf_view.get('byteOffset', 0)
        acc_offset = acc.get('byteOffset', 0)
        total_offset = base_offset + acc_offset

        count = acc['count']
        comp_type = acc['componentType']
        type_str = acc['type']

        # 维度映射
        num_comp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}.get(type_str, 1)

        # 类型映射
        if comp_type == 5126:  # FLOAT
            dtype = np.float32;
            comp_size = 4
        elif comp_type == 5123:  # UNSIGNED_SHORT
            dtype = np.uint16;
            comp_size = 2
        elif comp_type == 5125:  # UNSIGNED_INT
            dtype = np.uint32;
            comp_size = 4
        elif comp_type == 5121:  # UNSIGNED_BYTE
            dtype = np.uint8;
            comp_size = 1
        elif comp_type == 5122:  # SHORT
            dtype = np.int16;
            comp_size = 2
        else:
            return None

        # 边界检查 (这里是之前报错的地方)
        expected_bytes = count * num_comp * comp_size
        if total_offset + expected_bytes > len(data):
            print(
                f"Warning: Buffer overflow reading accessor {acc_idx}. Data len: {len(data)}, Req: {total_offset + expected_bytes}")
            return None

        # 处理步长 (Byte Stride)
        byte_stride = buf_view.get('byteStride', None)
        elem_size = num_comp * comp_size

        if byte_stride is None or byte_stride == elem_size:
            arr = np.frombuffer(data, dtype=dtype, count=count * num_comp, offset=total_offset)
            return arr.reshape((count, num_comp))
        else:
            out = np.empty((count, num_comp), dtype=dtype)
            for i in range(count):
                start = total_offset + i * byte_stride
                out[i] = np.frombuffer(data, dtype=dtype, count=num_comp, offset=start)
            return out

    # 4. 解析图片路径 (Images)
    image_uris = []
    for img in gltf.get('images', []):
        uri = img.get('uri', '')
        image_uris.append(uri if uri else None)

    # 5. 解析纹理映射 (Textures -> Images)
    textures_image_idx = []
    for tex in gltf.get('textures', []):
        src = tex.get('source', None)
        textures_image_idx.append(src)

    # 6. 解析材质 (Materials)
    all_materials = []
    for mat_def in gltf.get('materials', []):
        name = mat_def.get('name', f'mat_{len(all_materials)}')
        m = material.Material({'name': name})
        m['bsdf'] = 'pbr'

        m['kd'] = texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda'))
        m['ks'] = texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))

        pbr = mat_def.get('pbrMetallicRoughness', {})

        # BaseColor
        base_color_tex_info = pbr.get('baseColorTexture', None)
        if base_color_tex_info is not None:
            tex_idx = base_color_tex_info.get('index', -1)
            if 0 <= tex_idx < len(textures_image_idx):
                img_idx = textures_image_idx[tex_idx]
                if img_idx is not None and 0 <= img_idx < len(image_uris) and image_uris[img_idx]:
                    img_path = os.path.join(gltf_path, image_uris[img_idx])
                    if os.path.exists(img_path):
                        kd_tex = texture.load_texture2D(img_path)
                        kd_tex = texture.srgb_to_rgb(kd_tex)
                        m['kd'] = kd_tex

        # MetallicRoughness
        mr_tex_info = pbr.get('metallicRoughnessTexture', None)
        if mr_tex_info is not None:
            tex_idx = mr_tex_info.get('index', -1)
            if 0 <= tex_idx < len(textures_image_idx):
                img_idx = textures_image_idx[tex_idx]
                if img_idx is not None and 0 <= img_idx < len(image_uris) and image_uris[img_idx]:
                    img_path = os.path.join(gltf_path, image_uris[img_idx])
                    if os.path.exists(img_path):
                        m['ks'] = texture.load_texture2D(img_path, channels=3)

        # Normal Map
        norm_tex_info = mat_def.get('normalTexture', None)
        if norm_tex_info is not None:
            tex_idx = norm_tex_info.get('index', -1)
            if 0 <= tex_idx < len(textures_image_idx):
                img_idx = textures_image_idx[tex_idx]
                if img_idx is not None and 0 <= img_idx < len(image_uris) and image_uris[img_idx]:
                    img_path = os.path.join(gltf_path, image_uris[img_idx])
                    if os.path.exists(img_path):
                        m['normal'] = texture.load_texture2D(img_path, lambda_fn=lambda x: x * 2.0 - 1.0, channels=3)

        all_materials.append(m)

    if not all_materials:
        default_mat = material.Material({'name': 'default'})
        default_mat['bsdf'] = 'pbr'
        default_mat['kd'] = texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda'))
        default_mat['ks'] = texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        all_materials.append(default_mat)

    # 7. 解析网格
    pos_list = []
    nrm_list = []
    tan_list = []
    uv_list = []
    idx_list = []
    mat_idx_list = []

    global_v_offset = 0

    for mesh_def in gltf.get('meshes', []):
        for prim in mesh_def.get('primitives', []):
            mat_idx = prim.get('material', 0)
            if mat_idx >= len(all_materials): mat_idx = 0

            attrs = prim.get('attributes', {})
            pos = read_accessor(attrs.get('POSITION'))
            nrm = read_accessor(attrs.get('NORMAL'))
            tan = read_accessor(attrs.get('TANGENT'))
            uv0 = read_accessor(attrs.get('TEXCOORD_0'))
            indices = read_accessor(prim.get('indices'))

            if pos is None: continue

            v_pos = torch.tensor(pos, dtype=torch.float32, device='cuda')
            v_nrm = torch.tensor(nrm, dtype=torch.float32, device='cuda') if nrm is not None else None
            v_tan = torch.tensor(tan[:, :3], dtype=torch.float32, device='cuda') if tan is not None else None
            v_uv = torch.tensor(uv0, dtype=torch.float32, device='cuda') if uv0 is not None else None

            if indices is not None:
                idx = torch.tensor(indices.astype(np.int64).flatten(), dtype=torch.int64, device='cuda').reshape(-1, 3)
            else:
                count = v_pos.shape[0]
                idx = torch.arange(count, dtype=torch.int64, device='cuda').reshape(-1, 3)

            idx += global_v_offset

            pos_list.append(v_pos)
            if v_nrm is not None: nrm_list.append(v_nrm)
            if v_tan is not None: tan_list.append(v_tan)
            if v_uv is not None: uv_list.append(v_uv)

            idx_list.append(idx)

            num_faces = idx.shape[0]
            mat_idx_list.append(torch.full((num_faces,), mat_idx, dtype=torch.int64, device='cuda'))

            global_v_offset += v_pos.shape[0]

    if not pos_list:
        raise RuntimeError("No valid geometry found in glTF")

    V = torch.cat(pos_list, dim=0)
    F = torch.cat(idx_list, dim=0)
    N = torch.cat(nrm_list, dim=0) if len(nrm_list) == len(pos_list) else None
    T = torch.cat(tan_list, dim=0) if len(tan_list) == len(pos_list) else None
    UV = torch.cat(uv_list, dim=0) if len(uv_list) == len(pos_list) else None
    MF = torch.cat(mat_idx_list, dim=0)

    if len(uv_list) > 0 and len(uv_list) != len(pos_list):
        print("Warning: Some primitives are missing UVs. UVs will be discarded.")
        UV = None

    mesh_obj = mesh.Mesh(
        v_pos=V, t_pos_idx=F,
        v_nrm=N, t_nrm_idx=F if N is not None else None,
        v_tex=UV, t_tex_idx=F if UV is not None else None,
        v_tng=T, t_tng_idx=F if T is not None else None,
        material=all_materials[0],
        materials=all_materials,
        face_material_idx=MF
    )

    return mesh_obj


# ==============================================================================================
#  Save glTF (Single Material)
# ==============================================================================================

@torch.no_grad()
def save_gltf(folder, mesh_obj, diffuse_only=False):
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(folder))
    bin_name = base_name + '.bin'
    gltf_name = base_name + '.gltf'
    bin_filename = os.path.join(folder, bin_name)
    gltf_filename = os.path.join(folder, gltf_name)

    V = mesh_obj.v_pos.detach().cpu().numpy().astype(np.float32)
    UV = mesh_obj.v_tex.detach().cpu().numpy().astype(np.float32) if mesh_obj.v_tex is not None else None
    F = mesh_obj.t_pos_idx.detach().cpu().numpy()

    buffer_data = bytearray()
    buffer_views = []
    accessors = []

    def add_buffer_view(data_bytes, target):
        view_idx = len(buffer_views)
        offset = len(buffer_data)
        buffer_data.extend(data_bytes)
        while len(buffer_data) % 4 != 0: buffer_data.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})
        return view_idx

    def add_accessor(view_idx, comp_type, count, type_str, min_val=None, max_val=None):
        acc = {"bufferView": view_idx, "byteOffset": 0, "componentType": comp_type, "count": count, "type": type_str}
        if min_val is not None: acc["min"] = min_val
        if max_val is not None: acc["max"] = max_val
        accessors.append(acc)
        return len(accessors) - 1

    pos_idx = add_accessor(add_buffer_view(V.tobytes(), 34962), 5126, V.shape[0], "VEC3", V.min(axis=0).tolist(),
                           V.max(axis=0).tolist())

    uv_idx = None
    if UV is not None:
        uv_idx = add_accessor(add_buffer_view(UV.tobytes(), 34962), 5126, UV.shape[0], "VEC2")

    indices_flat = F.flatten()
    if indices_flat.max() < 65536:
        ind_bytes = indices_flat.astype(np.uint16).tobytes()
        comp_type = 5123
    else:
        ind_bytes = indices_flat.astype(np.uint32).tobytes()
        comp_type = 5125
    ind_idx = add_accessor(add_buffer_view(ind_bytes, 34963), comp_type, indices_flat.shape[0], "SCALAR")

    images = []
    textures = []
    materials = []

    mat = mesh_obj.material
    tex_name = "texture_base.png"
    tex_path = os.path.join(folder, tex_name)
    texture.save_texture2D(tex_path, texture.rgb_to_srgb(mat['kd']))

    images.append({"uri": tex_name})
    textures.append({"sampler": 0, "source": 0})

    materials.append({
        "name": "baked_material",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "doubleSided": True
    })

    gltf_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": bin_name, "byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": images,
        "textures": textures,
        "materials": materials,
        "meshes": [{"name": "mesh", "primitives": [
            {"attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx}, "indices": ind_idx, "material": 0,
             "mode": 4}]}],
        "nodes": [{"name": "root", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0
    }

    with open(gltf_filename, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    with open(bin_filename, 'wb') as f:
        f.write(buffer_data)


# ==============================================================================================
#  Save glTF (Multi-Material)
# ==============================================================================================

@torch.no_grad()
def save_gltf_multi(folder, mesh_obj, diffuse_only=False):
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(folder))
    bin_name = base_name + '.bin'
    gltf_name = base_name + '.gltf'
    bin_filename = os.path.join(folder, bin_name)
    gltf_filename = os.path.join(folder, gltf_name)

    V = mesh_obj.v_pos.detach().cpu().numpy().astype(np.float32)
    UV = mesh_obj.v_tex.detach().cpu().numpy().astype(np.float32)
    F = mesh_obj.t_pos_idx.detach().cpu().numpy()
    MF = mesh_obj.face_material_idx.detach().cpu().numpy()

    buffer_data = bytearray()
    buffer_views = []
    accessors = []

    def add_buffer_view(data_bytes, target):
        view_idx = len(buffer_views)
        offset = len(buffer_data)
        buffer_data.extend(data_bytes)
        while len(buffer_data) % 4 != 0: buffer_data.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})
        return view_idx

    def add_accessor(view_idx, comp_type, count, type_str, min_val=None, max_val=None):
        acc = {"bufferView": view_idx, "byteOffset": 0, "componentType": comp_type, "count": count, "type": type_str}
        if min_val is not None: acc["min"] = min_val
        if max_val is not None: acc["max"] = max_val
        accessors.append(acc)
        return len(accessors) - 1

    pos_view = add_buffer_view(V.tobytes(), 34962)
    pos_idx = add_accessor(pos_view, 5126, V.shape[0], "VEC3", V.min(axis=0).tolist(), V.max(axis=0).tolist())
    uv_view = add_buffer_view(UV.tobytes(), 34962)
    uv_idx = add_accessor(uv_view, 5126, UV.shape[0], "VEC2")

    images = []
    textures = []
    materials = []

    for i, mat in enumerate(mesh_obj.materials):
        tex_name = f"mat_{i}_diffuse.png"
        tex_path = os.path.join(folder, tex_name)
        texture.save_texture2D(tex_path, texture.rgb_to_srgb(mat['kd']))
        images.append({"uri": tex_name})
        textures.append({"sampler": 0, "source": i})
        materials.append({
            "name": getattr(mat, 'name', f"material_{i}"),
            "pbrMetallicRoughness": {"baseColorTexture": {"index": i}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "doubleSided": True
        })

    primitives = []
    unique_mats = np.unique(MF)
    for m_id in unique_mats:
        mask = (MF == m_id)
        sub_faces = F[mask].flatten()
        if sub_faces.max() < 65536:
            ind_bytes = sub_faces.astype(np.uint16).tobytes();
            comp = 5123
        else:
            ind_bytes = sub_faces.astype(np.uint32).tobytes();
            comp = 5125
        ind_idx = add_accessor(add_buffer_view(ind_bytes, 34963), comp, sub_faces.shape[0], "SCALAR")
        primitives.append(
            {"attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx}, "indices": ind_idx, "material": int(m_id),
             "mode": 4})

    gltf_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": bin_name, "byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": images,
        "textures": textures,
        "materials": materials,
        "meshes": [{"name": "baked_mesh", "primitives": primitives}],
        "nodes": [{"name": "root", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0
    }

    with open(gltf_filename, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    with open(bin_filename, 'wb') as f:
        f.write(buffer_data)