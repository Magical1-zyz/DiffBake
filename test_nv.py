# 新建一个 test_nv.py 测试
import torch
import nvdiffrast.torch as dr


def test():
    glctx = dr.RasterizeGLContext()
    print("Context created successfully")
    # 简单的三角形光栅化测试
    pos = torch.tensor([[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]], dtype=torch.float32, device='cuda')
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[256, 256])
    print("Rasterization success")


if __name__ == "__main__":
    test()
