#!/usr/bin/env python3
"""用 FoundationPose 自己的渲染器渲染 mesh，检查贴图/几何是否正确。

用法：
    python3 tools/render_mesh.py --mesh .../energy_unit_textured.obj --out /tmp/view

会输出三张图：
    <out>_top.png    顶视图
    <out>_cam.png    模拟相机视角（按 vis_dir 放相机）
    <out>_iso.png    45 度斜视
"""

import argparse
import sys

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, "/home/cheng/test/RCIA_Benmao_Vision/foundation_pose")
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402


def look_at(eye, target, up):
    """世界 -> 相机（OpenCV 约定：+z 朝前，+y 朝下）。"""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    # 基向量必须放成行：cam = (r·p, d·p, f·p)
    T[:3, :3] = np.stack([r, d, f], axis=0)
    T[:3, 3] = -T[:3, :3] @ eye
    return T


def render(mt, pose, K, H, W, glctx):
    color, depth, _ = nvdiffrast_render(
        K=K, H=H, W=W,
        ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float, device="cuda"),
        glctx=glctx, mesh_tensors=mt, output_size=(H, W), use_light=True)
    rgb = (color[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    d = depth[0].detach().cpu().numpy()
    return rgb, d


def save(rgb, d, path, pad=12):
    ys, xs = np.where(d > 0)
    if not len(xs):
        print("  [警告] 该视角没有几何，跳过", path)
        return
    H, W = d.shape
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    vis = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
    sil = ((d[y0:y1, x0:x1] > 0) * 255).astype(np.uint8)
    cont, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cont, -1, (0, 255, 0), 1)
    vis = cv2.resize(vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, vis)
    print("  已保存", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", default="/tmp/mesh_view")
    ap.add_argument("--vis-dir", nargs=3, type=float,
                    default=[-0.389, -0.195, 0.900])
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)])
    mat = getattr(m.visual, "material", None)
    print("面数:", len(m.faces), "| visual:", type(m.visual).__name__,
          "| 材质:", type(mat).__name__ if mat else None)
    if mat is not None and getattr(mat, "image", None) is not None:
        print("贴图:", mat.image.size)

    mt = make_mesh_tensors(m)
    glctx = dr.RasterizeCudaContext()
    K = np.array([[700., 0, 400], [0, 700, 300], [0, 0, 1]])
    H, W = 600, 800
    center = np.array([0.0, 0.0, 0.05])
    d = np.array(args.vis_dir, dtype=float)
    d /= np.linalg.norm(d)

    poses = {
        "top": (center + np.array([0, 0, 0.45]), center, np.array([0, 1., 0])),
        "cam": (center + d * 0.45, center, np.array([0, 0, 1.])),
        "iso": (center + np.array([0.20, -0.20, 0.18]), center,
                np.array([0, 0, 1.])),
    }
    for name, (eye, tgt, up) in poses.items():
        pose = look_at(eye, tgt, up)
        rgb, depth = render(mt, pose, K, H, W, glctx)
        save(rgb, depth, f"{args.out}_{name}.png")


if __name__ == "__main__":
    main()
