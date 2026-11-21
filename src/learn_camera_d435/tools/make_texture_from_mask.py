#!/usr/bin/env python3
"""用一张已有二值蒙版（灯窗轮廓）生成带贴图的 mesh。

蒙版的像素坐标按 --mask-bounds 映射到物体 XY 平面（单位 mm）。
贴图采用平面投影 UV，uv_span 必须大于物体最大尺寸，保证 UV 落在 [0,1]。

用法：
    python3 tools/make_texture_from_mask.py \
        --mesh energy_unit.obj \
        --mask /tmp/window_keep.npy \
        --mask-bounds -50 -55 50 55 \
        --out energy_unit_textured.obj
"""

import argparse
import os

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial


def build_texture(mask, bounds, uv_span, size):
    """把 mm 坐标下的蒙版重采样到 uv 空间的方形贴图。"""
    x1, y1, x2, y2 = bounds
    mh, mw = mask.shape
    u = (np.arange(size) + 0.5) / size
    v = (np.arange(size) + 0.5) / size
    X = (u - 0.5) * uv_span                      # mm
    Y = (0.5 - v) * uv_span                      # 图像 y 向下
    mx = (X - x1) / (x2 - x1) * (mw - 1)
    my = (y2 - Y) / (y2 - y1) * (mh - 1)
    mx = np.clip(mx, 0, mw - 1).astype(int)
    my = np.clip(my, 0, mh - 1).astype(int)
    sel = mask[np.ix_(my, mx)] > 0
    img = np.zeros((size, size, 3), np.uint8)
    img[sel] = (255, 0, 0)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mask", required=True, help=".npy 或图片")
    ap.add_argument("--mask-bounds", nargs=4, type=float, required=True,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="蒙版覆盖的物体 XY 范围（mm）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--uv-span", type=float, default=150.0)
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)])

    if args.mask.endswith(".npy"):
        mask = np.load(args.mask)
    else:
        mask = np.array(Image.open(args.mask).convert("L"))
    print("蒙版:", mask.shape, "非零像素", int((mask > 0).sum()))

    normals = m.face_normals
    cent = m.triangles_center * 1000.0
    # 灯窗是通孔/深槽，顶面在窗口处没有几何，光从下面的内表面透出来。
    # 所以 z 下限放到 3mm，把窗口底下的内表面也纳入平面投影。
    on_top = ((np.abs(normals[:, 2]) > 0.5)
              & (cent[:, 2] > 3.0) & (cent[:, 2] < 20.0))
    print(f"顶面面数: {on_top.sum()} / {len(m.faces)}")

    top = m.submesh([np.where(on_top)[0]], append=True)
    vt = np.asarray(top.vertices)
    ft = np.asarray(top.faces)
    uv_top = np.stack([0.5 + vt[:, 0] * 1000.0 / args.uv_span,
                       0.5 + vt[:, 1] * 1000.0 / args.uv_span], axis=1)

    rest = m.faces[~on_top]
    vr = m.vertices[rest].reshape(-1, 3)
    fr = np.arange(len(vr)).reshape(-1, 3)
    uv_rest = np.tile([0.5, 0.005], (len(vr), 1))

    mesh = trimesh.Trimesh(
        vertices=np.vstack([vt, vr]),
        faces=np.vstack([ft, fr + len(vt)]).astype(np.int64),
        process=False)
    tex = build_texture(mask, args.mask_bounds, args.uv_span, args.size)
    mesh.visual = TextureVisuals(
        uv=np.vstack([uv_top, uv_rest]),
        material=SimpleMaterial(image=tex, name="part4_mark"))

    out = os.path.abspath(args.out)
    mesh.export(out)
    print("已导出:", out, "| 面数:", len(mesh.faces))


if __name__ == "__main__":
    main()
