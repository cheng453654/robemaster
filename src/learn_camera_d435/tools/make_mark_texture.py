#!/usr/bin/env python3
"""按 RM 规则图 5-8（科技核心 MARK 灯）生成带 L 形灯带贴图的 mesh。

尺寸取自规则手册：
    MARK 端面 80×80 mm，灯带外沿到角点 30 mm，灯带宽 8 mm，距边缘 8 mm。
顶视图排布：左上 / 左下 / 右下 是 L，右上是一组独立小方块（不对称，
用来打破柱体的绕轴朝向歧义）。

输出：OBJ + MTL + PNG，坐标系保持原始 mesh（Z 向上），不改几何。

用法：
    python3 tools/make_mark_texture.py \
        --mesh /home/cheng/test/RCIA_Benmao_Vision/models/object/part4_cylinder_base.obj \
        --out  /home/cheng/test/RCIA_Benmao_Vision/models/object/part4_textured.obj

    # 180 度对不上实物时：--rot 180
"""

import argparse
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial


def pattern_rects(face, margin, arm, width):
    """返回若干 (x1, y1, x2, y2)，单位 mm，原点在 MARK 面中心。"""
    h = face / 2.0
    W, A, M = width, arm, margin
    r = []
    # 左上 L
    r.append((-h + M, h - M - A, -h + M + W, h - M))          # 竖臂
    r.append((-h + M, h - M - W, -h + M + A, h - M))          # 横臂
    # 左下 L
    r.append((-h + M, -h + M, -h + M + W, -h + M + A))
    r.append((-h + M, -h + M, -h + M + A, -h + M + W))
    # 右下 L
    r.append((h - M - W, -h + M, h - M, -h + M + A))
    r.append((h - M - A, -h + M, h - M, -h + M + W))
    # 右上：两个方块 + 一个下方块（和另外三个角不同，制造不对称）
    r.append((12.0, h - M - W, 20.0, h - M))                  # 靠中
    r.append((24.0, h - M - W, 32.0, h - M))                  # 靠右
    r.append((24.0, h - M - 3 * W, 32.0, h - M - 2 * W))      # 下方
    return r


def build_texture(rects, uv_span, size, rot):
    """uv_span：整张 UV 图在物体坐标里覆盖的毫米数（图案只用中间一部分）。"""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    ppm = size / uv_span                           # 每毫米像素数
    cx = cy = size / 2.0
    for x1, y1, x2, y2 in rects:
        # 物体坐标 +y 向上 -> 图像 y 向下
        px1, px2 = cx + x1 * ppm, cx + x2 * ppm
        py1, py2 = cy - y2 * ppm, cy - y1 * ppm
        d.rectangle([px1, py1, px2, py2], fill=(255, 0, 0))
    if rot:
        img = img.rotate(rot, resample=Image.NEAREST)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face", type=float, default=80.0, help="MARK 端面边长 mm")
    ap.add_argument("--margin", type=float, default=8.0, help="距边缘 mm")
    ap.add_argument("--arm", type=float, default=30.0, help="L 臂长 mm")
    ap.add_argument("--width", type=float, default=8.0, help="灯带宽 mm")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--rot", type=float, default=0.0,
                    help="图案旋转角度（实物方向对不上时用 90/180/270）")
    ap.add_argument("--subdiv", type=float, default=0.0,
                    help="细分最大边长 mm，0 表示不细分（默认不细分）")
    ap.add_argument("--top-z-min", type=float, default=5.0,
                    help="MARK 面所在的高度下限 mm")
    ap.add_argument("--top-z-max", type=float, default=20.0,
                    help="MARK 面所在的高度上限 mm")
    ap.add_argument("--uv-span", type=float, default=150.0,
                    help="UV 图覆盖的物体尺寸 mm，必须大于底座最大尺寸，"
                         "保证所有顶点 UV 落在 [0,1] 内")
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)])

    normals = m.face_normals
    cent = m.triangles_center * 1000.0            # mm
    # STL 的绕序可能不一致，用 |nz| 判断水平面；z 范围可调
    on_top = ((np.abs(normals[:, 2]) > 0.5)
              & (cent[:, 2] > args.top_z_min)
              & (cent[:, 2] < args.top_z_max))
    print(f"参与贴图的顶面: {on_top.sum()} / {len(m.faces)}")

    top_mesh = m.submesh([np.where(on_top)[0]], append=True)
    if args.subdiv > 0:
        vt, ft = trimesh.remesh.subdivide_to_size(
            top_mesh.vertices, top_mesh.faces, max_edge=args.subdiv * 0.001)
    else:
        vt, ft = np.asarray(top_mesh.vertices), np.asarray(top_mesh.faces)
    # 平面投影，但缩放到整张 UV 空间，保证所有顶点都落在 [0,1] 内。
    # 这样不需要 clip，大三角形也不会拉伸，可以不细分。
    uv_top = np.stack([0.5 + vt[:, 0] * 1000.0 / args.uv_span,
                       0.5 + vt[:, 1] * 1000.0 / args.uv_span], axis=1)
    print(f"顶面: {len(vt)} 顶点 / {len(ft)} 面  "
          f"UV 范围 u[{uv_top[:,0].min():.2f},{uv_top[:,0].max():.2f}] "
          f"v[{uv_top[:,1].min():.2f},{uv_top[:,1].max():.2f}]")

    # 其余面：逐面拆顶点，统一压到纹理底部的黑色区域
    rest = m.faces[~on_top]
    vr = m.vertices[rest].reshape(-1, 3)
    fr = np.arange(len(vr)).reshape(-1, 3)
    uv_rest = np.tile([0.5, 0.005], (len(vr), 1))

    v = np.vstack([vt, vr])
    f = np.vstack([ft, fr + len(vt)]).astype(np.int64)
    uv = np.vstack([uv_top, uv_rest])
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    tex = build_texture(pattern_rects(args.face, args.margin, args.arm,
                                      args.width), args.uv_span, args.size,
                        args.rot)
    mat = SimpleMaterial(image=tex, name="part4_mark")
    mesh.visual = TextureVisuals(uv=uv, material=mat)

    out = os.path.abspath(args.out)
    mesh.export(out)
    print("已导出:", out)
    for f2 in sorted(os.listdir(os.path.dirname(out))):
        if f2.startswith("part4_mark") or f2.startswith("material"):
            print("  ", f2)


if __name__ == "__main__":
    main()
