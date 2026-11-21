#!/usr/bin/env python3
"""从 CAD 提取灯窗轮廓，补成平面，再按轮廓生成红色贴图。

流程：
  1. 从 mesh 的顶面（z≈17mm）提取开孔轮廓
  2. 在 z=17mm 处补上平面（灯窗原来是通孔，没有面可涂）
  3. 按同一轮廓生成 2048² 贴图
  4. 平面投影 UV，导出 OBJ

用法：
    python3 tools/make_mark_from_cad.py \
        --mesh .../energy_unit.obj \
        --out  .../energy_unit_textured.obj
"""

import argparse
import os

import cv2
import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

RES = 0.25          # mm/px
X0, X1 = -50.0, 50.0
Y0, Y1 = 55.0, -55.0   # 图像第 0 行 = y=+55
TOP_Z = 17.0        # 顶面高度 mm
UV_SPAN = 150.0


def to_px(x, y):
    return (x - X0) / RES, (Y0 - y) / RES


def from_px(col, row):
    return X0 + col * RES, Y0 - row * RES


def top_raster(m):
    W = int(round((X1 - X0) / RES)) + 1
    H = int(round((Y0 - Y1) / RES)) + 1
    img = np.zeros((H, W), np.uint8)
    v = m.vertices * 1000.0
    f = m.faces
    c = m.triangles_center * 1000.0
    n = m.face_normals
    sel = (np.abs(n[:, 2]) > 0.5) & (c[:, 2] > 16.5) & (c[:, 2] < 17.5)
    for i in np.where(sel)[0]:
        pts = np.array([to_px(*v[fi][:2]) for fi in f[i]])
        cv2.fillPoly(img, [pts.astype(np.int32)], 255)
    return img


def window_mask(top):
    cont, _ = cv2.findContours(top, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    filled = top.copy()
    cv2.drawContours(filled, cont, -1, 255, -1)
    win = cv2.bitwise_and(filled, cv2.bitwise_not(top))

    H, W = win.shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    X = X0 + cols * RES
    Y = Y0 - rows * RES
    win[(X ** 2 + Y ** 2) < 27.0 ** 2] = 0          # 中心柱

    n, lab, stats, _ = cv2.connectedComponentsWithStats(win, 8)
    keep = np.zeros_like(win)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * RES * RES >= 30:
            keep[lab == i] = 255
    return keep


def fill_geometry(win, z=TOP_Z):
    """用 earcut 把每个灯窗轮廓三角化成平面。"""
    contours, hier = cv2.findContours(win, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0] if hier is not None else np.zeros((len(contours), 4), int)
    verts, faces = [], []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:          # 内轮廓由父轮廓处理
            continue
        shell = cv2.approxPolyDP(cnt, 0.8, True).reshape(-1, 2)
        if len(shell) < 3:
            continue
        poly = Polygon([from_px(c, r) for c, r in shell])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 20:
            continue
        try:
            v2, f2 = trimesh.creation.triangulate_polygon(poly, engine="earcut")
        except Exception:  # noqa: BLE001
            continue
        base = len(verts)
        verts.extend([[p[0], p[1], z] for p in v2])
        faces.extend(f2 + base)
    return np.array(verts, float), np.array(faces, int)


def build_texture(win, size):
    h, w = win.shape
    u = (np.arange(size) + 0.5) / size
    X = (u - 0.5) * UV_SPAN
    Y = (0.5 - u) * UV_SPAN
    cols = np.clip(((X - X0) / RES).astype(int), 0, w - 1)
    rows = np.clip(((Y0 - Y) / RES).astype(int), 0, h - 1)
    sel = win[np.ix_(rows, cols)] > 0
    img = np.zeros((size, size, 3), np.uint8)
    img[sel] = (255, 0, 0)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)])
    print("原 mesh:", len(m.faces), "面")

    top = top_raster(m)
    win = window_mask(top)
    print("灯窗面积: %.0f mm²" % (win.sum() / 255 * RES * RES))

    fv, ff = fill_geometry(win)
    print("补面: %d 顶点 / %d 面" % (len(fv), len(ff)))

    # 合并：原几何（米） + 补面（mm → 米）
    verts = np.vstack([m.vertices, fv * 0.001])
    faces = np.vstack([m.faces, ff + len(m.vertices)]).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    n = mesh.face_normals
    c = mesh.triangles_center * 1000.0
    on_tex = (np.abs(n[:, 2]) > 0.5) & (c[:, 2] > 3.0) & (c[:, 2] < 20.0)
    print("参与贴图的面:", int(on_tex.sum()), "/", len(mesh.faces))

    top_mesh = mesh.submesh([np.where(on_tex)[0]], append=True)
    vt = np.asarray(top_mesh.vertices)
    ft = np.asarray(top_mesh.faces)
    uv_top = np.stack([0.5 + vt[:, 0] * 1000.0 / UV_SPAN,
                       0.5 + vt[:, 1] * 1000.0 / UV_SPAN], axis=1)

    rest = mesh.faces[~on_tex]
    vr = mesh.vertices[rest].reshape(-1, 3)
    fr = np.arange(len(vr)).reshape(-1, 3)
    uv_rest = np.tile([0.5, 0.005], (len(vr), 1))

    out_mesh = trimesh.Trimesh(
        vertices=np.vstack([vt, vr]),
        faces=np.vstack([ft, fr + len(vt)]).astype(np.int64),
        process=False)
    out_mesh.visual = TextureVisuals(
        uv=np.vstack([uv_top, uv_rest]),
        material=SimpleMaterial(image=build_texture(win, args.size),
                                name="part4_mark"))

    out = os.path.abspath(args.out)
    out_mesh.export(out)
    print("已导出:", out, "| 面数:", len(out_mesh.faces))


if __name__ == "__main__":
    main()
