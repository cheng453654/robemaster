#!/usr/bin/python3
"""D435i + FoundationPose：YOLO/手动框选 + 可选 SAM 分割，实时 6DoF 位姿。

运行时只需要：
  - conda 环境 foundationpose_ros（torch / nvdiffrast / mycuda / SAM 都在里面）
  - 目标物体的 CAD 模型（默认用能量单元 Str3D.obj）

用法（注意必须先在终端里激活环境）：
    conda activate foundationpose_ros
    source /home/cheng/test/RCIA_Benmao_Vision/scripts/foundationpose_env.sh
    python learn_foundationpose.py
    python learn_foundationpose.py --mask-mode sam     # 默认全自动，无需人工画框
    # 全自动：YOLO 搜索 -> SAM -> 注册 -> 跟踪中 YOLO 校验 -> 跟丢自动重注册
    python learn_foundationpose.py --yolo \
        --yolo-model /home/cheng/datasets/part4_yolo/runs/part4_v2_132/weights/best.pt \
        --yolo-class part4 --yolo-conf 0.7 --mask-mode sam \
        --mesh /home/cheng/test/RCIA_Benmao_Vision/models/object/part4_cylinder_base.obj \
        --axes-frame origin --vis-mesh

常用参数：
    --mesh /path/to/object.obj     换目标物体
    --mesh-scale 0.001             模型单位不是米时换算（mm -> m）
    --rect 200,120,240,240         跳过鼠标框选，直接给定 (x,y,w,h)
    --frames 30 --no-display       无窗口自检
    --mask-mode rect|grabcut|sam   mask 生成方式（默认 rect 直接填框）
    --yolo                         自动搜索（默认开启，保留只是为了兼容旧命令）
    --no-yolo                      关闭自动搜索，改用手动框选
    --manual-fallback              YOLO 找不到目标时才允许手动框选（默认不弹框）
    --yolo-model PATH              YOLO 权重，默认 models/yolo/best-v2.pt
    --yolo-class 0|armor           只保留指定类别（id 或名字）
    --yolo-conf 0.7                搜索/首次锁定置信度（小数据集建议 0.7）
    --yolo-verify-conf 0.25        跟踪校验置信度（低一些，避免误判丢失）
    --yolo-track-check             跟踪期间持续跑 YOLO 校验是否跟丢（默认开）
    --no-yolo-track-check          关闭跟踪期间的 YOLO 丢失校验
    --yolo-check-every 1           每 N 帧校验一次
    --yolo-lock-iou 0.25           YOLO 框与 FP 投影框的 IoU 低于此值算不一致
    --yolo-lost-frames 8           连续 N 次不一致 -> 判定跟丢并重新注册
    --yolo-miss-frames 15          连续 N 次没检出目标 -> 判定跟丢并重新注册
    --yolo-mask-preview            YOLO 框也显示 mask 预览（默认自动接受）
    --sam-checkpoint PATH          SAM 权重，默认用项目里的 vit_b
    --sam-select score|area        SAM 多候选选择，默认 score
    --depth-filter                 深度空间+时间滤波，减少跟踪抖动
    --smooth 0.5                   位姿平滑（0=关闭），抑制逐帧抖动
    --track-refine-iter 5          每帧多迭代几次，位姿更稳但更慢
    --motion-predict 1.0           快速运动补偿：用前两帧外推本帧跟踪初值
    --crop-ratio 1.5               放大跟踪裁剪窗，快速运动时物体不易跑出窗口
    --no-depth-temporal            快速运动时关掉深度时间滤波，避免拖影
    --vis-mesh                     跟踪窗口实时叠加 mesh 投影轮廓，直观看位姿贴合度
    --axes-frame obb|mesh|origin   画 OBB 轴（默认）/ AABB 中心的 mesh 轴 / mesh 原点的 mesh 轴
    --eval --eval-frames 100       评估模式：IoU/覆盖率、深度误差、静止抖动、diff 图

运行流程：
    1. YOLO 搜索目标 -> 2. 把框交给 SAM 分割 -> 3. FoundationPose 注册
    4. 跟踪期间 YOLO 继续运行，比较 YOLO 框和 FP 投影框
    5. 连续不一致/连续检不到目标 -> 判定跟丢 -> 回到第 1 步自动重注册
    搜索和注册期间窗口会持续显示实时画面/状态，不再卡住

窗口操作：
    默认不需要画框：YOLO 找不到目标时会持续自动重试
    --no-yolo：手动拖框，SPACE/ENTER 确认
    --manual-fallback：只有 YOLO 失败时才弹出鼠标框选
    跟踪时：q / ESC 退出，r 立即触发重新搜索/分割/注册
    （手动 mask 预览：SPACE=接受，r=重新框选，ESC=退出；
      加 --no-mask-preview 可跳过；YOLO 框默认不预览）
输出：
    终端逐帧打印 4x4 位姿（object-in-camera）；--pose-output 可存文件
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# ---- FoundationPose 代码路径（可用环境变量覆盖）----
PROJECT_ROOT = os.environ.get("RCIA_ROOT", "/home/cheng/test/RCIA_Benmao_Vision")
FP_ROOT = os.environ.get("FOUNDATIONPOSE_ROOT", os.path.join(PROJECT_ROOT, "foundation_pose"))
for _p in (PROJECT_ROOT, FP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fp_matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/fp_ultralytics")
# 6GB 显存容易碎片化，减少 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import trimesh

# torch 2.6 起 torch.load 默认 weights_only=True，官方权重加载不了
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import nvdiffrast.torch as dr
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis

DEFAULT_MESH = os.path.join(PROJECT_ROOT, "models/object/Str3D.obj")
DEFAULT_SAM = os.path.join(PROJECT_ROOT, "models/sam/sam_vit_b_01ec64.pth")
WINDOW_NAME = "FoundationPose (rect mask)"


class UserQuit(Exception):
    """用户在搜索/注册过程中按 q/ESC。"""
DEFAULT_YOLO = os.path.join(PROJECT_ROOT, "models/yolo/best-v2.pt")


def parse_args():
    p = argparse.ArgumentParser(description="D435i + FoundationPose（YOLO/手动框选）")
    p.add_argument("--mesh", default=DEFAULT_MESH, help="目标物体 CAD 模型 (.obj)")
    p.add_argument("--mesh-scale", type=float, default=1.0,
                   help="mesh 单位换算到米，例如 mm 模型用 0.001；默认 1.0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--est-refine-iter", type=int, default=5)
    p.add_argument("--track-refine-iter", type=int, default=2)
    p.add_argument("--crop-ratio", type=float, default=1.2,
                   help="跟踪裁剪窗口相对物体直径的倍数；快速运动可加到 1.5~2.0，"
                        "但太大会降低定位分辨率")
    p.add_argument("--motion-predict", type=float, default=0.0,
                   help="快速运动补偿：按最近两段位移外推跟踪初值，0=关闭；"
                        "建议 0.5~1.0，配合 --track-refine-iter 4~5")
    p.add_argument("--no-depth-temporal", action="store_true",
                   help="配合 --depth-filter：只做空间滤波，关闭时间滤波，"
                        "快速运动时避免深度拖影")
    p.add_argument("--depth-filter", action="store_true",
                   help="对深度做空间+时间滤波，明显减少抖动（会稍增延迟）")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="位姿平滑系数 0~1，0=关闭；建议 0.4~0.6")
    p.add_argument("--rect", default=None, help="跳过框选，格式 x,y,w,h")
    p.add_argument("--yolo", dest="yolo", action="store_true", default=True,
                   help="启动后自动用 YOLO 搜索目标，把框交给 SAM/注册；"
                        "跟踪期间持续校验，跟丢自动重新注册（默认开启）")
    p.add_argument("--no-yolo", dest="yolo", action="store_false",
                   help="关闭 YOLO 自动搜索，退回手动框选")
    p.add_argument("--manual-fallback", action="store_true",
                   help="YOLO 找不到目标时允许手动框选；默认不弹框，"
                        "一直自动重试搜索")
    p.add_argument("--yolo-model", default=DEFAULT_YOLO,
                   help="YOLO 权重 (.pt)")
    p.add_argument("--yolo-class", default=None,
                   help="只保留指定类别，写类别 id（如 0）或名字（如 armor）；"
                        "默认保留所有类别")
    p.add_argument("--yolo-conf", type=float, default=0.5,
                   help="YOLO 搜索/首次锁定用的置信度阈值，默认 0.5")
    p.add_argument("--yolo-verify-conf", type=float, default=0.25,
                   help="跟踪校验用的 YOLO 置信度，默认 0.25；"
                        "比 --yolo-conf 低，避免正常跟踪时因置信度略低被误判丢失")
    p.add_argument("--yolo-select", choices=["conf", "area"], default="conf",
                   help="多个候选时选置信度最高（默认）还是面积最大")
    p.add_argument("--yolo-wait", type=float, default=2.0,
                   help="等待 YOLO 检出目标的最长时间（秒），超时回退手动框选")
    p.add_argument("--yolo-stable", type=int, default=1,
                   help="连续多少帧检测框 IoU>0.5 才采用，默认 1（立即采用）")
    p.add_argument("--yolo-track-check", dest="yolo_track_check",
                   action="store_true", default=True,
                   help="跟踪期间持续运行 YOLO，校验 FoundationPose 是否跟丢（默认开）")
    p.add_argument("--no-yolo-track-check", dest="yolo_track_check",
                   action="store_false",
                   help="关闭跟踪期间的 YOLO 丢失校验")
    p.add_argument("--yolo-check-every", type=int, default=1,
                   help="每多少帧运行一次 YOLO 校验，默认 1（每帧）")
    p.add_argument("--yolo-lock-iou", type=float, default=0.25,
                   help="YOLO 框与 FoundationPose 投影框的 IoU 低于该值时记为不一致，"
                        "默认 0.25")
    p.add_argument("--yolo-lost-frames", type=int, default=8,
                   help="连续多少次 YOLO 框与 FP 投影不一致后判定跟丢，默认 8")
    p.add_argument("--yolo-miss-frames", type=int, default=15,
                   help="连续多少次 YOLO 没检出目标后判定跟丢，默认 15")
    p.add_argument("--yolo-mask-preview", action="store_true",
                   help="YOLO 自动框也显示 mask 预览；默认自动接受，"
                        "便于跟丢后自动重新注册")
    p.add_argument("--mask-mode", choices=["rect", "grabcut", "sam"], default="rect",
                   help="rect=直接填框，grabcut=颜色细化，sam=SAM 分割")
    p.add_argument("--no-mask-preview", action="store_true",
                   help="框选后不显示 mask 预览，直接开始注册")
    p.add_argument("--sam-checkpoint", default=DEFAULT_SAM)
    p.add_argument("--sam-select", choices=["score", "area"], default="score",
                   help="SAM 多候选选择：score=置信度最高（默认）；"
                        "area=面积最大（有时能多包含低对比部分）")
    p.add_argument("--sam-model-type", default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--frames", type=int, default=0, help="采集多少帧后退出，0=一直跑")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--vis-mesh", action="store_true",
                   help="跟踪窗口叠加 mesh 投影轮廓 + 坐标轴，直观检查位姿贴合度")
    p.add_argument("--axes-frame", choices=["obb", "mesh", "origin"], default="obb",
                   help="--vis-mesh/--eval 的坐标轴参考：obb=有向包围盒轴（默认，原行为）；"
                        "mesh=mesh 文件自身的 X/Y/Z 轴，画在 AABB 中心；"
                        "origin=mesh 自身 X/Y/Z 轴，画在 mesh 原点 (0,0,0)"
                        "（自定义原点的网格用这个）")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--debug-dir", default="/tmp/learn_fp_debug")
    p.add_argument("--pose-output", default=None, help="每帧位姿保存路径 (.txt)")
    p.add_argument("--save-vis", default=None, help="可视化图片保存目录")
    p.add_argument("--eval", action="store_true",
                   help="评估模式：投影线框、算 mask IoU / 深度误差 / 静止稳定性")
    p.add_argument("--eval-frames", type=int, default=100,
                   help="评估帧数，静态场景建议 100（评估模式会覆盖 --frames）")
    p.add_argument("--eval-out", default="/tmp/fp_eval",
                   help="评估可视化输出目录")
    return p.parse_args()


def load_sam(checkpoint, model_type="vit_b", device="cuda"):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"找不到 SAM 权重: {checkpoint}")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


_sam_state = {"predictor": None}


def get_sam(checkpoint, model_type):
    """懒加载 SAM：只在需要分割时加载。"""
    if _sam_state["predictor"] is None:
        print(f"[fp] 加载 SAM ({model_type})...")
        _sam_state["predictor"] = load_sam(checkpoint, model_type)
        print("[fp] SAM 就绪")
    return _sam_state["predictor"]


def release_sam():
    """注册前释放 SAM，把显存让给 FoundationPose（6GB 卡必须这么做）。"""
    if _sam_state["predictor"] is not None:
        _sam_state["predictor"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_mask(color_bgr, rect, mode, sam_checkpoint, sam_model_type,
              sam_select="score"):
    """生成 mask；SAM 模式用完立即释放显存。"""
    predictor = get_sam(sam_checkpoint, sam_model_type) if mode == "sam" else None
    try:
        return build_mask(color_bgr, rect, mode, predictor, sam_select)
    finally:
        if mode == "sam":
            release_sam()


def build_mask(color_bgr, rect, mode, sam_predictor=None, sam_select="score"):
    """在给定矩形内生成物体 mask。"""
    x, y, w, h = [int(v) for v in rect]
    H, W = color_bgr.shape[:2]
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)
    mask = np.zeros((H, W), np.uint8)

    if mode == "rect":
        mask[y:y + h, x:x + w] = 255
        return mask

    if mode == "sam":
        if sam_predictor is None:
            raise RuntimeError("mask-mode=sam 但没有加载 SAM")
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(rgb)
        masks, scores, _ = sam_predictor.predict(
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
            multimask_output=True,
        )
        if sam_select == "area":
            # 面积最大的候选有时能多包含暗色/低对比的附属部分
            best = int(np.argmax([m.sum() for m in masks]))
        else:
            best = int(np.argmax(scores))
        mask = (masks[best].astype(np.uint8)) * 255
        # SAM 有时会把背景一起框进来，限制在框附近
        pad = 10
        keep = np.zeros((H, W), np.uint8)
        keep[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad] = 255
        mask[keep == 0] = 0
        # 兜底：分割太小就退回矩形
        if mask.sum() < 0.03 * 255 * w * h:
            mask[:] = 0
            mask[y:y + h, x:x + w] = 255
        return mask

    # GrabCut：用矩形初始化，再按颜色收边界
    gc = np.zeros((H, W), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(color_bgr, gc, (x, y, w, h), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask[np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))] = 255
    # 失败兜底：分割太小就退回矩形
    if mask.sum() < 0.2 * 255 * w * h:
        mask[:] = 0
        mask[y:y + h, x:x + w] = 255
    return mask


def mask_overlay(color_bgr, mask):
    """把 mask 半透明绿色叠加到原图上，返回 RGB 图。"""
    vis = color_bgr.copy()
    green = np.zeros_like(vis)
    green[:] = (0, 255, 0)
    sel = mask > 0
    vis[sel] = (0.5 * vis[sel] + 0.5 * green[sel]).astype(np.uint8)
    cv2.putText(vis, f"mask: {int(sel.sum())} px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def save_mask_vis(color_bgr, mask, out_dir, name):
    vis = mask_overlay(color_bgr, mask)
    cv2.imwrite(os.path.join(out_dir, name), vis[..., ::-1])


def confirm_mask_preview(color_bgr, mask, mode):
    """显示 mask 预览，等待用户确认。返回 accept / retry / quit。"""
    vis = mask_overlay(color_bgr, mask)
    cv2.putText(vis, f"{mode} | SPACE=accept  r=reselect  ESC=quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow("mask preview", vis[..., ::-1])
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32):      # ENTER / SPACE
            action = "accept"
            break
        if key == ord("r"):
            action = "retry"
            break
        if key in (27, ord("q")):    # ESC / q
            action = "quit"
            break
        # 用户点了窗口右上角的关闭
        if cv2.getWindowProperty("mask preview", cv2.WND_PROP_VISIBLE) < 1:
            action = "quit"
            break
    cv2.destroyWindow("mask preview")
    return action


# ---------------- 位姿评估工具 ----------------

def project_mesh(mesh, pose, K):
    v = np.asarray(mesh.vertices)
    vh = np.hstack([v, np.ones((v.shape[0], 1))])
    cam = (pose @ vh.T).T[:, :3]
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return cam, uv


def mesh_silhouette(mesh, pose, K, shape):
    cam, uv = project_mesh(mesh, pose, K)
    sil = np.zeros(shape[:2], np.uint8)
    polys = []
    for f in np.asarray(mesh.faces):
        if np.any(cam[f, 2] <= 0):
            continue
        polys.append(np.round(uv[f]).astype(np.int32))
    if polys:
        cv2.fillPoly(sil, polys, 255)
        # 相邻三角形各自取整后填充，会在公共边上留下 1~2px 裂缝，
        # 不补掉会把 IoU 压低（实测 0.78 -> 0.94）。闭运算只补缝隙。
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return sil


def silhouette_iou(sil, mask):
    inter = np.count_nonzero((sil > 0) & (mask > 0))
    union = np.count_nonzero((sil > 0) | (mask > 0))
    return float(inter) / float(union) if union else 0.0


def silhouette_stats(sil, mask):
    """返回 (IoU, coverage, precision)。

    coverage = |sil ∩ mask| / |mask|   mask 有多少被投影模型覆盖（召回）
    precision = |sil ∩ mask| / |sil|   投影有多少落在 mask 内
    coverage 高、IoU 低 => mask 只是物体的一部分（漏分割），不是位姿错。
    """
    inter = np.count_nonzero((sil > 0) & (mask > 0))
    sil_area = np.count_nonzero(sil > 0)
    mask_area = np.count_nonzero(mask > 0)
    union = sil_area + mask_area - inter
    iou = float(inter) / float(union) if union else 0.0
    cover = float(inter) / float(mask_area) if mask_area else 0.0
    prec = float(inter) / float(sil_area) if sil_area else 0.0
    return iou, cover, prec


def depth_consistency(mesh, pose, K, depth, n_samples=4000):
    """采样 mesh 正面点，和投影像素处的实测深度比较。

    返回 (中位误差 m, 90 分位误差 m, 1cm 内比例, 样本数)，样本不足返回 None。
    """
    pts, fidx = mesh.sample(n_samples, return_index=True)
    normals = np.asarray(mesh.face_normals)[fidx]
    cam = (pose @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    n_cam = normals @ pose[:3, :3].T
    front = np.einsum("ij,ij->i", n_cam, -cam) > 0   # 法线朝向相机
    valid = front & (cam[:, 2] > 0.01)
    cam = cam[valid]
    if len(cam) < 20:
        return None
    uv = cam @ K.T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    H, W = depth.shape[:2]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, cam = u[inside], v[inside], cam[inside]
    d = depth[v, u]
    ok = d > 0.05
    if ok.sum() < 20:
        return None
    err = np.abs(cam[ok, 2] - d[ok])
    return (float(np.median(err)), float(np.percentile(err, 90)),
            float((err < 0.01).mean()), int(ok.sum()))


def pose_delta(T0, T):
    """相对首帧的平移 (m) 和旋转角 (deg)。"""
    Trel = np.linalg.inv(T0) @ T
    t = float(np.linalg.norm(Trel[:3, 3]))
    c = (np.trace(Trel[:3, :3]) - 1.0) / 2.0
    ang = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return t, ang


def draw_mesh_outline(img_rgb, mesh, pose, K, color=(0, 255, 255), thickness=2):
    """画投影轮廓。网格面数很多，画全部线框会糊成一团，轮廓最直观。"""
    sil = mesh_silhouette(mesh, pose, K, img_rgb.shape)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color, thickness, cv2.LINE_AA)
    return img_rgb


def silhouette_diff_overlay(mask, sil):
    """绿=SAM 独有，蓝=mesh 投影独有，白=重合。"""
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    vis[mask > 0] = (0, 255, 0)
    vis[sil > 0] = (255, 0, 0)
    vis[(mask > 0) & (sil > 0)] = (255, 255, 255)
    return vis


def build_eval_overlay(rgb, mesh, pose, K, to_origin, bbox_vis, text):
    vis = rgb.copy()
    draw_mesh_outline(vis, mesh, pose, K)
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox_vis)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                        thickness=3, transparency=0, is_input_rgb=True)
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def predict_pose(prev3, prev2, prev1, factor=1.0):
    """用最近两段位移的均值外推下一帧，作为跟踪初值。

    只看一段位移容易把单帧抖动/收敛跳变当成速度，导致来回过冲；
    两段平均相当于给速度做低通。速度低于死区时直接保持上一帧。
    """
    from scipy.spatial.transform import Rotation

    v1 = prev1[:3, 3] - prev2[:3, 3]
    v0 = prev2[:3, 3] - prev3[:3, 3]
    n0, n1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
    dt = 0.5 * (v0 + v1)
    n = float(np.linalg.norm(dt))

    r1 = Rotation.from_matrix(prev1[:3, :3] @ prev2[:3, :3].T).as_rotvec()
    r0 = Rotation.from_matrix(prev2[:3, :3] @ prev3[:3, :3].T).as_rotvec()
    d_rot = 0.5 * (r0 + r1)
    ang = float(np.linalg.norm(d_rot))

    # 两段位移必须方向一致、大小相近，才认为是真的匀速运动。
    # 跟踪收敛跳变、对称位姿翻转、静止抖动都不满足，直接保持上一帧。
    ok_t = True
    if n0 > 1e-6 and n1 > 1e-6:
        cos_t = float(np.dot(v0, v1) / (n0 * n1))
        ratio = n1 / n0
        ok_t = cos_t > 0.5 and 0.5 < ratio < 2.0
    ok_r = True
    a0, a1 = float(np.linalg.norm(r0)), float(np.linalg.norm(r1))
    if a0 > 1e-6 and a1 > 1e-6:
        cos_r = float(np.dot(r0, r1) / (a0 * a1))
        ratio_r = a1 / a0
        ok_r = cos_r > 0.5 and 0.5 < ratio_r < 2.0
    if not (ok_t and ok_r):
        return prev1.copy()

    # 低于死区（<1.5cm 且 <2.5deg/帧）不预测，避免放大静止抖动
    if n < 0.015 and ang < np.radians(2.5):
        return prev1.copy()

    if n > 0.10:                       # 单帧超过 10cm 视为异常
        dt *= 0.10 / n
    if ang > np.radians(45):           # 单帧超过 45deg 视为异常
        d_rot *= np.radians(45) / ang

    out = np.eye(4)
    out[:3, 3] = prev1[:3, 3] + factor * dt
    out[:3, :3] = (Rotation.from_rotvec(factor * d_rot).as_matrix()
                   @ prev1[:3, :3])
    return out


def smooth_pose(prev, cur, alpha):
    """平移线性插值 + 旋转四元数插值。alpha 越大越跟随当前帧。"""
    from scipy.spatial.transform import Rotation

    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * prev[:3, 3] + alpha * cur[:3, 3]

    q_prev = Rotation.from_matrix(prev[:3, :3]).as_quat()
    q_cur = Rotation.from_matrix(cur[:3, :3]).as_quat()
    if np.dot(q_prev, q_cur) < 0:
        q_cur = -q_cur
    q = (1.0 - alpha) * q_prev + alpha * q_cur
    q /= np.linalg.norm(q)
    out[:3, :3] = Rotation.from_quat(q).as_matrix()
    return out


def start_camera(args):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(cfg)

    depth_filters = None
    if args.depth_filter:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        depth_filters = [spatial]
        if not args.no_depth_temporal:
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            depth_filters.append(temporal)
            print("[fp] 深度滤波: spatial + temporal")
        else:
            print("[fp] 深度滤波: spatial（已关闭 temporal，适合快速运动）")

    device = profile.get_device()
    depth_scale = device.first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)

    print(f"[fp] camera {args.width}x{args.height}@{args.fps}  K: "
          f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    if device.supports(rs.camera_info.usb_type_descriptor):
        usb = device.get_info(rs.camera_info.usb_type_descriptor)
        print(f"[fp] USB: {usb}")
        if usb.startswith("2"):
            print("[fp] ⚠ USB 2.x，帧率和分辨率受限，建议换 USB3 蓝口")

    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale, K, depth_filters


def wait_rgbd(pipeline, align, depth_scale, depth_filters=None, timeout_ms=5000):
    frames = pipeline.wait_for_frames(timeout_ms)
    # 只处理最新帧：YOLO/SAM/注册期间相机会继续出帧，如果一直取队列
    # 最旧的帧，画面会越来越滞后，看起来像卡住。
    while True:
        try:
            newer = pipeline.poll_for_frames()
        except RuntimeError:
            break
        if newer is None or newer.size() == 0:
            break
        frames = newer
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    if depth_filters:
        for f in depth_filters:
            depth_frame = f.process(depth_frame)
    color_bgr = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    depth[~np.isfinite(depth)] = 0
    depth[depth < 0.05] = 0
    return color_bgr, depth


def select_rect_interactive(color_bgr):
    print("[fp] 请在窗口中拖框圈住目标，然后按 SPACE/ENTER 确认")
    roi = cv2.selectROI("select object", color_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("select object")
    return roi


# ---------------- YOLO 自动检测 ----------------
_yolo_state = {"model": None, "path": None}


def get_yolo(model_path):
    """懒加载 Ultralytics YOLO；只在 --yolo 时导入，避免无关启动开销。"""
    if _yolo_state["model"] is None or _yolo_state["path"] != model_path:
        cfg_dir = os.environ.get("YOLO_CONFIG_DIR")
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "导入 ultralytics 失败。请在 foundationpose_ros 环境里安装：\n"
                "  pip install ultralytics\n"
                f"原始错误: {exc}"
            )
        print(f"[yolo] 加载模型: {model_path}")
        _yolo_state["model"] = YOLO(model_path)
        _yolo_state["path"] = model_path
    return _yolo_state["model"]


def resolve_yolo_class(text, names):
    """把 --yolo-class 的 id 或名字解析成 int id。"""
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    for cls_id, cls_name in (names or {}).items():
        if str(cls_name) == str(text):
            return int(cls_id)
    raise SystemExit(
        f"YOLO 类别 {text!r} 不存在；模型里的类别是: {names}"
    )


def yolo_detect(model, color_bgr, conf=0.5, target_class=None):
    """运行 YOLO，返回 (detections, names)。

    detection = (x1, y1, x2, y2, confidence, class_id)

    注意：Ultralytics 对 numpy 输入的约定是 BGR，直接传 color_bgr，
    不要先转成 RGB，否则通道顺序反了，检测效果会明显变差。
    """
    kwargs = {
        "conf": float(conf),
        "verbose": False,
        "device": 0 if torch.cuda.is_available() else "cpu",
    }
    if target_class is not None:
        kwargs["classes"] = [int(target_class)]

    results = model.predict(color_bgr, **kwargs)
    if not results:
        return [], {}
    res = results[0]
    names = getattr(res, "names", {}) or {}
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], names

    dets = []
    for box in boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
        conf_v = float(box.conf[0]) if box.conf is not None else 1.0
        dets.append((x1, y1, x2, y2, conf_v, cls_id))
    return dets, names


def det_xywh(det, width, height):
    """把 YOLO 的 (x1,y1,x2,y2,...) 裁剪并转成 (x,y,w,h)。"""
    x1 = int(np.clip(round(det[0]), 0, width - 1))
    y1 = int(np.clip(round(det[1]), 0, height - 1))
    x2 = int(np.clip(round(det[2]), 0, width - 1))
    y2 = int(np.clip(round(det[3]), 0, height - 1))
    rect = (x1, y1, x2 - x1, y2 - y1)
    if rect[2] <= 1 or rect[3] <= 1:
        return None
    return rect


def pick_yolo_rect(dets, width, height, select="conf"):
    """从检测结果里选一个框，裁剪到图像范围内并转成 x,y,w,h。"""
    if not dets:
        return None, None
    if select == "area":
        key = lambda d: (d[2] - d[0]) * (d[3] - d[1])
    else:
        key = lambda d: d[4]
    best = max(dets, key=key)
    return det_xywh(best, width, height), best


def xywh_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def projected_rect(mesh, pose, K, shape):
    """FoundationPose 位姿下 mesh 投影的 2D 包围框 (x,y,w,h)。

    用于和 YOLO 框比较：两者 IoU 持续偏低，说明 FoundationPose 已经
    不在目标上，需要重新注册。
    """
    cam, uv = project_mesh(mesh, pose, K)
    ok = np.isfinite(uv).all(axis=1) & (cam[:, 2] > 1e-4)
    if np.count_nonzero(ok) < 3:
        return None
    uv = uv[ok]
    h, w = shape[:2]
    x1 = int(np.clip(np.floor(uv[:, 0].min()), 0, w - 1))
    y1 = int(np.clip(np.floor(uv[:, 1].min()), 0, h - 1))
    x2 = int(np.clip(np.ceil(uv[:, 0].max()), 0, w - 1))
    y2 = int(np.clip(np.ceil(uv[:, 1].max()), 0, h - 1))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def draw_yolo_frame(color_bgr, dets, names, selected_rect=None, title=""):
    """画 YOLO 候选框和状态文字，返回 BGR 图。"""
    vis = color_bgr.copy()
    sx = sy = sw = sh = None
    if selected_rect is not None:
        sx, sy, sw, sh = selected_rect
    for x1, y1, x2, y2, conf_v, cls_id in dets:
        is_selected = (selected_rect is not None
                       and abs(x1 - sx) <= 2 and abs(y1 - sy) <= 2
                       and abs((x2 - x1) - sw) <= 2
                       and abs((y2 - y1) - sh) <= 2)
        color = (0, 255, 0) if is_selected else (0, 180, 255)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{names.get(cls_id, cls_id)} {conf_v:.2f}"
        cv2.putText(vis, label, (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if selected_rect is not None:
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
    if title:
        cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def show_yolo_frame(color_bgr, dets, names, selected_rect=None, title="",
                    wait_ms=1):
    """在跟踪窗口里实时显示搜索/注册状态，并处理 q/ESC。"""
    vis = draw_yolo_frame(color_bgr, dets, names, selected_rect, title)
    cv2.imshow(WINDOW_NAME, vis)
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), 27):
        raise UserQuit()


def detect_rect_yolo(model, args, pipeline, align, depth_scale,
                     depth_filters, target_class=None):
    """等一个稳定的 YOLO 框。

    返回 (rect, color_bgr, depth, detections, names)；超时 rect=None。
    """
    need = max(1, int(args.yolo_stable))
    deadline = time.time() + max(0.0, float(args.yolo_wait))
    streak = 0
    last_rect = None
    last = (None, None, None, [], {})

    while True:
        frame = wait_rgbd(pipeline, align, depth_scale, depth_filters)
        if frame is not None:
            color_bgr, depth = frame
            dets, names = yolo_detect(model, color_bgr, args.yolo_conf,
                                      target_class)
            rect, best = pick_yolo_rect(dets, color_bgr.shape[1],
                                        color_bgr.shape[0], args.yolo_select)
            if rect is not None:
                if last_rect is not None and xywh_iou(last_rect, rect) >= 0.5:
                    streak += 1
                else:
                    streak = 1
                last_rect = rect
                last = (rect, color_bgr, depth, dets, names)
                if best is not None:
                    cls_id, conf_v = best[5], best[4]
                    cls_name = names.get(cls_id, cls_id)
                    print(f"[yolo] {len(dets)} 个候选，选中 "
                          f"cls={cls_id}({cls_name}) conf={conf_v:.3f} "
                          f"box={rect} 稳定 {streak}/{need}")
            else:
                streak = 0
                last_rect = None
                last = (None, color_bgr, depth, dets, names)

            # 搜索期间持续刷新窗口，避免目标消失后画面看起来卡死
            if not args.no_display:
                title = f"YOLO searching... {len(dets)} candidate(s)"
                if rect is not None:
                    title += f"  box={rect}  stable {streak}/{need}"
                show_yolo_frame(color_bgr, dets, names, rect, title)

            if rect is not None and streak >= need:
                return last

        if time.time() >= deadline:
            if last[0] is None:
                print(f"[yolo] {args.yolo_wait:.1f}s 内没有检测到目标")
            return last


def save_yolo_vis(color_bgr, dets, names, selected_rect, out_dir, name):
    """把 YOLO 候选框画出来保存，方便检查检测质量。"""
    vis = draw_yolo_frame(color_bgr, dets, names, selected_rect)
    cv2.imwrite(os.path.join(out_dir, name), vis)


def acquire_lock(args, pipeline, align, depth_scale, depth_filters, est, K,
                 yolo=None, yolo_class_id=None, use_rect=True,
                 allow_manual=False, tag="000000"):
    """执行一轮：YOLO 搜索 -> mask -> FoundationPose 注册。

    返回 dict(pose, mask, rect, color_bgr, depth, dets, names, from_yolo)；
    用户按 ESC/关闭窗口返回 None。

    没检测到目标且无法手动框选时抛 RuntimeError，外层可以选择重试。
    allow_manual=False 时，YOLO 失败不再转手动，直接抛错让外层继续搜索
    （用于跟丢后的自动重注册）。
    """
    pending_rect = args.rect if use_rect else None
    use_yolo = yolo is not None

    while True:
        frame = wait_rgbd(pipeline, align, depth_scale, depth_filters)
        if frame is None:
            raise RuntimeError("拿不到 RGB-D 帧")
        color_bgr, depth = frame

        rect = None
        from_yolo = False
        dets, names = [], {}

        if pending_rect:
            rect = tuple(int(v) for v in pending_rect.split(","))
            print(f"[fp] 使用给定框: {rect}")
        elif use_yolo:
            rect, det_color, det_depth, dets, names = detect_rect_yolo(
                yolo, args, pipeline, align, depth_scale, depth_filters,
                yolo_class_id)
            if det_color is not None:
                color_bgr, depth = det_color, det_depth
            if rect is not None:
                from_yolo = True
                print(f"[fp] 使用 YOLO 框: {rect}")
            else:
                msg = (f"YOLO 在 {args.yolo_wait:.1f}s 内没有检测到目标；"
                       "可降低 --yolo-conf、检查 --yolo-class，"
                       "或去掉 --yolo 改用手动框选")
                if args.no_display or not allow_manual:
                    raise RuntimeError(msg)
                print(f"[yolo] {msg}")
                print("[yolo] 转入手动框选")
                rect = select_rect_interactive(color_bgr)
                use_yolo = False
        elif args.no_display:
            raise ValueError("--no-display 模式下必须提供 --rect，或加 --yolo")
        else:
            rect = select_rect_interactive(color_bgr)

        if rect is None or rect[2] <= 0 or rect[3] <= 0:
            raise RuntimeError("没有选择有效区域")

        if not from_yolo:
            # 框选期间相机还在出帧；wait_rgbd 会自动丢弃积压帧取最新帧
            fresh = wait_rgbd(pipeline, align, depth_scale, depth_filters)
            if fresh is not None:
                color_bgr, depth = fresh

        if not args.no_display:
            show_yolo_frame(color_bgr, dets, names, rect,
                            f"YOLO locked {rect} -> {args.mask_mode} segmenting ...")

        mask = make_mask(color_bgr, rect, args.mask_mode,
                         args.sam_checkpoint, args.sam_model_type,
                         args.sam_select)
        print(f"[fp] mask 面积 {int((mask > 0).sum())} 像素")
        if args.save_vis:
            os.makedirs(args.save_vis, exist_ok=True)
            save_mask_vis(color_bgr, mask, args.save_vis, f"mask_{tag}.png")
            if dets:
                save_yolo_vis(color_bgr, dets, names, rect, args.save_vis,
                              f"yolo_{tag}.png")

        # YOLO 自动框默认直接接受，保证跟丢后能自动重新注册；
        # --yolo-mask-preview 可以恢复手动确认。
        preview = not args.no_display and not args.no_mask_preview
        if from_yolo and not args.yolo_mask_preview:
            preview = False
        if not preview:
            break

        action = confirm_mask_preview(color_bgr, mask, args.mask_mode)
        if action == "accept":
            break
        if action == "quit":
            return None
        pending_rect = None   # retry：重新框选
        use_yolo = False
        print("[fp] 已切回手动框选")

    if not args.no_display:
        show_yolo_frame(color_bgr, [], {}, rect,
                        "Registering FoundationPose ... (~3s, please wait)")

    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    print("[fp] 开始注册位姿（约 3 秒，期间窗口会短暂无响应）...")
    t0 = time.time()
    est.pose_last = None
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                        iteration=args.est_refine_iter)
    print(f"[fp] 注册完成，用时 {time.time() - t0:.1f}s，初始位姿:\n{pose}")
    return {
        "pose": pose,
        "mask": mask,
        "rect": rect,
        "color_bgr": color_bgr,
        "depth": depth,
        "dets": dets,
        "names": names,
        "from_yolo": from_yolo,
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，FoundationPose 需要 GPU")

    mesh_path = args.mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"找不到 mesh: {mesh_path}")
    for w in ("2024-01-11-20-02-45", "2023-10-28-18-33-37"):
        if not os.path.exists(os.path.join(FP_ROOT, "weights", w, "model_best.pth")):
            raise FileNotFoundError(f"缺少权重: {FP_ROOT}/weights/{w}/model_best.pth")

    # ---- 加载模型 ----
    print(f"[fp] 加载 mesh: {mesh_path}")
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values()
                 if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise TypeError(f"mesh 里没有可用的三角网格: {mesh_path}")
        mesh = trimesh.util.concatenate(parts)
        print(f"[fp] mesh 含 {len(parts)} 个子网格，已合并")
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"不支持的 mesh 类型: {type(mesh)}")
    if args.mesh_scale != 1.0:
        mesh.apply_scale(args.mesh_scale)
        print(f"[fp] mesh 缩放 x{args.mesh_scale}（换算到米）")
    if mesh.vertex_normals is None:
        mesh.compute_vertex_normals()
    tex_img = getattr(getattr(mesh.visual, "material", None), "image", None)
    if tex_img is not None and getattr(mesh.visual, "uv", None) is not None:
        print(f"[fp] 贴图: {tex_img.size[0]}x{tex_img.size[1]}, "
              f"UV {np.asarray(mesh.visual.uv).shape[0]} 个")
    else:
        print("[fp] ⚠ mesh 没有贴图/UV，渲染只能用几何灰度；"
              "对称物体会更容易产生等价解")
    if args.axes_frame == "origin":
        # 不平移：画出来的轴就是 mesh 文件自身的 X/Y/Z，原点在 mesh 原点 (0,0,0)。
        to_origin = np.eye(4)
        bbox_vis = np.array(mesh.bounds, dtype=float).reshape(2, 3)
        print("[fp] 坐标轴参考: mesh 原点 (0,0,0)（轴 = mesh 自身 X/Y/Z）")
    elif args.axes_frame == "mesh":
        # 只做 AABB 中心平移，不旋转：画出来的轴就是 mesh 文件的 X/Y/Z。
        center = mesh.bounds.mean(axis=0)
        to_origin = np.eye(4)
        to_origin[:3, 3] = -center
        bbox_vis = np.stack([mesh.bounds[0] - center,
                             mesh.bounds[1] - center], axis=0).reshape(2, 3)
        print("[fp] 坐标轴参考: mesh AABB 中心（轴 = mesh 自身 X/Y/Z）")
    else:
        # 原行为：oriented_bounds 的有向包围盒轴（可能会交换/翻转轴）
        to_origin, _obb_extents = trimesh.bounds.oriented_bounds(mesh)
        bbox_vis = np.stack([-_obb_extents / 2, _obb_extents / 2],
                            axis=0).reshape(2, 3)
        print("[fp] 坐标轴参考: OBB（有向包围盒轴）")
    print(f"[fp] mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面, "
          f"尺寸 {np.round(mesh.extents * 100, 1)} cm")

    print("[fp] 初始化 FoundationPose（加载权重，可能需要 1-2 分钟）...")
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    if hasattr(est, "refiner") and est.refiner is not None:
        est.refiner.cfg["crop_ratio"] = args.crop_ratio
        print(f"[fp] refiner crop_ratio = {args.crop_ratio}")
    print("[fp] 模型就绪")

    pipeline, align, depth_scale, K, depth_filters = start_camera(args)

    # 等自动曝光稳定
    for _ in range(10):
        pipeline.wait_for_frames()

    # ---- YOLO 自动检测（可选）----
    yolo = None
    yolo_class_id = None
    if args.yolo:
        if not os.path.exists(args.yolo_model):
            raise FileNotFoundError(f"找不到 YOLO 权重: {args.yolo_model}")
        yolo = get_yolo(args.yolo_model)
        yolo_class_id = resolve_yolo_class(args.yolo_class,
                                           getattr(yolo, "names", {}))
        print(f"[yolo] 类别过滤: {args.yolo_class!r} -> {yolo_class_id}")

    # 跟踪期间是否用 YOLO 校验 FoundationPose 是否还锁在目标上
    yolo_verify = bool(args.yolo and args.yolo_track_check and yolo is not None)
    if args.eval and yolo_verify:
        print("[fp] eval 模式关闭 YOLO 跟踪丢失检测（评估时物体应保持静止）")
        yolo_verify = False

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)
    if args.debug_dir:
        os.makedirs(os.path.join(args.debug_dir, "ob_in_cam"), exist_ok=True)

    # ---- 状态机：搜索/注册 -> 跟踪 -> 丢失 -> 重新搜索/注册 ----
    frame_idx = 0
    lock_idx = 0
    need_lock = True
    verify_lock = False
    disagree = 0
    miss = 0
    eval_stats = None
    eval_limit = args.frames
    prev_pose = None      # 平滑后的输出位姿
    prev_raw = None       # 上一帧原始跟踪位姿（用于运动外推）
    prev_raw2 = None      # 上上帧
    prev_raw3 = None      # 上上上帧
    track_frames = 0      # 本次锁定内已跟踪的帧数
    tf_center = est.get_tf_to_centered_mesh().data.cpu().numpy()
    t_start = None
    try:
        while True:
            if need_lock:
                try:
                    state = acquire_lock(
                        args, pipeline, align, depth_scale, depth_filters,
                        est, K, yolo=yolo, yolo_class_id=yolo_class_id,
                        use_rect=(lock_idx == 0),
                        allow_manual=args.manual_fallback,
                        tag=f"{lock_idx:06d}")
                except UserQuit:
                    print("[fp] 用户退出搜索/注册")
                    break
                except RuntimeError as exc:
                    print(f"[fp] 搜索/注册失败: {exc}")
                    print("[fp] 0.5s 后重试 ...")
                    if args.no_display:
                        time.sleep(0.5)
                    else:
                        # 保持窗口响应，等待期间也能按 q/ESC 退出
                        key = cv2.waitKey(500) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    continue

                if state is None:      # 用户在预览窗口退出
                    break

                lock_idx += 1
                if t_start is None:
                    t_start = time.time()
                pose = state["pose"]
                mask = state["mask"]
                rect = state["rect"]
                color_bgr = state["color_bgr"]
                depth = state["depth"]
                rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                verify_lock = yolo_verify and state["from_yolo"]
                disagree = 0
                miss = 0
                track_frames = 0
                prev_pose = prev_raw = prev_raw2 = prev_raw3 = None
                need_lock = False

                if args.pose_output:
                    Path(args.pose_output).parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(args.pose_output, pose.reshape(4, 4))
                    print(f"[fp] 初始位姿已保存到 {args.pose_output}")
                print(f"[fp] 已锁定 #{lock_idx} ({'YOLO' if state['from_yolo'] else '手动'})"
                      f"{'，YOLO 持续校验中' if verify_lock else ''}")

                # ---- 评估模式初始化 ----
                eval_stats = None
                eval_limit = args.frames
                if args.eval:
                    os.makedirs(args.eval_out, exist_ok=True)
                    cv2.imwrite(os.path.join(args.eval_out, "mask_binary.png"), mask)
                    np.savetxt(os.path.join(args.eval_out, "pose_register.txt"),
                               pose.reshape(4, 4))
                    sil0 = mesh_silhouette(mesh, pose, K, mask.shape)
                    iou0, cover0, prec0 = silhouette_stats(sil0, mask)
                    cv2.imwrite(os.path.join(args.eval_out, "diff_000000.png"),
                                silhouette_diff_overlay(mask, sil0))
                    dc0 = depth_consistency(mesh, pose, K, depth)
                    eval_stats = {"iou": [iou0], "cover": [cover0],
                                  "prec": [prec0], "dmed": [], "p90": [],
                                  "frac": [], "t": [], "r": [],
                                  "T0": pose.copy()}
                    if dc0:
                        dmed, p90, frac, n = dc0
                        eval_stats["dmed"].append(dmed)
                        eval_stats["p90"].append(p90)
                        eval_stats["frac"].append(frac)
                        print(f"[fp] eval 注册帧: mask IoU={iou0:.3f}  "
                              f"覆盖={cover0:.3f}  投影精度={prec0:.3f}  "
                              f"深度中位={dmed * 1000:.1f}mm  90分位={p90 * 1000:.1f}mm  "
                              f"1cm内={frac * 100:.1f}%  样本={n}")
                    else:
                        print(f"[fp] eval 注册帧: mask IoU={iou0:.3f}  "
                              f"覆盖={cover0:.3f}  投影精度={prec0:.3f}  "
                              "深度有效样本不足")
                    vis = build_eval_overlay(rgb, mesh, pose, K, to_origin,
                                             bbox_vis, f"IoU {iou0:.3f}")
                    cv2.imwrite(os.path.join(args.eval_out, "eval_000000.png"),
                                vis[..., ::-1])
                    eval_limit = frame_idx + args.eval_frames
                    print(f"[fp] 评估模式：保持物体和相机静止，采集 "
                          f"{args.eval_frames} 帧")
                    print("[fp] 注意：IoU 用的是注册帧 mask，只反映拟合度，"
                          "不是独立真值")

                # 注册期间积压的旧帧由 wait_rgbd 统一丢弃

            # ---- 跟踪一帧 ----
            frame = wait_rgbd(pipeline, align, depth_scale, depth_filters)
            if frame is None:
                continue
            color_bgr, depth = frame
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            # 快速运动：用最近两段位移外推作为本帧跟踪初值
            if (args.motion_predict > 0 and track_frames >= 5
                    and prev_raw is not None
                    and prev_raw2 is not None and prev_raw3 is not None):
                pose_pred = predict_pose(prev_raw3, prev_raw2, prev_raw,
                                         args.motion_predict)
                est.pose_last = torch.as_tensor(
                    pose_pred @ np.linalg.inv(tf_center),
                    device='cuda', dtype=torch.float)

            pose_raw = est.track_one(rgb=rgb, depth=depth, K=K,
                                     iteration=args.track_refine_iter)
            pose = pose_raw
            if args.smooth > 0:
                if prev_pose is None:
                    prev_pose = pose_raw.copy()
                else:
                    pose = smooth_pose(prev_pose, pose_raw, args.smooth)
                    prev_pose = pose.copy()
                    if args.motion_predict <= 0:
                        # 没有运动预测时，把平滑后的位姿作为下一帧初值
                        est.pose_last = torch.as_tensor(
                            pose @ np.linalg.inv(tf_center),
                            device='cuda', dtype=torch.float)
            if prev_raw is not None:
                jump_t, jump_r = pose_delta(prev_raw, pose_raw)
                if jump_t > 0.05 or jump_r > 30.0:
                    print(f"[fp] ⚠ frame {frame_idx}: 帧间跳变 "
                          f"{jump_t * 1000:.0f}mm / {jump_r:.1f}deg，可能跟丢")
            prev_raw3, prev_raw2, prev_raw = prev_raw2, prev_raw, pose_raw.copy()

            # ---- YOLO 持续校验：FoundationPose 是否还锁在目标上 ----
            # 校验用较低的 --yolo-verify-conf：正常跟踪时置信度略低不算丢失。
            # 匹配时对所有候选取“与 FP 投影框的最大 IoU”，避免其他
            # 高置信度误检把真正的目标挤掉。
            if (verify_lock and yolo is not None
                    and frame_idx % max(1, args.yolo_check_every) == 0):
                dets_y, _ = yolo_detect(yolo, color_bgr,
                                        args.yolo_verify_conf, yolo_class_id)
                fp_rect = projected_rect(mesh, pose_raw, K, color_bgr.shape)
                if fp_rect is None:
                    disagree += 1
                    miss = 0
                    print(f"[yolo-check] frame {frame_idx}: FP 投影不可用 "
                          f"({disagree}/{args.yolo_lost_frames})")
                elif not dets_y:
                    miss += 1
                    disagree = 0
                    print(f"[yolo-check] frame {frame_idx}: 未检出目标 "
                          f"({miss}/{args.yolo_miss_frames})")
                else:
                    H, W = color_bgr.shape[:2]
                    best_iou = 0.0
                    for d in dets_y:
                        r = det_xywh(d, W, H)
                        if r is not None:
                            best_iou = max(best_iou, xywh_iou(r, fp_rect))
                    if best_iou >= args.yolo_lock_iou:
                        if disagree or miss:
                            print(f"[yolo-check] frame {frame_idx}: "
                                  f"重新对上目标 IoU={best_iou:.2f}")
                        disagree = miss = 0
                    else:
                        disagree += 1
                        miss = 0
                        print(f"[yolo-check] frame {frame_idx}: "
                              f"FP 投影与所有 YOLO 框都不一致 "
                              f"max IoU={best_iou:.2f} "
                              f"< {args.yolo_lock_iou:.2f} "
                              f"({disagree}/{args.yolo_lost_frames})")

                if (disagree >= args.yolo_lost_frames
                        or miss >= args.yolo_miss_frames):
                    print("[fp] YOLO 确认跟踪丢失，重新执行搜索/分割/注册")
                    need_lock = True
                    disagree = miss = 0
            if need_lock:
                continue

            print(f"[fp] frame {frame_idx:04d} pose:\n{np.array2string(pose, precision=4)}\n")

            if args.pose_output:
                np.savetxt(args.pose_output, pose.reshape(4, 4))
                np.savetxt(os.path.join(args.debug_dir, "ob_in_cam", f"{frame_idx:06d}.txt"),
                           pose.reshape(4, 4))

            if args.eval:
                sil = mesh_silhouette(mesh, pose, K, mask.shape)
                iou, cover, prec = silhouette_stats(sil, mask)
                dc = depth_consistency(mesh, pose, K, depth)
                t_err, r_err = pose_delta(eval_stats["T0"], pose)
                eval_stats["iou"].append(iou)
                eval_stats["cover"].append(cover)
                eval_stats["prec"].append(prec)
                eval_stats["t"].append(t_err)
                eval_stats["r"].append(r_err)
                if dc:
                    eval_stats["dmed"].append(dc[0])
                    eval_stats["p90"].append(dc[1])
                    eval_stats["frac"].append(dc[2])
                if frame_idx % 10 == 0 or frame_idx == eval_limit - 1:
                    txt = f"f{frame_idx} IoU {iou:.3f}"
                    if dc:
                        txt += f" depth {dc[0] * 1000:.0f}mm"
                    vis_eval = build_eval_overlay(rgb, mesh, pose, K, to_origin,
                                                  bbox_vis, txt)
                    cv2.imwrite(os.path.join(args.eval_out, f"eval_{frame_idx + 1:06d}.png"),
                                vis_eval[..., ::-1])
                    cv2.imwrite(os.path.join(args.eval_out, f"diff_{frame_idx + 1:06d}.png"),
                                silhouette_diff_overlay(mask, sil))

            if not args.no_display or args.save_vis:
                if args.vis_mesh:
                    c = (pose @ np.linalg.inv(to_origin))[:3, 3]
                    txt = (f"center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})m "
                           f"f{frame_idx} | q:quit r:re-search/register")
                    vis = build_eval_overlay(rgb, mesh, pose, K, to_origin,
                                             bbox_vis, txt)
                else:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose,
                                            bbox=bbox_vis)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    cv2.putText(vis, f"frame {frame_idx} | q:quit  r:re-search/register",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                if args.save_vis:
                    cv2.imwrite(os.path.join(args.save_vis, f"track_{frame_idx:06d}.png"),
                                vis[..., ::-1])
                if not args.no_display:
                    cv2.imshow("FoundationPose (rect mask)", vis[..., ::-1])
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        print("[fp] 手动触发重新搜索/分割/注册")
                        need_lock = True

            frame_idx += 1
            track_frames += 1
            if eval_limit and frame_idx >= eval_limit:
                break

        # ---- 评估统计 ----
        if args.eval and eval_stats is not None:
            print("\n========== 位姿精度评估 ==========")
            iou_mean = float(np.mean(eval_stats["iou"])) if eval_stats["iou"] else 0.0
            cover_mean = float(np.mean(eval_stats["cover"])) if eval_stats["cover"] else 0.0
            prec_mean = float(np.mean(eval_stats["prec"])) if eval_stats["prec"] else 0.0
            print(f"mask IoU        平均 {iou_mean:.3f}  (阈值 >=0.85)  "
                  f"{'PASS' if iou_mean >= 0.85 else 'FAIL'}")
            print(f"mask 覆盖率     平均 {cover_mean:.3f}  (mask 落在投影内的比例)")
            print(f"投影精度        平均 {prec_mean:.3f}  (投影落在 mask 内的比例)")
            if iou_mean < 0.85 and cover_mean >= 0.90:
                print("诊断：mask 几乎全部被模型投影覆盖，但 IoU 仍低 =>")
                print("      mask 不完整（常见于 SAM 漏掉物体暗色/低对比部分），")
                print("      不代表位姿错。看 diff_*.png：绿=mask 独有，蓝=投影独有。")
            elif iou_mean < 0.85:
                print("诊断：mask 未被投影完全覆盖，位姿或 mask 二者之一有问题，")
                print("      对照 diff_*.png 和 eval_*.png 线框判断。")
            if eval_stats["dmed"]:
                dmed = float(np.mean(eval_stats["dmed"])) * 1000
                p90 = float(np.mean(eval_stats["p90"])) * 1000
                frac = float(np.mean(eval_stats["frac"])) * 100
                print(f"深度中位误差    平均 {dmed:.1f} mm  (阈值 <=10 mm)  "
                      f"{'PASS' if dmed <= 10 else 'FAIL'}")
                print(f"深度 90 分位    平均 {p90:.1f} mm")
                print(f"1cm 内点比例    平均 {frac:.1f}%")
                if p90 > 30:
                    print("提示：90 分位偏大常由物体暗色/反光表面（深度空洞或偏差）引起，")
                    print("      不一定是位姿错；结合 diff_*.png 看投影与 mask 的重合情况。")
            else:
                print("深度误差        无有效样本")
            if len(eval_stats["t"]) > 1:
                ts = np.array(eval_stats["t"]) * 1000
                rr = np.array(eval_stats["r"])
                print(f"静止平移抖动    std {ts.std():.2f} mm  mean {ts.mean():.2f} mm  "
                      f"max {ts.max():.2f} mm  (阈值 std <=2 mm)  "
                      f"{'PASS' if ts.std() <= 2 else 'FAIL'}")
                print(f"静止旋转抖动    std {rr.std():.2f} deg  mean {rr.mean():.2f} deg  "
                      f"max {rr.max():.2f} deg  (阈值 std <=1.0 deg)  "
                      f"{'PASS' if rr.std() <= 1.0 else 'FAIL'}")
            if args.smooth > 0:
                print(f"注意：已开启 --smooth {args.smooth}，抖动指标会偏低，"
                      "测原始精度请加 --smooth 0")
            print(f"可视化输出      {args.eval_out}")
            print("   eval_*.png = 原图 + mesh 线框；diff_*.png = 绿 mask 独有 / 蓝 投影独有 / 白 重合")
            print("提示：以上是自洽性指标；真值精度请用 ArUco/物理量测算 ADD")
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        dt = time.time() - t_start if t_start is not None else 0.0
        if dt > 0:
            print(f"[fp] 共跟踪 {frame_idx} 帧，平均 {frame_idx / dt:.1f} fps")


if __name__ == "__main__":
    main()
