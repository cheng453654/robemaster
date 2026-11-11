#!/usr/bin/env python3
"""FoundationPose ROS 节点共用的几何 / 分割 / 位姿工具。

这个文件不依赖 ROS，摄像头节点、YOLO 节点和 FoundationPose 节点都可以
直接 import。内容从 learn_foundationpose.py 里抽出来，保持行为一致。
"""

import csv
import gc
import json
import os
import sys
import time

import cv2
import numpy as np
import torch


# ---------------- SAM 分割 ----------------

# MobileSAM（vit_t）单独放在一个目录，不和 Meta 原版 segment_anything 冲突。
# 目录结构：MOBILE_SAM_ROOT/mobile_sam/...，权重默认 models/sam/mobile_sam.pt。
MOBILE_SAM_ROOT = os.environ.get(
    "MOBILE_SAM_ROOT", os.path.expanduser("~/mobile_sam_pkg"))


def _ensure_mobile_sam():
    root = os.path.expanduser(MOBILE_SAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import mobile_sam  # noqa: F401  确认包可导入
    return root


def load_sam(checkpoint, model_type="vit_b", device="cuda"):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"找不到 SAM 权重: {checkpoint}")
    if model_type == "vit_t":
        _ensure_mobile_sam()
        from mobile_sam import SamPredictor, sam_model_registry
    else:
        from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


_sam_state = {"predictor": None, "checkpoint": None, "model_type": None}


def get_sam(checkpoint, model_type):
    """懒加载 SAM：只在需要分割时加载。"""
    if (_sam_state["predictor"] is None
            or _sam_state["checkpoint"] != checkpoint
            or _sam_state["model_type"] != model_type):
        print(f"[fp] 加载 SAM ({model_type})...")
        release_sam()
        _sam_state["predictor"] = load_sam(checkpoint, model_type)
        _sam_state["checkpoint"] = checkpoint
        _sam_state["model_type"] = model_type
        print("[fp] SAM 就绪")
    return _sam_state["predictor"]


def release_sam():
    """注册前释放 SAM，把显存让给 FoundationPose（6GB 卡必须这么做）。"""
    if _sam_state["predictor"] is not None:
        _sam_state["predictor"] = None
        _sam_state["checkpoint"] = None
        _sam_state["model_type"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_mask(color_bgr, rect, mode, sam_predictor=None, sam_select="score"):
    """在给定矩形内生成物体 mask。

    rect 是 (x, y, w, h)，像素坐标。mode: rect / sam / grabcut。
    """
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
    gc_mask = np.zeros((H, W), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(color_bgr, gc_mask, (x, y, w, h), bgd, fgd, 5,
                cv2.GC_INIT_WITH_RECT)
    mask[np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))] = 255
    if mask.sum() < 0.2 * 255 * w * h:
        mask[:] = 0
        mask[y:y + h, x:x + w] = 255
    return mask


def make_mask(color_bgr, rect, mode, sam_checkpoint, sam_model_type,
              sam_select="score", keep_loaded=False):
    """生成 mask。

    keep_loaded=False 时 SAM 用完立即释放显存（6GB 桌面卡需要）；
    Jetson 这类统一内存、显存充足或想加快重注册的平台可以设为 True。
    """
    predictor = get_sam(sam_checkpoint, sam_model_type) if mode == "sam" else None
    try:
        return build_mask(color_bgr, rect, mode, predictor, sam_select)
    finally:
        if mode == "sam" and not keep_loaded:
            release_sam()


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


# ---------------- 投影 / 轮廓 ----------------

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
    """返回 (IoU, coverage, precision)。"""
    inter = np.count_nonzero((sil > 0) & (mask > 0))
    sil_area = np.count_nonzero(sil > 0)
    mask_area = np.count_nonzero(mask > 0)
    union = sil_area + mask_area - inter
    iou = float(inter) / float(union) if union else 0.0
    cover = float(inter) / float(mask_area) if mask_area else 0.0
    prec = float(inter) / float(sil_area) if sil_area else 0.0
    return iou, cover, prec


def draw_mesh_outline(img_rgb, mesh, pose, K, color=(0, 255, 255), thickness=2):
    """画投影轮廓。网格面数很多，画全部线框会糊成一团，轮廓最直观。"""
    sil = mesh_silhouette(mesh, pose, K, img_rgb.shape)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color, thickness, cv2.LINE_AA)
    return img_rgb


def draw_axes(img_rgb, pose, K, length=0.05, thickness=2):
    """在 mesh 原点画物体坐标系 X/Y/Z 轴（RGB: 红=X 绿=Y 蓝=Z）。

    length 是轴长（米），默认 5cm，按物体尺寸调整。
    """
    pts = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ])
    cam = (pose @ np.hstack([pts, np.ones((4, 1))]).T).T[:, :3]
    if np.any(cam[:, 2] <= 1e-4):
        return img_rgb
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    if not np.all(np.isfinite(uv)):
        return img_rgb

    origin = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]   # RGB
    labels = ["X", "Y", "Z"]
    for i in range(3):
        tip = (int(round(uv[i + 1, 0])), int(round(uv[i + 1, 1])))
        cv2.arrowedLine(img_rgb, origin, tip, colors[i], thickness,
                        cv2.LINE_AA, tipLength=0.2)
        cv2.putText(img_rgb, labels[i], (tip[0] + 3, tip[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)
    cv2.circle(img_rgb, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img_rgb


def depth_consistency(mesh, pose, K, depth, n_samples=4000):
    """采样 mesh 正面点，和投影像素处的实测深度比较。"""
    pts, fidx = mesh.sample(n_samples, return_index=True)
    normals = np.asarray(mesh.face_normals)[fidx]
    cam = (pose @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    n_cam = normals @ pose[:3, :3].T
    front = np.einsum("ij,ij->i", n_cam, -cam) > 0
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


# ---------------- 位姿平滑 / 预测 ----------------

def pose_delta(T0, T):
    """相对首帧的平移 (m) 和旋转角 (deg)。"""
    Trel = np.linalg.inv(T0) @ T
    t = float(np.linalg.norm(Trel[:3, 3]))
    c = (np.trace(Trel[:3, :3]) - 1.0) / 2.0
    ang = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return t, ang


def predict_pose(prev3, prev2, prev1, factor=1.0):
    """用最近两段位移的均值外推下一帧，作为跟踪初值。"""
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

    if n < 0.015 and ang < np.radians(2.5):
        return prev1.copy()

    if n > 0.10:
        dt *= 0.10 / n
    if ang > np.radians(45):
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


# ---------------- 检测框工具 ----------------

def xyxy_to_xywh(x1, y1, x2, y2, width, height):
    """裁剪到图像内并转成 (x, y, w, h)；框太小返回 None。"""
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))
    rect = (x1, y1, x2 - x1, y2 - y1)
    if rect[2] <= 1 or rect[3] <= 1:
        return None
    return rect


def pick_rect(dets, select="conf"):
    """从 (x, y, w, h, score, class_id) 列表里选一个框。"""
    if not dets:
        return None
    if select == "area":
        key = lambda d: d[2] * d[3]
    else:
        key = lambda d: d[4]
    return max(dets, key=key)


def xywh_iou(a, b):
    ax1, ay1, aw, ah = a[:4]
    bx1, by1, bw, bh = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def max_iou(rect, dets):
    """rect 与所有检测框的最大 IoU；dets 为空返回 (0.0, None)。"""
    best, best_det = 0.0, None
    for d in dets:
        iou = xywh_iou(rect, d)
        if iou > best:
            best, best_det = iou, d
    return best, best_det


def projected_rect(mesh, pose, K, shape):
    """FoundationPose 位姿下 mesh 投影的 2D 包围框 (x,y,w,h)。"""
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


def draw_detections_rgb(img_rgb, dets, selected=None, color=(0, 255, 0)):
    """在 RGB 图上画检测框；selected 用另一种颜色高亮。"""
    vis = img_rgb
    for d in dets:
        x, y, w, h = [int(v) for v in d[:4]]
        score = float(d[4]) if len(d) > 4 else 0.0
        col = (0, 255, 255) if selected is not None and d is selected else color
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cv2.putText(vis, f"{score:.2f}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return vis


# ---------------- FoundationPose 注册 ----------------

def register_pose(est, K, rgb, depth, ob_mask, iteration=5, topk=12):
    """两阶段注册：先给全部朝向假设打粗分，只精修前 topk 个。

    与 estimater.FoundationPose.register 等价，唯一区别是 refiner 从
    「全部假设各精修 iteration 轮」改成「只精修粗分最高的 topk 个」。
    实测（96 个假设，est_refine_iter=5）：8.6s -> 2.8s，位姿差
    <1.1mm / <1.6°。topk<=0 时直接退回原版 register()。

    会维护 est 的 pose_last / poses / scores / best_id，
    之后可以照常调用 est.track_one()。
    """
    if topk <= 0:
        return est.register(K=K, rgb=rgb, depth=depth, ob_mask=ob_mask,
                            iteration=iteration)

    from Utils import (bilateral_filter_depth, depth2xyzmap, erode_depth,
                       set_seed)

    set_seed(0)
    if est.glctx is None:
        import nvdiffrast.torch as dr
        est.glctx = dr.RasterizeCudaContext()

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if ((depth >= 0.001) & (ob_mask > 0)).sum() < 4:
        pose = np.eye(4)
        pose[:3, 3] = est.guess_translation(depth=depth, mask=ob_mask, K=K)
        est.pose_last = None
        return pose

    est.H, est.W = depth.shape[:2]
    est.K = K
    est.ob_id = None
    est.ob_mask = ob_mask

    # 1) 全部朝向假设（平移已放到 mask 中心）先打一次粗分
    poses = est.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth,
                                          mask=ob_mask, scene_pts=None)
    scores, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb, depth=depth, K=K,
        ob_in_cams=poses.data.cpu().numpy(), normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx,
        mesh_diameter=est.diameter)
    order = torch.argsort(scores, descending=True)[:int(topk)]

    # 2) 只精修前 topk 个，再打分选最优
    xyz_map = depth2xyzmap(depth, K)
    poses, _ = est.refiner.predict(
        mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb, depth=depth,
        K=K, ob_in_cams=poses[order].data.cpu().numpy(), normal_map=None,
        xyz_map=xyz_map, glctx=est.glctx, mesh_diameter=est.diameter,
        iteration=iteration)
    scores, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb, depth=depth, K=K,
        ob_in_cams=poses.data.cpu().numpy(), normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx,
        mesh_diameter=est.diameter)

    ids = torch.argsort(scores, descending=True)
    scores = scores[ids]
    poses = poses[ids]
    src_ids = order[ids]

    est.pose_last = poses[0]          # centered-mesh 坐标，track_one 要用
    est.poses = poses
    est.scores = scores
    est.best_id = int(src_ids[0])
    best_pose = poses[0] @ est.get_tf_to_centered_mesh()
    return best_pose.data.cpu().numpy()


# ---------------- 运行期诊断记录 ----------------

class DebugRecorder:
    """把注册 / 稳定性验证 / 跟踪过程写盘，供离线分析。

    root/
      events.log   一行一个事件：`<墙钟时间> kind=... k=v ...`
      frames.csv   每次跟踪一行：耗时、相对参考帧的平移/旋转偏差
      frames/      注册时刻保存的 rgb / depth / mask（npz，可选）
    由 tools/analyze_fp_log.py 解析，不需要 ROS 和 GPU。
    """

    FIELDS = ("wall", "stamp", "phase", "n", "track_ms",
              "dt_mm", "da_deg", "max_dt_mm", "max_da_deg",
              "iou", "det_cx", "det_cy", "note")

    def __init__(self, root, save_frames=True):
        self.root = root
        self.save_frames = bool(save_frames)
        self.enabled = True
        os.makedirs(root, exist_ok=True)
        self._csv_path = os.path.join(root, "frames.csv")
        new = not os.path.exists(self._csv_path)
        self._csv = open(self._csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=self.FIELDS)
        if new:
            self._writer.writeheader()
        self._events = open(os.path.join(root, "events.log"), "a")

    # -- 文本事件 --
    def event(self, kind, **kv):
        if not self.enabled:
            return
        parts = [f"kind={kind}"]
        for key in sorted(kv):
            val = kv[key]
            if isinstance(val, float):
                val = f"{val:.6g}"
            parts.append(f"{key}={val}")
        self._events.write(f"{time.time():.6f} " + " ".join(parts) + "\n")
        self._events.flush()

    # -- 数值行 --
    def row(self, phase, **kv):
        if not self.enabled:
            return
        rec = {f: "" for f in self.FIELDS}
        rec["wall"] = f"{time.time():.6f}"
        rec["phase"] = phase
        for key, val in kv.items():
            if key in rec and val is not None:
                rec[key] = f"{val:.6g}" if isinstance(val, float) else val
        self._writer.writerow(rec)
        self._csv.flush()

    # -- 保存一帧原始数据（离线复现注册/跟踪用）--
    def save_frame(self, tag, rgb=None, depth=None, mask=None, **meta):
        if not (self.enabled and self.save_frames):
            return None
        out_dir = os.path.join(self.root, "frames")
        os.makedirs(out_dir, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}_{tag}"
        arrays = {}
        if rgb is not None:
            arrays["rgb"] = np.asarray(rgb)
        if depth is not None:
            arrays["depth"] = np.asarray(depth, dtype=np.float32)
        if mask is not None:
            arrays["mask"] = np.asarray(mask, dtype=np.uint8)
        if arrays:
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
        if meta:
            with open(os.path.join(out_dir, name + ".json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        return name

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        try:
            self._csv.close()
            self._events.close()
        except Exception:  # noqa: BLE001
            pass
