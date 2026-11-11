#!/usr/bin/env python3
"""单张图片的 SAM 框选分割演示。

只依赖 segment_anything / torch / opencv，不加载相机和 FoundationPose，
所以显存占用小，适合单独检查 SAM 的分割效果。

运行（必须用装了 SAM 的 conda 环境）：
    conda activate foundationpose_ros
    source /home/cheng/test/RCIA_Benmao_Vision/scripts/foundationpose_env.sh
    python learn_sam_demo.py                    # 弹文件选择框
    python learn_sam_demo.py --image /path/to/picture.jpg

交互：
    1. 窗口里拖框圈住目标，按 SPACE/ENTER 确认（c/ESC 取消退出）
    2. 分割完成后：
         1/2/3  选择第 1/2/3 个候选 mask
         n / p  下一个 / 上一个候选
         s      保存当前结果到 --out 目录
         r      重新框选并重新分割
         m      切换：叠加 / 纯 mask / 只看轮廓
         q/ESC  退出
    3. 保存的文件：
         <名字>_mask.png        二值 mask（0/255）
         <名字>_overlay.png     原图 + 绿色 mask 叠加
         <名字>_candidates.png  三个候选的轮廓对比
         <名字>_info.txt        框坐标、分数、面积

无人值守（不弹窗，必须给框）：
    python learn_sam_demo.py --image x.png --box 250,150,120,175 \
        --no-display --out ./sam_out
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as exc:
    raise SystemExit(
        "导入 segment_anything 失败。请先激活 foundationpose_ros 环境：\n"
        "  conda activate foundationpose_ros\n"
        "  source /home/cheng/test/RCIA_Benmao_Vision/scripts/foundationpose_env.sh\n"
        f"原始错误: {exc}"
    )

PROJECT_ROOT = os.environ.get("RCIA_ROOT", "/home/cheng/test/RCIA_Benmao_Vision")
DEFAULT_SAM = os.path.join(PROJECT_ROOT, "models/sam/sam_vit_b_01ec64.pth")
WINDOW = "SAM box demo"

# torch 2.6 起 torch.load 默认 weights_only=True
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def parse_args():
    p = argparse.ArgumentParser(description="单张图片 SAM 框选分割演示")
    p.add_argument("--image", default=None,
                   help="输入图片路径；不填则弹出文件选择框")
    p.add_argument("--box", default=None,
                   help="跳过鼠标框选，格式 x,y,w,h（原图坐标）")
    p.add_argument("--checkpoint", default=DEFAULT_SAM, help="SAM 权重")
    p.add_argument("--model-type", default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    p.add_argument("--out", default="sam_demo_out", help="保存目录")
    p.add_argument("--no-display", action="store_true",
                   help="不弹窗，直接保存分数最高的 mask（需要 --box）")
    p.add_argument("--max-display", type=int, default=1280,
                   help="显示窗口最长边，原图更大时等比缩小")
    p.add_argument("--save-all", action="store_true",
                   help="额外保存三个候选各自的二值 mask")
    return p.parse_args()


def choose_image():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise SystemExit("当前环境没有 tkinter，请用 --image 指定图片路径")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择一张图片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                   ("所有文件", "*.*")],
    )
    root.destroy()
    return path


def parse_box(text, width, height):
    vals = [int(round(float(v))) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--box 格式应为 x,y,w,h")
    x, y, w, h = vals
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def fit_display(img, max_side):
    """把图缩到窗口能放下，返回 (显示图, 缩放比)。"""
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return img.copy(), 1.0
    disp = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)
    return disp, scale


def select_box_interactive(img, max_side):
    disp, scale = fit_display(img, max_side)
    print("[sam] 在窗口中拖框圈住目标，按 SPACE/ENTER 确认，c 取消")
    roi = cv2.selectROI(WINDOW, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise SystemExit("没有选择有效区域，已退出")
    # 显示坐标换回原图坐标
    x = int(round(x / scale))
    y = int(round(y / scale))
    w = int(round(w / scale))
    h = int(round(h / scale))
    return parse_box(f"{x},{y},{w},{h}", img.shape[1], img.shape[0])


def load_predictor(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到 SAM 权重: {args.checkpoint}")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam] 加载 {args.model_type}: {args.checkpoint}  device={device}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam), device


def predict_masks(predictor, img_bgr, box):
    x, y, w, h = box
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    t0 = time.time()
    masks, scores, _ = predictor.predict(
        box=np.array([x, y, x + w, y + h], dtype=np.float32),
        multimask_output=True,
    )
    dt = time.time() - t0
    print(f"[sam] 分割完成 {dt:.2f}s")
    for i, (m, s) in enumerate(zip(masks, scores)):
        print(f"  候选 {i + 1}: score={s:.4f}  area={int(m.sum())} px")
    return masks, scores


def draw_display(img, masks, scores, box, sel, mode, max_side):
    disp, scale = fit_display(img, max_side)
    dh, dw = disp.shape[:2]
    dmasks = [
        cv2.resize(m.astype(np.uint8) * 255, (dw, dh),
                   interpolation=cv2.INTER_NEAREST) > 0
        for m in masks
    ]

    if mode == "mask":
        vis = np.zeros_like(disp)
    else:
        vis = disp.copy()

    if mode != "contour":
        m = dmasks[sel]
        if mode == "mask":
            vis[m] = (255, 255, 255)
        else:
            vis[m] = (0.55 * vis[m] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0)]  # BGR: 红 / 黄 / 蓝
    for i, dm in enumerate(dmasks):
        contours, _ = cv2.findContours(dm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        thick = 3 if i == sel else 1
        cv2.drawContours(vis, contours, -1, colors[i], thick, cv2.LINE_AA)

    x, y, w, h = [int(round(v * scale)) for v in box]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

    lines = []
    for i, s in enumerate(scores):
        mark = ">>" if i == sel else "  "
        lines.append(f"{mark} [{i + 1}] score {s:.3f}  area {int(masks[i].sum())} px")
    lines.append(f"selected {sel + 1} | 1/2/3 n/p | s save | r reselect | m view | q quit")

    for i, text in enumerate(lines):
        pos = (10, 26 + 24 * i)
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def save_result(img, masks, scores, box, sel, out_dir, stem, save_all=False):
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{stem}_{stamp}")

    mask = masks[sel]
    cv2.imwrite(base + "_mask.png", mask.astype(np.uint8) * 255)

    overlay = img.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(base + "_overlay.png", overlay)

    cand = img.copy()
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0)]
    for i, (m, s) in enumerate(zip(masks, scores)):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cand, cs, -1, colors[i], 2, cv2.LINE_AA)
        cv2.putText(cand, f"{i + 1}: {s:.3f}", (10, 26 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)
    cv2.imwrite(base + "_candidates.png", cand)

    if save_all:
        for i, m in enumerate(masks):
            cv2.imwrite(base + f"_mask_c{i + 1}.png",
                        m.astype(np.uint8) * 255)

    with open(base + "_info.txt", "w", encoding="utf-8") as f:
        f.write(f"box: {box[0]},{box[1]},{box[2]},{box[3]}\n")
        f.write(f"selected: {sel + 1}\n")
        for i, (m, s) in enumerate(zip(masks, scores)):
            f.write(f"candidate {i + 1}: score={s:.6f} area={int(m.sum())}\n")

    print(f"[sam] 已保存: {base}_mask.png / _overlay.png / _candidates.png")
    return base


def run_ui(img, masks, scores, box, args, stem):
    sel = int(np.argmax(scores))
    mode = "overlay"
    while True:
        vis = draw_display(img, masks, scores, box, sel, mode, args.max_display)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            return "quit"
        if key in (ord("1"), ord("2"), ord("3")):
            sel = key - ord("1")
        elif key == ord("n"):
            sel = (sel + 1) % len(masks)
        elif key == ord("p"):
            sel = (sel - 1) % len(masks)
        elif key == ord("m"):
            mode = {"overlay": "mask", "mask": "contour",
                    "contour": "overlay"}[mode]
        elif key == ord("s"):
            save_result(img, masks, scores, box, sel, args.out, stem,
                        args.save_all)
        elif key == ord("r"):
            return "reselect"

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"


def main():
    args = parse_args()
    if args.no_display and not args.box:
        raise SystemExit("--no-display 模式必须提供 --box x,y,w,h")
    if not args.image:
        if args.no_display:
            raise SystemExit("--no-display 模式必须用 --image 指定图片")
        args.image = choose_image()
        if not args.image:
            raise SystemExit("没有选择图片")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {args.image}")
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(args.image))[0]
    print(f"[sam] 图片 {args.image}  {w}x{h}")

    predictor, _ = load_predictor(args)

    pending_box = args.box
    while True:
        if pending_box:
            box = parse_box(pending_box, w, h)
            print(f"[sam] 使用框: {box}")
        else:
            box = select_box_interactive(img, args.max_display)
            print(f"[sam] 框选结果: {box}")

        masks, scores = predict_masks(predictor, img, box)

        if args.no_display:
            sel = int(np.argmax(scores))
            save_result(img, masks, scores, box, sel, args.out, stem,
                        args.save_all)
            break

        action = run_ui(img, masks, scores, box, args, stem)
        if action == "reselect":
            pending_box = None
            continue
        break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
