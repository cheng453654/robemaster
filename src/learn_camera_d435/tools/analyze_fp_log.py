#!/usr/bin/env python3
"""分析 FoundationPose 节点 debug_record 产生的日志。

用法：
    python3 tools/analyze_fp_log.py                # 自动选最新一次运行
    python3 tools/analyze_fp_log.py --dir /tmp/fp_debug_ros/20261111-101500
    python3 tools/analyze_fp_log.py --all          # 一次性分析所有运行

输出：跟踪耗时分布、稳定性验证每个窗口的帧数与位姿偏差、注册耗时与结果，
以及是"帧率不够"还是"位姿在抖"的判断建议。

只用标准库，不需要 ROS / torch / GPU。
"""

import argparse
import csv
import glob
import os
import statistics
import sys

DEFAULT_DIR = "/tmp/fp_debug_ros"


# ---------------- 读取 ----------------

def latest_run(root):
    runs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d))
    return runs[-1] if runs else None


def read_events(path):
    events = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                wall = float(parts[0])
            except ValueError:
                continue
            kv = {}
            for token in parts[1:]:
                if "=" in token:
                    key, _, val = token.partition("=")
                    kv[key] = val
            events.append((wall, kv))
    return events


def read_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                row["wall"] = float(row["wall"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def fnum(row, key, default=None):
    val = row.get(key, "")
    if val in ("", None):
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ---------------- 统计 ----------------

def pct(values, p):
    vals = sorted(values)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def fmt_ms(values):
    if not values:
        return "n/a"
    return (f"中位/均值/95分位 = "
            f"{statistics.median(values):.0f} / "
            f"{statistics.mean(values):.0f} / "
            f"{pct(values, 0.95):.0f} ms")


def settle_runs(rows):
    """把连续出现的 settle 行切成一次次验证窗口（n 从 1 开始）。"""
    runs = []
    current = []
    for row in rows:
        if row.get("phase") != "settle":
            continue
        n = fnum(row, "n", 0)
        if n == 1 and current:
            runs.append(current)
            current = []
        current.append(row)
        note = row.get("note", "")
        # 窗口满了或换了一次锁，下一次 n=1 会自动切
        if note and "lock=" in note:
            pass
    if current:
        runs.append(current)
    return runs


def analyze_run(run_dir):
    print(f"\n=== {run_dir} ===")
    events = read_events(os.path.join(run_dir, "events.log"))
    rows = read_rows(os.path.join(run_dir, "frames.csv"))
    settle_frames = 10
    for _, kv in events:
        if kv.get("kind") == "start":
            try:
                settle_frames = int(kv.get("settle_frames", settle_frames))
            except ValueError:
                pass

    # --- 事件 ---
    kinds = {}
    for _, kv in events:
        kinds[kv.get("kind", "?")] = kinds.get(kv.get("kind", "?"), 0) + 1
    if kinds:
        print("事件统计: " + "  ".join(f"{k}×{v}" for k, v in
                                     sorted(kinds.items())))
    for _, kv in events:
        if kv.get("kind") == "settle_timeout":
            print(f"  超时: 帧数={kv.get('frames')} "
                  f"耗时={kv.get('elapsed', '?')}s "
                  f"最大偏差 Δt={kv.get('max_dt_mm', '?')}mm "
                  f"Δr={kv.get('max_da_deg', '?')}°")
        elif kv.get("kind") == "register_done":
            print(f"  注册成功: {kv.get('elapsed', '?')}s rect={kv.get('rect')}")
        elif kv.get("kind") == "register_fail":
            print(f"  注册失败: {kv.get('elapsed', '?')}s {kv.get('error')}")
        elif kv.get("kind") == "register_bad_iou":
            print(f"  注册 IoU 不合格: {kv.get('iou')}")
        elif kv.get("kind") == "start":
            print(f"  启动参数: settle_frames={kv.get('settle_frames')} "
                  f"阈值 {kv.get('settle_dt_mm')}mm / "
                  f"{kv.get('settle_ddeg')}° 超时 {kv.get('settle_timeout')}s  "
                  f"est_iter={kv.get('est_refine_iter')} "
                  f"track_iter={kv.get('track_refine_iter')}")

    # --- 跟踪耗时 ---
    track = [r for r in rows if r.get("phase") == "track"]
    track_ms = [fnum(r, "track_ms") for r in track]
    track_ms = [v for v in track_ms if v is not None]
    if track_ms:
        print(f"跟踪（locked）: {len(track_ms)} 帧, {fmt_ms(track_ms)}")
        steps_t = [fnum(r, "dt_mm") for r in track]
        steps_a = [fnum(r, "da_deg") for r in track]
        steps_t = [v for v in steps_t if v is not None]
        steps_a = [v for v in steps_a if v is not None]
        if steps_t:
            print(f"  相邻帧位姿变化: Δt 95分位={pct(steps_t, 0.95):.2f}mm "
                  f"Δr 95分位={pct(steps_a, 0.95):.2f}°")

    # --- 稳定性验证窗口 ---
    runs = settle_runs(rows)
    if not runs:
        print("没有稳定性验证记录（还没进入 verifying 阶段）")
    for i, run in enumerate(runs, 1):
        frames = len(run)
        ms = [fnum(r, "track_ms") for r in run]
        ms = [v for v in ms if v is not None]
        dts = [fnum(r, "dt_mm") for r in run]
        das = [fnum(r, "da_deg") for r in run]
        dts = [v for v in dts if v is not None]
        das = [v for v in das if v is not None]
        elapsed = (run[-1]["wall"] - run[0]["wall"]) if len(run) > 1 else 0.0
        text = f"验证窗口 #{i}: {frames} 帧 / {elapsed:.2f}s"
        if ms:
            text += f", 跟踪 {statistics.median(ms):.0f}ms/帧"
        if dts:
            text += f", 相对首帧最大 Δt={max(dts):.1f}mm"
        if das:
            text += f" Δr={max(das):.2f}°"
        print(text)
        cxs = [fnum(r, "det_cx") for r in run]
        cys = [fnum(r, "det_cy") for r in run]
        ious = [fnum(r, "iou") for r in run]
        cxs = [v for v in cxs if v is not None]
        cys = [v for v in cys if v is not None]
        ious = [v for v in ious if v is not None]
        if len(cxs) >= 2 and len(cys) >= 2:
            move = ((cxs[-1] - cxs[0]) ** 2 + (cys[-1] - cys[0]) ** 2) ** 0.5
            print(f"    YOLO 框中心位移 {move:.1f}px", end="")
            if ious:
                print(f", YOLO/FP 框 IoU 中位 {statistics.median(ious):.2f}")
            else:
                print()
        elif ious:
            print(f"    YOLO/FP 框 IoU 中位 {statistics.median(ious):.2f}")

    # --- 判断 ---
    print("判断:")
    if not runs:
        print("  未进入验证阶段，无法判断。先确认 YOLO conf 能到阈值。")
        return
    full = 0
    slow = 0
    for run in runs:
        ms = [fnum(r, "track_ms") for r in run]
        ms = [v for v in ms if v is not None]
        if ms and statistics.median(ms) > 150:
            slow += 1
        if len(run) >= settle_frames:
            full += 1
    if slow:
        print(f"  {slow}/{len(runs)} 个窗口单帧跟踪 >150ms：Orin 上跟踪偏慢，"
              "先考虑把 settle_timeout 调大或 settle_frames 调小。")
    if full:
        print(f"  {full}/{len(runs)} 个窗口凑满了帧数：跟踪速度够，"
              "问题在位姿抖动，看窗口内的 Δt/Δr 与阈值对比。")
    else:
        print("  没有窗口凑满帧数：超时前帧数不够，是速度问题而不是抖动。")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="指定一次运行的目录")
    ap.add_argument("--root", default=DEFAULT_DIR,
                    help=f"运行目录的父目录（默认 {DEFAULT_DIR}）")
    ap.add_argument("--all", action="store_true", help="分析所有运行")
    args = ap.parse_args(argv)

    if args.dir:
        dirs = [args.dir]
    elif args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(d))
    else:
        d = latest_run(args.root)
        dirs = [d] if d else []
    if not dirs:
        print(f"没找到日志目录: {args.root}")
        return 1
    for d in dirs:
        analyze_run(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
