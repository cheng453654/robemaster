#!/usr/bin/python3
"""打开 Intel RealSense D435i，显示彩色图 + 深度图，并打印画面中心点的深度。

依赖：pyrealsense2 + opencv（已装在系统 python3.10 的 ~/.local 下）。
注意：当前 PATH 里默认的 python3 是 miniconda 的 3.14，里面没有这两个包，
      所以直接写 python3 会报 ModuleNotFoundError，请显式用 /usr/bin/python3。

用法：
    /usr/bin/python3 learn_d435.py
    /usr/bin/python3 learn_d435.py --frames 30 --no-display   # 无窗口自检
按键：
    q / ESC 退出
    s       保存当前一帧 color/depth 图片到 ./d435_snapshots/
    p       打印相机内参
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
SNAPSHOT_DIR = "d435_snapshots"


def parse_args():
    p = argparse.ArgumentParser(description="D435i 彩色+深度预览")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--frames", type=int, default=0, help="采集多少帧后退出，0 表示一直运行")
    p.add_argument("--no-display", action="store_true", help="不开窗口（无显示器/自检用）")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 创建管道和配置：同时打开 RGB 和深度两个流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    # 2. 启动，获取实际生效的流参数
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 深度单位：米/计数
    print(f"[d435] 已启动 {args.width}x{args.height}@{args.fps}  depth_scale={depth_scale:.6f} m/unit")

    # D435i 需要 USB 3.x；插在 USB 2 口上带宽不够，帧率会明显掉
    if device.supports(rs.camera_info.usb_type_descriptor):
        usb = device.get_info(rs.camera_info.usb_type_descriptor)
        print(f"[d435] USB 连接类型: {usb}")
        if usb.startswith("2"):
            print("[d435] ⚠ 检测到 USB 2.x，请换到主机/USB3 集线器的蓝色 USB 3 口，并使用原装 USB-C 线")

    # 3. 对齐：把深度图重投影到彩色相机视角，两图逐像素一一对应
    align = rs.align(rs.stream.color)
    # 深度图上色（近=暖色，远=冷色）
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    intrinsics_printed = False
    frame_count = 0
    t_start = time.time()
    last_saved = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())          # BGR uint8
            depth_image = np.asanyarray(depth_frame.get_data())          # uint16, 单位=depth_scale
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # 画面中心点深度（米）
            cx, cy = args.width // 2, args.height // 2
            center_dist = depth_frame.get_distance(cx, cy)

            vis = np.hstack((color_image, depth_colormap))
            cv2.putText(vis, f"center: {center_dist:.3f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(vis, (cx + args.width, cy), 5, (0, 255, 0), -1)

            if not intrinsics_printed:
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                print(f"[d435] color 内参: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                      f"cx={intr.ppx:.1f} cy={intr.ppy:.1f} "
                      f"distortion={intr.model}")
                intrinsics_printed = True

            if not args.no_display:
                cv2.imshow("D435i  color | depth", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and time.time() - last_saved > 0.5:
                    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"{SNAPSHOT_DIR}/color_{stamp}.png", color_image)
                    cv2.imwrite(f"{SNAPSHOT_DIR}/depth_{stamp}.png", depth_image)
                    print(f"[d435] 已保存到 {SNAPSHOT_DIR}/")
                    last_saved = time.time()
                if key == ord("p"):
                    print(f"[d435] 中心深度 = {center_dist:.4f} m "
                          f"({center_dist / depth_scale:.0f} 计数)")

            frame_count += 1
            if args.frames and frame_count >= args.frames:
                break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        dt = time.time() - t_start
        if dt > 0:
            print(f"[d435] 共 {frame_count} 帧，平均 {frame_count / dt:.1f} fps")


if __name__ == "__main__":
    main()
