#!/usr/bin/env python3
"""检查物体投影区域内的深度质量（只读，不影响正在跑的节点）。

做法：订阅当前位姿，把 mesh 投影成轮廓，统计轮廓内的有效深度比例。
黑色物体最常见的失败模式就是这里比例太低，导致 FoundationPose 注册
退化成"猜平移"。

用法：
    source /opt/ros/humble/setup.bash
    source ~/learn_ws/install/setup.bash
    conda activate foundationpose_ros
    python3 tools/depth_check.py

输出：
    - 轮廓像素数、有效深度比例
    - 深度中位数/分位数
    - 可视化图 /tmp/depth_check.png（红 = 轮廓内无深度）
"""

import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
import trimesh
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import fp_common as fp  # noqa: E402


class DepthCheck(Node):
    def __init__(self):
        super().__init__("depth_check")
        self.bridge = CvBridge()
        self.color = self.depth = self.info = self.pose = None
        q = qos_profile_sensor_data
        self.create_subscription(Image, "/camera/color/image_raw",
                                 self._color_cb, q)
        self.create_subscription(Image, "/camera/depth/image_raw",
                                 self._depth_cb, q)
        self.create_subscription(CameraInfo, "/camera/color/camera_info",
                                 self._info_cb, q)
        self.create_subscription(PoseStamped, "/foundationpose/object_pose",
                                 self._pose_cb, q)

    def _color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _depth_cb(self, msg):
        try:
            if msg.encoding == "32FC1":
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                d = raw.astype(np.float32) * 0.001
        except Exception:  # noqa: BLE001
            return
        d = np.asarray(d, dtype=np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0.05] = 0.0
        self.depth = d

    def _info_cb(self, msg):
        if len(msg.k) == 9:
            self.info = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = [p.x, p.y, p.z]
        self.pose = T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(
        HERE, "..", "config", "params.yaml"))
    ap.add_argument("--out", default="/tmp/depth_check.png")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.params))
    mesh_path = cfg["foundationpose"]["ros__parameters"]["mesh_path"]
    print(f"mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()
             if isinstance(g, trimesh.Trimesh)])

    rclpy.init()
    node = DepthCheck()
    import time
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (node.color is not None and node.depth is not None
                and node.info is not None and node.pose is not None):
            break

    if node.pose is None:
        print("没收到 /foundationpose/object_pose —— 先完成注册再来测")
        return
    if node.color is None or node.depth is None:
        print("没收到图像/深度话题")
        return
    if node.depth.shape[:2] != node.color.shape[:2]:
        print(f"color {node.color.shape[:2]} 与 depth "
              f"{node.depth.shape[:2]} 尺寸不一致")
        return

    K = node.info
    sil = fp.mesh_silhouette(mesh, node.pose, K, node.color.shape)
    roi = sil > 0
    n = int(roi.sum())
    if n == 0:
        print("mesh 投影到画面外了，位姿可能不对")
        return

    valid = node.depth > 0
    ratio = float(valid[roi].mean())
    d_in = node.depth[roi]
    d_ok = d_in[d_in > 0]

    print(f"\n轮廓像素: {n}  ({100*n/(roi.size):.2f}% 画面)")
    print(f"有效深度比例: {ratio*100:.1f}%")
    if d_ok.size:
        p = np.percentile(d_ok, [10, 50, 90])
        print(f"深度 (m): p10={p[0]:.3f}  中位={p[1]:.3f}  p90={p[2]:.3f}")

    frac = 100 * n / roi.size
    if ratio >= 0.80:
        verdict = "深度充足，FoundationPose 可以稳定工作"
    elif ratio >= 0.60:
        verdict = ("基本可用，但跟踪可能不稳：退远一点 / 斜一点 / 补光，"
                   "再测一次看能否到 80%")
    else:
        verdict = ("过低，黑色表面吸收太严重：提高激光功率、补侧光、"
                   "调整拍摄角度；仍不行就要考虑不依赖深度的方案")
    print(f"判断: {verdict}")
    if frac < 3.0:
        print(f"提示: 物体只占画面 {frac:.1f}%，太远了——位姿精度和"
              f"YOLO 检测都会受影响，试着靠近一点")

    # 可视化：红 = 轮廓内无深度
    vis = node.color.copy()
    vis[roi & ~valid] = (0, 0, 255)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    cv2.putText(vis, f"depth valid {ratio*100:.1f}%", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(args.out, vis)
    print(f"可视化: {args.out}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
