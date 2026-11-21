#!/usr/bin/env python3
"""在相机画面上拖框，直接把 manual_rect 设到 FoundationPose 节点。

用法：
    source /opt/ros/humble/setup.bash
    source ~/learn_ws/install/setup.bash
    conda activate foundationpose_ros
    python3 tools/pick_box.py

操作：
    左键拖动 = 画框，松手后立即生效（设置 manual_rect + 触发 reacquire）
    r        = 清空 manual_rect（恢复 YOLO 检测）
    q / ESC  = 退出

坐标是彩色相机原始分辨率（640x480）下的像素值，左上角原点。

如果 FoundationPose 节点没在运行，工具只打印坐标和手动命令，不会报错退出。
"""

import argparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class Picker(Node):
    def __init__(self, topic, scale, node_name, apply):
        super().__init__("box_picker")
        self.bridge = CvBridge()
        self.scale = max(scale, 0.1)
        self.target = node_name
        self.apply = apply
        self.frame = None
        self.start = None
        self.end = None
        self.rect = None

        # 相机节点用 sensor_data QoS（BEST_EFFORT），订阅端必须一致，
        # 否则会报 "incompatible QoS"，一帧都收不到
        self.create_subscription(
            Image, topic, self._cb, qos_profile_sensor_data)

        self.set_param_cli = self.create_client(
            SetParameters, f"{self.target}/set_parameters")
        self.reacquire_cli = self.create_client(
            Trigger, f"{self.target}/reacquire")

    def _cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:  # noqa: BLE001
            pass

    # ---------------- 把框写进节点 ----------------

    def _call_reacquire(self, _future=None):
        if not self.reacquire_cli.service_is_ready():
            if not self.reacquire_cli.wait_for_service(timeout_sec=1.0):
                print("  [警告] 找不到 reacquire 服务", flush=True)
                return
        self.reacquire_cli.call_async(Trigger.Request())
        print("  已触发重新注册", flush=True)

    def _push_param(self, raw):
        if not self.apply:
            print("  (--no-apply，未发送)", flush=True)
            return
        if not self.set_param_cli.service_is_ready():
            if not self.set_param_cli.wait_for_service(timeout_sec=1.0):
                print(f"  [警告] 找不到 {self.target}/set_parameters，"
                      f"确认节点在运行；可手动执行下面的命令", flush=True)
                return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="manual_rect",
            value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                 string_value=raw))]
        fut = self.set_param_cli.call_async(req)

        def _done(f):
            try:
                res = f.result()
                ok = bool(res and all(r.successful for r in res.results))
            except Exception:  # noqa: BLE001
                ok = False
            print(f"  manual_rect 设置{'成功' if ok else '失败'}",
                  flush=True)
            if ok:
                self._call_reacquire()
        fut.add_done_callback(_done)

    # ---------------- 鼠标 ----------------

    def mouse(self, event, x, y, _flags, _param):
        # 鼠标坐标是缩放后窗口里的，换算回原图
        x = int(x / self.scale)
        y = int(y / self.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            self.end = (x, y)
            x1, y1 = self.start
            x2, y2 = self.end
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            if rw > 4 and rh > 4:
                self.rect = (rx, ry, rw, rh)
                raw = f"{rx},{ry},{rw},{rh}"
                print(f"\n框: {raw}", flush=True)
                self._push_param(raw)
            self.start = None


def main():
    ap = argparse.ArgumentParser(description="在相机画面上拖框，设置 manual_rect")
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--scale", type=float, default=1.6,
                    help="窗口放大倍数，默认 1.6")
    ap.add_argument("--node", default="/foundationpose",
                    help="FoundationPose 节点名，默认 /foundationpose")
    ap.add_argument("--no-apply", action="store_true",
                    help="只打印坐标，不自动设置参数")
    args = ap.parse_args()

    rclpy.init()
    node = Picker(args.topic, args.scale, args.node, not args.no_apply)

    win = "pick box  (drag / r reset / q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, node.mouse)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is None:
                continue
            vis = node.frame.copy()
            if node.start is not None and node.end is not None:
                cv2.rectangle(vis, node.start, node.end, (0, 255, 0), 2)
            if node.rect is not None:
                x, y, w, h = node.rect
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(vis, f"{x},{y},{w},{h}", (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_LINEAR)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                node.start = node.end = node.rect = None
                node._push_param("")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
