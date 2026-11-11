#!/usr/bin/env python3
"""OpenCV 图像查看节点。

订阅一个 sensor_msgs/Image 话题并用窗口显示，默认显示 FoundationPose 的
调试图（检测框 + mesh 轮廓 + 状态）。也可以用参数切到原始彩色图：

    ros2 run learn_camera_d435 image_view_node.py \
      --ros-args -p image_topic:=/camera/color/image_raw

按键：q / ESC 关闭窗口并退出本节点（不会影响其他三个节点）。
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageViewNode(Node):
    def __init__(self):
        super().__init__("image_viewer")

        self.image_topic = str(self.declare_parameter(
            "image_topic", "/foundationpose/debug_image").value)
        self.window_name = str(self.declare_parameter(
            "window_name", "FoundationPose ROS").value)
        rate = float(self.declare_parameter("display_rate", 30.0).value)

        self.bridge = CvBridge()
        self.latest = None
        self.window_created = False
        self.sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._show)
        self.get_logger().info(
            f"显示 {self.image_topic}（窗口: {self.window_name}，q/ESC 退出）")

    def _image_cb(self, msg):
        # 只保留最新帧，显示慢时直接丢旧帧
        self.latest = msg

    def _show(self):
        try:
            if not self.window_created:
                # 延迟到收到第一帧再建窗口，避免无显示器时在初始化阶段报错
                if self.latest is None:
                    return
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_created = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.get_logger().info("关闭显示窗口")
                rclpy.shutdown()
                return

            if self.latest is None:
                return
            msg = self.latest
            self.latest = None
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"图像转换失败: {exc}")
                return
            cv2.imshow(self.window_name, img)
        except cv2.error as exc:
            self.get_logger().error(
                f"无法创建/更新窗口（没有显示器？可设 use_viewer:=false）: {exc}")
            rclpy.shutdown()

    def destroy_node(self):
        if self.window_created:
            cv2.destroyWindow(self.window_name)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageViewNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
