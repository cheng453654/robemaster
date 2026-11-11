#!/usr/bin/env python3
"""D435i 摄像头节点。

发布：
  /camera/color/image_raw      sensor_msgs/Image    bgr8
  /camera/depth/image_raw      sensor_msgs/Image    32FC1（单位：米，已对齐到彩色）
  /camera/color/camera_info    sensor_msgs/CameraInfo

深度已经用 rs.align(rs.stream.color) 对齐到彩色相机，并把 RealSense 的
16UC1 计数乘以 depth_scale 转成 32FC1 米制，下游节点不需要再换算。
"""

import threading

import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class D435CameraNode(Node):
    def __init__(self):
        super().__init__("d435_camera")

        self.width = int(self.declare_parameter("width", 640).value)
        self.height = int(self.declare_parameter("height", 480).value)
        self.fps = int(self.declare_parameter("fps", 30).value)
        self.frame_id = str(self.declare_parameter(
            "frame_id", "camera_color_optical_frame").value)
        self.depth_filter = bool(self.declare_parameter(
            "depth_filter", False).value)
        self.depth_temporal = bool(self.declare_parameter(
            "depth_temporal", True).value)
        # 彩色相机曝光/画质。默认自动曝光；运动模糊明显时可以改成手动，
        # 把 exposure 调小（单位 100µs，80 = 8ms）并适当提高 gain。
        self.auto_exposure = bool(self.declare_parameter(
            "auto_exposure", True).value)
        self.exposure = float(self.declare_parameter(
            "exposure", 166.0).value)
        self.gain = float(self.declare_parameter(
            "gain", 64.0).value)
        # 0=关闭，1=50Hz，2=60Hz，3=自动；国内市电选 1 可减少频闪条纹
        self.power_line_frequency = int(self.declare_parameter(
            "power_line_frequency", 3).value)
        self.sharpness = int(self.declare_parameter(
            "sharpness", 50).value)
        color_topic = str(self.declare_parameter(
            "color_topic", "/camera/color/image_raw").value)
        depth_topic = str(self.declare_parameter(
            "depth_topic", "/camera/depth/image_raw").value)
        info_topic = str(self.declare_parameter(
            "camera_info_topic", "/camera/color/camera_info").value)

        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(
            Image, color_topic, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(
            Image, depth_topic, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(
            CameraInfo, info_topic, qos_profile_sensor_data)

        # 没有插相机时 rs.pipeline.start() 会一直阻塞，先明确检查一次
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "没有检测到 RealSense 设备。请检查 USB 连接（建议 USB3 蓝口）")
        self.get_logger().info(
            f"检测到 RealSense: {devices[0].get_info(rs.camera_info.name)}")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16, self.fps)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                f"启动 D435i 失败: {exc}（相机可能被其他程序占用）") from exc

        self.align = rs.align(rs.stream.color)
        self.depth_filters = self._make_depth_filters()

        device = profile.get_device()
        self.device = device
        self._apply_color_options(device)
        self.add_on_set_parameters_callback(self._on_param_update)
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().intrinsics)
        self.intr = intr
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        usb = "unknown"
        if device.supports(rs.camera_info.usb_type_descriptor):
            usb = device.get_info(rs.camera_info.usb_type_descriptor)
        self.get_logger().info(
            f"camera {self.width}x{self.height}@{self.fps}  "
            f"fx={intr.fx:.1f} fy={intr.fy:.1f} "
            f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}  "
            f"depth_scale={self.depth_scale:.6f}  USB={usb}")
        if usb.startswith("2"):
            self.get_logger().warn(
                "检测到 USB 2.x，帧率/分辨率受限；建议换 USB3 蓝口和原装线")

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _on_param_update(self, params):
        """运行时调曝光参数，立即作用到相机。"""
        changed = False
        for prm in params:
            if prm.name == "auto_exposure":
                self.auto_exposure = bool(prm.value)
                changed = True
            elif prm.name == "exposure":
                self.exposure = float(prm.value)
                changed = True
            elif prm.name == "gain":
                self.gain = float(prm.value)
                changed = True
            elif prm.name == "power_line_frequency":
                self.power_line_frequency = int(prm.value)
                changed = True
            elif prm.name == "sharpness":
                self.sharpness = int(prm.value)
                changed = True
        if changed:
            self._apply_color_options(self.device)
        return SetParametersResult(successful=True)

    def _apply_color_options(self, device):
        """设置彩色相机曝光/频闪/锐度，并把最终生效值打印出来。"""
        try:
            sensor = device.first_color_sensor()
        except RuntimeError as exc:
            self.get_logger().warn(f"拿不到彩色 sensor，跳过曝光设置: {exc}")
            return

        def set_opt(name, value):
            try:
                if sensor.supports(name):
                    sensor.set_option(name, value)
                else:
                    self.get_logger().warn(f"该相机不支持 {name}")
            except RuntimeError as exc:
                self.get_logger().warn(f"设置 {name} 失败: {exc}")

        set_opt(rs.option.power_line_frequency,
                float(self.power_line_frequency))
        set_opt(rs.option.sharpness, float(self.sharpness))
        set_opt(rs.option.enable_auto_exposure,
                1.0 if self.auto_exposure else 0.0)
        if not self.auto_exposure:
            set_opt(rs.option.gain, self.gain)
            set_opt(rs.option.exposure, self.exposure)

        try:
            ex = sensor.get_option(rs.option.exposure)
            gn = sensor.get_option(rs.option.gain)
            self.get_logger().info(
                f"彩色相机: auto_exposure={self.auto_exposure} "
                f"exposure={ex:.0f}(100us) gain={gn:.0f} "
                f"power_line={sensor.get_option(rs.option.power_line_frequency):.0f}")
        except RuntimeError as exc:
            self.get_logger().warn(f"读取曝光参数失败: {exc}")

    def _make_depth_filters(self):
        if not self.depth_filter:
            return []
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        filters = [spatial]
        if self.depth_temporal:
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            filters.append(temporal)
        return filters

    def _capture_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError as exc:
                self.get_logger().warn(f"wait_for_frames: {exc}")
                continue

            # 只发布最新帧。下游处理慢时不能让相机队列越积越多，
            # 否则发布出去的是几秒前的画面。
            while True:
                try:
                    newer = self.pipeline.poll_for_frames()
                except RuntimeError:
                    break
                if newer.size() == 0:
                    break
                frames = newer

            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            for f in self.depth_filters:
                depth_frame = f.process(depth_frame)

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_m = (np.asanyarray(depth_frame.get_data()).astype(np.float32)
                       * self.depth_scale)
            depth_m[~np.isfinite(depth_m)] = 0.0
            depth_m[depth_m < 0.05] = 0.0

            stamp = self.get_clock().now().to_msg()

            color_msg = self.bridge.cv2_to_imgmsg(color_bgr, encoding="bgr8")
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = self.frame_id

            depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self.frame_id

            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = self.frame_id
            info.width = self.width
            info.height = self.height
            info.distortion_model = "plumb_bob"
            info.d = [float(v) for v in self.intr.coeffs]
            info.k = [float(v) for v in self.K.reshape(-1)]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [self.intr.fx, 0.0, self.intr.ppx, 0.0,
                      0.0, self.intr.fy, self.intr.ppy, 0.0,
                      0.0, 0.0, 1.0, 0.0]

            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            self.info_pub.publish(info)

    def destroy_node(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = D435CameraNode()
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
