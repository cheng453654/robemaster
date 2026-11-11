#!/usr/bin/env python3
"""YOLO 检测节点。

订阅彩色图，发布 vision_msgs/Detection2DArray（每帧都发，没检测到就发空数组）。
默认置信度故意设得低一些（0.25），这样 FoundationPose 跟踪阶段还能拿到框
做一致性校验；FoundationPose 节点自己用 det_acquire_conf 决定注册门槛。
"""

import os

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/fp_ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)


def resolve_class_id(text, names):
    if text is None or str(text) == "":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    for cls_id, cls_name in (names or {}).items():
        if str(cls_name) == str(text):
            return int(cls_id)
    raise ValueError(f"YOLO 类别 {text!r} 不存在；模型类别: {names}")


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        model_path = str(self.declare_parameter("model_path", "").value)
        class_name = str(self.declare_parameter("class_name", "").value)
        self.conf = float(self.declare_parameter("conf", 0.25).value)
        self.device = str(self.declare_parameter("device", "0").value)
        process_rate = float(self.declare_parameter("process_rate", 30.0).value)
        self.publish_debug = bool(self.declare_parameter(
            "publish_debug_image", True).value)
        image_topic = str(self.declare_parameter(
            "image_topic", "/camera/color/image_raw").value)
        det_topic = str(self.declare_parameter(
            "detections_topic", "/yolo/detections").value)
        debug_topic = str(self.declare_parameter(
            "debug_image_topic", "/yolo/debug_image").value)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 YOLO 权重: {model_path!r}")
        self.get_logger().info(f"加载 YOLO: {model_path}")
        from ultralytics import YOLO
        self.model = YOLO(model_path)

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA 不可用，YOLO 改用 CPU")
            self.device = "cpu"

        self.class_id = resolve_class_id(class_name, self.model.names)
        self.classes = [self.class_id] if self.class_id is not None else None
        self.get_logger().info(
            f"YOLO 就绪，classes={self.classes} conf={self.conf} "
            f"device={self.device}")

        self.bridge = CvBridge()
        self.latest_msg = None
        self.sub = self.create_subscription(
            Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self.det_pub = self.create_publisher(
            Detection2DArray, det_topic, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(
            Image, debug_topic, qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / max(process_rate, 1.0),
                                       self._process)

    def _image_cb(self, msg):
        # 只保留最新帧；推理慢时宁可丢帧也不排队
        self.latest_msg = msg

    def _process(self):
        msg = self.latest_msg
        if msg is None:
            return
        self.latest_msg = None

        try:
            color_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"图像转换失败: {exc}")
            return

        try:
            kwargs = {
                "conf": self.conf,
                "verbose": False,
                "device": self.device,
            }
            if self.classes is not None:
                kwargs["classes"] = self.classes
            results = self.model.predict(color_bgr, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"YOLO 推理失败: {exc}")
            return

        out = Detection2DArray()
        out.header = msg.header

        names = {}
        boxes = None
        if results:
            res = results[0]
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)

        vis = color_bgr.copy() if (self.publish_debug and boxes is not None) else None
        n = 0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                n += 1
                det = Detection2D()
                det.header = msg.header
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(names.get(cls_id, cls_id))
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                det.bbox.center.position.x = float((x1 + x2) * 0.5)
                det.bbox.center.position.y = float((y1 + y2) * 0.5)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)
                out.detections.append(det)

                if vis is not None:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(vis, f"{score:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.det_pub.publish(out)

        if self.publish_debug:
            if vis is None:
                vis = color_bgr.copy()
            cv2.putText(vis, f"YOLO: {n} det  conf>={self.conf:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            dbg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            dbg.header = msg.header
            self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectNode()
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
