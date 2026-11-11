#!/usr/bin/env python3
"""FoundationPose ROS 节点。

订阅：
  /camera/color/image_raw        sensor_msgs/Image  bgr8
  /camera/depth/image_raw        sensor_msgs/Image  32FC1 (米)
  /camera/color/camera_info      sensor_msgs/CameraInfo
  /yolo/detections               vision_msgs/Detection2DArray

发布：
  /foundationpose/object_pose    geometry_msgs/PoseStamped  (mesh -> camera)
  /foundationpose/status         std_msgs/String
  /foundationpose/debug_image    sensor_msgs/Image  rgb8
  TF: <camera frame> -> <object_frame>

服务：
  /foundationpose/reacquire      std_srvs/Trigger  强制重新搜索/注册

工作流程和单进程脚本一致：
  YOLO 检测 -> 选一个框 -> SAM 分割 -> FoundationPose 注册 -> 逐帧跟踪
  跟踪期间持续拿 YOLO 框和 FP 投影框比 IoU，连续不一致/漏检就自动重新注册。
注册是在本节点的定时器里同步执行的，只会阻塞本节点；摄像头和 YOLO 节点
在各自进程里继续运行，所以画面不会像单进程脚本那样整窗卡住。
"""

import os
import sys
import threading
import time

# ---- FoundationPose 代码路径（和单进程脚本一致，可用环境变量覆盖）----
PROJECT_ROOT = os.environ.get("RCIA_ROOT", "/home/cheng/test/RCIA_Benmao_Vision")
FP_ROOT = os.environ.get("FOUNDATIONPOSE_ROOT",
                         os.path.join(PROJECT_ROOT, "foundation_pose"))
for _p in (PROJECT_ROOT, FP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fp_matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/fp_ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import rclpy
import torch
import trimesh
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection2DArray

import fp_common as fp

# torch 2.6 起 torch.load 默认 weights_only=True，官方权重加载不了
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import nvdiffrast.torch as dr  # noqa: E402
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402


def _header_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__("foundationpose")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，FoundationPose 需要 GPU")
        self.bridge = CvBridge()

        self._declare_params()
        self._load_mesh()
        self._init_estimator()
        self._init_state()
        self._init_recorder()
        self._init_ros()

    # ---------------- 诊断记录 ----------------

    def _init_recorder(self):
        self.rec = None
        if not self.debug_record:
            return
        run_dir = os.path.join(
            self.debug_record_dir, time.strftime("%Y%m%d-%H%M%S"))
        try:
            self.rec = fp.DebugRecorder(
                run_dir, save_frames=self.debug_record_frames)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"诊断记录初始化失败: {exc}")
            return
        self.rec.event(
            "start",
            mesh=self.mesh_path,
            settle_frames=self.settle_frames,
            settle_dt_mm=self.settle_dt * 1000,
            settle_ddeg=self.settle_ddeg,
            settle_timeout=self.settle_timeout,
            est_refine_iter=self.est_refine_iter,
            track_refine_iter=self.track_refine_iter,
            det_acquire_conf=self.det_acquire_conf,
            device=torch.cuda.get_device_name(0),
        )
        self.get_logger().info(f"诊断记录已开启: {run_dir}")

    # ---------------- 参数 / 模型 ----------------

    def _declare_params(self):
        p = self.declare_parameter
        self.mesh_path = str(p("mesh_path", "").value)
        self.mesh_scale = float(p("mesh_scale", 1.0).value)

        self.mask_mode = str(p("mask_mode", "sam").value)
        self.sam_checkpoint = str(p("sam_checkpoint", "").value)
        self.sam_model_type = str(p("sam_model_type", "vit_b").value)
        self.sam_select = str(p("sam_select", "score").value)
        # 6GB 桌面卡默认用完释放；Jetson 16GB 统一内存可设 true，
        # 省掉每次注册重新加载 SAM 的 1~2 秒
        self.sam_keep_loaded = bool(p("sam_keep_loaded", False).value)

        self.est_refine_iter = int(p("est_refine_iter", 5).value)
        # 两阶段注册：先给全部朝向假设打粗分，只精修前 reg_topk 个。
        # 实测 96 个假设 8.6s -> 2.8s（位姿差 <1.1mm/<1.6°）。
        # reg_topk<=0 或 reg_two_stage=false 时退回原版 register()。
        self.reg_two_stage = bool(p("reg_two_stage", True).value)
        self.reg_topk = int(p("reg_topk", 12).value)
        self.track_refine_iter = int(p("track_refine_iter", 2).value)
        self.crop_ratio = float(p("crop_ratio", 1.2).value)
        # 相机看不到物体背面时，只保留落在可见面一侧的初始朝向假设。
        # vis_dir 是 mesh 坐标系里「从物体指向相机」的方向；
        # rot_filter_deg=90 表示可见半球，0 表示关闭（保留全部 252 个）。
        self.rot_filter_deg = float(p("rot_filter_deg", 90.0).value)
        self.vis_dir = [float(v) for v in p(
            "vis_dir", [0.03, -0.50, 0.87]).value]
        # 注册完成后先跟踪一小段，位姿在窗口内足够稳定才对外发布。
        # settle_frames=0 关闭验证，注册完立即发布。
        self.settle_frames = int(p("settle_frames", 10).value)
        self.settle_dt = float(p("settle_dt", 0.005).value)
        self.settle_ddeg = float(p("settle_ddeg", 1.5).value)
        self.settle_timeout = float(p("settle_timeout", 3.0).value)
        self.smooth = float(p("smooth", 0.0).value)
        self.motion_predict = float(p("motion_predict", 0.0).value)
        self.fp_debug = bool(p("fp_debug", False).value)
        self.fp_debug_dir = str(p("fp_debug_dir", "/tmp/fp_debug").value)
        # 诊断记录：把注册/验证/跟踪过程写到 CSV + 事件日志，离线分析用
        self.debug_record = bool(p("debug_record", False).value)
        self.debug_record_dir = str(p(
            "debug_record_dir", "/tmp/fp_debug_ros").value)
        self.debug_record_frames = bool(
            p("debug_record_frames", True).value)

        self.yolo_track_check = bool(p("yolo_track_check", True).value)
        self.yolo_lock_iou = float(p("yolo_lock_iou", 0.25).value)
        self.yolo_lost_frames = int(p("yolo_lost_frames", 8).value)
        self.yolo_miss_frames = int(p("yolo_miss_frames", 15).value)
        self.det_acquire_conf = float(p("det_acquire_conf", 0.7).value)
        self.det_max_age = float(p("det_max_age", 0.5).value)
        self.det_timeout = float(p("det_timeout", 1.0).value)
        self.det_select = str(p("det_select", "conf").value)
        self.det_class = str(p("det_class", "").value)
        self.det_stable_frames = int(p("det_stable_frames", 3).value)
        self.det_stable_iou = float(p("det_stable_iou", 0.5).value)
        self.reg_min_iou = float(p("reg_min_iou", 0.15).value)
        self.acquire_cooldown = float(p("acquire_cooldown", 1.5).value)
        # 注册失败的框先记住，短时间内不再拿它反复试（避免一直卡在注册）
        self.bad_rect_ttl = float(p("bad_rect_ttl", 10.0).value)
        self.bad_rect_iou = float(p("bad_rect_iou", 0.4).value)

        self.depth_scale = float(p("depth_scale", 0.001).value)
        self.max_stamp_diff = float(p("max_stamp_diff", 0.05).value)
        self.camera_frame = str(p(
            "camera_frame", "camera_color_optical_frame").value)
        self.object_frame = str(p("object_frame", "object").value)

        self.publish_tf = bool(p("publish_tf", True).value)
        self.publish_debug_image = bool(p(
            "publish_debug_image", True).value)
        self.debug_rate = float(p("debug_rate", 15.0).value)
        self.processing_rate = float(p("processing_rate", 30.0).value)
        self.rate_log_period = float(p("rate_log_period", 5.0).value)
        # 调试图里画 mesh 投影轮廓 + 物体坐标系（红X 绿Y 蓝Z）
        self.draw_axes = bool(p("draw_axes", True).value)
        self.axis_length = float(p("axis_length", 0.05).value)

        self.color_topic = str(p(
            "color_topic", "/camera/color/image_raw").value)
        self.depth_topic = str(p(
            "depth_topic", "/camera/depth/image_raw").value)
        self.camera_info_topic = str(p(
            "camera_info_topic", "/camera/color/camera_info").value)
        self.detections_topic = str(p(
            "detections_topic", "/yolo/detections").value)
        self.pose_topic = str(p(
            "pose_topic", "/foundationpose/object_pose").value)
        self.status_topic = str(p(
            "status_topic", "/foundationpose/status").value)
        self.debug_topic = str(p(
            "debug_image_topic", "/foundationpose/debug_image").value)

    def _load_mesh(self):
        if not self.mesh_path or not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"找不到 mesh: {self.mesh_path!r}")
        if self.mask_mode == "sam" and not os.path.exists(self.sam_checkpoint):
            raise FileNotFoundError(
                f"找不到 SAM 权重: {self.sam_checkpoint!r}")
        for w in ("2024-01-11-20-02-45", "2023-10-28-18-33-37"):
            weight = os.path.join(FP_ROOT, "weights", w, "model_best.pth")
            if not os.path.exists(weight):
                raise FileNotFoundError(f"缺少 FoundationPose 权重: {weight}")

        self.get_logger().info(f"加载 mesh: {self.mesh_path}")
        loaded = trimesh.load(self.mesh_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise TypeError(f"mesh 里没有可用的三角网格: {self.mesh_path}")
            self.mesh = trimesh.util.concatenate(parts)
            self.get_logger().info(
                f"mesh 含 {len(parts)} 个子网格，已合并")
        else:
            self.mesh = loaded
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"不支持的 mesh 类型: {type(self.mesh)}")
        if self.mesh_scale != 1.0:
            self.mesh.apply_scale(self.mesh_scale)
            self.get_logger().info(f"mesh 缩放 x{self.mesh_scale}（换算到米）")
        if self.mesh.vertex_normals is None:
            self.mesh.compute_vertex_normals()

        tex_img = getattr(getattr(self.mesh.visual, "material", None),
                          "image", None)
        if tex_img is None:
            self.get_logger().warn(
                "mesh 没有贴图/UV，渲染只能用几何灰度；对称物更容易产生等价解")
        self.get_logger().info(
            f"mesh: {len(self.mesh.vertices)} 顶点, "
            f"{len(self.mesh.faces)} 面, "
            f"尺寸 {np.round(self.mesh.extents * 100, 1)} cm")

    def _init_estimator(self):
        self.get_logger().info(
            "初始化 FoundationPose（加载权重，可能需要 1-2 分钟）...")
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=self.fp_debug,
            debug_dir=self.fp_debug_dir,
        )
        if getattr(self.est, "refiner", None) is not None:
            self.est.refiner.cfg["crop_ratio"] = self.crop_ratio
        self.tf_center = self.est.get_tf_to_centered_mesh().data.cpu().numpy()
        self.tf_center_inv = np.linalg.inv(self.tf_center)
        self._filter_rot_grid()
        self.get_logger().info(
            f"FoundationPose 就绪，crop_ratio={self.crop_ratio}")

    def _filter_rot_grid(self):
        """只保留相机落在可见面一侧的初始朝向假设。

        相机永远看不到物体背面时，可以把 register() 的初始朝向从整个
        球面缩到可见半球，refiner/scorer 的工作量按假设数线性下降。
        vis_dir 用 mesh 坐标系里「从物体指向相机」的方向；实际值可以
        从一次成功注册的位姿反算：d = normalize(-R^T t)。
        """
        deg = float(self.rot_filter_deg)
        if deg <= 0:
            self.get_logger().info("朝向过滤已关闭，使用全部初始假设")
            return
        d = np.asarray(self.vis_dir, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            self.get_logger().warn("vis_dir 无效，跳过朝向过滤")
            return
        d /= n

        grid = self.est.rot_grid.detach().cpu().numpy()
        # 每个假设都把相机放在球面上的一个位置；旋转部分的逆给出
        # 相机在物体坐标系里的方向。
        cam_dir = np.linalg.inv(grid)[:, :3, 3]
        norm = np.linalg.norm(cam_dir, axis=1)
        ok = norm > 1e-9
        cam_dir[ok] /= norm[ok, None]
        cos_max = np.cos(np.radians(min(deg, 180.0)))
        keep = ok & (cam_dir @ d >= cos_max)
        n_keep = int(keep.sum())
        if n_keep < 6:
            self.get_logger().warn(
                f"朝向过滤 {deg:.0f}° 后只剩 {n_keep} 个假设，太少，"
                f"保留全部 {len(grid)} 个")
            return

        self.est.rot_grid = torch.as_tensor(
            grid[keep], device=self.est.rot_grid.device, dtype=torch.float)
        self.get_logger().info(
            f"朝向过滤 {deg:.0f}°: {len(grid)} -> {n_keep} 个初始假设 "
            f"(vis_dir={np.round(d, 3).tolist()})")

    def _init_state(self):
        self.K = None
        self.last_color = None
        self.last_depth = None
        self.last_det = None
        self.last_processed_stamp = None
        self.last_debug_time = 0.0
        self.last_status = ""
        # 速率统计：每隔 rate_log_period 秒打印一次实际 tick/debug/pose 频率
        self.tick_count = 0
        self.debug_count = 0
        self.pose_count = 0
        self.tick_busy = 0.0
        self.last_rate_log = time.time()
        # 调试画面由独立定时器渲染；这里只保存最新状态，渲染时再取图像
        self.debug_lock = threading.Lock()
        self.debug_selected = None
        self.debug_text = "searching"
        self.last_search_log = 0.0
        self.last_color_shape = None
        self.last_acquire_attempt = 0.0
        self.last_candidate_rect = None
        self.candidate_streak = 0
        self.bad_rect = None
        self.bad_rect_time = 0.0

        self.locked = False
        self.settling = False       # 注册后、提交前的稳定性验证阶段
        self.settle_start = 0.0
        self.settle_poses = []      # 稳定性窗口内的跟踪位姿
        self.pose = None            # 对外发布的位姿（可能已平滑）
        self.pose_smooth = None
        self.prev_raws = []         # 最近 3 帧原始跟踪位姿
        self.track_frames = 0
        self.disagree = 0
        self.miss = 0
        self.lock_index = 0
        self.last_iou = 0.0
        self._shape_warned = False

    def _init_ros(self):
        # depth=1：注册会阻塞本节点几秒，恢复后只要最新帧，不要在旧帧上
        # 逐帧补跟踪。
        latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # 回调分组：注册/跟踪会阻塞主定时器几秒，所以
        # - io 组：图像/检测订阅，继续接收最新帧
        # - main 组：处理与 reacquire 服务，同一时间只跑一个
        # - debug 组：独立线程渲染调试图，注册期间画面也不停
        self.io_cb_group = ReentrantCallbackGroup()
        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.debug_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Image, self.color_topic, self._color_cb, latest_qos,
            callback_group=self.io_cb_group)
        self.create_subscription(
            Image, self.depth_topic, self._depth_cb, latest_qos,
            callback_group=self.io_cb_group)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._info_cb, latest_qos,
            callback_group=self.io_cb_group)
        self.create_subscription(
            Detection2DArray, self.detections_topic, self._det_cb, latest_qos,
            callback_group=self.io_cb_group)

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.status_pub = self.create_publisher(
            String, self.status_topic, 10)
        self.debug_pub = self.create_publisher(
            Image, self.debug_topic, qos_profile_sensor_data)
        self.tf_broadcaster = (TransformBroadcaster(self)
                               if self.publish_tf else None)
        self.create_service(Trigger, "~/reacquire", self._reacquire_cb,
                            callback_group=self.main_cb_group)
        # 允许运行时用 ros2 param set 调这些参数，不用重启
        self.add_on_set_parameters_callback(self._on_param_update)
        self.timer = self.create_timer(
            1.0 / max(self.processing_rate, 1.0), self._tick,
            callback_group=self.main_cb_group)
        self.debug_timer = self.create_timer(
            1.0 / max(self.debug_rate, 1.0), self._debug_tick,
            callback_group=self.debug_cb_group)
        self.get_logger().info(
            f"订阅 {self.color_topic} + {self.depth_topic} + "
            f"{self.camera_info_topic} + {self.detections_topic}")
        self.get_logger().info(
            f"参数: debug_rate={self.debug_rate:.1f} "
            f"processing_rate={self.processing_rate:.1f} "
            f"track_refine_iter={self.track_refine_iter} "
            f"est_refine_iter={self.est_refine_iter} "
            f"det_acquire_conf={self.det_acquire_conf:.2f} "
            f"det_stable_frames={self.det_stable_frames} "
            f"acquire_cooldown={self.acquire_cooldown:.1f}s "
            f"settle_frames={self.settle_frames} "
            f"settle_dt={self.settle_dt * 1000:.1f}mm "
            f"settle_ddeg={self.settle_ddeg:.1f}°")

    # ---------------- ROS 回调 ----------------

    # 这些参数每一帧都会用到，运行时 set 后立即生效
    TUNABLES = {
        "det_acquire_conf": float,
        "det_stable_frames": int,
        "det_stable_iou": float,
        "reg_min_iou": float,
        "acquire_cooldown": float,
        "bad_rect_ttl": float,
        "bad_rect_iou": float,
        "yolo_track_check": bool,
        "yolo_lock_iou": float,
        "yolo_lost_frames": int,
        "yolo_miss_frames": int,
        "det_max_age": float,
        "det_timeout": float,
        "est_refine_iter": int,
        "track_refine_iter": int,
        "settle_frames": int,
        "settle_dt": float,
        "settle_ddeg": float,
        "settle_timeout": float,
        "smooth": float,
        "motion_predict": float,
        "draw_axes": bool,
        "axis_length": float,
    }

    def _on_param_update(self, params):
        for prm in params:
            cast = self.TUNABLES.get(prm.name)
            if cast is None:
                continue
            setattr(self, prm.name, cast(prm.value))
            self.get_logger().info(
                f"参数已更新: {prm.name}={getattr(self, prm.name)}")
        return SetParametersResult(successful=True)

    def _color_cb(self, msg):
        self.last_color = msg

    def _depth_cb(self, msg):
        self.last_depth = msg

    def _info_cb(self, msg):
        if len(msg.k) == 9:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _det_cb(self, msg):
        self.last_det = msg

    def _reacquire_cb(self, request, response):
        self._lose_lock("service reacquire")
        response.success = True
        response.message = "reacquiring"
        return response

    # ---------------- 主循环 ----------------

    def _tick(self):
        # 记录每次回调耗时，用于区分「处理太慢」和「DDS 丢消息」
        t0 = time.time()
        try:
            self._tick_once()
        finally:
            self.tick_count += 1
            self.tick_busy += time.time() - t0
            self._log_rates()

    def _log_rates(self):
        period = self.rate_log_period
        if period <= 0:
            return
        now = time.time()
        elapsed = now - self.last_rate_log
        if elapsed < period:
            return
        n = max(self.tick_count, 1)
        self.get_logger().info(
            f"速率: tick={self.tick_count / elapsed:.1f}Hz "
            f"debug={self.debug_count / elapsed:.1f}Hz "
            f"pose={self.pose_count / elapsed:.1f}Hz "
            f"回调平均耗时={1000.0 * self.tick_busy / n:.1f}ms")
        self.tick_count = 0
        self.debug_count = 0
        self.pose_count = 0
        self.tick_busy = 0.0
        self.last_rate_log = now

    def _tick_once(self):
        color_msg = self.last_color
        depth_msg = self.last_depth
        if color_msg is None or depth_msg is None or self.K is None:
            return

        key = (color_msg.header.stamp.sec, color_msg.header.stamp.nanosec)
        if key == self.last_processed_stamp:
            return
        dt = abs(_header_ns(color_msg.header.stamp)
                 - _header_ns(depth_msg.header.stamp)) * 1e-9
        if dt > self.max_stamp_diff:
            return

        self.last_processed_stamp = key
        try:
            color_bgr = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="bgr8")
            depth = self._depth_to_meters(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"图像转换失败: {exc}")
            return
        if depth.shape[:2] != color_bgr.shape[:2]:
            if not self._shape_warned:
                self.get_logger().warn(
                    f"color {color_bgr.shape[:2]} 和 depth "
                    f"{depth.shape[:2]} 尺寸不一致，跳过")
                self._shape_warned = True
            return

        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        if not self.locked:
            self._try_acquire(color_bgr, rgb, depth, color_msg)
        elif self.settling:
            self._settle(color_bgr, rgb, depth, color_msg)
        else:
            self._track(color_bgr, rgb, depth, color_msg)

    def _depth_to_meters(self, msg):
        enc = msg.encoding
        if enc == "32FC1":
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        elif enc in ("16UC1", "mono16"):
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth = raw.astype(np.float32) * self.depth_scale
        else:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.05] = 0.0
        return depth

    # ---------------- 检测结果 ----------------

    def _get_dets(self, color_msg):
        """返回 (dets, age)。dets: (x,y,w,h,score,class_id) 列表。"""
        msg = self.last_det
        if msg is None or self.last_color_shape is None:
            return [], float("inf")
        age = (_header_ns(color_msg.header.stamp)
               - _header_ns(msg.header.stamp)) * 1e-9
        age = max(0.0, age)
        h, w = self.last_color_shape[:2]
        dets = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            if self.det_class and str(hyp.class_id) != self.det_class:
                continue
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y
            rect = fp.xyxy_to_xywh(cx - sx * 0.5, cy - sy * 0.5,
                                   cx + sx * 0.5, cy + sy * 0.5, w, h)
            if rect is None:
                continue
            dets.append((rect[0], rect[1], rect[2], rect[3],
                         float(hyp.score), str(hyp.class_id)))
        return dets, age

    # ---------------- 搜索 / 注册 ----------------

    def _try_acquire(self, color_bgr, rgb, depth, color_msg):
        self.last_color_shape = color_bgr.shape
        dets, age = self._get_dets(color_msg)
        self._publish_status("searching")

        if age > self.det_timeout:
            self.last_candidate_rect = None
            self.candidate_streak = 0
            self._set_debug(None,
                            "searching: waiting for YOLO detections")
            self._log_search("等待 YOLO 检测结果...")
            return
        candidates = [d for d in dets if d[4] >= self.det_acquire_conf]
        # 刚注册失败过的框先跳过，否则每过一个冷却时间就会再卡 3 秒
        if candidates and self.bad_rect is not None:
            if time.time() - self.bad_rect_time < self.bad_rect_ttl:
                good = [d for d in candidates
                        if fp.xywh_iou(d[:4], self.bad_rect)
                        < self.bad_rect_iou]
                if not good:
                    self.last_candidate_rect = None
                    self.candidate_streak = 0
                    self._set_debug(
                        None, "searching: 忽略上次注册失败的框")
                    self._log_search("上次注册失败的框仍在画面里，暂时跳过")
                    return
                candidates = good
            else:
                self.bad_rect = None
        if not candidates:
            self.last_candidate_rect = None
            self.candidate_streak = 0
            self._set_debug(
                None, f"searching: no det >= {self.det_acquire_conf:.2f}")
            self._log_search(
                f"未发现目标（{len(dets)} 个候选，"
                f"conf>={self.det_acquire_conf:.2f}）")
            return

        selected = fp.pick_rect(candidates, self.det_select)
        rect = selected[:4]

        # 同一个框连续稳定出现若干帧才注册，过滤偶发误检
        if (self.last_candidate_rect is not None
                and fp.xywh_iou(rect, self.last_candidate_rect)
                >= self.det_stable_iou):
            self.candidate_streak += 1
        else:
            self.candidate_streak = 1
        self.last_candidate_rect = rect
        if self.candidate_streak < self.det_stable_frames:
            self._set_debug(
                selected,
                f"candidate {self.candidate_streak}/"
                f"{self.det_stable_frames} conf={selected[4]:.2f}")
            return

        # 注册/分割很贵：失败后不要每一帧都重试，等冷却时间再说
        now = time.time()
        if now - self.last_acquire_attempt < self.acquire_cooldown:
            return
        self.last_acquire_attempt = now

        self.get_logger().info(
            f"[fp] YOLO 锁定 {rect} conf={selected[4]:.2f}，开始 "
            f"{self.mask_mode} 分割...")
        self._set_debug(
            selected, f"locked {rect} -> {self.mask_mode} segmenting")

        try:
            mask = fp.make_mask(color_bgr, rect, self.mask_mode,
                                self.sam_checkpoint, self.sam_model_type,
                                self.sam_select,
                                keep_loaded=self.sam_keep_loaded)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"分割失败: {exc}")
            return
        area = int((mask > 0).sum())
        self.get_logger().info(f"[fp] mask 面积 {area} 像素")
        if area < 50:
            self.get_logger().warn("mask 太小，跳过本次注册")
            if self.rec:
                self.rec.event("mask_too_small", rect=rect, area=area)
            self._mark_bad_rect(rect)
            return

        if self.rec:
            self.rec.event("mask", rect=rect, conf=float(selected[4]),
                           area=area)
            # 保存注册用的原始帧，便于离线复现注册/跟踪
            self.rec.save_frame("register", rgb=rgb, depth=depth, mask=mask,
                                rect=[float(v) for v in rect],
                                conf=float(selected[4]), area=area)

        self._set_debug(selected,
                        "Registering FoundationPose (~3s)")
        self.get_logger().info("开始注册位姿（约 3 秒）...")
        t0 = time.time()
        self.est.pose_last = None
        try:
            if self.reg_two_stage:
                pose = fp.register_pose(
                    self.est, K=self.K, rgb=rgb, depth=depth, ob_mask=mask,
                    iteration=self.est_refine_iter, topk=self.reg_topk)
            else:
                pose = self.est.register(
                    K=self.K, rgb=rgb, depth=depth, ob_mask=mask,
                    iteration=self.est_refine_iter)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"注册失败: {exc}")
            if self.rec:
                self.rec.event("register_fail", rect=rect,
                               elapsed=time.time() - t0, error=str(exc))
            self.est.pose_last = None
            return
        pose = np.asarray(pose)
        if self.rec:
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            self.rec.event("register_done", rect=rect,
                           elapsed=time.time() - t0,
                           t=[float(v) for v in pose[:3, 3]],
                           q=[float(v) for v in quat])

        # 注册质量门限：注册出来的 mesh 投影必须和 YOLO 框对得上。
        # 否则多半是框在背景上/目标不在画面里，直接丢弃，别进入跟踪。
        fp_rect = fp.projected_rect(self.mesh, pose, self.K,
                                    color_bgr.shape)
        reg_iou = fp.xywh_iou(fp_rect, rect) if fp_rect is not None else 0.0
        if fp_rect is None or reg_iou < self.reg_min_iou:
            self.get_logger().warn(
                f"注册结果与 YOLO 框不一致 IoU={reg_iou:.2f} "
                f"< {self.reg_min_iou:.2f}，丢弃本次注册")
            if self.rec:
                self.rec.event("register_bad_iou", rect=rect,
                               iou=reg_iou)
            self.est.pose_last = None
            self._mark_bad_rect(rect)
            return

        self.get_logger().info(
            f"注册完成，用时 {time.time() - t0:.1f}s，"
            f"框 IoU={reg_iou:.2f}\n{np.round(pose, 4)}")

        self.locked = True
        self.last_candidate_rect = None
        self.candidate_streak = 0
        self.pose = pose.copy()      # 仅用于调试图；验证通过前不发布
        self.pose_smooth = pose.copy()
        self.prev_raws = []
        self.track_frames = 0
        self.disagree = 0
        self.miss = 0
        self.last_iou = 0.0
        self.lock_index += 1

        if self.settle_frames > 0:
            self.settling = True
            self.settle_start = time.time()
            self.settle_poses = []
            self._publish_status(f"verifying #{self.lock_index}")
            self._set_debug(
                selected,
                f"verifying #{self.lock_index} 0/{self.settle_frames}")
            self.get_logger().info(
                f"注册完成，开始稳定性验证：{self.settle_frames} 帧，"
                f"阈值 {self.settle_dt * 1000:.1f}mm / "
                f"{self.settle_ddeg:.1f}°，超时 {self.settle_timeout:.1f}s")
            return

        self._publish_pose(pose, color_msg)
        self._publish_status(f"locked #{self.lock_index}")
        self._set_debug(selected,
                        f"locked #{self.lock_index}")

    def _mark_bad_rect(self, rect):
        self.bad_rect = tuple(float(v) for v in rect)
        self.bad_rect_time = time.time()
        self.last_candidate_rect = None
        self.candidate_streak = 0

    def _log_search(self, text):
        now = time.time()
        if now - self.last_search_log >= 2.0:
            self.get_logger().info(text)
            self.last_search_log = now

    # ---------------- 跟踪 ----------------

    def _track(self, color_bgr, rgb, depth, color_msg):
        self.last_color_shape = color_bgr.shape

        if (self.motion_predict > 0 and self.track_frames >= 5
                and len(self.prev_raws) >= 3):
            pred = fp.predict_pose(self.prev_raws[-3], self.prev_raws[-2],
                                   self.prev_raws[-1], self.motion_predict)
            self.est.pose_last = torch.as_tensor(
                pred @ self.tf_center_inv, device="cuda", dtype=torch.float)

        t_track = time.time()
        try:
            pose_raw = self.est.track_one(
                rgb=rgb, depth=depth, K=self.K,
                iteration=self.track_refine_iter)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"跟踪失败: {exc}")
            self._lose_lock(f"track error: {exc}")
            self._set_debug(None, "lost: track error")
            return
        track_dt = time.time() - t_track
        pose_raw = np.asarray(pose_raw)
        if self.rec:
            if self.prev_raws:
                dt, da = fp.pose_delta(self.prev_raws[-1], pose_raw)
                self.rec.row("track", stamp=_header_ns(color_msg.header.stamp),
                             n=self.track_frames + 1,
                             track_ms=track_dt * 1000,
                             dt_mm=dt * 1000, da_deg=da)
            else:
                self.rec.row("track", stamp=_header_ns(color_msg.header.stamp),
                             n=self.track_frames + 1,
                             track_ms=track_dt * 1000)

        pose_out = pose_raw
        if self.smooth > 0:
            if self.pose_smooth is None:
                self.pose_smooth = pose_raw.copy()
            else:
                pose_out = fp.smooth_pose(
                    self.pose_smooth, pose_raw, self.smooth)
                self.pose_smooth = pose_out.copy()
                if self.motion_predict <= 0:
                    self.est.pose_last = torch.as_tensor(
                        pose_out @ self.tf_center_inv,
                        device="cuda", dtype=torch.float)

        self.prev_raws.append(pose_raw.copy())
        if len(self.prev_raws) > 3:
            self.prev_raws.pop(0)
        self.track_frames += 1
        self.pose = pose_out

        # ---- YOLO 持续校验 ----
        dets, age = self._get_dets(color_msg)
        lost_reason = None
        if self.yolo_track_check:
            fp_rect = fp.projected_rect(
                self.mesh, pose_raw, self.K, color_bgr.shape)
            if age > self.det_max_age:
                self.miss += 1
                self.disagree = 0
                self.last_iou = 0.0
            elif fp_rect is None:
                self.disagree += 1
                self.miss = 0
            elif not dets:
                self.miss += 1
                self.disagree = 0
                self.last_iou = 0.0
            else:
                best, _ = fp.max_iou(fp_rect, dets)
                self.last_iou = best
                if best >= self.yolo_lock_iou:
                    if self.disagree or self.miss:
                        self.get_logger().info(
                            f"YOLO 重新对上目标 IoU={best:.2f}")
                    self.disagree = 0
                    self.miss = 0
                else:
                    self.disagree += 1
                    self.miss = 0

            if self.disagree >= self.yolo_lost_frames:
                lost_reason = (
                    f"YOLO 框不匹配 IoU={self.last_iou:.2f} "
                    f"({self.disagree} 帧)")
            elif self.miss >= self.yolo_miss_frames:
                lost_reason = f"YOLO 连续漏检 ({self.miss} 帧)"

        if lost_reason is not None:
            self.get_logger().warn(f"{lost_reason}，重新搜索/注册")
            self._lose_lock(lost_reason)
            self._set_debug(None, f"lost: {lost_reason}")
            self._publish_status(f"lost: {lost_reason}")
            return

        self._publish_pose(pose_out, color_msg)
        self._publish_status(f"locked #{self.lock_index}")
        iou_text = (f"IoU={self.last_iou:.2f}" if self.yolo_track_check
                    else "IoU=off")
        self._set_debug(
            None,
            f"locked #{self.lock_index}  track={self.track_frames}  "
            f"{iou_text}")

    # ---------------- 稳定性验证 ----------------

    def _det_geometry(self, color_msg, pose, shape):
        """返回 (iou, cx, cy)：YOLO 框与 FP 投影框的 IoU、YOLO 框中心。

        诊断用：稳定性验证期间同时记录 YOLO 框中心。若 YOLO 框在动而 FP
        位姿也在动 → 物体真的在被移动；YOLO 框不动、FP 位姿在动 → 跟踪漂了。
        """
        dets, _ = self._get_dets(color_msg)
        if not dets:
            return None, None, None
        best = max(dets, key=lambda d: d[4])
        cx = best[0] + best[2] * 0.5
        cy = best[1] + best[3] * 0.5
        fp_rect = fp.projected_rect(self.mesh, pose, self.K, shape)
        iou = fp.xywh_iou(fp_rect, best[:4]) if fp_rect is not None else 0.0
        return iou, cx, cy

    def _settle(self, color_bgr, rgb, depth, color_msg):
        """注册后先跟踪一小段，位姿在窗口内稳定才提交。

        窗口是滑动的：物体先动后停时，等最近的 settle_frames 帧不再
        变化再提交；一直不稳定就按 settle_timeout 丢弃本次锁定。
        """
        self.last_color_shape = color_bgr.shape

        if time.time() - self.settle_start > self.settle_timeout:
            self.get_logger().warn(
                f"稳定性验证超时（{self.settle_timeout:.1f}s，"
                f"{len(self.settle_poses)} 帧），丢弃本次锁定")
            if len(self.settle_poses) >= 2:
                ref = self.settle_poses[0]
                max_t, max_a = 0.0, 0.0
                for p in self.settle_poses[1:]:
                    dt, da = fp.pose_delta(ref, p)
                    max_t = max(max_t, dt)
                    max_a = max(max_a, da)
                self.get_logger().warn(
                    f"[诊断] settle 窗口内最大偏差 "
                    f"Δt={max_t * 1000:.1f}mm Δr={max_a:.2f}° "
                    f"(阈值 {self.settle_dt * 1000:.1f}mm / "
                    f"{self.settle_ddeg:.1f}°)")
                if self.rec:
                    self.rec.event(
                        "settle_timeout",
                        lock=self.lock_index,
                        frames=len(self.settle_poses),
                        elapsed=time.time() - self.settle_start,
                        max_dt_mm=max_t * 1000, max_da_deg=max_a)
            elif self.rec:
                self.rec.event(
                    "settle_timeout", lock=self.lock_index,
                    frames=len(self.settle_poses),
                    elapsed=time.time() - self.settle_start)
            self._lose_lock("settle timeout")
            self._set_debug(None, "lost: settle timeout")
            self._publish_status("lost: settle timeout")
            return

        t_track = time.time()
        try:
            pose_raw = self.est.track_one(
                rgb=rgb, depth=depth, K=self.K,
                iteration=self.track_refine_iter)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"验证阶段跟踪失败: {exc}")
            self._lose_lock(f"settle track error: {exc}")
            self._set_debug(None, "lost: settle track error")
            self._publish_status("lost: settle track error")
            return
        track_dt = time.time() - t_track
        pose_raw = np.asarray(pose_raw)

        if self.settle_poses:
            dt, da = fp.pose_delta(self.settle_poses[0], pose_raw)
            self.get_logger().info(
                f"[诊断] settle {len(self.settle_poses) + 1}/"
                f"{self.settle_frames}  跟踪 {track_dt * 1000:.0f}ms  "
                f"自首帧 Δt={dt * 1000:.1f}mm Δr={da:.2f}°")
            if self.rec:
                iou, dcx, dcy = self._det_geometry(
                    color_msg, pose_raw, color_bgr.shape)
                self.rec.row(
                    "settle", stamp=_header_ns(color_msg.header.stamp),
                    n=len(self.settle_poses) + 1,
                    track_ms=track_dt * 1000,
                    dt_mm=dt * 1000, da_deg=da,
                    iou=iou, det_cx=dcx, det_cy=dcy,
                    note=f"lock={self.lock_index}")
        else:
            self.get_logger().info(
                f"[诊断] settle 1/{self.settle_frames}  "
                f"跟踪 {track_dt * 1000:.0f}ms")
            if self.rec:
                iou, dcx, dcy = self._det_geometry(
                    color_msg, pose_raw, color_bgr.shape)
                self.rec.row(
                    "settle", stamp=_header_ns(color_msg.header.stamp),
                    n=1, track_ms=track_dt * 1000,
                    iou=iou, det_cx=dcx, det_cy=dcy,
                    note=f"lock={self.lock_index}")

        self.settle_poses.append(pose_raw.copy())
        if len(self.settle_poses) > self.settle_frames:
            self.settle_poses.pop(0)

        n = len(self.settle_poses)
        if n < self.settle_frames:
            self._set_debug(
                None,
                f"verifying #{self.lock_index} {n}/{self.settle_frames}")
            return

        ref = self.settle_poses[0]
        max_t, max_a = 0.0, 0.0
        for p in self.settle_poses[1:]:
            dt, da = fp.pose_delta(ref, p)
            max_t = max(max_t, dt)
            max_a = max(max_a, da)

        if max_t > self.settle_dt or max_a > self.settle_ddeg:
            self._set_debug(
                None,
                f"verifying #{self.lock_index}  "
                f"Δt={max_t * 1000:.1f}mm Δr={max_a:.2f}°")
            if self.rec:
                self.rec.row(
                    "settle_window", stamp=_header_ns(color_msg.header.stamp),
                    n=n, max_dt_mm=max_t * 1000, max_da_deg=max_a,
                    note=f"lock={self.lock_index} unstable")
            return

        pose_commit = self._average_poses(self.settle_poses)
        self.settling = False
        self.settle_poses = []
        self.pose = pose_commit
        self.pose_smooth = pose_commit.copy()
        self.prev_raws = [pose_commit.copy()]
        self.track_frames = 0
        # 后续跟踪从提交位姿继续（pose_last 用 centered-mesh 坐标）
        self.est.pose_last = torch.as_tensor(
            pose_commit @ self.tf_center_inv,
            device="cuda", dtype=torch.float)

        self.get_logger().info(
            f"位姿稳定，已提交：{self.settle_frames} 帧内最大偏差 "
            f"{max_t * 1000:.1f}mm / {max_a:.2f}°")
        if self.rec:
            self.rec.row(
                "settle_window", stamp=_header_ns(color_msg.header.stamp),
                n=n, max_dt_mm=max_t * 1000, max_da_deg=max_a,
                note=f"lock={self.lock_index} committed")
            self.rec.event(
                "settle_commit", lock=self.lock_index, frames=n,
                max_dt_mm=max_t * 1000, max_da_deg=max_a,
                elapsed=time.time() - self.settle_start)
        self._publish_pose(pose_commit, color_msg)
        self._publish_status(f"locked #{self.lock_index}")
        self._set_debug(None, f"locked #{self.lock_index}")

    @staticmethod
    def _average_poses(poses):
        """平移取均值，旋转取四元数均值（先对齐符号）。"""
        t = np.mean([p[:3, 3] for p in poses], axis=0)
        quats = Rotation.from_matrix(
            np.stack([p[:3, :3] for p in poses])).as_quat()
        q0 = quats[0]
        for i in range(1, len(quats)):
            if float(np.dot(quats[i], q0)) < 0.0:
                quats[i] = -quats[i]
        q = quats.mean(axis=0)
        q /= np.linalg.norm(q)
        out = np.eye(4)
        out[:3, :3] = Rotation.from_quat(q).as_matrix()
        out[:3, 3] = t
        return out

    def _lose_lock(self, reason):
        if self.locked:
            self.get_logger().warn(f"丢失锁定: {reason}")
        if self.rec:
            self.rec.event("lose_lock", reason=reason,
                           locked=self.locked, settling=self.settling)
        self.locked = False
        self.settling = False
        self.settle_poses = []
        self.pose = None
        self.pose_smooth = None
        self.prev_raws = []
        self.track_frames = 0
        self.disagree = 0
        self.miss = 0
        self.last_iou = 0.0
        self.last_candidate_rect = None
        self.candidate_streak = 0
        # 主动跟丢后允许立刻重新注册同一个位置
        self.bad_rect = None
        self.bad_rect_time = 0.0
        self.est.pose_last = None

    # ---------------- 发布 ----------------

    def _publish_pose(self, pose, color_msg):
        header = color_msg.header
        if not header.frame_id:
            header.frame_id = self.camera_frame
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = float(pose[0, 3])
        msg.pose.position.y = float(pose[1, 3])
        msg.pose.position.z = float(pose[2, 3])
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.pose_pub.publish(msg)
        self.pose_count += 1

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header = header
            tf.child_frame_id = self.object_frame
            tf.transform.translation.x = float(pose[0, 3])
            tf.transform.translation.y = float(pose[1, 3])
            tf.transform.translation.z = float(pose[2, 3])
            tf.transform.rotation = msg.pose.orientation
            self.tf_broadcaster.sendTransform(tf)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _set_debug(self, selected, text):
        """处理线程只更新状态，真正的渲染在独立回调组的 _debug_tick 里做。"""
        with self.debug_lock:
            self.debug_selected = selected
            self.debug_text = text

    def _debug_tick(self):
        """独立的调试图发布定时器。

        注册/跟踪阻塞主定时器时，这个回调在另一个线程里继续用最新相机帧
        渲染，所以窗口里的画面不会停在注册前那一帧。
        """
        if not self.publish_debug_image or not rclpy.ok():
            return
        color_msg = self.last_color
        if color_msg is None:
            return
        now = time.time()
        # 定时器周期等于 1/debug_rate，留 10% 余量，避免抖动导致隔帧丢图
        if now - self.last_debug_time < 0.9 / max(self.debug_rate, 1.0):
            return
        self.last_debug_time = now

        try:
            vis = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"debug 图像转换失败: {exc}")
            return

        with self.debug_lock:
            selected = self.debug_selected
            text = self.debug_text
        dets, _age = self._get_dets(color_msg)
        fp.draw_detections_rgb(vis, dets, selected)
        pose = self.pose
        if pose is not None and self.K is not None:
            fp.draw_mesh_outline(vis, self.mesh, pose, self.K)
            if self.draw_axes:
                fp.draw_axes(vis, pose, self.K, self.axis_length)
        cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        try:
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"debug 图像发布失败: {exc}")
            return
        msg.header = color_msg.header
        try:
            self.debug_pub.publish(msg)
        except Exception:  # noqa: BLE001  节点正在关闭
            return
        self.debug_count += 1

    def destroy_node(self):
        if self.rec:
            self.rec.event("stop")
            self.rec.close()
        fp.release_sam()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = FoundationPoseNode()
        # 多线程执行器：注册/跟踪在 main 组阻塞时，io 组继续收图，
        # debug 组继续发布调试图，窗口画面不会停住。
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
