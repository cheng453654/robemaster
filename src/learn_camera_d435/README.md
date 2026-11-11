# learn_camera_d435

D435i + YOLO + SAM + FoundationPose 的 ROS 2 版本。原来单进程脚本
`learn_foundationpose.py` 里的功能拆成了三个独立节点：

| 节点 | 可执行文件 | 作用 |
| --- | --- | --- |
| `d435_camera` | `d435_camera_node.py` | 打开 D435i，发布彩色图、对齐后的深度图、相机内参 |
| `yolo_detector` | `yolo_detect_node.py` | 跑 YOLO，发布 `vision_msgs/Detection2DArray` |
| `foundationpose` | `foundationpose_node.py` | 选框 -> SAM 分割 -> 注册 -> 跟踪 -> YOLO 持续校验丢失 |
| `image_viewer` | `image_view_node.py` | 用 OpenCV 窗口显示调试图（launch 默认启动） |

YOLO 和 FoundationPose 在各自进程里，注册时只有 FoundationPose 节点会阻塞
约 3 秒，摄像头和 YOLO 继续运行。FoundationPose 节点内部又用多线程执行器
把调试图发布拆到独立回调组，所以注册期间显示窗口仍然刷新最新相机画面，
只是 mesh 轮廓要等注册完成才出现。

## 文件结构

```
learn_camera_d435/
├── src/
│   ├── d435_camera_node.py
│   ├── yolo_detect_node.py
│   ├── foundationpose_node.py
│   ├── fp_common.py            # 三个节点共用：SAM/投影/位姿平滑/框 IoU
│   ├── learn_d435.py           # 原来的单进程预览脚本（保留）
│   └── learn_foundationpose.py # 原来的单进程完整脚本（保留）
├── launch/foundationpose.launch.py
├── config/params.yaml
├── CMakeLists.txt
└── package.xml
```

## 环境

三个节点都用 `#!/usr/bin/env python3`，所以启动前要激活装了
`pyrealsense2 / ultralytics / torch / trimesh / segment_anything` 的 conda
环境，并 source ROS：

```bash
source /opt/ros/humble/setup.bash
source /home/cheng/learn_ws/install/setup.bash
source /home/cheng/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose_ros
```

## 编译

```bash
cd /home/cheng/learn_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select learn_camera_d435 --symlink-install
source install/setup.bash
```

## 启动

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py
```

启动后会弹出一个 OpenCV 窗口，默认显示
`/foundationpose/debug_image`（YOLO 框 + mesh 轮廓 + 物体坐标轴 + 状态文字）。
窗口里按 `q` 或 `ESC` 只关闭显示节点，摄像头/YOLO/FoundationPose 继续运行；
在启动终端按 `Ctrl-C` 才会停止整套系统。

没有显示器、或者不想要窗口时：

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py use_viewer:=false
```

只想看原始摄像头画面，不改代码的话单独开一个显示节点：

```bash
ros2 run learn_camera_d435 image_view_node.py \
  --ros-args -p image_topic:=/camera/color/image_raw
```

默认读取 `config/params.yaml`。换参数文件：

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py \
  params_file:=/绝对路径/params.yaml
```

也可以单独启动某个节点，例如只开摄像头：

```bash
ros2 run learn_camera_d435 d435_camera_node.py
```

## 话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | bgr8 |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | 32FC1，单位米，已对齐到彩色 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 彩色相机内参 |
| `/yolo/detections` | `vision_msgs/Detection2DArray` | 每帧都发，没检测到就是空数组 |
| `/yolo/debug_image` | `sensor_msgs/Image` | YOLO 候选框 |
| `/foundationpose/object_pose` | `geometry_msgs/PoseStamped` | mesh -> camera 位姿 |
| `/foundationpose/status` | `std_msgs/String` | `searching` / `verifying #n` / `locked #n` / `lost: ...` |
| `/foundationpose/debug_image` | `sensor_msgs/Image` | 检测框 + mesh 轮廓 + 坐标轴 + 状态（默认显示这个） |

TF：`camera_color_optical_frame -> object`（名字由 `object_frame` 参数控制）。

查看效果：

```bash
ros2 run rqt_image_view rqt_image_view /foundationpose/debug_image
ros2 topic echo /foundationpose/object_pose
ros2 run tf2_ros tf2_echo camera_color_optical_frame object
```

强制重新搜索/注册：

```bash
ros2 service call /foundationpose/reacquire std_srvs/srv/Trigger
```

## 关键参数

都在 `config/params.yaml`，按节点名分段。

### d435_camera

- `width` / `height` / `fps`：默认 640x480@30。
- `depth_filter`：默认 `false`。快速运动时不要开时间滤波，否则深度拖影。
- `depth_temporal`：只在 `depth_filter: true` 时生效。
- `auto_exposure` / `exposure` / `gain`：彩色相机曝光。默认自动；运动模糊
  明显时改成 `auto_exposure: false` 并把 `exposure` 调小（单位 100µs，
  80 = 8ms），同时适当提高 `gain`。
- `power_line_frequency`：0 关闭 / 1 50Hz / 2 60Hz / 3 自动。国内市电选 1
  可减少灯光频闪造成的条纹。
- `sharpness`：相机内部锐度，默认 50。

### yolo_detector

- `model_path`：训练好的 `best.pt`。
- `class_name`：只保留这个类别，空字符串表示不过滤。
- `conf`：默认 **0.25**。故意设低，保证跟踪阶段还能拿到框做校验；
  注册门槛由 FoundationPose 节点的 `det_acquire_conf` 控制。
- `device`：`"0"` 用 GPU，`"cpu"` 用 CPU。

### foundationpose

- `mesh_path` / `mesh_scale`：模型路径和缩放（已经是米就保持 1.0）。
- `mask_mode`：`sam` / `rect` / `grabcut`。
- `sam_checkpoint` / `sam_model_type` / `sam_select`：SAM 配置。
- `sam_keep_loaded`：SAM 是否常驻。默认 `false`（6GB 桌面卡每次用完释放）；
  Jetson 16GB 统一内存建议 `true`，省掉每次注册重新加载的 1~2 秒。
- `rot_filter_deg` / `vis_dir`：初始朝向过滤。相机看不到物体背面时，
  只保留落在 `vis_dir` 周围 `rot_filter_deg` 度锥体内的初始朝向假设。
  `90` 表示可见半球，252 个假设减到约 126 个，注册时间约减半；
  `60` 更快但要求物体放置朝向更固定；`0` 关闭。
  `vis_dir` 是 mesh 坐标系里「从物体指向相机」的方向，可由一次成功注册
  的位姿反算：`d = normalize(-R^T t)`。改这两个参数要重启节点。
- `settle_frames` / `settle_dt` / `settle_ddeg` / `settle_timeout`：
  注册完成后的稳定性验证。节点先用注册位姿持续跟踪 `settle_frames` 帧，
  最近这个窗口内所有位姿相对首帧的平移/旋转偏差都不超过
  `settle_dt`（米）/ `settle_ddeg`（度），才把窗口位姿取平均后发布，
  状态从 `verifying #n` 变成 `locked #n`。窗口期间不发布位姿和 TF。
  窗口是滑动的，物体先动后停也能等到稳定；超过 `settle_timeout` 仍
  不稳定则丢弃本次锁定、回到搜索。默认 10 帧 / 5mm / 1.5° / 3s；
  `settle_frames: 0` 关闭验证（注册完立即发布）。
- `est_refine_iter`：注册迭代次数，默认 5；想快可以降到 3。
- `track_refine_iter`：跟踪迭代次数，默认 1（实测跟踪约 26Hz）；改成 2 更准，
  但跟踪只有约 15Hz，画面会更顿。
- `crop_ratio`：跟踪裁剪窗，快速运动可以调到 1.5。
- `smooth`：位姿平滑，0 关闭；抖动明显可试 0.3~0.5。
- `motion_predict`：快速运动外推，0 关闭；可试 1.0。
- `det_acquire_conf`：YOLO 置信度达到多少才提交 SAM/注册，默认 0.85。
  背景误检通常在 0.7 左右，调高可以避免反复注册导致画面卡。
- `det_stable_frames` / `det_stable_iou`：同一个框连续稳定多少帧、IoU 多少
  才注册，默认 3 帧 / 0.5。
- `reg_min_iou`：注册后 mesh 投影和 YOLO 框的最小 IoU，默认 0.15；
  对不上就丢弃本次注册。
- `yolo_track_check`：跟踪期间是否用 YOLO 校验。
- `yolo_lock_iou`：FP 投影框和 YOLO 框的最小 IoU，默认 0.15。放宽是为了
  容忍快速运动时投影的短暂滞后。
- `yolo_lost_frames`：连续多少帧 IoU 不达标才重新注册，默认 15。
- `yolo_miss_frames`：连续多少帧没检测到才重新注册，默认 30。
- `det_max_age` / `det_timeout`：检测结果最大允许延迟（秒），默认 1.0 / 0.5。
- `acquire_cooldown`：分割/注册失败后的重试冷却时间，默认 1.5 秒。
- `bad_rect_ttl` / `bad_rect_iou`：注册失败（mesh 投影和 YOLO 框对不上）的框，
  在 `bad_rect_ttl` 秒内不再重试，避免反复卡 3 秒；与失败框 IoU 大于
  `bad_rect_iou` 的检测都会被跳过。默认 10 秒 / 0.4。
- `processing_rate`：FP 节点处理频率，默认 30 Hz。
- `publish_debug_image` / `debug_rate`：调试图开关和发布频率。
- `draw_axes` / `axis_length`：调试图里是否在 mesh 原点画物体坐标系
  （红 X、绿 Y、蓝 Z），轴长单位米，默认 0.05（5cm）。
- `rate_log_period`：每隔多少秒打印一次实际 `tick/debug/pose` 频率，
  默认 5 秒，设 0 关闭。

上面这些和检测/注册/跟踪相关的参数都支持运行时修改，不用重启：

```bash
ros2 param set /foundationpose det_acquire_conf 0.9
ros2 param set /foundationpose bad_rect_ttl 20.0
```

`debug_rate`、`processing_rate` 这类频率参数改了要重启节点才生效。

### image_viewer

- `image_topic`：要显示的话题，默认 `/foundationpose/debug_image`；
  改成 `/camera/color/image_raw` 可看原始画面。
- `window_name`：窗口标题。
- `display_rate`：刷新频率。

## 部署到 Jetson Orin NX Super

目标平台：JetPack 6.2（Ubuntu 22.04，aarch64，Python 3.10，TensorRT 10.x），
Orin NX 16GB / 157 TOPS(INT8, Super 模式)。

### 环境（必须换）

- **不要用现在这台 x86 的 conda 环境**。用系统 `python3.10` + venv
  （加 `--system-site-packages` 以便用 ROS 的 rclpy/cv_bridge），
  或者直接用 NVIDIA L4T ROS 容器。
- **PyTorch 用 NVIDIA 给 JetPack 的 wheel**，不要从 PyPI 装。
- **pyrealsense2 要源码编译 librealsense**（Jetson 上没有现成 wheel），
  必要时加 `-DFORCE_RSUSB_BACKEND=ON`。
- **FoundationPose 的 CUDA 扩展在板子上编译**：
  `export TORCH_CUDA_ARCH_LIST="8.7"`。nvdiffrast 也要在板子上装。
- **TensorRT engine 必须在板子上导出**，x86 上导出的不能用：
  ```bash
  yolo export model=best.pt format=engine half=True imgsz=640 device=0
  ```
  然后把 `yolo_detector.model_path` 指向 `best.engine`。
- 所有路径（mesh / SAM / YOLO 权重）改成板子上的绝对路径，或者用
  `RCIA_ROOT` / `FOUNDATIONPOSE_ROOT` 环境变量覆盖。

### 性能（按这个顺序调）

1. **YOLO 换 TensorRT FP16**，并把 `yolo_detector.process_rate` 从 30 降到
   10~15（跟踪校验不需要 30Hz，把 GPU 让给 FoundationPose）。
2. **`sam_keep_loaded: true`**：Orin 是统一内存，不用每次注册都重新加载
   SAM，单次注册能省 1~2 秒。桌面 6GB 卡保持 `false`。
3. **`est_refine_iter: 3`、`track_refine_iter: 1`、`crop_ratio: 1.1`**：
   降低注册/跟踪计算量，先验证精度是否够。
4. **`debug_rate: 10~15`**，无显示器时 `use_viewer:=false`，
   远程用 `rqt_image_view` 看。
5. **开满功耗并锁频**：`sudo nvpmodel -m 0`（MAXN Super / 40W）、
   `sudo jetson_clocks`，确保散热。用 `tegrastats` 看 GPU/CPU/温度/功耗。
6. **USB3**：D435i 接蓝色 USB3 口；USB2 会限制帧率。
7. **内存**：16GB 是 CPU/GPU 共用的，建议开 8~16GB swap/zram，
   无头运行（关桌面）能省 1~2GB。

### 预期

Orin 的显存带宽（约 102GB/s）和 FP32 算力都明显低于桌面独显，模型推理
大概慢 2~4 倍。建议先用自带的 `速率: tick=… debug=… pose=…` 日志和
`tegrastats` 实测，再决定是否继续上 FP16/TensorRT 版 FoundationPose。

## 常见问题

**启动时报没有 RealSense 设备**：摄像头节点会先检查设备再启动，
`rs.pipeline.start()` 在没有设备时会一直阻塞，所以这里直接报错。检查 USB
连接，建议 USB3 蓝口和原装线。

**USB 2.x 警告**：带宽不够，分辨率和帧率会受限。

**画面偶尔模糊**：

1. 运动模糊——自动曝光在室内常见 1/60~1/30 秒，曝光时间内物体或相机一动
   就拖影。加灯最有效；也可以改手动曝光：
   ```bash
   # 先看当前曝光（单位 100µs）
   ros2 param set /d435_camera auto_exposure false
   ros2 param set /d435_camera exposure 80
   ros2 param set /d435_camera gain 90
   ```
2. 定焦镜头——D435i 的彩色镜头是定焦，离目标太近（大约 0.3m 以内）会虚焦，
   保持 0.5m 左右最清晰。
3. 逆光/暗光——相机会拉长曝光、提高 gain，噪声大看起来发虚；避免对着窗户拍。
4. 灯光频闪——把 `power_line_frequency` 设为 1（50Hz）可以减少横条纹。
5. 显示窗口被拉大——OpenCV 窗口放大是插值缩放，会变软。保存一帧原图对比
   就能排除（原图清楚就是显示问题，不是相机问题）。

**显存不够**：SAM 在 `make_mask` 用完立即释放；如果仍然 OOM，先关掉
`/yolo/debug_image` 或用 `track_refine_iter: 1`。也可以在独立的终端里用
`nvidia-smi` 看占用。

**没有画面**：ROS 节点没有 OpenCV 窗口，用
`rqt_image_view /foundationpose/debug_image` 看。调试图发布在独立线程里，
注册期间画面也会继续刷新；如果仍然卡住，看终端里的
`速率: tick=... debug=... 回调平均耗时=...` 日志判断是处理慢还是 DDS 丢帧。

**想改回单进程调试**：原来的 `src/learn_foundationpose.py` 仍然保留，可按原
方式运行，但它默认会开 OpenCV 窗口。
