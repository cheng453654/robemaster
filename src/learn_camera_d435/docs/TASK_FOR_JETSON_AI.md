# 任务书：把 learn_camera_d435 移植到本机（Jetson Orin）

> 这份文件是给**运行在 Jetson 上的 AI 助手**看的执行指令。
> 请按阶段推进，每个阶段结束都要验证并汇报，不要跳步。

---

## 0. 任务目标

本机是一台 Jetson Orin（JetPack 6.x / Ubuntu 22.04 / aarch64 / Python 3.10）。
源机器是一台 x86_64 笔记本。目标是把源机器上的 ROS 2 视觉项目完整移植过来并跑通：

```
D435i 相机 → YOLO 检测 → SAM 分割 → FoundationPose 6D 注册/跟踪 → ROS 话题输出
```

最终要能执行：

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py
```

---

## 1. 源机器信息（传输时需要）

| 项目 | 值 |
|---|---|
| 用户名 | `cheng` |
| 当前 IP | `10.75.112.133`（手机热点，可能变化，以用户现场确认为准） |
| ROS 包 | `/home/cheng/learn_ws/src/learn_camera_d435` |
| FoundationPose | `/home/cheng/test/RCIA_Benmao_Vision` |
| YOLO 权重 | `/home/cheng/datasets/part4_yolo/runs/part4_v2_132/weights/best.pt` |
| 源机 ROS | Humble（和本机一致） |
| 源机 Python | 3.10（和本机一致） |

**本机目标路径**：

```
~/learn_ws/src/learn_camera_d435
~/RCIA_Benmao_Vision
~/models/part4/best.pt
```

---

## 2. 硬性约束（先读完再动手）

1. **不要用 conda。** JetPack 的 CUDA / cuDNN / TensorRT 是系统级的，conda 会覆盖它们，numpy 也会 ABI 冲突。用 `venv --system-site-packages`。
2. **不要从 PyPI 直接装 torch。** aarch64 + JetPack 必须用 NVIDIA 的轮子。
3. **不要拷贝编译产物。** 这些在源机器上是 x86_64 的，拷过来会 `ImportError: invalid ELF header`：
   - 任何 `*.so`
   - `build/`、`install/`、`log/`
   - 源机的 `yolo_venv/`（5.2 GB 的 x86 Python 环境）
4. **不要删除本机已有文件。** 需要覆盖时先备份（`mv x x.bak`）或先问用户。
5. **需要 sudo 的操作**：先输出要执行的命令和它的影响，等用户确认后再执行。
6. **每一步都要验证**，不要因为命令没报错就认为成功。
7. **遇到本文件没有覆盖的情况，停下来问用户**，不要凭猜测处理。
8. 不要改动源机器上的任何东西（传输只读）。

---

## 3. 阶段 0：只读勘察（先做，不装任何东西）

先收集事实并汇报，让用户确认方向：

```bash
cat /etc/nv_tegra_release
uname -m
python3 --version
lsb_release -a
df -h /
free -h
nvpmodel -q --verbose 2>/dev/null | head -5
nvcc --version 2>/dev/null || echo "nvcc 未安装"
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "torch 未安装"
ls /opt/ros 2>/dev/null || echo "ROS 未安装"
```

**汇报重点**：
- 是不是 `aarch64`、Python 是不是 3.10
- JetPack 版本（R36.x = 6.x，决定 CUDA 版本和 torch wheel 地址）
- ROS 是否已装、torch 是否已装
- 磁盘剩余空间（需要至少 10 GB）

如果发现本机已经装了某部分，记录并跳过对应阶段，不要重复安装。

---

## 4. 阶段 1：系统基础

```bash
# 换 apt 源（国内）
sudo sed -i 's@//ports.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
sudo apt update && sudo apt upgrade -y

sudo apt install -y build-essential cmake git curl wget vim \
  python3-pip python3-dev python3-venv \
  libssl-dev libusb-1.0-0-dev libudev-dev pkg-config \
  libgtk-3-dev libgl1-mesa-dev libglu1-mesa-dev \
  libjpeg-dev libpng-dev libavcodec-dev libavformat-dev
```

功耗与 swap：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks

sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**验证**：`tegrastats` 能看到 GPU 频率；`free -h` 显示 swap 8G。

---

## 5. 阶段 2：ROS 2 Humble

```bash
sudo add-apt-repository universe -y
sudo apt install -y software-properties-common curl

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
sudo apt install -y ros-humble-cv-bridge ros-humble-vision-msgs \
                    ros-humble-image-transport ros-humble-rqt-image-view
```

**验证**：

```bash
source /opt/ros/humble/setup.bash
ros2 --version
python3 -c "import rclpy; print('rclpy OK')"
```

---

## 6. 阶段 3：Python 环境 + PyTorch

```bash
python3 -m venv --system-site-packages ~/venv/fp
source ~/venv/fp/bin/activate
pip install -U pip setuptools wheel
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

装 PyTorch（**必须用 Jetson 专用源**，按阶段 0 的 JetPack 版本选）：

```bash
# JetPack 6.2（CUDA 12.6）
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# JetPack 6.1（CUDA 12.2）改用：
# pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu122
```

如果该源不可用，到 `https://developer.download.nvidia.com/compute/redist/jp/` 找对应版本的
`torch-*.whl` / `torchvision-*.whl` 下载后 `pip install <本地文件>`。

**验证**：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 必须输出 True
```

写入 `~/.bashrc`（**顺序很重要：先 ROS，再 venv**）：

```bash
cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
source ~/venv/fp/bin/activate
export TORCH_CUDA_ARCH_LIST="8.7"
export RCIA_ROOT=$HOME/RCIA_Benmao_Vision
export FOUNDATIONPOSE_ROOT=$RCIA_ROOT/foundation_pose
EOF
```

> 源机器用的是 torch 2.5.1，Jetson 上优先选 2.5.0；若后面 CUDA 扩展编译报 torch API 不兼容，
> 在 2.5.0 / 2.6.0 之间换成另一个再试。

---

## 7. 阶段 4：librealsense + pyrealsense2

Jetson 上没有 pip 可用的 aarch64 轮子，必须源码编译，且**必须开 RSUSB 后端**：

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.5

mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF
make -j$(nproc)
sudo make install
sudo ldconfig

sudo cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

> `make -j$(nproc)` 在 Orin NX 上可能因内存不足失败，改用 `make -j4`。

Python 绑定：

```bash
source ~/venv/fp/bin/activate
pip install pybind11
cd ~/librealsense/wrappers/python
pip install .
```

**验证**：

```bash
python - <<'EOF'
import pyrealsense2 as rs, numpy as np
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p.start(c)
for _ in range(3):
    print(np.asanyarray(p.wait_for_frames().get_color_frame().get_data()).shape)
p.stop()
EOF
```

> 相机必须插**蓝色 USB 3 口**。若无设备，先 `lsusb | grep Intel` 确认识别。

---

## 8. 阶段 5：其余 Python 依赖

```bash
source ~/venv/fp/bin/activate
pip install "numpy==1.26.4"
pip install opencv-python==4.11.0.86
pip install scipy trimesh pyyaml imageio
pip install ultralytics==8.4.98
pip install pybind11 ninja
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**验证**：

```bash
python -c "import cv2, numpy, scipy, trimesh, ultralytics, segment_anything; print('ok', numpy.__version__)"
```

> numpy 必须是 1.26.x。2.x 会导致 torch / opencv 的二进制接口报错。

---

## 9. 阶段 6：传输代码与权重

### 9.1 先测连通

```bash
ping -c 3 10.75.112.133
```

不通时按顺序排查：
1. 源机 IP 变了 → 让用户在源机执行 `hostname -I` 获取新 IP；
2. 热点客户端隔离 → 换网线直连，或改用 U 盘；
3. 源机没开 sshd → 让用户在源机执行
   `sudo apt install -y openssh-server && sudo systemctl enable --now ssh`。

### 9.2 拉取

```bash
mkdir -p ~/learn_ws/src ~/models/part4

rsync -avP --exclude build --exclude install --exclude log \
  cheng@<源机IP>:/home/cheng/learn_ws/src/learn_camera_d435/ \
  ~/learn_ws/src/learn_camera_d435/

rsync -avP \
  --exclude build --exclude '*.so' --exclude '*.egg-info' \
  --exclude __pycache__ --exclude yolo_venv --exclude webots_sim \
  --exclude datasets --exclude install --exclude log \
  --exclude 'OpenNI_*.zip*' --exclude runs \
  cheng@<源机IP>:/home/cheng/test/RCIA_Benmao_Vision/ \
  ~/RCIA_Benmao_Vision/

rsync -avP \
  cheng@<源机IP>:/home/cheng/datasets/part4_yolo/runs/part4_v2_132/weights/best.pt \
  ~/models/part4/
```

如果本机没有 rsync：`sudo apt install -y rsync`。

### 9.3 验证

```bash
du -sh ~/RCIA_Benmao_Vision ~/learn_ws/src/learn_camera_d435 ~/models
find ~/RCIA_Benmao_Vision -name '*.so'          # 必须为空
ls ~/RCIA_Benmao_Vision/foundation_pose/        # 应有 mycpp nvdiffrast bundlesdf Utils.py estimater.py
ls ~/RCIA_Benmao_Vision/models/sam/             # 应有 sam_vit_b_01ec64.pth (358M)
ls ~/RCIA_Benmao_Vision/foundation_pose/weights # 两个 model_best.pth (182M / 66M)
```

预期总量约 700 MB。

---

## 10. 阶段 7：编译 FoundationPose 扩展

```bash
source ~/venv/fp/bin/activate
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd $RCIA_ROOT/foundation_pose

# (1) mycpp
cd mycpp && rm -rf build && mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python) \
         -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j4
cd $RCIA_ROOT/foundation_pose

# (2) nvdiffrast
cd nvdiffrast && rm -rf build *.egg-info && pip install --no-build-isolation -e .
cd $RCIA_ROOT/foundation_pose

# (3) bundlesdf/mycuda
cd bundlesdf/mycuda && rm -rf build *.egg-info *.so && pip install -e .
cd $RCIA_ROOT/foundation_pose
```

**验证**：

```bash
cd $RCIA_ROOT && python - <<'EOF'
import os, sys
root = os.environ["RCIA_ROOT"]
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "foundation_pose"))
import torch
import mycpp.build.mycpp as mycpp
import nvdiffrast.torch as dr
from estimater import FoundationPose
print("mycpp / nvdiffrast / estimater OK, cuda:", torch.cuda.is_available())
EOF
```

编译约 20–40 分钟，`mycuda` 最慢。期间不要切功耗模式。

---

## 11. 阶段 8：编译 ROS 包

```bash
source /opt/ros/humble/setup.bash
cd ~/learn_ws
colcon build --packages-select learn_camera_d435 --symlink-install
source install/setup.bash
```

**验证**：

```bash
ros2 pkg prefix learn_camera_d435
ros2 run learn_camera_d435 d435_camera_node.py --help 2>&1 | head -3
```

---

## 12. 阶段 9：改配置路径

编辑 `~/learn_ws/src/learn_camera_d435/config/params.yaml`，这三条是源机器的绝对路径：

```yaml
yolo_detector:
  ros__parameters:
    model_path: /home/<本机用户名>/models/part4/best.pt

foundationpose:
  ros__parameters:
    mesh_path: /home/<本机用户名>/RCIA_Benmao_Vision/models/object/part4_cylinder_base.obj
    sam_checkpoint: /home/<本机用户名>/RCIA_Benmao_Vision/models/sam/sam_vit_b_01ec64.pth
```

> `RCIA_ROOT` 已由 `~/.bashrc` 提供，代码会自动读取，不需要改代码。
> 改完后若用的是 `--symlink-install`，不需要重新 build；否则重新 `colcon build`。

---

## 13. 阶段 10：分步验证（严禁直接跑整套）

按顺序，每一步单独起终端：

**① 相机**

```bash
source /opt/ros/humble/setup.bash && source ~/learn_ws/install/setup.bash
ros2 run learn_camera_d435 d435_camera_node.py
```

另一终端：

```bash
ros2 topic hz /camera/color/image_raw     # 接近 30
ros2 topic hz /camera/depth/image_raw
```

**② 相机 + YOLO**

```bash
ros2 run learn_camera_d435 yolo_detect_node.py
ros2 topic echo /yolo/detections --once
```

对着实物，应能检测到框。

**③ 只测 FoundationPose**

```bash
ros2 run learn_camera_d435 foundationpose_node.py
ros2 topic echo /foundationpose/status
# searching → verifying #n → locked #n
```

**④ 整套**

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py
# 无显示器时：
ros2 launch learn_camera_d435 foundationpose.launch.py use_viewer:=false
```

---

## 14. 阶段 11：性能调优（可选，基础跑通后再做）

1. YOLO 换 TensorRT（**engine 必须在板子上导**）：

   ```bash
   cd ~/models/part4
   yolo export model=best.pt format=engine half=True imgsz=640 device=0
   ```

   然后把 `model_path` 指向 `best.engine`，`process_rate` 从 30 降到 10~15。

2. `sam_keep_loaded: true`（Orin 统一内存，省掉每次注册重载 SAM 的 1–2 秒）。

3. `est_refine_iter: 3`、`track_refine_iter: 1`、`crop_ratio: 1.1`。

4. `debug_rate: 10`，不用调试时 `publish_debug_image: false`。

5. 用 `tegrastats` 确认温度 60–80 ℃、无降频。

---

## 15. 验收清单

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `import pyrealsense2, nvdiffrast, mycpp.build.mycpp` 全部成功
- [ ] `ros2 topic hz /camera/color/image_raw` → ~30
- [ ] `/yolo/detections` 有稳定输出
- [ ] `/foundationpose/status` 能到 `locked #n`
- [ ] `/foundationpose/object_pose` 有 ~30 Hz 输出
- [ ] `tegrastats` 无降频
- [ ] `use_viewer:=false` 时无显示器也能运行

---

## 16. 已知坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `invalid ELF header` / `wrong ELF class` | 拷了 x86 的 `.so` | 删除后按阶段 7 重编 |
| `No module named 'mycpp'` | C++ 扩展没编 | 检查 `mycpp/build/mycpp.cpython-310-aarch64-linux-gnu.so` |
| `torch.cuda.is_available()` 为 False | 装了 PyPI 的 torch | 卸载，换 NVIDIA wheel |
| `unsupported gpu architecture` | 架构号写错 | `TORCH_CUDA_ARCH_LIST=8.7` |
| `cv2.imshow` Qt/GTK 报错 | 无显示器或缺 GTK | `use_viewer:=false`，远程用 `rqt_image_view` |
| 注册要十几秒 | 未锁频 / 迭代次数高 / 没换 TensorRT | 阶段 11 |
| 内存 OOM | 16 GB CPU/GPU 共用 | 加 swap、关桌面、`sam_keep_loaded: false` |
| `rclpy` 导入失败 | source 顺序错 | 先 ROS、再 venv |
| numpy ABI 报错 | 装了 numpy 2.x | 固定 `numpy==1.26.4` |
| 相机只有 15 fps | USB 2 口 | 换蓝色 USB 3 |
| 两台机器话题不通 | `ROS_DOMAIN_ID` 不同 | 设成同一个值 |

---

## 17. 汇报格式

每完成一个阶段，用这个格式汇报：

```
[阶段 N] 名称
状态：完成 / 部分完成 / 失败
执行的命令：（关键命令）
验证结果：（贴关键输出）
遇到的问题：
下一步：
```

遇到以下情况**必须停下来问用户**，不要自行决定：
- 需要 sudo 且本文件没列出该命令
- 需要覆盖或删除已有文件
- 验证结果和预期不一致，且原因不明
- 需要下载超过 1 GB 的内容
