# 移植到 Jetson Orin（JetPack 6.x / Ubuntu 22.04 / aarch64）

目标机：Jetson Orin 系列，全新系统。
源机器：x86_64 + Ubuntu 22.04 + ROS 2 Humble + conda（Python 3.10）。

---

## 0. 先搞清楚什么能拷、什么必须重编

| 内容 | 处理方式 | 原因 |
|---|---|---|
| `learn_camera_d435/` 代码 | ✅ 直接拷 | 纯 Python |
| `foundation_pose/` 源码 | ✅ 拷（排除 build 产物） | 纯 Python + C++ 源码 |
| `foundation_pose/**/*.so` | ❌ 删掉重编 | 当前是 `cpython-310-x86_64-linux-gnu.so` |
| `foundation_pose/weights/` (247 M) | ✅ 直接拷 | 数据文件，与架构无关 |
| `models/` 下 object + sam (358 M) | ✅ 直接拷 | 同上 |
| `best.pt` (6 M) | ✅ 直接拷 | PyTorch 权重可跨架构 |
| `learn_ws/build/ install/ log/` | ❌ 不要拷 | CMake 缓存里写死了 x86 路径 |
| `RCIA/yolo_venv/` (5.2 G) | ❌ 不要拷 | x86 的 Python 虚拟环境 |
| `RCIA/build/ install/ log/ webots_sim/ datasets/` | ❌ 不要拷 | 用不到或需重编 |
| conda 环境 | ❌ 不要重建 | JetPack 上 numpy/torch 会打架，用 venv |

总传输量约 **700 MB**，其中 605 MB 是权重。

---

## 1. 目标机基础准备

确认系统版本：

```bash
cat /etc/nv_tegra_release          # 应显示 R36.x = JetPack 6.x
jetson_release                     # 可选，需要 jetson-stats
python3 --version                  # 必须是 3.10.x
uname -m                           # aarch64
```

**换源 + 更新**（国内必做，否则 apt 会慢到怀疑人生）：

```bash
sudo sed -i 's@//ports.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git curl wget vim \
                    python3-pip python3-dev python3-venv \
                    libssl-dev libusb-1.0-0-dev libudev-dev pkg-config \
                    libgtk-3-dev libgl1-mesa-dev libglu1-mesa-dev \
                    libjpeg-dev libpng-dev libavcodec-dev libavformat-dev
```

**功耗与散热**（Orin NX/Nano 默认是低功耗模式，不调体验会很差）：

```bash
sudo nvpmodel -m 0        # MAXN（Orin Nano 是 15W→MAXN，Orin NX 是 40W）
sudo jetson_clocks        # 锁频
tegrastats                # 另开终端看 GPU/CPU/温度/功耗
```

**加 swap**（16 GB 统一内存，ROS + 两个模型容易吃紧）：

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 2. 安装 ROS 2 Humble

Ubuntu 22.04 对应 Humble，arm64 有官方 apt 包：

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

# 本项目实际用到的
sudo apt install -y ros-humble-cv-bridge ros-humble-vision-msgs \
                    ros-humble-image-transport ros-humble-rqt-image-view
```

写进 `~/.bashrc`：

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
```

验证：

```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker   # 能跑就说明 ROS 正常
```

---

## 3. Python 环境（venv，不要用 conda）

JetPack 的 CUDA / cuDNN / TensorRT 是系统级的，conda 会把它们盖掉，numpy 也容易出 ABI 问题。用 venv 并把系统包透传进来：

```bash
python3 -m venv --system-site-packages ~/venv/fp
source ~/venv/fp/bin/activate
pip install -U pip setuptools wheel
```

写进 `~/.bashrc`（**注意顺序：先 ROS 再 venv**）：

```bash
cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
source ~/venv/fp/bin/activate
export TORCH_CUDA_ARCH_LIST="8.7"
export RCIA_ROOT=$HOME/RCIA_Benmao_Vision
export FOUNDATIONPOSE_ROOT=$RCIA_ROOT/foundation_pose
EOF
```

> ROS 的 Python 包（`rclpy`、`cv_bridge`）通过 `PYTHONPATH` 注入，和 venv 的 Python 3.10 是同一版本，可以直接 import。这点和源机器上的 conda 方案原理相同。

---

## 4. 安装 PyTorch（必须用 NVIDIA 的 wheel）

**不要** `pip install torch`，PyPI 上的是 x86 或没用 Jetson 的 CUDA。

JetPack 6.2（CUDA 12.6）：

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

JetPack 6.1（CUDA 12.2）：把 `cu126` 改成 `cu122`。
如果该源不可用，退回到 NVIDIA 官方 redist：

```bash
# 到 https://developer.download.nvidia.com/compute/redist/jp/ 找对应版本的
# torch-*.whl 和 torchvision-*.whl，wget 下来后 pip install 本地文件
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 期望: 2.5.0 / 2.6.0, True, Orin
```

**版本选择**：源机器用的是 torch 2.5.1，Jetson 上没有完全对应的版本，2.5.0 最接近，优先选它。如果后面 `nvdiffrast`/`mycuda` 编译报 torch API 不兼容，就在 2.5.0 / 2.6.0 之间换一个再试。

---

## 5. 编译 librealsense + pyrealsense2

Jetson 上没有 pip 可用的 aarch64 轮子，只能源码编译。**必须开 RSUSB 后端**，否则内核 patch 会失败：

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.5        # 与源机器的 2.59.1 同代，稳定

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

# udev 规则（不加会权限不足）
sudo cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Python 绑定
pip install pybind11
cd ../wrappers/python
pip install .
```

验证：

```bash
python - <<'EOF'
import pyrealsense2 as rs, numpy as np
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p.start(c)
for _ in range(3):
    f = p.wait_for_frames(); print(np.asanyarray(f.get_color_frame().get_data()).shape)
p.stop()
EOF
```

> D435i 必须插**蓝色 USB 3 口**。插 USB 2 会限制到 640×480@15，深度质量也会掉。

---

## 6. 安装其余 Python 依赖

```bash
pip install "numpy==1.26.4"          # 不要升 2.x，torch/opencv 都没完全适配
pip install opencv-python==4.11.0.86
pip install scipy trimesh pyyaml imageio
pip install ultralytics==8.4.98
pip install pybind11 ninja
pip install git+https://github.com/facebookresearch/segment-anything.git
```

如果 `import cv2` 报 GTK / 显示相关错误，补装：

```bash
sudo apt install -y libqt5gui5 python3-pyqt5
```

---

## 7. 编译 FoundationPose 的 CUDA 扩展

先拷代码（见第 8 节），然后：

```bash
source ~/venv/fp/bin/activate
export TORCH_CUDA_ARCH_LIST="8.7"          # Orin = sm_87，写错会白编半小时
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd $RCIA_ROOT/foundation_pose

# (1) mycpp —— pybind11 C++ 扩展
cd mycpp && rm -rf build && mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python) \
         -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cd $RCIA_ROOT/foundation_pose

# (2) nvdiffrast —— 光栅化器
cd nvdiffrast && rm -rf build *.egg-info && pip install --no-build-isolation -e .
cd $RCIA_ROOT/foundation_pose

# (3) bundlesdf/mycuda —— CUDA 扩展
cd bundlesdf/mycuda && rm -rf build *.egg-info *.so && pip install -e .
cd $RCIA_ROOT/foundation_pose
```

验证：

```bash
cd $RCIA_ROOT && python - <<'EOF'
import os, sys
root = os.environ["RCIA_ROOT"]
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "foundation_pose"))
import torch
import mycpp.build.mycpp as mycpp      # 注意是 mycpp.build.mycpp
import nvdiffrast.torch as dr
from estimater import FoundationPose
print("mycpp / nvdiffrast / estimater OK, cuda:", torch.cuda.is_available())
EOF
```

> 编译约 20–40 分钟，`mycuda` 最慢。中途不要切功耗模式。

---

## 8. 传输代码与权重

在**源机器**上执行：

```bash
# 1) ROS 包
rsync -avP --exclude build --exclude install --exclude log \
  /home/cheng/learn_ws/src/learn_camera_d435/ \
  cheng@<板子IP>:~/learn_ws/src/learn_camera_d435/

# 2) FoundationPose 源码 + 权重（跳过编译产物和用不到的目录）
rsync -avP \
  --exclude 'build' --exclude '*.so' --exclude '*.egg-info' \
  --exclude '__pycache__' --exclude 'yolo_venv' \
  --exclude 'webots_sim' --exclude 'datasets' --exclude 'install' --exclude 'log' \
  --exclude 'OpenNI_*.zip*' --exclude 'runs' \
  /home/cheng/test/RCIA_Benmao_Vision/ \
  cheng@<板子IP>:~/RCIA_Benmao_Vision/

# 3) 模型文件
ssh cheng@<板子IP> 'mkdir -p ~/models/part4'
scp /home/cheng/datasets/part4_yolo/runs/part4_v2_132/weights/best.pt \
    cheng@<板子IP>:~/models/part4/
```

注意：`--exclude '*.so'` 要保留，x86 的扩展带过去只会导致 `ImportError: invalid ELF header`。

板子上检查一下到底有什么：

```bash
ls $RCIA_ROOT/foundation_pose/mycpp/*.so $RCIA_ROOT/foundation_pose/nvdiffrast/*.so 2>/dev/null
# 应该是空的，有就删掉
```

---

## 9. 编译 ROS 包

```bash
source /opt/ros/humble/setup.bash
cd ~/learn_ws
colcon build --packages-select learn_camera_d435 --symlink-install
source install/setup.bash
```

> 如果 `~/learn_ws/src` 里还有别的包（比如源机器上的 `franka-color-sorting-robot`），只 build 这一个就行。

---

## 10. 改路径

`config/params.yaml` 里有三条绝对路径是源机器的，必须改：

```yaml
yolo_detector:
  ros__parameters:
    model_path: /home/cheng/models/part4/best.pt          # ← 改

foundationpose:
  ros__parameters:
    mesh_path: /home/cheng/RCIA_Benmao_Vision/models/object/part4_cylinder_base.obj   # ← 改
    sam_checkpoint: /home/cheng/RCIA_Benmao_Vision/models/sam/sam_vit_b_01ec64.pth    # ← 改
```

> 用户目录若不同（比如 `jetson` 而不是 `cheng`），一律替换。
> `RCIA_ROOT` / `FOUNDATIONPOSE_ROOT` 已经在 `~/.bashrc` 里设好，`foundationpose_node.py` 会自动读取，不用改代码。

---

## 11. 分步验证（不要一上来就跑整套）

**第 1 步：只有相机**

```bash
ros2 run learn_camera_d435 d435_camera_node.py
```

另一个终端：

```bash
ros2 topic hz /camera/color/image_raw      # 应接近 30
ros2 topic hz /camera/depth/image_raw
```

**第 2 步：相机 + YOLO**

```bash
ros2 run learn_camera_d435 yolo_detect_node.py
ros2 topic echo /yolo/detections --once
```

拿实物对着镜头，应该能看到框。

**第 3 步：只跑 FoundationPose（订阅上一步的话题）**

```bash
ros2 run learn_camera_d435 foundationpose_node.py
```

首次会打印 `[fp] 加载 SAM`，注册一次约 3–15 秒（Orin 比桌面慢）。看状态：

```bash
ros2 topic echo /foundationpose/status
# searching → verifying #n → locked #n
```

**第 4 步：整套**

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py
```

无显示器时：

```bash
ros2 launch learn_camera_d435 foundationpose.launch.py use_viewer:=false
```

---

## 12. 上板调优（按这个顺序）

1. **YOLO 换 TensorRT**——engine 必须在板子上导，x86 导的不能用：

   ```bash
   cd ~/models/part4
   yolo export model=best.pt format=engine half=True imgsz=640 device=0
   ```

   然后 `model_path` 改成 `~/models/part4/best.engine`。
   再把 `yolo_detector.process_rate` 从 30 降到 10~15，把 GPU 让给 FoundationPose。

2. **`sam_keep_loaded: true`**——Orin 是统一内存，不用每次注册都重载 SAM，单次注册省 1–2 秒。桌面 6 GB 卡才需要 `false`。

3. **降低迭代次数**：`est_refine_iter: 3`、`track_refine_iter: 1`、`crop_ratio: 1.1`。

4. **`debug_rate: 10`**，不调试时 `publish_debug_image: false`。

5. **锁频 + MAXN**（第 1 节已做），用 `tegrastats` 确认温度没到降频线。

6. **无头运行**：`sudo systemctl set-default multi-user.target`，能省 1–2 GB 内存。

---

## 13. 已知坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `ImportError: invalid ELF header` 或 `wrong ELF class` | 拷了 x86 的 `.so` | 删掉重编（第 7 节） |
| `No module named 'mycpp'` | C++ 扩展没编或没装 | 检查 `mycpp/build/mycpp.cpython-310-aarch64-linux-gnu.so` 是否存在 |
| `torch.cuda.is_available()` 为 False | 装了 PyPI 的 torch | 卸掉，换成 NVIDIA wheel |
| nvdiffrast 编译报 `unsupported gpu architecture 'compute_XX'` | `TORCH_CUDA_ARCH_LIST` 不对 | 设成 `8.7` |
| `cv2.imshow` 报 Qt 错误 | 无显示器 / 缺 GTK | `use_viewer:=false`，远程用 `rqt_image_view` |
| 注册一次要十几秒 | 未锁频 / 迭代次数高 / 没换 TensorRT | 见第 12 节 |
| 显存/内存 OOM | 16 GB 是 CPU/GPU 共用 | 加 swap、关桌面、`sam_keep_loaded: false` 试试 |
| `rclpy` 导入失败 | venv 激活顺序错 | 先 `source /opt/ros/humble/setup.bash`，再 activate venv |
| `numpy` 相关 ABI 报错 | 装了 numpy 2.x | 固定 `numpy==1.26.4` |
| D435i 掉帧 / 只有 15 fps | 插在 USB 2 口 | 换蓝色 USB 3 口 + 原装线 |
| ROS 话题连不上 | `ROS_DOMAIN_ID` 不同 | 两台机器设成同一个值 |

---

## 14. 验收清单

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `python -c "import pyrealsense2, nvdiffrast, mycpp"` 全部成功
- [ ] `ros2 topic hz /camera/color/image_raw` → ~30
- [ ] `/yolo/detections` 有稳定输出
- [ ] `/foundationpose/status` 能到 `locked #n`
- [ ] `/foundationpose/object_pose` 有 30 Hz 左右的位姿
- [ ] `tegrastats` 温度 60–80 ℃，无降频
- [ ] 拔掉显示器也能跑（`use_viewer:=false`）

预计耗时：系统与 ROS 1–2 h，librealsense 30 min，FoundationPose 编译 20–40 min，调试 1–2 h。半天到一天。
