# 能量单元 6D 检测：使用说明与注意事项

本文记录能量单元（吸取棍）检测方案的模型来源、贴图生成、相机与注册参数、
已知限制和迁移注意事项。**改动前先读完第 7 节"已知限制"。**

---

## 1. 方案概览

```
D435i 彩色+深度
   │
   ├─ 手动框选（pick_box.py）──→ /foundationpose/manual_rect
   │                                  │
   └─ SAM 分割 ──→ FoundationPose 注册/跟踪 ──→ /foundationpose/object_pose (TF: object)
```

- 检测端目前**没有可用的 YOLO 模型**，用 `manual_rect` 手动框选代替（见 6.1）
- 注册与跟踪由 FoundationPose 完成，输出物体位姿
- 物体尺寸：**100 × 109 × 117 mm**

---

## 2. 模型

### 2.1 来源

原始模型是三个 STL（同一装配坐标系，单位 mm）：

```
能量单元装 - 能量单元吸棍-1.STL        2372 面  100×109×113
能量单元装 - 能量单元吸棍中心柱-1.STL   188 面   22×22×84
能量单元装 - 能量单元吸棍底板-2.STL     620 面  100×109×10
```

原始文件在微信接收目录（`~/文档/xwechat_files/.../2026-11/`），**该目录会随聊天记录清理，
不要依赖它**。以仓库里的 `models/energy_unit.obj` 为准。

### 2.2 合并与归一化

```
三个 STL
  │ 合并 → 3180 面
  │ XY 居中、Z 底面对齐到 0
  │ ×0.001（mm → m）
  ▼
models/energy_unit.obj        3180 面，bounds [-0.05,-0.0545,0] ~ [0.05,0.0545,0.117]
```

**单位必须是米**。FoundationPose 不做尺度估计，单位错了位姿直接错。

### 2.3 带贴图版本

```
models/energy_unit_textured.obj   3232 面（补了 52 个灯窗面）
models/material.mtl
models/part4_mark.png
```

三个文件必须同目录。OBJ 本身不存颜色。

---

## 3. 灯窗贴图

### 3.1 图案来源

图案**不是按规则图纸估的**，而是从 CAD 模型的实际灯窗开孔提取的。

关键事实：

- 灯窗在 CAD 里是**通孔/深槽**，顶面（z=17 mm）在窗口处没有几何
- 直接给顶面贴图，红色会"涂在洞里"，渲染出来只剩轮廓线
- 所以流程是：**先提取开孔 → 补成平面 → 再上色**

### 3.2 实测尺寸

| 项目 | 实测 | 图纸 |
|---|---|---|
| MARK 整体 | 79.8 × 79.5 mm | 80 × 80 |
| 灯带宽度 | 8.2 mm | 8 |
| L 臂长 | 30.8 mm | 30 |
| 小方块 | 8.2 × 8.2 mm | 8 |

偏差 <1 mm，来自 0.25 mm/px 的栅格化，不是建模误差。

布局：**左上、左下、右下 三个 L + 右上角一组方块**（3 个小块）。
MARK 在 y 方向不居中（中心偏 +4.75 mm），这是 CAD 原样，不是错误。

### 3.3 重新生成

```bash
cd /home/cheng/learn_ws
/home/cheng/miniconda3/envs/foundationpose_ros/bin/python \
  src/learn_camera_d435/tools/make_mark_from_cad.py \
  --mesh src/learn_camera_d435/models/energy_unit.obj \
  --out  src/learn_camera_d435/models/energy_unit_textured.obj
```

脚本做的事：射线/栅格提取顶面 → 找开孔 → earcut 补面 → 平面投影 UV → 生成 2048² 贴图。

### 3.4 检查贴图是否生效

```bash
/home/cheng/miniconda3/envs/foundationpose_ros/bin/python \
  src/learn_camera_d435/tools/render_mesh.py \
  --mesh src/learn_camera_d435/models/energy_unit_textured.obj \
  --out /tmp/mark_view
```

输出 `_top.png`（顶视）、`_cam.png`（标定相机视角）、`_iso.png`。
顶视图应看到三个红色 L + 右上方块，位置与 CAD 开孔重合。

---

## 4. 相机设置

`config/params.yaml` → `d435_camera`：

| 参数 | 值 | 原因 |
|---|---|---|
| `depth_filter` | `true` | 黑色表面深度空洞，空间滤波填洞 |
| `depth_temporal` | `false` | 时间滤波会拖影 |
| `laser_power` | `360` | D435i 上限，黑色物体必须给满 |
| `emitter_enabled` | `true` | 关了就没有深度 |
| `auto_exposure` | `false` | 灯条会把自动曝光带偏 |
| `exposure` | `150` (15 ms) | 超过 166（约 1/60 s）运动就糊 |
| `gain` | `120` | 把黑色本体抬到灰度 40–90 |
| `power_line_frequency` | `1` | 国内 50 Hz，减少横纹 |

`depth_filter` 只在启动时读取，运行中改不生效。

### 距离

- **0.4 m**：深度有效率 66.7%，物体占画面 7.2%
- **0.78 m**：深度有效率 83.9%，物体占画面 2.3%
- 推荐 **0.55–0.65 m**，兼顾深度覆盖和物体像素数

用 `tools/depth_check.py` 实测当前画面：

```bash
python3 src/learn_camera_d435/tools/depth_check.py
```

---

## 5. 注册参数与 vis_dir

### 5.1 当前值

```yaml
rot_filter_deg: 90.0
vis_dir: [-0.389, -0.195, 0.900]
```

朝向过滤把初始假设从 252 降到 126，注册时间和显存减半。

### 5.2 vis_dir 的含义

在**物体坐标系**里，"物体指向相机"的方向。用于砍掉"相机在物体背面"的假设。

**它是针对固定安装姿态标定的**。物体坐标系跟着物体转，所以：

| 情况 | 是否要重标 |
|---|---|
| 相机挪位置/角度 | ✅ |
| 物体默认摆放方向变了 | ✅ |
| 换模型（坐标系变了） | ✅ |
| 物体被拿起来随意翻转 | ❌ 关掉过滤（`rot_filter_deg: 0`） |
| 只是平移 | ❌ |

### 5.3 重标方法

注册成功一次后：

```bash
source /opt/ros/humble/setup.bash
/home/cheng/miniconda3/envs/foundationpose_ros/bin/python - <<'EOF'
import numpy as np, rclpy, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped

class G(Node):
    def __init__(self):
        super().__init__('vd'); self.ds=[]
        self.create_subscription(PoseStamped,'/foundationpose/object_pose',self.cb,qos_profile_sensor_data)
    def cb(self,m):
        p,q=m.pose.position,m.pose.orientation
        x,y,z,w=q.x,q.y,q.z,q.w
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        d=-R.T@np.array([p.x,p.y,p.z]); self.ds.append(d/np.linalg.norm(d))

rclpy.init(); n=G(); t=time.time()
while rclpy.ok() and len(n.ds)<30 and time.time()-t<12:
    rclpy.spin_once(n, timeout_sec=0.1)
a=np.array(n.ds); d=a.mean(0); d/=np.linalg.norm(d)
print("vis_dir = [%.3f, %.3f, %.3f]" % tuple(d))
print("样本偏差 最大 %.2f°" % np.degrees(np.arccos(np.clip(a@d,-1,1))).max())
n.destroy_node(); rclpy.shutdown()
EOF
```

把结果写进 `params.yaml`，`rot_filter_deg` 保持 90。

---

## 6. 手动框选（当前检测入口）

### 6.1 为什么用手动框

现有 YOLO 模型（`part4_v2_132`）是在**白色零件合成图**上训练的，
在黑色实拍上 mAP50 只有 0.035，基本检测不到。所以暂用 `manual_rect`。

### 6.2 用法

```bash
python3 src/learn_camera_d435/tools/pick_box.py
```

窗口里左键拖框圈住物体，松手后脚本自动：

1. 调用 `/foundationpose/set_parameters` 设置 `manual_rect`
2. 调用 `/foundationpose/reacquire` 触发注册

终端会打印 `manual_rect 设置成功` / `已触发重新注册`。

按 `r` 清空 `manual_rect`（恢复 YOLO），`q` 退出。

### 6.3 参数

```yaml
manual_rect: ""        # 空 = 用 YOLO；"x,y,w,h" 像素坐标 = 手动
```

- 坐标是彩色图像原始分辨率（640×480），左上角原点
- 手动模式下自动跳过 YOLO 跟踪校验（没有真检测器，固定框会误判跟丢）
- 换框时自动清除 `bad_rect` 冷却，避免新框被跳过 10 s

### 6.4 框怎么画

- **宁松勿紧**：SAM 用框做提示，切掉边缘会分割不全
- 框住整个物体（底座 + 柱体）
- 尽量别带进大片背景

---

## 7. 已知限制

### 7.1 YOLO 不可用

需要用**实拍**重新训练。合成数据训的模型在这个物体上无效。
数据集建议：开灯状态为主（这才是最终形态），关灯状态各 20%。

### 7.2 深度受黑色表面影响

- 有效深度 66–84%，取决于距离和角度
- 有约 1/3 表面没有深度约束，跟踪时位姿可能抖动或绕轴跳变
- 补光只改善 RGB，不改善深度；深度靠角度和距离

### 7.3 显存 OOM

252 个假设一次性进 scorer 会撑爆 6 GB 卡。已给 FoundationPose 打补丁：

```python
# foundation_pose/learning/training/predict_score.py
bs = min(32, pose_data.rgbAs.shape[0])   # 原为全部
bs = 64                                  # make_crop_data_batch 里原为 512
```

**这个补丁不在本仓库里**，换机器/重装 FoundationPose 后需要重新打。
备份文件：同目录 `predict_score.py.bak`。

### 7.4 补面改变了局部几何

灯窗补成**平齐的平面**（z=17 mm），实物可能有 1–2 mm 台阶。
外形轮廓不变，不影响 6D 位姿；如果发现精度受影响，可把补面下移 1 mm 做浅槽。

### 7.5 贴图依赖三件套

`energy_unit_textured.obj` + `material.mtl` + `part4_mark.png` 必须同目录。
拷模型时漏了 PNG 会变成纯灰几何。

---

## 8. 迁移到小电脑（Jetson）

1. `git pull` 拿到模型和工具
2. **改 `params.yaml` 里的绝对路径**：`mesh_path`、`sam_checkpoint` 等
   目前写的是 `/home/cheng/...`，小电脑上是 `/home/jetson/...`
3. 重新打 FoundationPose 的 scorer 分批补丁（见 7.3）
4. `vis_dir` 需要重新标定（坐标系相同，但相机安装姿态可能不同）
5. 首次启动检查日志：
   ```
   加载 mesh: .../energy_unit_textured.obj
   mesh: 9604 顶点, 3232 面, 尺寸 [10.0 10.9 11.7] cm
   朝向过滤 90°: 252 -> 126 个初始假设
   FoundationPose 就绪
   ```
   **不应出现** `mesh 没有贴图/UV` 警告。

---

## 9. 排查清单

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动报找不到 mesh | `mesh_path` 路径错 | 改成实际路径 |
| 日志出现 `mesh 没有贴图/UV` | 三件套不全 | 检查 MTL/PNG 是否同目录 |
| 注册 OOM | 补丁未打 / 其他进程占显存 | 打 7.3 补丁，关显示节点 |
| 一直 `searching` | `manual_rect` 没设上 | 看 pick_box 终端是否打印成功 |
| 锁上但位姿绕轴乱翻 | 深度不足 / vis_dir 错 | 退远到 0.6 m；重标 vis_dir |
| 位姿持续漂移 | 深度空洞 | 提高激光功率（已满）、换角度 |
| 按 `r` 后检测不到 | 手动框已清空，YOLO 又无效 | 重新拖框 |
| Blender 里看不到红色 | 实体模式不显示贴图 | `Z` → 材质预览 |
| 红色 L 偏移/翻转 | 用了旧的贴图生成脚本 | 用 `make_mark_from_cad.py` 重新生成 |

---

## 10. 文件清单

```
src/learn_camera_d435/
├── models/
│   ├── energy_unit.obj              合并后的无贴图模型
│   ├── energy_unit_textured.obj     带 MARK 贴图（配置指向它）
│   ├── material.mtl
│   └── part4_mark.png               2048²，黑底红色灯窗
├── tools/
│   ├── pick_box.py                  手动框选（设置 manual_rect + reacquire）
│   ├── depth_check.py               统计投影区域内的深度有效率
│   ├── render_mesh.py               用 FP 渲染器出图检查贴图
│   ├── make_mark_from_cad.py        从 CAD 灯窗生成贴图（主用）
│   ├── make_texture_from_mask.py    用现成蒙版生成贴图
│   └── make_mark_texture.py         按图纸尺寸画图案（已被 CAD 方案取代）
├── config/params.yaml               全部参数（含 mesh_path、vis_dir）
└── docs/
    ├── ENERGY_UNIT_DETECTION.md     本文
    ├── DEPLOY_JETSON.md             Jetson 部署流程
    └── TASK_FOR_JETSON_AI.md        给小电脑 AI 的任务书
```
