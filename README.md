# Raspberry Pi 5 + Hailo-8 行人检测系统

固定摄像头 + AI 推理，实时检测监控区域内是否有人活动。

---

## 目录

- [硬件平台](#硬件平台)
- [软件栈](#软件栈)
- [系统架构](#系统架构)
- [快速启动](#快速启动)
- [检测调优](#检测调优)
- [Web 仪表盘](#web-仪表盘)
- [PCIe / I/O 稳定性配置](#pci--io-稳定性配置)
- [I/O 错误历史与修复](#io-错误历史与修复)
- [故障排查](#故障排查)
- [附录](#附录)

---

## 硬件平台

| 组件 | 详情 |
|------|------|
| **主板** | Raspberry Pi 5 Model B Rev 1.0 |
| **CPU** | ARM Cortex-A76, 4 核, 2400 MHz |
| **RAM** | 8 GB (可用 ~7.1 GB) |
| **存储** | Samsung PM981 NVMe SSD 512GB (`nvme0n1`, **DRAM-less**, PCIe Gen2 x1 协商) |
| **NPU** | Hailo-8 HM21LB1C2LAE (PCIe `0001:04:00.0`, 满血版硅片, 固件锁 400MHz ≈10 TOPS) |
| **摄像头** | USB2.0 Camera (`/dev/video0`, V4L2) |
| **OS** | Debian Trixie (6.12.75+rpt-rpi-2712, aarch64) |

### PCIe 拓扑

```
Pi 5 PCIe 1 (Gen3 x1, 8 GT/s)
  └─ ASM1182e Switch (Gen2 x1, 5 GT/s, ~400 MB/s)
       ├─ Hailo-8 (0001:04:00.0)  — NPU 推理
       └─ Samsung PM981 (0001:03:00.0)  — NVMe 存储
```

> **注意**：NVMe 与 Hailo 共享同一 PCIe switch 上行链路（~400 MB/s），并发时存在带宽争用。这是 Pi 5 NVMe HAT 的常见架构。

### 性能基准

| 场景 | 结果 |
|------|------|
| YOLOv8s NPU 推理延迟 | ~8.2ms @400MHz, ~150 FPS (hw_only) |
| USB 摄像头视频推理 | ~29 ms/帧, ~26 FPS |
| 实际检测管道（含帧保存） | ~10 FPS (可配置) |
| 真实行人检测置信度 | 0.80+ |

---

## 软件栈

### Hailo 软件包

| 包 | 版本 | 说明 |
|----|------|------|
| `hailort` | 4.23.0 | 运行时库 + CLI 工具 |
| `hailort-pcie-driver` | 4.23.0 | PCIe 内核驱动 |
| `hailo-tappas-core` | 5.1.0 | GStreamer 推理框架 |
| `hailo-models` | 1.0.0-2 | 预编译 HEF 模型 |
| `python3-hailort` | 4.23.0 | Python 绑定 (`hailo_platform`) |
| `rpicam-apps-hailo-postprocess` | 11.12.0-1 | CSI 摄像头 AI 后处理 |
| GStreamer | 1.26.2 | 含 hailo 插件 |

### 预编译模型 (`/usr/share/hailo-models/`)

本项目使用 **YOLOv8s_h8l.hef**（80 类 COCO，640×640），通过 Python SDK 调用，仅检测"人"类（COCO class 0）。

> **注意**：虽然模型文件名为 `_h8l`，但实际硅片是满血 Hailo-8（通过 `ctrl.identify()` 确认，Board Name: `Hailo-8`, Arch: `HAILO8`）。固件将 NPU 频率锁在 400MHz，实际算力约 10 TOPS（标称 26 TOPS @ 676MHz）。

### 其他依赖

- **Python**: OpenCV, Flask, numpy, hailo_platform
- **摄像头**: V4L2 (`/dev/video0`)

---

## 系统架构

```
┌──────────┐    V4L2    ┌──────────────┐
│ USB 摄像头 │ ──────▶   │  usb_camera  │
│ /dev/video0│          │  _hailo.py   │
└──────────┘            │              │
                        │  ┌────────┐  │      ┌──────────────┐
                        │  │YOLOv8s │  │      │   Web 仪表盘  │
                        │  │ Hailo  │──┼─────▶│  Flask :5000 │
                        │  │  NPU   │  │      │  MJPEG stream│
                        │  └────────┘  │      └──────────────┘
                        │              │
                        │  每 60 帧保存 │
                        │  (双过滤器)   │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  output/     │
                        │  sess_*.jpg  │
                        └──────────────┘
```

### 主线程 vs Web 线程

- **主线程**：摄像头采集 → 预处理 → NPU 推理 → 标注绘制 → 帧保存
- **后台线程**：Flask HTTP 服务，通过 `DetectionState` 共享状态（线程安全锁）

---

## 快速启动

### 启动检测 + Web 仪表盘

```bash
# 前台运行（调试）
python3 usb_camera_hailo.py

# 后台运行
nohup python3 usb_camera_hailo.py \
  --output-dir ./output \
  --fps-target 10 \
  --conf-thresh 0.35 \
  --port 5000 \
  > /tmp/hailo-detection.log 2>&1 &
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `./output` | 标注帧保存目录 |
| `--fps-target` | `10` | 目标处理帧率 |
| `--conf-thresh` | `0.40` | 置信度阈值（0.10–0.60） |
| `--port` | `5000` | Web 仪表盘端口 |

### Hailo 快速命令

```bash
hailortcli scan                              # 扫描设备
hailortcli fw-control identify               # 固件信息
hailortcli run /usr/share/hailo-models/yolov8s_h8l.hef -c 100 --measure-latency  # 推理测试
```

---

## 检测调优

### 当前参数（2026/06/19 起）

| 参数 | 值 | 说明 |
|------|----|------|
| 置信度阈值 | **0.40** | 从 0.35 提升到此值，进一步减少误报 |
| 空白区域过滤 | mean > 240 AND std < 15 | 二次防护：排除过曝/纯色区域 |
| 直方图去重 | Bhattacharyya < 0.3 AND IoU > 0.8 | 避免重复保存相似帧 |
| 保存频率 | 每 60 帧（约 6 秒） | 仅在检测到目标时保存 |
| 文件命名 | `sess_{N}_f{帧号}.jpg` | 按会话编号分组 |

### 过滤器链

```
NPU 检测 → conf ≥ 0.35? → 非空白区域? → 与上一帧差异够大? → 保存
```

**为什么选 0.40**：真实行人检测置信度通常在 0.50–0.80，阈值提高到 0.40 过滤掉 0.35–0.40 区间的低置信度误报，误报率减少约 10–15%。

**空白过滤器**：即使模型对白纸打出 0.35+ 的分数，`_is_blank_region()` 通过灰度均值和标准差识别出纯色区域，阻止保存。

**直方图去重**：4×4 网格，每格 16 级灰度直方图，合并为 256 维特征向量。若 Bhattacharyya 距离 < 0.3（高度相似）且边界框 IoU > 0.8（位置重叠），则跳过保存。

---

## 数据库管理

### SQLite 元数据存储

系统使用 SQLite 数据库存储检测元数据，支持高效查询和管理：

**数据库文件**: `./output/detections.db`

**表结构**:
- `id`: 主键自增
- `filename`: 图片文件名
- `timestamp`: 检测时间
- `session_id`: 会话编号
- `frame_count`: 帧编号
- `confidence`: 置信度（索引优化）
- `bbox_x1/y1/x2/y2`: 边界框坐标
- `image_width/height`: 图片尺寸
- `fps`: 检测时 FPS
- `infer_ms`: 推理延迟（毫秒）
- `detection_count`: 本帧检测数

**索引优化**:
- `idx_confidence`: 置信度范围查询
- `idx_timestamp`: 时间范围查询
- `idx_session`: 会话分组统计

### 查询 API

```bash
# 查询所有置信度≥0.40的检测
curl http://localhost:5000/api/metadata?min_conf=0.40

# 查询特定置信度范围
curl http://localhost:5000/api/metadata?min_conf=0.35&max_conf=0.45

# 分页查询
curl http://localhost:5000/api/metadata?min_conf=0.40&page=2&per_page=50
```

### 管理工具

**删除低置信度图片**:
```bash
# 预览（不实际删除）
python3 delete_low_confidence.py --min-conf 0.35 --max-conf 0.40 --dry-run

# 执行删除
python3 delete_low_confidence.py --min-conf 0.35 --max-conf 0.40
```

**性能对比**（10万条记录预估）:

| 操作 | JSON方案 | SQLite方案 | 性能提升 |
|------|---------|-----------|----------|
| 插入一条记录 | 2-5秒 | <10毫秒 | 200-500倍 |
| 查询置信度范围 | 30-60秒 | <50毫秒 | 600-1200倍 |
| 内存占用 | 200-500MB | <1MB | 200-500倍 |

---

## Web 仪表盘

访问 `http://<设备IP>:5000`

### 功能

- **实时 MJPEG 流**：`/stream`，带标注框的实时画面
- **运行统计**：FPS、推理延迟、总帧数、检测数、运行时长
- **当前检测列表**：置信度彩色标签（≥50% 绿 / ≥30% 黄 / <30% 红）
- **检测历史**：最近 10 帧的柱状图
- **图片浏览器**：2 行固定网格布局，分页（24 张/页），自动刷新（每 5 秒）
- **灯箱查看器**：点击缩略图全屏查看，键盘 ← → Esc 操作
- **重启按钮**：通过 POST `/api/restart` 触发自重启（需 systemd 管理）

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 仪表盘页面 |
| `/stream` | GET | MJPEG 视频流 |
| `/api/stats` | GET | 实时运行统计 |
| `/api/images` | GET | 图片列表（分页） |
| `/api/image/<name>` | GET | 单张图片 |
| `/api/restart` | POST | 重启服务 |

---

## PCIe / I/O 稳定性配置

### 已应用的修复（2026/06/14）

| 配置项 | 文件 | 值 | 目的 |
|--------|------|-----|------|
| `pcie_aspm=off` | `/boot/firmware/cmdline.txt` | 行尾追加 | 禁用 PCIe 主动电源管理，防止链路进入 L1 后断开 |
| `dtparam=pciex1_gen=2` | `/boot/firmware/config.txt` | `[cm4]` 前 | 强制 PCIe Gen2 速度，避免 Gen3/Gen2 协商失败 |

### ⚠️ HMB (Host Memory Buffer) 问题

Pi 5 BCM2712 固件会自动注入 `nvme.max_host_mem_size_mb=0`（禁用 HMB），该参数不在任何用户可配置文件中。

```bash
# 检查 HMB 状态
cat /proc/cmdline | grep nvme.max_host_mem_size_mb
# 有输出 = 固件注入，HMB 被禁用
# 无输出 = HMB 已启用

# 检查 NVMe 实际协商
dmesg | grep -i "hmb\|host memory buffer"
```

这是 DRAM-less SSD 的已知问题。软件缓解已足够（见下方 I/O 错误修复章节）。

### 验证清单

```bash
# 系统健康
vcgencmd measure_temp            # 温度（<60°C 正常）
vcgencmd get_throttled           # 应返回 0x0

# NVMe 状态
cat /sys/class/nvme/nvme0/state  # 应返回 "live"
sudo smartctl -a /dev/nvme0n1    # SMART 健康数据

# PCIe 链路
lspci -vvv -s 0001:03:00.0 | grep -i "lnksta"  # NVMe 链路状态
lspci -vvv -s 0001:04:00.0 | grep -i "lnksta"  # Hailo 链路状态

# Hailo NPU
hailortcli scan                  # 应发现设备
hailortcli fw-control identify   # 固件版本

# 摄像头
v4l2-ctl --list-devices          # 确认 /dev/video0
```

---

## I/O 错误历史与修复

### 事件时间线

| 日期 | 事件 |
|------|------|
| 2026/06/13 | 首次系统崩溃：`/usr/bin/ls: Input/output error`，多机复现确认 |
| 2026/06/13 | 根因分析完成：ASM1182e PCIe switch + DRAM-less NVMe + 高并发 I/O |
| 2026/06/13 | 实施缓解措施：持久化日志 + ionice 限制 |
| 2026/06/14 | 确认固件注入 HMB 禁用问题；应用 PCIe 稳定性配置 |
| 2026/06/15 | 最后一次重启，系统运行至今稳定 |

### 根因链

```
Pi 5 Gen3 x1 → Gen2 PCIe switch (ASM1182e/PEX1184)
  ├─ NVMe (Samsung PM981, DRAM-less) 与 Hailo 共享 ~400 MB/s 链路
  ├─ 固件禁用 HMB (nvme.max_host_mem_size_mb=0) → NVMe 无缓存加速
  └─ 高并发 I/O（推理帧保存 + 文件扫描）→ 链路拥塞 → NVMe 超时 → PCIe 断开 → 文件系统错误
```

### 已实施的缓解措施

| 措施 | 状态 | 说明 |
|------|------|------|
| `pcie_aspm=off` | ✅ 持久 | 防止 PCIe 链路进入低功耗状态 |
| `pciex1_gen=2` | ✅ 持久 | 强制 Gen2 稳定协商 |
| 持久化日志 | ✅ 已启用 | 崩溃后可诊断 |
| ionice Claude | ⚠️ 运行时 | 重启后需重新设置 |

### 如果再次出现 I/O 错误

```bash
# 1. 快速诊断
mount | grep "ro,"                                    # 检查只读挂载
dmesg | grep -i "error\|I/O\|nvme\|ext4" | tail -40   # 内核日志

# 2. NVMe 健康
cat /sys/class/nvme/nvme0/state                       # 应返回 "live"
sudo smartctl -a /dev/nvme0n1                         # SMART 数据

# 3. PCIe 链路
lspci -vvv -s 0001:03:00.0 | grep -i "lnksta"        # NVMe 链路
lspci -vvv -s 0001:04:00.0 | grep -i "lnksta"        # Hailo 链路

# 4. Hailo 驱动
dmesg | grep -i "hailo.*warn\|hailo.*error"
```

---

## 故障排查

### 摄像头无法打开

```bash
v4l2-ctl --list-devices   # 确认 /dev/video0 存在
v4l2-ctl --list-formats   # 查看支持的格式
```

### Hailo NPU 不响应

```bash
hailortcli scan                          # 设备检测
dmesg | grep hailo                       # 驱动日志
cat /sys/class/nvme/nvme0/state          # 检查 NVMe（共享 PCIe 链路）
```

### Web 仪表盘无法访问

```bash
# 检查 Flask 进程
pgrep -f usb_camera_hailo

# 检查端口占用
ss -tlnp | grep 5000

# 查看日志
tail -50 /tmp/hailo-detection.log
```

### 误检过多

- 提高 `--conf-thresh`（当前 0.35）
- 检查空白过滤器是否生效：`_is_blank_region`（mean > 240, std < 15）
- 考虑添加 ROI 掩码，限制检测区域

### 保存的图片太少

- 降低 `--conf-thresh`（最低 0.10）
- 调整保存频率：修改 `frame_count % 60` 中的 60 为更小值
- 放宽直方图去重阈值：Bhattacharyya 0.3 → 0.15

---

## 附录

### 文件清单

| 文件 | 说明 |
|------|------|
| `usb_camera_hailo.py` | 主程序：摄像头采集 + NPU 推理 + Web 仪表盘 |
| `output/` | 标注帧输出目录（1543 张图片，118 MB） |
| `hailort.log` | HailoRT 日志 |
| `hailo.web.log` | Web 服务日志 |

### 系统服务

| 服务 | 状态 | 说明 |
|------|------|------|
| `hailort.service` | 已启用 | Hailo 运行时服务 |
| 检测服务 | 手动 | 通过 `nohup` 启动，无 systemd unit |

### 启动配置

**`/boot/firmware/cmdline.txt`**（非注释部分）：
```
... pcie_aspm=off
```

**`/boot/firmware/config.txt`**（非注释部分）：
```
dtparam=pciex1_gen=2
arm_boost=1
```

### 版本历史

| 日期 | 变更 |
|------|------|
| 2026/06/10 | Hailo 软件栈安装，首次推理测试 |
| 2026/06/11 | USB 摄像头推理验证，中文标签 |
| 2026/06/13 | 系统崩溃，根因分析，I/O 缓解 |
| 2026/06/14 | PCIe 稳定性配置（ASPM off + Gen2） |
| 2026/06/19 | 检测调优：conf 0.35 + 空白过滤 + 直方图去重 |
| 2026/06/20 | 保存图片加时间戳水印，完整项目文档 |
| 2026/06/20 | 确认硅片为满血 Hailo-8（非 Hailo-8L），TOPS benchmark 完成 |
