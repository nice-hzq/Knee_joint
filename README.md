# 膝关节角度实时测量 (Knee Joint Angle)

基于双 IMU 的实时关节角度测量模块。两个 WT901 传感器分别固定在大腿（近端）和小腿上（远端），
通过 ESP32 汇聚后经串口上传到上位机；上位机以 Mahony 与四元数 EKF 并行解算两传感器的
**相对姿态**，同时输出两套屈伸 / 外展 / 旋转三轴角度。

本目录是一个**自包含的独立项目**，不依赖也不修改任何其他工程。

姿态坐标系、四元数方向、Mahony/EKF 数学模型、参数定义、欧拉角顺序和论文引用说明见：
[双 IMU 姿态解算、坐标系与关节角定义](docs/orientation_and_coordinate_systems.md)。

---

## 功能

- **实时串口采集**：两个 IMU 的帧按 `esp32_rx_ms` 一对一近邻配对（容差 30 ms），输出实时角度
- **离线 CSV 回放**：把录制的原始数据当数据源重放，用于复现与回归验证
- **动态标定**：站立静止 3 秒，取平均相对姿态作为零点（`q_rel_0`），之后的角度都是相对零点
- **双姿态解算**：同一组 Acc + Gyr 数据并行运行 Mahony 与 quaternion EKF
- **实时曲线界面**：Tkinter 三面板曲线（Mahony 实线 / EKF 虚线）+ 双套三轴数值与 ROM
- **自动落盘**：角度序列 CSV + 会话元数据 JSON，原始同步帧可选保存

---

## 安装

```bash
pip install -r requirements.txt
```

依赖仅三项：`numpy`、`AHRS`（Mahony + quaternion EKF）、`pyserial`。
曲线用 Tkinter Canvas 自绘，**不需要** matplotlib / pandas / scipy。

---

## 运行

### 图形界面

```bash
python run.py
```

左侧选串口与波特率 → 点「配置设备映射」把两个传感器的 `WT****` 编号分别绑定到大腿 / 小腿别名
→ 「▶ 开始测量」→ 保持静止站立后点「开始校准」→ 开始活动，右侧实时出曲线。

### 命令行

```bash
# 实时串口
python run_cli.py --port COM8 --joint right_knee

# 离线回放（全速跑完，用已有标定文件复现）
python run_cli.py --replay samples/joint_angle_20260716_093022 \
                  --joint right_knee \
                  --calib-file temp/right_knee_pose_calib.json

# 离线回放（按真实时间 1:1 播放，便于观察曲线）
python run_cli.py --replay data/raw/joint_angle_20260910_101500 --speed 1
```

`--replay` 既可以传单个 `<device_id>.csv`，也可以传一个目录（目录内每个 `<device_id>.csv` 视为一路数据流）。

参数优先级：**命令行 > config.json > 内置默认值**。`python run_cli.py --help` 查看全部参数。

---

## 配置 (`config.json`)

| 键 | 说明 | 默认 |
|---|---|---|
| `joint_key` | 默认关节 | `right_knee` |
| `fs` | 采样率 Hz，决定标定所需帧数 | `100.0` |
| `serial.port` / `serial.baud` | 串口与波特率 | `COM8` / `921600` |
| `device_map_file` | 设备映射文件 | `temp/device_alias_map.json` |
| `device_map` | 内联映射（优先于文件） | `{}` |
| `output_root` | 输出根目录 | `data` |
| `calib_dir` | 标定与设备映射目录 | `temp` |
| `calib.mode` / `calib.dur_s` | 标定模式与窗口时长 | `lower_body_standing` / `3.0` |
| `sync.max_dt_ms` | 双 IMU 配对容差 | `30.0` |
| `kalman.var_gyr` | EKF 陀螺过程噪声方差 | `0.09` |
| `kalman.var_acc` | EKF 加速度测量噪声方差 | `0.25` |
| `save_raw` | 是否保存原始同步帧 | `true` |

相对路径一律按项目根目录解析。

---

## 支持的关节

| key | 近端 | 远端 |
|---|---|---|
| `left_knee` / `right_knee` | L4 / R4（大腿） | L5 / R5（小腿） |
| `left_ankle` / `right_ankle` | L5 / R5（小腿） | L6 / R6（脚掌） |
| `left_hip` / `right_hip` | S1（骨盆） | L4 / R4（大腿） |

**角度极性**：`flexion (+)` 屈曲 / `(-)` 伸展；`abduction (+)` 外展 / `(-)` 内收；
`rotation (+)` 外旋 / `(-)` 内旋。

**安装方向约定**：两个传感器的 X 轴应分别沿各自骨段指向远端关节（大腿指向膝、小腿指向踝），
理想情况下坐标系近似平行，标定时 `q_rel_0` 的 Roll ≈ 0°。

---

## 标定流程

1. 受试者保持静止站立，膝关节处于自然的 0° 位
2. 确保数据已积累 ≥ `calib.dur_s × fs` 帧（3 秒 @ 100 Hz = 300 帧）
3. 点击「开始校准」（或 CLI 的 `--calib-at`）
4. Mahony 与 EKF 分别取该窗口内两路数据的**平均四元数**，计算各自的
   `q_rel_0 = inv(q_prox_avg) · q_dist_avg`
5. 静态判定用陀螺仪**标准差**（< 10 dps）与加速度**标准差**（< 0.15 g）——
   用标准差而非中位数，是因为 WT901 存在恒定零偏，中位数会被零偏抬高而误判为运动

标定结果写入 `temp/<joint>_pose_calib.json`。`q_rel_0` 保持为 Mahony 零位，
`q_rel_0_kalman` 保存 EKF 零位；旧文件缺少后者时会自动复用 `q_rel_0`。

---

## 输出

```
data/
├─ raw/joint_angle_<YYYYMMDD_HHMMSS>/<device_id>.csv   原始 16 列，仅写实际参与配对的帧
└─ joint_angle_<YYYYMMDD_HHMMSS>/
   ├─ joint_angle_data.csv    time_s + Mahony 三轴角 + kalman_* 三轴角
   └─ session_info.json       会话元数据（ROM、同步统计、数据源等）
```

原始 CSV 的 16 列：

```
device_id, sensor_timestamp, sensor_ms, esp32_rx_ms,
Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z,
Angle_x, Angle_y, Angle_z, Geo_x, Geo_y, Geo_z
```

串口上传的每行以 `[DATA]` 开头，后接上述 16 列。
其中设备自带的 `Angle_*` 不参与这两套上位机姿态解算。

---

## 项目结构

```
Knee_joint/
├─ run.py                 图形界面入口
├─ run_cli.py             命令行入口（实时 / 回放）
├─ config.json            默认配置
├─ knee_joint/
│  ├─ quaternion.py       四元数运算
│  ├─ mahony.py           Mahony AHRS 姿态解算
│  ├─ kalman.py           Quaternion EKF 姿态解算
│  ├─ models.py           关节定义 / 标定 / 角度状态
│  ├─ calibration.py      标定算法
│  ├─ engine.py           关节角引擎（相对姿态 → 三轴角度）
│  ├─ protocol.py         16 列 CSV / [DATA] 解析、u32 时间戳工具
│  ├─ sync.py             双 IMU 近邻配对 + 会话时钟
│  ├─ sources.py          SerialSource / CsvReplaySource
│  ├─ recorder.py         会话落盘
│  ├─ worker.py           采集线程（队列消息契约）
│  └─ ui/                 Tkinter 界面
├─ temp/                  设备映射 + 标定文件
├─ samples/               回归测试用的实拍数据
└─ tests/                 单元测试与端到端回归
```

---

## 测试

```bash
python -m unittest discover -s tests -v
```

- `test_quaternion.py` — 四元数运算
- `test_sync.py` — 双 IMU 配对语义、30 ms 容差边界、uint32 回绕、统计量
- `test_kalman.py` — EKF 单元四元数、双状态同步、旧标定兼容、双套 CSV
- `test_replay.py` — 用 `samples/` 里的实拍数据做端到端回放回归

---

## 说明

- 数据源只做了串口与离线回放两种。ESP32 固件另有 TCP :8888 服务端，本模块未实现对应客户端。
- 本项目不包含步态分析、报告生成等与关节角度无关的功能。
