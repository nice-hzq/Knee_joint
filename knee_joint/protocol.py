# -*- coding: utf-8 -*-
"""
ESP32 [DATA] 帧协议解析
Frame protocol — 16-column CSV row parsing shared by the serial and replay sources.

ESP32 固件每帧输出一行 CSV（串口带 [DATA] 前缀，TCP 无前缀）:
  device_id, sensor_timestamp, sensor_ms, esp32_rx_ms,
  Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z,
  Angle_x, Angle_y, Angle_z, Geo_x, Geo_y, Geo_z

关节角度只用 acc / gyr / mag 九轴：
  imu9 = [Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z, Geo_x, Geo_y, Geo_z]
传感器自带的 Angle_* 列（WT901 内部解算角）在本路径中**刻意不使用**。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ============================================================
# 常量
# ============================================================
CSV_HEADER = [
    "device_id", "sensor_timestamp", "sensor_ms", "esp32_rx_ms",
    "Acc_x", "Acc_y", "Acc_z",
    "Gyr_x", "Gyr_y", "Gyr_z",
    "Angle_x", "Angle_y", "Angle_z",
    "Geo_x", "Geo_y", "Geo_z",
]

DATA_PREFIX = "[DATA]"
_MIN_COLUMNS = 16

# 加速度模长合理性区间（单位 g）
# 静止约 1.0 g；运动 + 冲击上限约 3.5 g。超出视为异常帧丢弃。
ACC_NORM_MIN_G = 0.3
ACC_NORM_MAX_G = 3.5

# imu9 三个分量在 CSV 列中的切片
_ACC_SLICE = slice(4, 7)
_GYR_SLICE = slice(7, 10)
_MAG_SLICE = slice(13, 16)


# ============================================================
# 时间戳工具（ESP32 millis() 是 32 位，约 49.7 天回绕一次）
# ============================================================
def signed_u32_delta_ms(lhs, rhs) -> int:
    """返回两个 ESP32 millis() 时间戳的有符号差，正确处理 32 位回绕。"""
    delta = (int(lhs) - int(rhs)) & 0xFFFFFFFF
    return delta - 0x100000000 if delta > 0x7FFFFFFF else delta


def abs_u32_delta_ms(lhs, rhs) -> int:
    return abs(signed_u32_delta_ms(lhs, rhs))


def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return safe.strip("_") or "unknown_device"


# ============================================================
# 帧
# ============================================================
@dataclass
class Frame:
    """一帧同步就绪的 IMU 数据。"""
    device_id: str
    esp32_rx_ms: int              # ESP32 millis()，跨传感器同步的时间基准
    imu9: np.ndarray              # (9,) = [acc(3), gyr(3), mag(3)]
    raw_row: Optional[List[str]] = None   # 原始 16 列，用于落盘

    @property
    def acc(self) -> np.ndarray:
        return self.imu9[0:3]

    @property
    def gyr(self) -> np.ndarray:
        return self.imu9[3:6]

    @property
    def mag(self) -> np.ndarray:
        return self.imu9[6:9]


# ============================================================
# 解析
# ============================================================
def _build_frame(device_id: str, cols: List[str]) -> Optional[Frame]:
    """从已切分的列中构造 Frame；任何异常一律返回 None（丢弃该帧）。"""
    try:
        esp32_rx_ms = int(cols[3])
        acc = np.array([float(x) for x in cols[_ACC_SLICE]], dtype=float)
        gyr = np.array([float(x) for x in cols[_GYR_SLICE]], dtype=float)
        mag = np.array([float(x) for x in cols[_MAG_SLICE]], dtype=float)
    except (ValueError, IndexError):
        return None

    imu9 = np.concatenate([acc, gyr, mag])   # (9,)

    if np.any(np.isnan(imu9)) or np.any(np.isinf(imu9)):
        return None

    acc_norm = float(np.linalg.norm(acc))
    if acc_norm < ACC_NORM_MIN_G or acc_norm > ACC_NORM_MAX_G:
        return None

    return Frame(device_id=device_id, esp32_rx_ms=esp32_rx_ms,
                 imu9=imu9, raw_row=list(cols[:_MIN_COLUMNS]))


def parse_data_line(line: str) -> Optional[Frame]:
    """
    解析一行串口输出。line 可带或不带 [DATA] 前缀；不匹配则返回 None。
    """
    line = line.strip()
    if line.startswith(DATA_PREFIX):
        line = line[len(DATA_PREFIX):]
    if not line:
        return None
    return parse_csv_row(line.split(","))


def parse_csv_row(cols: List[str]) -> Optional[Frame]:
    """解析已切分的一行 CSV 列。列数不足或数值非法时返回 None。"""
    if len(cols) < _MIN_COLUMNS:
        return None
    device_id = cols[0].strip()
    if not device_id:
        return None
    return _build_frame(device_id, cols)


def is_header_row(cols: List[str]) -> bool:
    """判断是否为 CSV 表头行（离线回放时跳过）。"""
    return bool(cols) and cols[0].strip() == CSV_HEADER[0]
