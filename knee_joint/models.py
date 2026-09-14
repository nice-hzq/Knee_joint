# -*- coding: utf-8 -*-
"""
关节角度数据模型
Joint angle data models — JointBinding, JointCalibration, JointAngleState.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np


# ============================================================
# 预设关节定义
# ============================================================
#
# 角度极性约定:
#   flexion (+)  = 屈膝 (knee bending)   |  flexion (-) = 伸膝
#   abduction (+) = 外展 (away from body)  |  abduction (-) = 内收
#   rotation (+)  = 外旋 (external)        |  rotation (-)  = 内旋 (internal)
#
# flexion_axis = 0 → roll = 屈伸主轴（绕传感器 X 轴旋转）
#
JOINT_DEFS = {
    "left_knee": {
        "name": "左膝",
        "name_en": "Left Knee",
        "proximal_default": "L4",   # 大腿
        "distal_default": "L5",     # 小腿
        "prox_label": "大腿 (Thigh)",
        "dist_label": "小腿 (Shank)",
        "flexion_axis": 0,          # roll = 屈伸
    },
    "right_knee": {
        "name": "右膝",
        "name_en": "Right Knee",
        "proximal_default": "R4",
        "distal_default": "R5",
        "prox_label": "大腿 (Thigh)",
        "dist_label": "小腿 (Shank)",
        "flexion_axis": 0,
    },
    "left_ankle": {
        "name": "左踝",
        "name_en": "Left Ankle",
        "proximal_default": "L5",
        "distal_default": "L6",
        "prox_label": "小腿 (Shank)",
        "dist_label": "脚掌 (Foot)",
        "flexion_axis": 0,
    },
    "right_ankle": {
        "name": "右踝",
        "name_en": "Right Ankle",
        "proximal_default": "R5",
        "distal_default": "R6",
        "prox_label": "小腿 (Shank)",
        "dist_label": "脚掌 (Foot)",
        "flexion_axis": 0,
    },
    "left_hip": {
        "name": "左髋",
        "name_en": "Left Hip",
        "proximal_default": "S1",
        "distal_default": "L4",
        "prox_label": "骨盆 (Pelvis)",
        "dist_label": "大腿 (Thigh)",
        "flexion_axis": 0,
    },
    "right_hip": {
        "name": "右髋",
        "name_en": "Right Hip",
        "proximal_default": "S1",
        "distal_default": "R4",
        "prox_label": "骨盆 (Pelvis)",
        "dist_label": "大腿 (Thigh)",
        "flexion_axis": 0,
    },
}


# ============================================================
# Data Classes
# ============================================================
@dataclass
class JointBinding:
    """关节—传感器绑定"""
    joint_name: str            # e.g. "left_knee"
    proximal_sensor: str       # e.g. "L4"
    distal_sensor: str         # e.g. "L5"
    flexion_axis: int = 0      # 屈伸轴在欧拉角中的索引 (0=roll, 1=pitch, 2=yaw)


@dataclass
class JointCalibration:
    """一次标定的结果"""
    joint_name: str
    proximal_sensor: str
    distal_sensor: str
    calibration_mode: str                      # "lower_body_standing" | "t_pose"
    q_rel_0: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    # EKF 独立零位。None 表示旧版标定文件，运行时回退到 Mahony 零位。
    q_rel_0_kalman: Optional[List[float]] = None
    zero_angle_deg: float = 0.0                # 默认 0
    calibration_duration_s: float = 3.0
    created_time: str = ""
    sample_count: int = 0

    def __post_init__(self):
        if not self.created_time:
            self.created_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def is_valid(self) -> bool:
        q = np.array(self.q_rel_0, dtype=float)
        # numpy 的比较结果是 np.bool_，不能直接由 json.dump 序列化。
        return bool(len(q) == 4 and np.linalg.norm(q) > 0.5)

    def kalman_q_rel_0(self) -> List[float]:
        """返回 EKF 零位；兼容只有 ``q_rel_0`` 的旧标定文件。"""
        q = self.q_rel_0_kalman
        if q is None:
            return self.q_rel_0
        arr = np.asarray(q, dtype=float)
        if len(arr) != 4 or np.linalg.norm(arr) <= 0.5:
            return self.q_rel_0
        return q

    def to_dict(self) -> dict:
        return {
            "joint_name": self.joint_name,
            "proximal_sensor": self.proximal_sensor,
            "distal_sensor": self.distal_sensor,
            "calibration_mode": self.calibration_mode,
            "q_rel_0": [float(x) for x in self.q_rel_0],
            "q_rel_0_kalman": [float(x) for x in self.kalman_q_rel_0()],
            "zero_angle_deg": float(self.zero_angle_deg),
            "calibration_duration_s": float(self.calibration_duration_s),
            "created_time": self.created_time,
            "sample_count": int(self.sample_count),
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "JointCalibration":
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            joint_name=d["joint_name"],
            proximal_sensor=d["proximal_sensor"],
            distal_sensor=d["distal_sensor"],
            calibration_mode=d.get("calibration_mode", "unknown"),
            q_rel_0=d.get("q_rel_0", [1.0, 0.0, 0.0, 0.0]),
            q_rel_0_kalman=d.get("q_rel_0_kalman"),
            zero_angle_deg=d.get("zero_angle_deg", 0.0),
            calibration_duration_s=d.get("calibration_duration_s", 3.0),
            created_time=d.get("created_time", ""),
            sample_count=d.get("sample_count", 0),
        )


@dataclass
class JointAngleState:
    """实时关节角度状态（线程安全由外部保证）"""
    joint_name: str = ""

    # ── 三轴角度 (roll=屈伸, pitch=外展内收, yaw=旋转) ──
    flexion_deg: float = 0.0       # roll  — 屈曲/伸展
    abduction_deg: float = 0.0     # pitch — 外展/内收
    rotation_deg: float = 0.0      # yaw   — 内旋/外旋

    # ── 屈伸 ROM ──
    max_flexion_deg: float = -180.0
    min_flexion_deg: float = 180.0
    rom_deg: float = 0.0

    timestamp: float = 0.0

    # ── 时间序列缓存（用于画曲线）──
    max_history: int = 3600          # 3600 帧 @ 100Hz 降采样 20Hz = 180 秒缓冲
    history_t: List[float] = field(default_factory=list)
    history_flexion: List[float] = field(default_factory=list)
    history_abduction: List[float] = field(default_factory=list)
    history_rotation: List[float] = field(default_factory=list)

    def update(self, flexion_deg: float, abduction_deg: float = 0.0,
               rotation_deg: float = 0.0, t: float = None):
        if t is None:
            t = time.time()
        self.flexion_deg = float(flexion_deg)
        self.abduction_deg = float(abduction_deg)
        self.rotation_deg = float(rotation_deg)
        self.timestamp = t

        if flexion_deg > self.max_flexion_deg:
            self.max_flexion_deg = flexion_deg
        if flexion_deg < self.min_flexion_deg:
            self.min_flexion_deg = flexion_deg
        self.rom_deg = self.max_flexion_deg - self.min_flexion_deg

        self.history_t.append(t)
        self.history_flexion.append(float(flexion_deg))
        self.history_abduction.append(float(abduction_deg))
        self.history_rotation.append(float(rotation_deg))
        if len(self.history_t) > self.max_history:
            self.history_t = self.history_t[-self.max_history:]
            self.history_flexion = self.history_flexion[-self.max_history:]
            self.history_abduction = self.history_abduction[-self.max_history:]
            self.history_rotation = self.history_rotation[-self.max_history:]

    def reset(self):
        self.flexion_deg = 0.0
        self.abduction_deg = 0.0
        self.rotation_deg = 0.0
        self.max_flexion_deg = -180.0
        self.min_flexion_deg = 180.0
        self.rom_deg = 0.0
        self.history_t.clear()
        self.history_flexion.clear()
        self.history_abduction.clear()
        self.history_rotation.clear()
