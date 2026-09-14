# -*- coding: utf-8 -*-
"""
膝关节角度测量 UI 包
Knee Joint Angle Measurement — UI package.

只包含实时关节角度所需的常量与界面，不含步态分析相关内容。
"""

import os
import sys

from ..models import JOINT_DEFS


# ============================================================
# 路径
# ============================================================
def get_base_dir():
    """返回项目根目录（开发时为 Knee_joint/，打包后为 exe 所在目录）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # knee_joint/ui/__init__.py → knee_joint/ → Knee_joint/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 共享常量
# ============================================================
DEFAULT_BAUD = 921600

VALID_ALIASES = [
    "H", "T1", "T12",
    "L1", "L2", "L3", "L4", "L5", "L6",
    "R1", "R2", "R3", "R4", "R5", "R6",
    "S1",
]

# ============================================================
# 关节角度常量
# ============================================================
#
# 角度极性约定 (Angle Polarity Convention):
#   flexion (+)  = 屈膝 / knee flexion (bending)
#   flexion (-)  = 伸膝 / knee extension (straightening, hyperextension)
#   abduction (+)  = 外展 / away from midline
#   abduction (-)  = 内收 / toward midline
#   rotation (+)   = 外旋 / external rotation
#   rotation (-)   = 内旋 / internal rotation
#
# 传感器安装方向约定:
#   大腿 IMU: 传感器 X 轴沿大腿骨方向指向膝关节
#   小腿 IMU: 传感器 X 轴沿小腿骨方向指向踝关节
#   （理想情况下两传感器坐标系应近似平行，标定时 q_rel_0 的 roll≈0°）
#
# 说明: 关节定义以 models.JOINT_DEFS 为唯一来源，这里只做界面友好的别名，
#       避免同一份数据在两处各写一遍。
JOINT_OPTIONS = {
    key: {
        "label": f"{d['name']} ({d['name_en']})",
        "proximal": d["proximal_default"],
        "distal": d["distal_default"],
        "prox_label": d["prox_label"],
        "dist_label": d["dist_label"],
    }
    for key, d in JOINT_DEFS.items()
}

CALIB_MODES = {
    "lower_body_standing": "下肢站立校准",
    "t_pose": "T-pose 全身校准",
}

CALIB_STATES = {
    "uncalibrated": "未标定",
    "calibrating":  "标定中...",
    "calibrated":   "已标定 ✓",
    "failed":       "标定失败 ✗",
}
