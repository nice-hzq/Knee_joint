# -*- coding: utf-8 -*-
"""
膝关节角度测量模块
Knee / Joint Angle Measurement — calibration + real-time dual-IMU relative orientation.

典型用法（纯算法，无串口）:

    from knee_joint import JointBinding, JointAngleEngine, calibrate_joint_from_arrays

    binding = JointBinding("right_knee", "R4", "R5")
    calib = calibrate_joint_from_arrays(binding, imu_prox, imu_dist, fs=100.0)
    engine = JointAngleEngine(binding, fs=100.0)
    engine.load_calibration(calib)
    flexion = engine.update(imu_prox[0], imu_dist[0])
"""

from .models import (
    JOINT_DEFS,
    JointBinding,
    JointCalibration,
    JointAngleState,
)
from .calibration import (
    calibrate_joint_from_arrays,
    calib_filepath,
    save_calibration,
    load_calibration,
)
from .engine import (
    JointAngleEngine,
    online_calibrate_from_buffers,
)
from .kalman import KalmanOrientationNode

__version__ = "1.1.0"

__all__ = [
    "JOINT_DEFS",
    "JointBinding",
    "JointCalibration",
    "JointAngleState",
    "calibrate_joint_from_arrays",
    "calib_filepath",
    "save_calibration",
    "load_calibration",
    "JointAngleEngine",
    "online_calibrate_from_buffers",
    "KalmanOrientationNode",
]
