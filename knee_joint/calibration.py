# -*- coding: utf-8 -*-
"""
关节姿态校准模块
Joint pose calibration — standing / T-pose calibration for joint angle zero-reference.

支持两种模式:
  1) lower_body_standing — 下肢站立校准（膝关节、踝关节、髋关节）
  2) t_pose — T-pose 全身校准（后续扩展）

校准流程:
  采集 3 秒内两个传感器的原始 IMU 数据 → Mahony / EKF 解算四元数
  → 分别求平均 → 两套 q_rel_0 = inv(q_proximal_0) * q_distal_0
  → 保存为 JSON
"""

import os
import time
import numpy as np

from .quaternion import quat_inv, quat_mul, quat_normalize, average_quaternions
from .mahony import MahonyOrientationNode
from .kalman import KalmanOrientationNode
from .models import JointBinding, JointCalibration


# ============================================================
# 主函数：从 IMU 数据计算初始相对姿态
# ============================================================
def calibrate_joint_from_arrays(
    binding: JointBinding,
    imu_proximal: np.ndarray,    # (N, 9) 近端传感器原始数据
    imu_distal: np.ndarray,      # (N, 9) 远端传感器原始数据
    fs: float = 100.0,
    calib_dur_s: float = 3.0,
    calib_mode: str = "lower_body_standing",
    *,
    gyr_thr_dps: float = 10.0,         # WT901 静止时陀螺噪声通常 < 2 dps
    acc_std_thr_g: float = 0.15,       # WT901 加速度噪声 std 约 0.05~0.12g
    kalman_var_gyr: float = 0.3 ** 2,
    kalman_var_acc: float = 0.5 ** 2,
) -> JointCalibration:
    """
    用静止站立段数据标定关节初始相对姿态。

    参数:
        binding: 关节-传感器绑定
        imu_proximal: (N,9) 近端 IMU [acc(3), gyr(3), mag(3)]
        imu_distal:   (N,9) 远端 IMU [acc(3), gyr(3), mag(3)]
        fs: 采样率
        calib_dur_s: 校准持续时间（秒）
        calib_mode: 校准模式名称
        gyr_thr_dps: 陀螺模长阈值 — 超过此值认为非静止
        acc_std_thr_g: 加速度模长标准差阈值 — 超过此值认为非静止

    返回:
        JointCalibration (含 q_rel_0)
    """
    n_samples = int(round(calib_dur_s * fs))

    # 截取最后 n_samples（确保是用户稳定站立后的数据）
    imu_proximal = np.asarray(imu_proximal[-n_samples:], dtype=float)
    imu_distal   = np.asarray(imu_distal[-n_samples:],   dtype=float)

    if len(imu_proximal) < n_samples or len(imu_distal) < n_samples:
        raise ValueError(
            f"校准数据不足: 需要 {n_samples} 帧, "
            f"近端={len(imu_proximal)}, 远端={len(imu_distal)}"
        )

    # ---- 静止检测 ----
    ok_p, msg_p = _check_static(imu_proximal, fs, gyr_thr_dps, acc_std_thr_g)
    ok_d, msg_d = _check_static(imu_distal, fs, gyr_thr_dps, acc_std_thr_g)
    if not ok_p:
        raise RuntimeError(f"近端传感器 [{binding.proximal_sensor}] 未保持静止: {msg_p}")
    if not ok_d:
        raise RuntimeError(f"远端传感器 [{binding.distal_sensor}] 未保持静止: {msg_d}")

    # ---- Mahony 解算 ----
    quats_p = _run_mahony(imu_proximal, fs)
    quats_d = _run_mahony(imu_distal,   fs)

    # ---- 平均四元数 ----
    q_proximal_0 = average_quaternions(quats_p)
    q_distal_0   = average_quaternions(quats_d)

    # ---- q_rel_0 = inv(q_proximal) * q_distal ----
    q_rel_0 = quat_mul(quat_inv(q_proximal_0), q_distal_0)
    q_rel_0 = quat_normalize(q_rel_0)

    # ---- Quaternion EKF 解算并计算独立零位 ----
    kalman_quats_p = _run_kalman(
        imu_proximal, fs, var_gyr=kalman_var_gyr, var_acc=kalman_var_acc)
    kalman_quats_d = _run_kalman(
        imu_distal, fs, var_gyr=kalman_var_gyr, var_acc=kalman_var_acc)
    kalman_q_proximal_0 = average_quaternions(kalman_quats_p)
    kalman_q_distal_0 = average_quaternions(kalman_quats_d)
    q_rel_0_kalman = quat_mul(
        quat_inv(kalman_q_proximal_0), kalman_q_distal_0)
    q_rel_0_kalman = quat_normalize(q_rel_0_kalman)

    calib = JointCalibration(
        joint_name=binding.joint_name,
        proximal_sensor=binding.proximal_sensor,
        distal_sensor=binding.distal_sensor,
        calibration_mode=calib_mode,
        q_rel_0=q_rel_0.tolist(),
        q_rel_0_kalman=q_rel_0_kalman.tolist(),
        calibration_duration_s=calib_dur_s,
        sample_count=n_samples,
    )

    print(f"[CALIB] {binding.joint_name}: "
          f"Mahony q_rel_0 = [{q_rel_0[0]:.4f}, {q_rel_0[1]:.4f}, "
          f"{q_rel_0[2]:.4f}, {q_rel_0[3]:.4f}] | "
          f"EKF q_rel_0 = [{q_rel_0_kalman[0]:.4f}, {q_rel_0_kalman[1]:.4f}, "
          f"{q_rel_0_kalman[2]:.4f}, {q_rel_0_kalman[3]:.4f}]")
    return calib


# ============================================================
# 文件 I/O 辅助
# ============================================================
def calib_filepath(joint_name: str, calib_dir: str = "./temp") -> str:
    """返回标定文件的默认路径，如 temp/left_knee_pose_calib.json"""
    return os.path.join(calib_dir, f"{joint_name}_pose_calib.json")


def save_calibration(calib: JointCalibration, calib_dir: str = "./temp") -> str:
    """保存标定到文件，返回文件路径"""
    path = calib_filepath(calib.joint_name, calib_dir)
    calib.save(path)
    print(f"[CALIB] 已保存: {path}")
    return path


def load_calibration(joint_name: str, calib_dir: str = "./temp") -> JointCalibration:
    """从文件加载标定"""
    path = calib_filepath(joint_name, calib_dir)
    calib = JointCalibration.load(path)
    print(f"[CALIB] 已加载: {path}  |  q_rel_0 valid={calib.is_valid()}")
    return calib


# ============================================================
# Internal helpers
# ============================================================
def _check_static(imu9: np.ndarray, fs: float,
                  gyr_thr_dps: float, acc_std_thr_g: float) -> tuple:
    """
    检查 IMU 数据段是否静止。返回 (is_static, message)。

    陀螺用 STD 而非中位数 —— 传感器有常值零偏（WT901 手册 ±20 dps），
    中位数会包含零偏，用标准差才能判断"传感器是否在转动"。
    """
    a = imu9[:, 0:3]
    w = imu9[:, 3:6]
    gyr_norm = np.linalg.norm(w, axis=1)
    acc_norm = np.linalg.norm(a, axis=1)

    gyr_med = float(np.median(gyr_norm))
    gyr_std = float(np.std(gyr_norm))
    acc_std = float(np.std(acc_norm))
    acc_mean = float(np.mean(acc_norm))

    gyr_ok = gyr_std < gyr_thr_dps
    acc_ok = acc_std < acc_std_thr_g

    if not gyr_ok:
        return False, (f"陀螺未静止: norm中位数={gyr_med:.2f}°/s (含零偏), "
                       f"std={gyr_std:.2f}°/s >= {gyr_thr_dps}°/s。"
                       f"请确保传感器完全静止放在桌面上。")
    if not acc_ok:
        return False, (f"加速度波动过大: norm均值={acc_mean:.3f}g, std={acc_std:.4f}g >= {acc_std_thr_g}g。"
                       f"请确保传感器不受振动干扰。")
    return True, "OK"


def _run_mahony(imu9: np.ndarray, fs: float) -> np.ndarray:
    """用 Mahony 滤波器解算四元数序列。返回 (N,4) wxyz。"""
    node = MahonyOrientationNode(
        name="calib_tmp",
        fs=fs,
        use_mag=False,
        acc_unit="g",
        gyr_unit="deg",
        kp=0.5,
        ki=0.1,
    )
    node.init_from_static(
        imu9=imu9,
        n_first=0,
        n_init=min(400, len(imu9)),
        estimate_gyro_bias=True,   # ★ 减去静止段陀螺均值，消除出厂零偏
    )
    return node.run_batch(imu9)


def _run_kalman(
    imu9: np.ndarray,
    fs: float,
    *,
    var_gyr: float = 0.3 ** 2,
    var_acc: float = 0.5 ** 2,
) -> np.ndarray:
    """用四元数 EKF 解算序列。返回 (N,4) wxyz。"""
    node = KalmanOrientationNode(
        name="calib_tmp_ekf",
        fs=fs,
        use_mag=False,
        acc_unit="g",
        gyr_unit="deg",
        var_gyr=var_gyr,
        var_acc=var_acc,
    )
    node.init_from_static(
        imu9=imu9,
        n_first=0,
        n_init=min(400, len(imu9)),
        estimate_gyro_bias=True,
    )
    return node.run_batch(imu9)
