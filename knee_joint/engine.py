# -*- coding: utf-8 -*-
"""
实时关节角度计算引擎
Real-time joint angle computation from dual-IMU relative orientation.

核心公式:
  q_rel_t   = inv(q_proximal_t) * q_distal_t       (当前相对姿态)
  q_joint_t = inv(q_rel_0) * q_rel_t               (标定补偿后的关节姿态)
  flexion   = euler_from_quat(q_joint_t)[flexion_axis]  (屈伸角, 度)
"""

import time
from typing import Optional

import numpy as np

from .quaternion import (
    quat_inv, quat_mul, quat_normalize, quat_to_euler, quat_conj,
)
from .mahony import MahonyOrientationNode
from .kalman import KalmanOrientationNode
from .models import JointBinding, JointCalibration, JointAngleState


# ============================================================
# JointAngleEngine — 单关节实时引擎
# ============================================================
class JointAngleEngine:
    """
    管理一个关节的实时角度计算。

    使用方式:
      engine = JointAngleEngine(binding, fs=100)
      engine.load_calibration(calib)            # 加载标定
      for each frame:
          engine.update(imu_prox, imu_dist)     # 输入 9 轴 IMU 数据
          state = engine.get_state()            # 获取当前角度状态
    """

    def __init__(self, binding: JointBinding, fs: float = 100.0, *,
                 kalman_var_gyr: float = 0.3 ** 2,
                 kalman_var_acc: float = 0.5 ** 2):
        self.binding = binding
        self.fs = float(fs)
        self.calibration: Optional[JointCalibration] = None
        # ``state`` 保留为 Mahony 状态，维持已有 API 与回放基准兼容。
        self.state = JointAngleState(joint_name=binding.joint_name)
        self.kalman_state = JointAngleState(joint_name=binding.joint_name)
        self.kalman_var_gyr = float(kalman_var_gyr)
        self.kalman_var_acc = float(kalman_var_acc)

        # 两个 Mahony 节点（在线更新模式）
        self._node_prox = MahonyOrientationNode(
            name=f"{binding.joint_name}_prox",
            fs=self.fs,
            use_mag=False,
            acc_unit="g",
            gyr_unit="deg",
            kp=0.5,
            ki=0.1,
        )
        self._node_dist = MahonyOrientationNode(
            name=f"{binding.joint_name}_dist",
            fs=self.fs,
            use_mag=False,
            acc_unit="g",
            gyr_unit="deg",
            kp=0.5,
            ki=0.1,
        )
        self._kalman_node_prox = KalmanOrientationNode(
            name=f"{binding.joint_name}_prox_ekf",
            fs=self.fs,
            use_mag=False,
            acc_unit="g",
            gyr_unit="deg",
            var_gyr=self.kalman_var_gyr,
            var_acc=self.kalman_var_acc,
        )
        self._kalman_node_dist = KalmanOrientationNode(
            name=f"{binding.joint_name}_dist_ekf",
            fs=self.fs,
            use_mag=False,
            acc_unit="g",
            gyr_unit="deg",
            var_gyr=self.kalman_var_gyr,
            var_acc=self.kalman_var_acc,
        )

        self._initialized = False
        self._init_buffer_p: list = []  # 仅用于自动初始化的临时缓冲
        self._init_buffer_d: list = []
        self._init_frames: int = 200    # 收集 200 帧后自动初始化

        # ★ 持久环形缓冲区 — 始终保留最近 N 秒数据，供在线标定使用
        self._rolling_buf_p: list = []
        self._rolling_buf_d: list = []
        self._rolling_max_s: float = 10.0   # 保留最近 10 秒

    # ---- 标定管理 ----
    def load_calibration(self, calib: JointCalibration):
        """加载标定结果"""
        self.calibration = calib
        q = np.array(calib.q_rel_0, dtype=float)
        print(f"[JOINT] {self.binding.joint_name}: 已加载标定 "
              f"q_rel_0=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

    def has_calibration(self) -> bool:
        return self.calibration is not None and self.calibration.is_valid()

    # ---- 初始化 ----
    def _try_auto_init(self):
        """收集足够帧后自动初始化 Mahony 与 EKF 节点。"""
        if self._initialized:
            return

        n_prox = len(self._init_buffer_p)
        n_dist = len(self._init_buffer_d)
        if n_prox >= self._init_frames and n_dist >= self._init_frames:
            init_data_p = np.array(self._init_buffer_p[-self._init_frames:], dtype=float)
            init_data_d = np.array(self._init_buffer_d[-self._init_frames:], dtype=float)
            self._node_prox.init_from_static(imu9=init_data_p, n_first=0,
                                             n_init=self._init_frames, estimate_gyro_bias=True)
            self._node_dist.init_from_static(imu9=init_data_d, n_first=0,
                                             n_init=self._init_frames, estimate_gyro_bias=True)
            self._kalman_node_prox.init_from_static(
                imu9=init_data_p, n_first=0, n_init=self._init_frames,
                estimate_gyro_bias=True)
            self._kalman_node_dist.init_from_static(
                imu9=init_data_d, n_first=0, n_init=self._init_frames,
                estimate_gyro_bias=True)
            self._initialized = True
            # 注意：不清空 _init_buffer（标定可能需要用），但后续不再增长
            print(f"[JOINT] {self.binding.joint_name}: Mahony/EKF 自动初始化完成 "
                  f"({self._init_frames} 帧)"
                  f" | 环形缓冲已有近端={len(self._rolling_buf_p)} 远端={len(self._rolling_buf_d)} 帧")

    def _add_to_rolling(self, imu_p: np.ndarray, imu_d: np.ndarray):
        """始终追加到环形缓冲区，超出上限自动丢弃旧数据"""
        self._rolling_buf_p.append(imu_p.copy())
        self._rolling_buf_d.append(imu_d.copy())
        max_frames = int(self._rolling_max_s * self.fs)
        if len(self._rolling_buf_p) > max_frames:
            self._rolling_buf_p = self._rolling_buf_p[-max_frames:]
        if len(self._rolling_buf_d) > max_frames:
            self._rolling_buf_d = self._rolling_buf_d[-max_frames:]

    def get_rolling_buffer_sizes(self) -> tuple:
        """返回 (prox_frames, dist_frames)"""
        return len(self._rolling_buf_p), len(self._rolling_buf_d)

    # ---- 主更新 ----
    def update(self,
               imu9_proximal: np.ndarray,   # (9,) 近端传感器一帧
               imu9_distal: np.ndarray,      # (9,) 远端传感器一帧
               timestamp_s: float = None,    # 已同步的传感器时间；缺省时使用系统时间
               ) -> float:
        """
        输入一帧 IMU 数据，返回当前屈伸角（度）。

        imu9_proximal: shape (9,) = [ax,ay,az, gx,gy,gz, mx,my,mz]
        imu9_distal:   shape (9,)
        """
        imu_p = np.asarray(imu9_proximal, dtype=float).flatten()
        imu_d = np.asarray(imu9_distal,   dtype=float).flatten()

        # 异常数据检测
        if np.any(np.isnan(imu_p)) or np.any(np.isinf(imu_p)):
            return self.state.flexion_deg if self._initialized else 0.0
        if np.any(np.isnan(imu_d)) or np.any(np.isinf(imu_d)):
            return self.state.flexion_deg if self._initialized else 0.0
        # 加速度模长合理性检查
        acc_norm_p = float(np.linalg.norm(imu_p[0:3]))
        acc_norm_d = float(np.linalg.norm(imu_d[0:3]))
        if acc_norm_p < 0.3 or acc_norm_p > 3.5:
            return self.state.flexion_deg if self._initialized else 0.0
        if acc_norm_d < 0.3 or acc_norm_d > 3.5:
            return self.state.flexion_deg if self._initialized else 0.0

        # ★ 持久环形缓冲（始终记录，供在线标定使用）
        self._add_to_rolling(imu_p, imu_d)

        # 初始化阶段：也放入 init 缓冲
        if not self._initialized:
            self._init_buffer_p.append(imu_p.copy())
            self._init_buffer_d.append(imu_d.copy())
            if len(self._init_buffer_p) > self._init_frames * 2:
                self._init_buffer_p = self._init_buffer_p[-self._init_frames * 2:]
            if len(self._init_buffer_d) > self._init_frames * 2:
                self._init_buffer_d = self._init_buffer_d[-self._init_frames * 2:]
            self._try_auto_init()
            return 0.0

        # Mahony 在线更新
        q_prox = self._node_prox.update_one(imu_p[0:3], imu_p[3:6], imu_p[6:9])
        q_dist = self._node_dist.update_one(imu_d[0:3], imu_d[3:6], imu_d[6:9])

        kalman_q_prox = self._kalman_node_prox.update_one(
            imu_p[0:3], imu_p[3:6], imu_p[6:9])
        kalman_q_dist = self._kalman_node_dist.update_one(
            imu_d[0:3], imu_d[3:6], imu_d[6:9])

        # ---- 关节角计算 ----
        flexion, abduction, rotation = self._compute_all_angles(q_prox, q_dist)
        self.state.update(flexion, abduction, rotation, t=timestamp_s)
        k_flexion, k_abduction, k_rotation = self._compute_all_angles(
            kalman_q_prox, kalman_q_dist, algorithm="kalman")
        self.kalman_state.update(
            k_flexion, k_abduction, k_rotation, t=timestamp_s)

        return flexion

    def _compute_all_angles(self, q_prox: np.ndarray, q_dist: np.ndarray,
                            algorithm: str = "mahony"):
        """
        核心公式:
          q_rel_t   = inv(q_prox) * q_dist
          q_joint_t = inv(q_rel_0) * q_rel_t
          返回 (roll=屈伸, pitch=外展内收, yaw=旋转) 单位: 度
        """
        q_rel_t = quat_mul(quat_inv(q_prox), q_dist)

        if self.calibration is not None and self.calibration.is_valid():
            q_rel_0 = np.array(
                self.calibration.kalman_q_rel_0()
                if algorithm == "kalman" else self.calibration.q_rel_0,
                dtype=float,
            )
            q_joint = quat_mul(quat_inv(q_rel_0), q_rel_t)
        else:
            q_joint = q_rel_t

        q_joint = quat_normalize(q_joint)
        euler_deg = quat_to_euler(q_joint, degrees=True)  # [roll, pitch, yaw]

        return float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])

    # ---- 状态获取 ----
    def get_state(self) -> JointAngleState:
        return self.state

    def get_kalman_state(self) -> JointAngleState:
        return self.kalman_state

    def reset_state(self):
        self.state.reset()
        self.kalman_state.reset()
        print(f"[JOINT] {self.binding.joint_name}: 角度状态已重置")

    def is_ready(self) -> bool:
        """是否已完成初始化，可以输出有效角度"""
        return self._initialized

    def get_debug_info(self) -> dict:
        """返回调试信息"""
        return {
            "joint": self.binding.joint_name,
            "proximal": self.binding.proximal_sensor,
            "distal": self.binding.distal_sensor,
            "calibrated": self.has_calibration(),
            "initialized": self._initialized,
            "flexion_deg": round(self.state.flexion_deg, 2),
            "abduction_deg": round(self.state.abduction_deg, 2),
            "rotation_deg": round(self.state.rotation_deg, 2),
            "max_deg": round(self.state.max_flexion_deg, 2),
            "min_deg": round(self.state.min_flexion_deg, 2),
            "rom_deg": round(self.state.rom_deg, 2),
            "kalman_flexion_deg": round(self.kalman_state.flexion_deg, 2),
            "kalman_abduction_deg": round(self.kalman_state.abduction_deg, 2),
            "kalman_rotation_deg": round(self.kalman_state.rotation_deg, 2),
            "kalman_rom_deg": round(self.kalman_state.rom_deg, 2),
            "rolling_p": len(self._rolling_buf_p),
            "rolling_d": len(self._rolling_buf_d),
        }


# ============================================================
# 便捷函数：在线实时校准（从 engine 持久环形缓冲区读取）
# ============================================================
def online_calibrate_from_buffers(
    engine: JointAngleEngine,
    calib_mode: str = "lower_body_standing",
    calib_dur_s: float = 3.0,
) -> JointCalibration:
    """
    从 engine 内部的持久环形缓冲区读取最近 calib_dur_s 秒数据执行标定。

    注意：调用此函数前 engine 必须已经运行了至少 calib_dur_s 秒。
    标定时需保持静止站立姿势。
    """
    n_needed = int(round(calib_dur_s * engine.fs))

    # ★ 从持久环形缓冲区读取（而非已清空的 _init_buffer）
    buf_p = np.array(engine._rolling_buf_p[-n_needed:], dtype=float)
    buf_d = np.array(engine._rolling_buf_d[-n_needed:], dtype=float)

    if len(buf_p) < n_needed or len(buf_d) < n_needed:
        raise ValueError(
            f"缓冲数据不足: 需要 {n_needed} 帧 (={calib_dur_s}s @ {engine.fs}Hz), "
            f"近端={len(buf_p)} 帧, 远端={len(buf_d)} 帧。"
            f"请等待至少 {calib_dur_s} 秒后再标定。"
        )

    from .calibration import calibrate_joint_from_arrays
    calib = calibrate_joint_from_arrays(
        binding=engine.binding,
        imu_proximal=buf_p,
        imu_distal=buf_d,
        fs=engine.fs,
        calib_dur_s=calib_dur_s,
        calib_mode=calib_mode,
        kalman_var_gyr=engine.kalman_var_gyr,
        kalman_var_acc=engine.kalman_var_acc,
    )
    engine.load_calibration(calib)
    return calib
