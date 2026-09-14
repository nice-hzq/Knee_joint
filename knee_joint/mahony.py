# -*- coding: utf-8 -*-
"""
Mahony 姿态解算节点
Mahony AHRS node — 单传感器 acc+gyr(+mag) → 四元数 (wxyz)。

本模块只保留关节角度计算所需的 MahonyOrientationNode。
原项目中同文件的步态专用辅助函数
(rotate_acc_to_world / remove_gravity_and_static_bias / plot_acc_world_check /
 QuaternionManager / 模块级 init_from_static / is_static_segment) 已剔除。
"""

import numpy as np

from ahrs.filters import Mahony
from ahrs.common.orientation import acc2q, am2q

G0 = 9.80665


class MahonyOrientationNode:
    """
    管理一个传感器节点的姿态（四元数序列 + 在线更新）
    - 支持 IMU 模式（acc+gyr）和 MARG 模式（acc+gyr+mag）
    - 输出四元数格式：wxyz
    """
    def __init__(
        self,
        name: str,
        fs: float,
        *,
        use_mag: bool = False,
        acc_unit: str = "g",     # "g" or "mps2"
        gyr_unit: str = "deg",   # "deg" or "rad"
        kp: float = 0.5,
        ki: float = 0.1,         # 零偏在线估计 — 原 0.05 过小导致收敛极慢
        q0: np.ndarray | None = None,
        gyro_bias: np.ndarray | None = None,
    ):
        self.name = name
        self.fs = float(fs)
        self.use_mag = bool(use_mag)
        self.acc_unit = acc_unit
        self.gyr_unit = gyr_unit
        self.kp = float(kp)
        self.ki = float(ki) if float(ki) > 0.0 else 1e-6
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)

        self.filter = Mahony(frequency=self.fs, k_P=self.kp, k_I=self.ki,
                             b0=self.gyro_bias.copy())

        if q0 is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.q = np.asarray(q0, dtype=float)
            self.q = self.q / (np.linalg.norm(self.q) + 1e-12)

        # 历史缓存（按需开）
        self.Q = None  # (N,4)

    def _prep_units(self, acc3, gyr3, mag3=None):
        acc3 = np.asarray(acc3, float)
        gyr3 = np.asarray(gyr3, float)

        # acc -> m/s^2
        if self.acc_unit.lower() in ["g", "grav"]:
            acc3 = acc3 * G0

        # gyr -> rad/s
        if self.gyr_unit.lower().startswith("deg"):
            gyr3 = np.deg2rad(gyr3)

        if mag3 is not None:
            mag3 = np.asarray(mag3, float)

        return acc3, gyr3, mag3

    def update_one(self, acc3, gyr3, mag3=None) -> np.ndarray:
        """
        更新一帧，返回当前四元数 wxyz
        """
        acc3, gyr3, mag3 = self._prep_units(acc3, gyr3, mag3)

        # 注意：陀螺零偏由 Mahony 滤波器内部 b 估计处理（已通过 b0 注入初始值）
        # 不在此处做外部减法，避免与 filter.b 叠加导致双重扣除

        if self.use_mag and mag3 is not None:
            q_new = self.filter.updateMARG(self.q, gyr=gyr3, acc=acc3, mag=mag3)
        else:
            q_new = self.filter.updateIMU(self.q, gyr=gyr3, acc=acc3)

        q_new = np.asarray(q_new, float)
        q_new = q_new / (np.linalg.norm(q_new) + 1e-12)

        self.q = q_new
        return self.q

    def run_batch(self, imu9: np.ndarray) -> np.ndarray:
        """
        imu9: (N,9) = [acc(3), gyr(3), mag(3)]
        返回 Q: (N,4) wxyz
        """
        imu9 = np.asarray(imu9, float)
        if imu9.ndim != 2 or imu9.shape[1] != 9:
            raise ValueError(f"{self.name}: imu9 must be (N,9)")

        N = imu9.shape[0]
        Q = np.zeros((N, 4), dtype=float)

        for i in range(N):
            acc3 = imu9[i, 0:3]
            gyr3 = imu9[i, 3:6]
            mag3 = imu9[i, 6:9] if self.use_mag else None
            Q[i] = self.update_one(acc3, gyr3, mag3)

        self.Q = Q
        return Q

    def init_from_static(
            self,
            imu9: np.ndarray,
            *,
            n_first: int = 200,
            n_init: int = 400,
            estimate_gyro_bias: bool = False,
    ):
        """
        使用一段静止 IMU 数据初始化姿态
        - 设置 self.q（传感器 → 世界）
        - （可选）设置 self.gyro_bias

        imu9: (N,9) = [acc, gyr, mag]
        """
        imu9 = np.asarray(imu9, float)

        a = imu9[n_first:n_init, 0:3]
        w = imu9[n_first:n_init, 3:6]
        m = imu9[n_first:n_init, 6:9] if self.use_mag else None

        # ===== 单位处理 =====
        if self.acc_unit.lower() in ["g", "grav"]:
            a = a * G0
        if self.gyr_unit.lower().startswith("deg"):
            w = np.deg2rad(w)

        a_mean = a.mean(axis=0)

        # ===== 姿态初始化 =====
        if self.use_mag and m is not None:
            m_mean = m.mean(axis=0)
            q0 = am2q(a_mean, m_mean)
        else:
            q0 = acc2q(a_mean)

        q0 = np.asarray(q0, float)
        self.q = q0 / (np.linalg.norm(q0) + 1e-12)

        # ===== 陀螺零偏（可选）=====
        if estimate_gyro_bias:
            self.gyro_bias = w.mean(axis=0)
            # 同步注入 Mahony 滤波器内部零偏估计，加速收敛
            self.filter.b = self.gyro_bias.copy()

        return self.q
