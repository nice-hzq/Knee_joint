# -*- coding: utf-8 -*-
"""
Kalman (quaternion EKF) 姿态解算节点。

与 :mod:`knee_joint.mahony` 使用相同的输入、初始化和输出约定：
单帧输入为 acc + gyr (+ mag)，输出四元数格式为 wxyz。
"""

import numpy as np

from ahrs.common.orientation import acc2q, am2q
from ahrs.filters import EKF

from .mahony import G0


class KalmanOrientationNode:
    """管理一个传感器节点的四元数 EKF 姿态。"""

    def __init__(
        self,
        name: str,
        fs: float,
        *,
        use_mag: bool = False,
        acc_unit: str = "g",
        gyr_unit: str = "deg",
        var_gyr: float = 0.3 ** 2,
        var_acc: float = 0.5 ** 2,
        var_mag: float = 0.8 ** 2,
        q0: np.ndarray | None = None,
        gyro_bias: np.ndarray | None = None,
    ):
        self.name = name
        self.fs = float(fs)
        self.use_mag = bool(use_mag)
        self.acc_unit = acc_unit
        self.gyr_unit = gyr_unit
        self.var_gyr = float(var_gyr)
        self.var_acc = float(var_acc)
        self.var_mag = float(var_mag)
        self.gyro_bias = (
            np.zeros(3, dtype=float)
            if gyro_bias is None
            else np.asarray(gyro_bias, dtype=float)
        )

        self.filter = EKF(
            frequency=self.fs,
            frame="NED",
            noises=np.array([self.var_gyr, self.var_acc, self.var_mag], dtype=float),
        )
        if q0 is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.q = self._normalize(q0)
        self.Q = None

    @staticmethod
    def _normalize(q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("EKF returned an invalid quaternion")
        return q / norm

    def _prep_units(self, acc3, gyr3, mag3=None):
        acc3 = np.asarray(acc3, dtype=float)
        gyr3 = np.asarray(gyr3, dtype=float)
        if self.acc_unit.lower() in ("g", "grav"):
            acc3 = acc3 * G0
        if self.gyr_unit.lower().startswith("deg"):
            gyr3 = np.deg2rad(gyr3)
        if mag3 is not None:
            mag3 = np.asarray(mag3, dtype=float)
        return acc3, gyr3, mag3

    def update_one(self, acc3, gyr3, mag3=None) -> np.ndarray:
        """更新一帧并返回当前四元数 wxyz。"""
        acc3, gyr3, mag3 = self._prep_units(acc3, gyr3, mag3)
        gyr3 = gyr3 - self.gyro_bias
        q_prev = self.q.copy()
        try:
            q_new = self.filter.update(
                self.q,
                gyr=gyr3,
                acc=acc3,
                mag=mag3 if self.use_mag else None,
            )
            self.q = self._normalize(q_new)
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            # 单帧数值异常时保持上一次有效姿态，避免整条实时链路中断。
            self.q = q_prev
            self.filter.P = np.identity(4)
        return self.q

    def run_batch(self, imu9: np.ndarray) -> np.ndarray:
        """批量运行；imu9 为 (N, 9)，返回 (N, 4) wxyz。"""
        imu9 = np.asarray(imu9, dtype=float)
        if imu9.ndim != 2 or imu9.shape[1] != 9:
            raise ValueError(f"{self.name}: imu9 must be (N,9)")

        Q = np.zeros((len(imu9), 4), dtype=float)
        for i, sample in enumerate(imu9):
            Q[i] = self.update_one(
                sample[0:3], sample[3:6], sample[6:9] if self.use_mag else None
            )
        self.Q = Q
        return Q

    def init_from_static(
        self,
        imu9: np.ndarray,
        *,
        n_first: int = 200,
        n_init: int = 400,
        estimate_gyro_bias: bool = False,
    ) -> np.ndarray:
        """使用一段静止 IMU 数据初始化姿态和可选的陀螺零偏。"""
        imu9 = np.asarray(imu9, dtype=float)
        if imu9.ndim != 2 or imu9.shape[1] != 9:
            raise ValueError(f"{self.name}: imu9 must be (N,9)")

        start = max(0, int(n_first))
        stop = min(len(imu9), int(n_init))
        if stop <= start:
            raise ValueError(f"{self.name}: empty static initialization window")

        acc = imu9[start:stop, 0:3]
        gyr = imu9[start:stop, 3:6]
        mag = imu9[start:stop, 6:9] if self.use_mag else None
        if self.acc_unit.lower() in ("g", "grav"):
            acc = acc * G0
        if self.gyr_unit.lower().startswith("deg"):
            gyr = np.deg2rad(gyr)

        if self.use_mag and mag is not None:
            q0 = am2q(acc.mean(axis=0), mag.mean(axis=0))
        else:
            q0 = acc2q(acc.mean(axis=0))
        self.q = self._normalize(q0)

        if estimate_gyro_bias:
            self.gyro_bias = gyr.mean(axis=0)

        # 每次重新初始化时也重置 EKF 协方差，避免沿用旧会话状态。
        self.filter.P = np.identity(4)
        return self.q
