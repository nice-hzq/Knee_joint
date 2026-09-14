# -*- coding: utf-8 -*-
"""统一四元数运算模块 (w,x,y,z 约定)"""

import numpy as np


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    return quat_conj(quat_normalize(q))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    vq = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=float)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_relative(q_parent: np.ndarray, q_child: np.ndarray) -> np.ndarray:
    return quat_mul(quat_inv(q_parent), quat_normalize(q_child))


def average_quaternions(qs: np.ndarray) -> np.ndarray:
    qs = quat_normalize(np.asarray(qs, dtype=float))
    q_ref = qs[0].copy()
    aligned = []
    for q in qs:
        q2 = q.copy()
        if np.dot(q2, q_ref) < 0:
            q2 = -q2
        aligned.append(q2)
    return quat_normalize(np.mean(np.asarray(aligned), axis=0))


def quat_align_hemisphere(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    if np.dot(q, q_ref) < 0.0:
        q = -q
    return q


def enforce_quat_continuity(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=float).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def quat_to_euler(q: np.ndarray, degrees: bool = False):
    if q.shape[-1] != 4:
        raise ValueError("q must have last dim 4: [w, x, y, z]")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    euler = np.stack([roll, pitch, yaw], axis=-1)
    return np.degrees(euler) if degrees else euler


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    angle_rad = np.asarray(angle_rad, dtype=float)
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_180(angle_deg):
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def R_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))
