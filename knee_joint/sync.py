# -*- coding: utf-8 -*-
"""
双 IMU 帧同步
Dual-IMU frame synchronization — nearest-neighbour pairing on `esp32_rx_ms`.

策略（与原实时实现完全一致）:
  - 以**远端**传感器最早待处理帧的 esp32_rx_ms 为锚点
  - 在近端传感器缓冲区中一对一近邻匹配，命中即从两侧消费
  - 最大允许时间差 max_dt_ms（默认 30 ms，约 3 个 100 Hz 采样周期）
  - 无法匹配的过期帧被丢弃并计入统计

所有时间比较都使用 32 位回绕安全的 `abs_u32_delta_ms`。
"""

from collections import deque
from typing import List, NamedTuple, Optional

from .protocol import Frame, abs_u32_delta_ms, signed_u32_delta_ms


class PairedFrame(NamedTuple):
    """一对已同步的近端/远端帧。字段顺序与原实现返回的元组一致。"""
    imu_proximal: object          # np.ndarray (9,)
    imu_distal: object            # np.ndarray (9,)
    t_s: float                    # 会话相对时间（秒）
    dt_ms: float                  # 该对的 esp32_rx_ms 时间差（毫秒）
    raw_proximal: Optional[List[str]]
    raw_distal: Optional[List[str]]


class SessionClock:
    """
    把 ESP32 uint32 millis() 展开为单调递增的会话秒数。

    millis() 约 49.7 天回绕一次；回绕后的时间戳比上一帧小，
    此时按无符号差值继续累加而不是倒退。
    """

    def __init__(self):
        self._last_ms: Optional[int] = None
        self._elapsed_ms: int = 0

    def unwrap(self, timestamp_ms) -> float:
        current = int(timestamp_ms) & 0xFFFFFFFF
        if self._last_ms is None:
            self._last_ms = current
            self._elapsed_ms = 0
        else:
            delta = (current - self._last_ms) & 0xFFFFFFFF
            if delta <= 0x7FFFFFFF:
                self._elapsed_ms += delta
                self._last_ms = current
        return self._elapsed_ms / 1000.0

    @property
    def elapsed_s(self) -> float:
        """自首个时间戳以来累计的会话时长（秒）。"""
        return self._elapsed_ms / 1000.0

    def reset(self):
        self._last_ms = None
        self._elapsed_ms = 0


class FrameSynchronizer:
    """按 device_id 缓冲帧，并一对一配对近端/远端。"""

    def __init__(self, max_dt_ms: float = 30.0, buffer_limit: int = 500):
        self.max_dt_ms = float(max_dt_ms)
        self.buffer_limit = int(buffer_limit)

        self.buffers: dict = {}          # {device_id: deque[Frame]}
        self.device_order: List[str] = []   # 设备首次上线顺序

        self.prox_id: Optional[str] = None
        self.dist_id: Optional[str] = None

        self.clock = SessionClock()

        # 统计
        self.matched_count = 0
        self.drop_prox_count = 0
        self.drop_dist_count = 0
        self.dt_sum_ms = 0.0
        self.dt_max_ms = 0.0

    # ---- 缓冲 ----
    def feed(self, frame: Frame) -> bool:
        """送入一帧。返回 True 表示这是一个此前未出现过的 device_id。"""
        is_new = frame.device_id not in self.buffers
        if is_new:
            self.buffers[frame.device_id] = deque(maxlen=self.buffer_limit)
            self.device_order.append(frame.device_id)
        self.buffers[frame.device_id].append(frame)
        return is_new

    def has_device(self, device_id: str) -> bool:
        return device_id in self.buffers

    def buffer_size(self, device_id: str) -> int:
        buf = self.buffers.get(device_id)
        return len(buf) if buf else 0

    def set_roles(self, proximal_id: str, distal_id: str):
        self.prox_id = proximal_id
        self.dist_id = distal_id

    def reset_roles(self):
        self.prox_id = None
        self.dist_id = None

    # ---- 配对 ----
    def next_pair(self) -> Optional[PairedFrame]:
        """
        取下一对已同步的帧；无可用配对时返回 None。
        已匹配的帧不会被再次使用。
        """
        if not self.prox_id or not self.dist_id:
            return None
        if self.prox_id not in self.buffers or self.dist_id not in self.buffers:
            return None

        buf_p = self.buffers[self.prox_id]
        buf_d = self.buffers[self.dist_id]
        max_dt = self.max_dt_ms

        while buf_p and buf_d:
            dist_frame = buf_d[0]
            t_d = dist_frame.esp32_rx_ms

            best_idx = min(
                range(len(buf_p)),
                key=lambda i: abs_u32_delta_ms(buf_p[i].esp32_rx_ms, t_d))
            best_dt = abs_u32_delta_ms(buf_p[best_idx].esp32_rx_ms, t_d)

            if best_dt <= max_dt:
                prox_frame = buf_p[best_idx]
                # 弹出匹配帧本身，以及比它更早的帧 ——
                # 更早的帧不可能再匹配后续远端帧，计入丢弃。
                for _ in range(best_idx + 1):
                    buf_p.popleft()
                self.drop_prox_count += best_idx
                buf_d.popleft()

                self.matched_count += 1
                self.dt_sum_ms += best_dt
                self.dt_max_ms = max(self.dt_max_ms, best_dt)

                return PairedFrame(
                    imu_proximal=prox_frame.imu9,
                    imu_distal=dist_frame.imu9,
                    t_s=self.clock.unwrap(t_d),
                    dt_ms=float(best_dt),
                    raw_proximal=prox_frame.raw_row,
                    raw_distal=dist_frame.raw_row,
                )

            newest_p_delta = signed_u32_delta_ms(buf_p[-1].esp32_rx_ms, t_d)
            if newest_p_delta > max_dt:
                # 近端数据已经越过此远端帧；该远端帧无法再匹配。
                buf_d.popleft()
                self.drop_dist_count += 1
                continue

            # 近端缓冲中的过旧帧不可能与当前或后续远端帧匹配。
            old_count = 0
            while buf_p and signed_u32_delta_ms(buf_p[0].esp32_rx_ms, t_d) < -max_dt:
                buf_p.popleft()
                old_count += 1
            self.drop_prox_count += old_count
            break

        return None

    # ---- 统计 ----
    @property
    def avg_dt_ms(self) -> float:
        return self.dt_sum_ms / self.matched_count if self.matched_count else 0.0

    def stats(self) -> dict:
        return {
            "sync_matched_count": self.matched_count,
            "sync_dropped_proximal": self.drop_prox_count,
            "sync_dropped_distal": self.drop_dist_count,
            "sync_avg_dt_ms": round(self.avg_dt_ms, 3),
            "sync_max_dt_ms": round(self.dt_max_ms, 3),
        }
