# -*- coding: utf-8 -*-
"""
数据源
Data sources — 串口实时采集 / 离线 CSV 回放，统一 `frames()` 迭代接口。

    src = SerialSource("COM8", 921600)
    for frame in src.frames(stop_event):
        ...

    src = CsvReplaySource("samples/joint_angle_20260716_093022", speed=1.0)
    for frame in src.frames():
        ...
"""

import csv
import glob
import heapq
import os
import threading
import time
from typing import Iterator, List, Optional

from .protocol import (
    DATA_PREFIX, Frame, is_header_row, parse_csv_row, parse_data_line,
    signed_u32_delta_ms,
)

try:
    import serial
except ImportError:      # 允许在无 pyserial 的环境下做离线回放
    serial = None


class SourceError(RuntimeError):
    """数据源致命错误（串口打不开、文件不存在等）。"""


class SerialSource:
    """从 ESP32 串口读取 [DATA] 行。"""

    def __init__(self, port: str, baud: int = 921600, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._buffer = b""

    def open(self):
        if serial is None:
            raise SourceError("未安装 pyserial，无法使用串口。请运行: pip install pyserial")
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except Exception as e:
            raise SourceError(f"串口打开失败: {e}") from e

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def frames(self, stop_event: Optional[threading.Event] = None) -> Iterator[Frame]:
        if self._ser is None:
            self.open()

        self._buffer = b""
        while stop_event is None or not stop_event.is_set():
            try:
                if self._ser.in_waiting:
                    data = self._ser.read(self._ser.in_waiting)
                else:
                    data = self._ser.read(1)
                    if not data:
                        continue
            except Exception as e:
                raise SourceError(f"串口读取失败: {e}") from e

            self._buffer += data
            while b"\n" in self._buffer:
                line_bytes, self._buffer = self._buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line.startswith(DATA_PREFIX):
                    continue
                frame = parse_data_line(line)
                if frame is not None:
                    yield frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class CsvReplaySource:
    """
    回放已录制的 16 列原始 CSV。

    path 可以是:
      - 目录：读取目录下所有 *.csv，每个文件视为一个 device_id 的数据流
        （正好对应 data/raw/joint_angle_<ts>/<device_id>.csv 的布局）
      - 单个文件：按 device_id 列自行分组

    speed:
      0     → 全速回放（不节流）
      1.0   → 按录制时的真实时间节流
      10.0  → 10 倍速
    """

    def __init__(self, path: str, speed: float = 0.0):
        self.path = path
        self.speed = float(speed)
        if not os.path.exists(path):
            raise SourceError(f"回放路径不存在: {path}")

    # ---- 读取 ----
    def _files(self) -> List[str]:
        if os.path.isdir(self.path):
            return sorted(glob.glob(os.path.join(self.path, "*.csv")))
        return [self.path]

    def _read_from_files(self) -> List[List[Frame]]:
        """返回按 device_id 分组的帧列表（每个列表保持录制顺序）。"""
        streams: dict = {}
        for filepath in self._files():
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                for cols in csv.reader(f):
                    if is_header_row(cols):
                        continue
                    frame = parse_csv_row(cols)
                    if frame is None:
                        continue
                    streams.setdefault(frame.device_id, []).append(frame)
        return list(streams.values())

    @staticmethod
    def _unwrap_stream(frames: List[Frame]) -> List[tuple]:
        """
        把单个设备的 esp32_rx_ms 展开成单调毫秒，处理 32 位回绕。

        返回 [(t_ms, frame), ...]，按 t_ms 升序（文件本身即录制顺序）。
        """
        out = []
        last = None
        elapsed = 0
        for frame in frames:
            current = int(frame.esp32_rx_ms) & 0xFFFFFFFF
            if last is None:
                elapsed = 0
            else:
                delta = (current - last) & 0xFFFFFFFF
                if delta <= 0x7FFFFFFF:
                    elapsed += delta
            last = current
            out.append((elapsed, frame))
        return out

    def frames(self, stop_event: Optional[threading.Event] = None) -> Iterator[Frame]:
        streams = self._read_from_files()
        if not streams:
            raise SourceError(f"未从 {self.path} 读到任何有效数据帧")

        # 每个设备各自展开时间轴后，做一次多路归并
        iters = [iter(self._unwrap_stream(s)) for s in streams]
        merged = heapq.merge(*iters, key=lambda item: item[0])

        t0 = None
        wall0 = None
        for t_ms, frame in merged:
            if stop_event is not None and stop_event.is_set():
                return

            if self.speed > 0.0:
                if t0 is None:
                    t0 = t_ms
                    wall0 = time.time()
                else:
                    target = (t_ms - t0) / 1000.0 / self.speed
                    sleep_s = target - (time.time() - wall0)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            yield frame

    def close(self):
        pass
