# -*- coding: utf-8 -*-
"""
会话记录
Session recording — 原始同步帧 + 关节角度序列落盘。

输出布局:
    <output_root>/raw/<采集名称>_<YYYYMMDD_HHMMSS>/<device_id>.csv
    <output_root>/joint_angles/<采集名称>_<YYYYMMDD_HHMMSS>/joint_angle_data.csv
    <output_root>/joint_angles/<采集名称>_<YYYYMMDD_HHMMSS>/session_info.json

原始 CSV **只记录真正参与了同步配对并用于输出角度的帧**，不是全量串口流。
"""

import csv
import json
import os
import time
from datetime import datetime
from typing import List, Optional

from .models import JointAngleState
from .protocol import CSV_HEADER, sanitize_filename

ANGLES_HEADER = ["time_s", "flexion_deg", "abduction_deg", "rotation_deg"]
KALMAN_ANGLES_HEADER = [
    "kalman_flexion_deg", "kalman_abduction_deg", "kalman_rotation_deg",
]

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_session_name(name: str) -> str:
    """生成可安全用作 Windows 目录名的采集名称，同时保留中文和空格。"""
    text = " ".join(str(name or "").strip().split())
    safe = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in text
        if ord(c) >= 32
    ).strip(" ._-")
    safe = safe[:80].rstrip(" ._-") or "session"
    if safe.upper() in _WINDOWS_RESERVED_NAMES:
        safe = f"session_{safe}"
    return safe


class SessionRecorder:
    """把一次测量会话的原始帧与角度序列写入磁盘。"""

    def __init__(self, output_root: str, session_timestamp: Optional[str] = None,
                 save_raw: bool = True, session_name: Optional[str] = None):
        self.output_root = os.path.abspath(output_root)
        self.session_timestamp = session_timestamp or time.strftime("%Y%m%d_%H%M%S")
        self.save_raw = save_raw

        self.session_name = str(session_name or "").strip() or "session"
        safe_name = sanitize_session_name(self.session_name)
        safe_timestamp = sanitize_session_name(self.session_timestamp).replace(" ", "_")
        self.raw_root = os.path.join(self.output_root, "raw")
        self.joint_angles_root = os.path.join(self.output_root, "joint_angles")
        self.session_id = self._unique_session_id(
            f"{safe_name}_{safe_timestamp}")
        self._raw_dir_path = os.path.join(self.raw_root, self.session_id)
        self._session_dir_path = os.path.join(
            self.joint_angles_root, self.session_id)

        self.raw_dir: Optional[str] = None
        self.raw_total_rows = 0
        self._raw_writers: dict = {}
        self._raw_save_failed = False

    def _unique_session_id(self, base_id: str) -> str:
        """若同名会话已存在，追加 _02、_03，且两类数据始终共享同一 ID。"""
        candidate = base_id
        suffix = 2
        while (os.path.exists(os.path.join(self.output_root, "raw", candidate)) or
               os.path.exists(os.path.join(
                   self.output_root, "joint_angles", candidate))):
            candidate = f"{base_id}_{suffix:02d}"
            suffix += 1
        return candidate

    # ========================================================
    # 原始同步帧
    # ========================================================
    def record_raw_row(self, row: List[str]):
        """记录一行已用于输出关节角度的同步原始传感器数据。"""
        if not self.save_raw or self._raw_save_failed:
            return
        if row is None or len(row) < len(CSV_HEADER):
            return
        device_id = row[0].strip()
        if not device_id:
            return

        try:
            if device_id not in self._raw_writers:
                if self.raw_dir is None:
                    self.raw_dir = self._raw_dir_path
                    os.makedirs(self.raw_dir, exist_ok=True)

                filepath = os.path.join(
                    self.raw_dir, f"{sanitize_filename(device_id)}.csv")
                f = open(filepath, "w", newline="", encoding="utf-8")
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                f.flush()
                self._raw_writers[device_id] = {
                    "file": f, "writer": writer, "rows": 0, "path": filepath,
                }

            entry = self._raw_writers[device_id]
            entry["writer"].writerow(row[:len(CSV_HEADER)])
            entry["rows"] += 1
            self.raw_total_rows += 1
            if entry["rows"] % 50 == 0:
                entry["file"].flush()
        except (OSError, csv.Error) as e:
            self._raw_save_failed = True
            print(f"[RECORDER] 原始数据保存失败: {type(e).__name__}: {e}")
            self.close_raw()

    def close_raw(self):
        """刷新并关闭所有原始数据文件。"""
        for entry in self._raw_writers.values():
            try:
                entry["file"].flush()
                entry["file"].close()
            except OSError:
                pass
        self._raw_writers.clear()

    # ========================================================
    # 角度序列
    # ========================================================
    def save_session(self, state: JointAngleState, meta: dict, *,
                     kalman_state: Optional[JointAngleState] = None) -> Optional[str]:
        """
        保存当前会话的角度数据。

        meta 至少应包含 joint_key / proximal_sensor / distal_sensor / fs；
        同步统计可通过 meta["sync"] 传入（见 FrameSynchronizer.stats()）。

        Returns:
            str: 保存目录路径；数据不足时返回 None。
        """
        try:
            n = len(state.history_t) if state.history_t else 0
            n = min(n, len(state.history_flexion),
                    len(state.history_abduction), len(state.history_rotation))
            if kalman_state is not None:
                n = min(n, len(kalman_state.history_t),
                        len(kalman_state.history_flexion),
                        len(kalman_state.history_abduction),
                        len(kalman_state.history_rotation))
            if n < 2:
                return None

            session_dir = self._session_dir_path
            os.makedirs(session_dir, exist_ok=True)

            # ---- 角度 CSV ----
            csv_path = os.path.join(session_dir, "joint_angle_data.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ANGLES_HEADER + KALMAN_ANGLES_HEADER
                    if kalman_state is not None else ANGLES_HEADER)
                t0 = state.history_t[0]
                for i in range(n):
                    row = [
                        f"{state.history_t[i] - t0:.3f}",
                        f"{state.history_flexion[i]:.4f}",
                        f"{state.history_abduction[i]:.4f}",
                        f"{state.history_rotation[i]:.4f}",
                    ]
                    if kalman_state is not None:
                        row.extend([
                            f"{kalman_state.history_flexion[i]:.4f}",
                            f"{kalman_state.history_abduction[i]:.4f}",
                            f"{kalman_state.history_rotation[i]:.4f}",
                        ])
                    writer.writerow(row)

            # ---- 会话信息 JSON ----
            sync = dict(meta.get("sync") or {})
            session_info = {
                "session_name": self.session_name,
                "session_id": self.session_id,
                "joint_key": meta.get("joint_key", ""),
                "proximal_sensor": meta.get("proximal_sensor", ""),
                "distal_sensor": meta.get("distal_sensor", ""),
                "proximal_label": meta.get("proximal_label", ""),
                "distal_label": meta.get("distal_label", ""),
                "proximal_device_id": meta.get("proximal_device_id", ""),
                "distal_device_id": meta.get("distal_device_id", ""),
                "calibrated": bool(meta.get("calibrated", False)),
                "calibration_mode": meta.get("calibration_mode", ""),
                "sample_count": n,
                "duration_s": round(state.history_t[n - 1] - state.history_t[0], 2),
                "fs": meta.get("fs", 0),
                "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": meta.get("source", ""),
                "flexion_rom_deg": round(state.rom_deg, 2),
                "flexion_max_deg": round(state.max_flexion_deg, 2),
                "flexion_min_deg": round(state.min_flexion_deg, 2),
                "sync_matched_count": sync.get("sync_matched_count", 0),
                "sync_dropped_proximal": sync.get("sync_dropped_proximal", 0),
                "sync_dropped_distal": sync.get("sync_dropped_distal", 0),
                "sync_avg_dt_ms": sync.get("sync_avg_dt_ms", 0.0),
                "sync_max_dt_ms": sync.get("sync_max_dt_ms", 0.0),
                "raw_data_dir": self.raw_dir or "",
                "joint_angle_data_dir": session_dir,
                "raw_sample_count": self.raw_total_rows,
            }
            if kalman_state is not None:
                session_info.update({
                    "kalman_flexion_rom_deg": round(kalman_state.rom_deg, 2),
                    "kalman_flexion_max_deg": round(
                        kalman_state.max_flexion_deg, 2),
                    "kalman_flexion_min_deg": round(
                        kalman_state.min_flexion_deg, 2),
                })
            json_path = os.path.join(session_dir, "session_info.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(session_info, f, ensure_ascii=False, indent=2)

            print(f"[RECORDER] 数据已保存: {csv_path} ({n} 帧)")
            return session_dir

        except Exception as e:
            print(f"[RECORDER] 保存数据失败: {type(e).__name__}: {e}")
            import traceback as _tb
            _tb.print_exc()
            return None
