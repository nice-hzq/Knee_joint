# -*- coding: utf-8 -*-
"""
关节角度采集线程
Joint angle worker — 把数据源、帧同步、角度引擎、会话记录串成一条流水线。

线程通过 `status_queue` 向上层（GUI 或 CLI）发布消息，消息契约与原项目一致:

    joint_status       state=connecting|connected|device_detected|identified|
                             uncalibrated|waiting_calib|devices|disconnected
    joint_angle        flexion_deg / abduction_deg / rotation_deg /
                       max_deg / min_deg / rom_deg / history_* / sync_dt_ms /
                       initialized / calibrated
    joint_calib_status state=calibrating
    joint_calib_done   q_rel_0 / euler_rel_0 / message
    joint_calib_error  message
    joint_error        message / traceback

用法:
    src = SerialSource("COM8", 921600)
    worker = JointWorker(src, "right_knee", device_map, output_root, queue)
    worker.start(); ...; worker.stop()
"""

import json
import os
import queue
import threading
import time
import traceback
from typing import Optional

import numpy as np

from .calibration import save_calibration
from .engine import JointAngleEngine, online_calibrate_from_buffers
from .models import JOINT_DEFS, JointBinding, JointCalibration
from .quaternion import quat_to_euler
from .recorder import SessionRecorder
from .sync import FrameSynchronizer

DEVICE_MAP_FILENAME = "device_alias_map.json"

# 未标定时也持续更新引擎（保持滤波器收敛），但只有标定后才推送曲线数据
UI_FIRST_N_FRAMES = 10      # 标定后前 N 帧全量推送，让数值/曲线快速出现
UI_FRAME_STRIDE = 5         # 之后每 N 帧推送一次（100Hz → 约 20Hz）
UI_HISTORY_CAP = 1200       # 单次推送携带的历史点数上限

DEVICE_REPORT_INTERVAL_S = 3.0
SENSOR_TIMEOUT_MS = 1000

AUTO_CALIB_RETRY_INTERVAL_S = 1.0   # 自动标定失败后的重试间隔
AUTO_CALIB_MAX_ATTEMPTS = 10        # 连续失败这么多次后放弃（改为手动校准）


class JointWorker(threading.Thread):
    """
    实时关节角度测量线程。

    数据通道由外部注入的 `source` 决定（串口或离线回放），
    本类只负责：帧同步 → Mahony/EKF 解算 → 相对姿态 → 关节角 → 落盘/上报。

    设备识别:
      - device_map: {device_id: sensor_alias}，用于识别近端/远端传感器
      - device_map 为空时尝试从 calib_dir/device_alias_map.json 加载
      - 映射不完整时不启动匹配，避免大腿/小腿因上线顺序而互换
    """

    def __init__(self, source, joint_key: str, output_root: str,
                 status_queue: queue.Queue, *,
                 device_map: Optional[dict] = None,
                 calib_dir: str = "./temp",
                 fs: float = 100.0,
                 calib_mode: str = "lower_body_standing",
                 calib_dur_s: float = 3.0,
                 max_sync_dt_ms: float = 30.0,
                 save_raw: bool = True,
                 calib_file: Optional[str] = None,
                 auto_calib_at_s: Optional[float] = None,
                 source_label: str = "serial",
                 kalman_var_gyr: float = 0.3 ** 2,
                 kalman_var_acc: float = 0.5 ** 2):
        super().__init__(daemon=True)

        if joint_key not in JOINT_DEFS:
            raise ValueError(f"未知关节: {joint_key} (可选: {list(JOINT_DEFS)})")

        self.source = source
        self.joint_key = joint_key
        self.output_root = output_root
        self.q = status_queue
        self.calib_dir = calib_dir
        self.fs = float(fs)
        self.calib_mode = calib_mode
        self.calib_dur_s = float(calib_dur_s)
        self.save_raw = save_raw
        self.calib_file = calib_file
        self.auto_calib_at_s = auto_calib_at_s
        self.source_label = source_label
        self.kalman_var_gyr = float(kalman_var_gyr)
        self.kalman_var_acc = float(kalman_var_acc)

        self._stop_event = threading.Event()

        # 设备映射
        self.device_map = dict(device_map) if device_map else {}
        info = JOINT_DEFS[joint_key]
        self.proximal_alias = info["proximal_default"]
        self.distal_alias = info["distal_default"]
        self.proximal_label = info["prox_label"]
        self.distal_label = info["dist_label"]

        # 运行时
        self._engine: Optional[JointAngleEngine] = None
        self._calib: Optional[JointCalibration] = None
        self._pairs = FrameSynchronizer(max_dt_ms=max_sync_dt_ms)
        self._recorder = SessionRecorder(output_root, save_raw=save_raw)

        self._pending_calib = False
        self._pending_calib_mode = calib_mode
        self._warned_uncalib = False
        # 自动标定：下一次尝试的会话时间（None = 不再尝试），以及连续失败次数
        self._auto_calib_next_s: Optional[float] = (
            float(auto_calib_at_s) if auto_calib_at_s is not None else None)
        self._auto_calib_attempts = 0
        self._angle_frame_count = 0
        self._last_frame_time: dict = {}

        self.session_dir: Optional[str] = None

    # ========================================================
    # 控制
    # ========================================================
    def stop(self):
        self._stop_event.set()

    def start_calibration(self, calib_mode: Optional[str] = None):
        """请求一次标定（由上层 UI 在用户点击时调用）。"""
        self._pending_calib_mode = calib_mode or self.calib_mode
        self._pending_calib = True

    # ========================================================
    # 消息上报
    # ========================================================
    def _post(self, msg_type, **kwargs):
        try:
            self.q.put_nowait({"type": msg_type, **kwargs})
        except queue.Full:
            pass

    def _post_critical(self, msg_type, **kwargs):
        payload = {"type": msg_type, **kwargs}
        try:
            self.q.put(payload, timeout=5.0)
        except queue.Full:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put(payload, timeout=1.0)
            except queue.Full:
                pass

    # ========================================================
    # 设备映射
    # ========================================================
    def _load_device_map(self):
        """尝试从 calib_dir/device_alias_map.json 加载设备映射（不覆盖已显式传入的项）。"""
        map_path = os.path.join(self.calib_dir, DEVICE_MAP_FILENAME)
        if not os.path.exists(map_path):
            return
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    self.device_map.setdefault(k, v)
                print(f"[WORKER] 已加载设备映射: {self.device_map}")
        except Exception as e:
            print(f"[WORKER] 加载设备映射失败: {e}")

    def _identify_sensors(self) -> bool:
        """
        仅使用显式 device_map 识别近端/远端，避免因设备上线顺序变化而交换大腿/小腿。
        返回是否已完成识别。
        """
        if self._pairs.prox_id and self._pairs.dist_id:
            return True

        prox_candidates = [did for did, alias in self.device_map.items()
                           if alias == self.proximal_alias and self._pairs.has_device(did)]
        dist_candidates = [did for did, alias in self.device_map.items()
                           if alias == self.distal_alias and self._pairs.has_device(did)]

        if prox_candidates and dist_candidates:
            self._pairs.set_roles(prox_candidates[0], dist_candidates[0])
            self._post("joint_status", state="identified",
                       message=f"传感器已识别: {self.proximal_alias}={prox_candidates[0]}, "
                               f"{self.distal_alias}={dist_candidates[0]}",
                       proximal_id=prox_candidates[0], distal_id=dist_candidates[0],
                       proximal_alias=self.proximal_alias,
                       distal_alias=self.distal_alias)
            return True

        # 映射不完整：给出可操作的诊断，而不是静默等待
        if self._pairs.device_order and not self.device_map:
            self._post("joint_status", state="waiting",
                       message="尚未配置设备映射 — 请先在「设备映射」对话框中把 "
                               f"{self.proximal_alias}(近端) 与 {self.distal_alias}(远端) "
                               "各自绑定到一个 device_id")
        return False

    # ========================================================
    # 标定
    # ========================================================
    def _apply_calibration(self, calib: JointCalibration):
        self._engine.load_calibration(calib)
        self._calib = calib
        save_calibration(calib, self.calib_dir)
        # 新零点生效后清除校准前角度/ROM，避免曲线出现旧值到 0° 的假跳变。
        self._engine.reset_state()
        self._angle_frame_count = 0

        q0 = np.array(calib.q_rel_0, dtype=float)
        q0_kalman = np.array(calib.kalman_q_rel_0(), dtype=float)
        euler0 = quat_to_euler(q0, degrees=True)
        euler0_kalman = quat_to_euler(q0_kalman, degrees=True)
        diag_msg = (
            f"标定完成: {calib.calibration_mode}\n"
            f"q_rel_0 = [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]\n"
            f"安装姿态: Roll(屈伸)={euler0[0]:.1f}°  "
            f"Pitch(外展)={euler0[1]:.1f}°  "
            f"Yaw(旋转)={euler0[2]:.1f}°\n"
            f"EKF零位: Roll={euler0_kalman[0]:.1f}°  "
            f"Pitch={euler0_kalman[1]:.1f}°  Yaw={euler0_kalman[2]:.1f}°\n"
            f"（站立时 knee angle=0°，传感器轴间夹角应接近 0°）"
        )
        self._post("joint_calib_done", joint=self.joint_key, message=diag_msg,
                   q_rel_0=q0.tolist(),
                   euler_rel_0=[float(euler0[0]), float(euler0[1]), float(euler0[2])],
                   q_rel_0_kalman=q0_kalman.tolist(),
                   euler_rel_0_kalman=[float(x) for x in euler0_kalman])
        return True

    def _run_calibration(self, mode: str) -> bool:
        """从引擎持久环形缓冲区执行一次标定。"""
        need = int(round(self.calib_dur_s * self.fs))
        rp, rd = self._engine.get_rolling_buffer_sizes()
        if rp < need or rd < need:
            self._post("joint_calib_error",
                       message=f"缓冲数据不足: {self.proximal_label}={rp} "
                               f"{self.distal_label}={rd} 帧, 需要各 {need} 帧 "
                               f"({self.calib_dur_s}秒)。请等待数据积累。")
            return False

        self._post("joint_calib_status", state="calibrating",
                   message=f"正在标定 ({mode})，请保持静止...")
        try:
            calib = online_calibrate_from_buffers(
                self._engine, calib_mode=mode, calib_dur_s=self.calib_dur_s)
        except Exception as e:
            self._post("joint_calib_error", message=f"标定失败: {e}")
            return False
        return self._apply_calibration(calib)

    # ========================================================
    # 主流程
    # ========================================================
    def _run_impl(self):
        info = JOINT_DEFS[self.joint_key]
        binding = JointBinding(
            joint_name=self.joint_key,
            proximal_sensor=self.proximal_alias,
            distal_sensor=self.distal_alias,
            flexion_axis=info["flexion_axis"],
        )
        self._engine = JointAngleEngine(
            binding,
            fs=self.fs,
            kalman_var_gyr=self.kalman_var_gyr,
            kalman_var_acc=self.kalman_var_acc,
        )
        self._pending_calib_mode = self.calib_mode

        self._load_device_map()

        # ---- 打开数据源 ----
        self._post("joint_status", state="connecting",
                   message=f"打开数据源 ({self.source_label})...")
        try:
            if hasattr(self.source, "open"):
                self.source.open()
        except Exception as e:
            self._post_critical("joint_error", message=str(e))
            return
        self._post("joint_status", state="connected",
                   message=f"已连接 ({self.source_label})，等待传感器数据...")

        # ---- 预加载标定文件（可选，离线回放常用）----
        if self.calib_file:
            try:
                calib = JointCalibration.load(self.calib_file)
                if calib.is_valid():
                    self._engine.load_calibration(calib)
                    self._calib = calib
                    self._post("joint_calib_done", joint=self.joint_key,
                               message=f"已加载标定文件: {self.calib_file}",
                               q_rel_0=list(calib.q_rel_0), euler_rel_0=None,
                               q_rel_0_kalman=list(calib.kalman_q_rel_0()),
                               euler_rel_0_kalman=None)
            except Exception as e:
                self._post("joint_calib_error",
                           message=f"标定文件加载失败: {e}")
        else:
            self._post("joint_status", state="uncalibrated",
                       message="未标定 — 引擎就绪后请保持静止站立并点击「开始校准」")

        last_device_report = time.time()

        for frame in self.source.frames(self._stop_event):
            if self._stop_event.is_set():
                break

            device_id = frame.device_id
            if self._pairs.feed(frame):
                alias = self.device_map.get(device_id) or "?"
                self._post("joint_status", state="device_detected",
                           message=f"检测到设备: {device_id} → {alias}",
                           device_id=device_id, alias=alias)
            self._last_frame_time[device_id] = time.time()

            self._identify_sensors()

            if self._pairs.prox_id and self._pairs.dist_id:
                self._drain_pairs()

            # ---- 自动标定（离线回放 / 无人值守）----
            self._maybe_auto_calibrate()

            # ---- 周期性设备状态 ----
            now = time.time()
            if now - last_device_report > DEVICE_REPORT_INTERVAL_S:
                last_device_report = now
                self._report_devices(now)

            # ---- 处理上层标定请求 ----
            if self._pending_calib and self._engine.is_ready():
                self._pending_calib = False
                self._run_calibration(self._pending_calib_mode)

        # ---- 收尾 ----
        self._save_session()

    def _maybe_auto_calibrate(self):
        """
        到点后自动标定一次，失败则按秒重试。

        为什么需要重试：触发点按**会话时间**判断，而标定要求攒够
        `calib_dur_s × fs` 帧。配对后的有效帧率可能明显低于 fs
        （两路并非每帧都能配上，实测约 86%），所以 3 秒的会话时间
        未必攒够 300 帧 —— 一次性触发会直接放弃，让回放永远标不上。
        """
        if self._auto_calib_next_s is None or not self._engine.is_ready():
            return
        if self._pairs.clock.elapsed_s < self._auto_calib_next_s:
            return

        self._auto_calib_attempts += 1
        if self._run_calibration(self.calib_mode):
            self._auto_calib_next_s = None
            return

        if self._auto_calib_attempts >= AUTO_CALIB_MAX_ATTEMPTS:
            self._auto_calib_next_s = None
            self._post("joint_calib_error",
                       message=f"自动标定连续失败 {self._auto_calib_attempts} 次，已放弃。"
                               "请确认数据稳定后手动点击「开始校准」。")
            return

        self._auto_calib_next_s = (self._pairs.clock.elapsed_s
                                   + AUTO_CALIB_RETRY_INTERVAL_S)

    def _drain_pairs(self):
        """把缓冲区里所有可配对的帧推过引擎。"""
        while True:
            pair = self._pairs.next_pair()
            if pair is None:
                return

            output_started = self._engine.has_calibration()
            self._engine.update(pair.imu_proximal, pair.imu_distal,
                                timestamp_s=pair.t_s)

            if output_started:
                # 每个角度样本仅写入对应的近端、远端同步原始帧。
                self._recorder.record_raw_row(pair.raw_proximal)
                self._recorder.record_raw_row(pair.raw_distal)

            state = self._engine.get_state()
            kalman_state = self._engine.get_kalman_state()
            calibrated = self._engine.has_calibration()

            if not calibrated:
                if self._engine.is_ready() and not self._warned_uncalib:
                    self._warned_uncalib = True
                    self._post("joint_status", state="waiting_calib",
                               message="引擎已就绪，请保持静止站立后点击「开始校准」")
                continue

            # 校准后的前 10 帧立即发送，让数值和曲线快速出现；
            # 稳定后每 5 帧发送一次（100Hz→约20Hz）。
            self._angle_frame_count += 1
            frame_count = self._angle_frame_count
            if frame_count <= UI_FIRST_N_FRAMES or frame_count % UI_FRAME_STRIDE == 0:
                step = 1 if frame_count <= UI_FIRST_N_FRAMES else UI_FRAME_STRIDE
                t_raw = state.history_t
                f_raw = state.history_flexion
                a_raw = state.history_abduction
                r_raw = state.history_rotation
                start = max(0, len(t_raw) - UI_HISTORY_CAP * step)
                self._post("joint_angle",
                           joint=self.joint_key,
                           flexion_deg=round(state.flexion_deg, 2),
                           abduction_deg=round(state.abduction_deg, 2),
                           rotation_deg=round(state.rotation_deg, 2),
                           max_deg=round(state.max_flexion_deg, 2),
                           min_deg=round(state.min_flexion_deg, 2),
                           rom_deg=round(state.rom_deg, 2),
                           kalman_flexion_deg=round(kalman_state.flexion_deg, 2),
                           kalman_abduction_deg=round(kalman_state.abduction_deg, 2),
                           kalman_rotation_deg=round(kalman_state.rotation_deg, 2),
                           kalman_max_deg=round(kalman_state.max_flexion_deg, 2),
                           kalman_min_deg=round(kalman_state.min_flexion_deg, 2),
                           kalman_rom_deg=round(kalman_state.rom_deg, 2),
                           history_t=t_raw[start::step],
                           history_flexion=f_raw[start::step],
                           history_abduction=a_raw[start::step],
                           history_rotation=r_raw[start::step],
                           kalman_history_t=kalman_state.history_t[start::step],
                           kalman_history_flexion=kalman_state.history_flexion[start::step],
                           kalman_history_abduction=kalman_state.history_abduction[start::step],
                           kalman_history_rotation=kalman_state.history_rotation[start::step],
                           sync_dt_ms=round(pair.dt_ms, 1),
                           sample_time_s=round(pair.t_s, 4),
                           initialized=self._engine.is_ready(),
                           calibrated=True)

    def _report_devices(self, now: float):
        if not self._pairs.device_order:
            self._post("joint_status", state="waiting", message="等待传感器数据...")
            return

        dev_info = []
        for did in self._pairs.device_order:
            alias = self.device_map.get(did) or "?"
            buf_n = self._pairs.buffer_size(did)
            last_t = self._last_frame_time.get(did, 0)
            age_s = now - last_t if last_t else 999
            role = ""
            if did == self._pairs.prox_id:
                role = f"→{self.proximal_alias}({self.proximal_label})"
            elif did == self._pairs.dist_id:
                role = f"→{self.distal_alias}({self.distal_label})"
            timeout_flag = " ⚠超时" if age_s > SENSOR_TIMEOUT_MS / 1000.0 else ""
            dev_info.append(f"{did}({alias}){role} buf={buf_n} age={age_s:.1f}s{timeout_flag}")

        if self._pairs.matched_count:
            dev_info.append(f"同步: 配对={self._pairs.matched_count} "
                            f"平均差={self._pairs.avg_dt_ms:.1f}ms "
                            f"最大差={self._pairs.dt_max_ms:.1f}ms")
        self._post("joint_status", state="devices", message=" | ".join(dev_info))

    def _save_session(self) -> Optional[str]:
        if self._engine is None:
            return None
        meta = {
            "joint_key": self.joint_key,
            "proximal_sensor": self.proximal_alias,
            "distal_sensor": self.distal_alias,
            "proximal_label": self.proximal_label,
            "distal_label": self.distal_label,
            "proximal_device_id": self._pairs.prox_id or "",
            "distal_device_id": self._pairs.dist_id or "",
            "calibrated": self._engine.has_calibration(),
            "calibration_mode": self._calib.calibration_mode if self._calib else "",
            "fs": int(self.fs),
            "source": self.source_label,
            "sync": self._pairs.stats(),
        }
        self.session_dir = self._recorder.save_session(
            self._engine.get_state(), meta,
            kalman_state=self._engine.get_kalman_state())
        return self.session_dir

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            self._post_critical("joint_error",
                                message=f"线程异常: {type(e).__name__}: {e}",
                                traceback=traceback.format_exc())
        finally:
            self._recorder.close_raw()
            if self._recorder.raw_dir:
                print(f"[WORKER] 原始数据已保存: {self._recorder.raw_dir} "
                      f"({self._recorder.raw_total_rows} 帧)")
            self._post("joint_status", state="disconnected", message="已停止")
