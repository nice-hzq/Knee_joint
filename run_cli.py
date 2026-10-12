# -*- coding: utf-8 -*-
"""
膝关节角度测量 — 命令行入口（无界面）

实时串口:
    python run_cli.py --port COM8 --joint right_knee --session-name "深蹲测试01"

离线回放:
    python run_cli.py --replay samples/joint_angle_20260716_093022 \
                      --joint right_knee --calib-file temp/right_knee_pose_calib.json

参数优先级: 命令行 > config.json > 内置默认值
"""

import argparse
import os
import queue
import sys
import time

from knee_joint import JOINT_DEFS
from knee_joint.config import (
    load_config, load_device_map, project_root, resolve_path,
)
from knee_joint.sources import CsvReplaySource, SerialSource, SourceError
from knee_joint.worker import JointWorker

# 控制台为 GBK 时避免中文/符号打印崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="膝关节角度测量（实时串口 / 离线回放）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--config", default=None, help="配置文件路径（默认 ./config.json）")
    p.add_argument("--joint", default=None, choices=sorted(JOINT_DEFS),
                   help="关节名称（默认取配置）")

    src = p.add_argument_group("数据源")
    src.add_argument("--port", default=None, help="串口，如 COM8")
    src.add_argument("--baud", type=int, default=None, help="波特率（默认 921600）")
    src.add_argument("--replay", default=None,
                     help="离线回放：原始 CSV 文件或目录（目录内每个 <device_id>.csv 为一个数据流）")
    src.add_argument("--speed", type=float, default=0.0,
                     help="回放速度倍率：0=全速（默认），1.0=按真实时间")
    src.add_argument("--fs", type=float, default=None, help="采样率 Hz（默认 100）")

    cal = p.add_argument_group("标定")
    cal.add_argument("--calib-file", default=None,
                     help="直接加载已有标定 JSON（离线复现推荐）")
    cal.add_argument("--calib-at", type=float, default=None,
                     help="在回放的第 N 秒自动标定（默认：未指定 --calib-file 时取 3.0 秒）")
    cal.add_argument("--calib-mode", default=None, help="标定模式名（默认 lower_body_standing）")
    cal.add_argument("--calib-dur", type=float, default=None, help="标定窗口时长秒（默认 3.0）")
    cal.add_argument("--calib-dir", default=None, help="标定文件目录（默认 ./temp）")

    filt = p.add_argument_group("姿态滤波")
    filt.add_argument("--kalman-var-gyr", type=float, default=None,
                      help="EKF 陀螺过程噪声方差（默认取配置）")
    filt.add_argument("--kalman-var-acc", type=float, default=None,
                      help="EKF 加速度测量噪声方差（默认取配置）")

    dev = p.add_argument_group("设备映射")
    dev.add_argument("--device-map", default=None,
                     help='设备映射 JSON 文件，或内联如 \'{"WT...":"R4","WT...":"R5"}\'')
    dev.add_argument("--proximal", default=None, help="近端传感器 device_id（等价于映射到 proximal 别名）")
    dev.add_argument("--distal", default=None, help="远端传感器 device_id")

    out = p.add_argument_group("输出")
    out.add_argument("--output", default=None, help="输出根目录（默认 ./data）")
    out.add_argument("--session-name", default=None,
                     help="本次采集名称；最终目录会自动追加时间戳（默认使用关节 key）")
    out.add_argument("--no-raw", action="store_true", help="不保存原始同步帧")
    out.add_argument("--quiet", action="store_true", help="不实时打印角度，只打印状态")

    return p


def _parse_inline_map(text: str) -> dict:
    import json
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    # ---- 参数合并 ----
    joint_key = args.joint or cfg["joint_key"]
    if joint_key not in JOINT_DEFS:
        print(f"[ERROR] 未知关节: {joint_key}")
        return 2

    fs = args.fs if args.fs is not None else cfg["fs"]
    calib_dir = resolve_path(args.calib_dir or cfg["calib_dir"])
    output_root = resolve_path(args.output or cfg["output_root"])
    calib_mode = args.calib_mode or cfg["calib"]["mode"]
    calib_dur = args.calib_dur if args.calib_dur is not None else cfg["calib"]["dur_s"]
    max_dt = cfg["sync"]["max_dt_ms"]
    save_raw = cfg["save_raw"] and not args.no_raw
    kalman_cfg = cfg.get("kalman") or {}
    kalman_var_gyr = (args.kalman_var_gyr if args.kalman_var_gyr is not None
                      else kalman_cfg.get("var_gyr", 0.3 ** 2))
    kalman_var_acc = (args.kalman_var_acc if args.kalman_var_acc is not None
                      else kalman_cfg.get("var_acc", 0.5 ** 2))

    # ---- 设备映射 ----
    device_map = load_device_map(cfg)
    if args.device_map:
        if os.path.exists(args.device_map):
            import json
            with open(args.device_map, "r", encoding="utf-8") as f:
                device_map.update(json.load(f))
        else:
            device_map.update(_parse_inline_map(args.device_map))

    info = JOINT_DEFS[joint_key]
    if args.proximal:
        device_map[args.proximal] = info["proximal_default"]
    if args.distal:
        device_map[args.distal] = info["distal_default"]

    # ---- 数据源 ----
    if args.replay:
        replay_path = resolve_path(args.replay)
        try:
            source = CsvReplaySource(replay_path, speed=args.speed)
        except SourceError as e:
            print(f"[ERROR] {e}")
            return 2
        source_label = f"replay:{args.replay}"
        auto_calib_at = args.calib_at
        if auto_calib_at is None and not args.calib_file:
            auto_calib_at = 3.0     # 回放默认用前 3 秒做零位标定
    else:
        port = args.port or cfg["serial"]["port"]
        baud = args.baud or cfg["serial"]["baud"]
        source = SerialSource(port, baud)
        source_label = f"serial:{port}"
        auto_calib_at = args.calib_at

    calib_file = resolve_path(args.calib_file) if args.calib_file else None
    if calib_file and not os.path.exists(calib_file):
        print(f"[WARN] 标定文件不存在，将改用自动标定: {calib_file}")
        calib_file = None
        if auto_calib_at is None:
            auto_calib_at = 3.0

    print(f"关节: {info['name']} ({joint_key})   近端={info['proximal_default']} "
          f"远端={info['distal_default']}   采样率={fs:g} Hz")
    print(f"数据源: {source_label}")
    print(f"设备映射: {device_map or '（空 — 将等待 device_alias_map.json）'}")
    print(f"输出目录: {output_root}")
    print(f"采集名称: {args.session_name or joint_key}")
    print("-" * 72)

    # ---- 运行 ----
    q: queue.Queue = queue.Queue()
    worker = JointWorker(
        source, joint_key, output_root, q,
        device_map=device_map,
        calib_dir=calib_dir,
        fs=fs,
        calib_mode=calib_mode,
        calib_dur_s=calib_dur,
        max_sync_dt_ms=max_dt,
        save_raw=save_raw,
        calib_file=calib_file,
        auto_calib_at_s=auto_calib_at,
        source_label=source_label,
        session_name=args.session_name or joint_key,
        kalman_var_gyr=kalman_var_gyr,
        kalman_var_acc=kalman_var_acc,
    )
    worker.start()

    printer = _MessagePrinter(quiet=args.quiet)
    try:
        while worker.is_alive() or not q.empty():
            try:
                msg = q.get(timeout=0.2)
            except queue.Empty:
                continue
            printer.handle(msg)
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断，正在停止...")
        worker.stop()

    worker.join(timeout=10.0)
    print("-" * 72)
    if worker.session_dir:
        print(f"会话数据: {worker.session_dir}")
    return 0


class _MessagePrinter:
    """把 worker 的队列消息打印到控制台；角度行做限速避免刷屏。"""

    PRINT_INTERVAL_S = 0.1

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._last_angle_print = 0.0

    def handle(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "joint_angle":
            if not self.quiet:
                now = time.time()
                if now - self._last_angle_print >= self.PRINT_INTERVAL_S:
                    self._last_angle_print = now
                    print(f"t={msg['sample_time_s']:8.3f}s  "
                          f"Mahony=({msg['flexion_deg']:7.2f},"
                          f"{msg['abduction_deg']:7.2f},"
                          f"{msg['rotation_deg']:7.2f})  "
                          f"EKF=({msg['kalman_flexion_deg']:7.2f},"
                          f"{msg['kalman_abduction_deg']:7.2f},"
                          f"{msg['kalman_rotation_deg']:7.2f})  "
                          f"ROM={msg['rom_deg']:6.2f}/{msg['kalman_rom_deg']:6.2f}  "
                          f"dSync={msg['sync_dt_ms']:.0f}ms")
        elif msg_type == "joint_status":
            if msg.get("state") in ("connecting", "connected", "device_detected",
                                    "identified", "uncalibrated", "waiting_calib",
                                    "disconnected"):
                print(f"[状态] {msg.get('message', '')}")
        elif msg_type == "joint_calib_status":
            print(f"[标定] {msg.get('message', '')}")
        elif msg_type == "joint_calib_done":
            print(f"[标定] {msg.get('message', '')}")
        elif msg_type == "joint_calib_error":
            print(f"[标定失败] {msg.get('message', '')}")
        elif msg_type == "joint_error":
            print(f"[错误] {msg.get('message', '')}")
            if msg.get("traceback"):
                print(msg["traceback"])


if __name__ == "__main__":
    sys.exit(main())
