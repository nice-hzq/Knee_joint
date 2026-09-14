# -*- coding: utf-8 -*-
"""
配置
Configuration — 读取 config.json，并把相对路径解析到项目根目录。

项目根 = knee_joint 包的上一级目录（即 Knee_joint/）。
"""

import json
import os
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

DEFAULTS = {
    "joint_key": "right_knee",
    "fs": 100.0,
    "serial": {"port": "COM8", "baud": 921600},
    "device_map": {},
    "device_map_file": "temp/device_alias_map.json",
    "output_root": "data",
    "calib_dir": "temp",
    "calib": {"mode": "lower_body_standing", "dur_s": 3.0},
    "sync": {"max_dt_ms": 30.0},
    "kalman": {"var_gyr": 0.3 ** 2, "var_acc": 0.5 ** 2},
    "save_raw": True,
}


def project_root() -> str:
    return PROJECT_ROOT


def resolve_path(path: str) -> str:
    """相对路径按项目根解析；绝对路径原样返回。"""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> dict:
    """加载配置文件；文件不存在时返回默认配置。"""
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(cfg_path):
        return json.loads(json.dumps(DEFAULTS))

    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return _merge(DEFAULTS, user_cfg)


def load_device_map(cfg: dict) -> dict:
    """
    解析设备映射：cfg["device_map"] 显式项优先，
    其余从 cfg["device_map_file"] 读取。
    """
    mapping = {}
    map_file = cfg.get("device_map_file")
    if map_file:
        full = resolve_path(map_file)
        if os.path.exists(full):
            try:
                with open(full, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    mapping.update(loaded)
            except Exception as e:
                print(f"[CONFIG] 读取设备映射失败 ({full}): {e}")
    mapping.update(cfg.get("device_map") or {})
    return mapping
