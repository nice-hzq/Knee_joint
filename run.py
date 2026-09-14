# -*- coding: utf-8 -*-
"""
膝关节角度测量 — 图形界面入口

    python run.py

只依赖 numpy / AHRS / pyserial；曲线由 Tkinter Canvas 直接绘制。
"""

import sys


def _check_dependencies():
    """启动前检查依赖，缺什么就明确告诉用户装什么。"""
    missing = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import ahrs  # noqa: F401
    except ImportError:
        missing.append("AHRS")

    try:
        import serial  # noqa: F401
    except ImportError:
        missing.append("pyserial")

    if missing:
        print("[ERROR] 缺少依赖: " + ", ".join(missing), file=sys.stderr)
        print("       请执行:  pip install -r requirements.txt", file=sys.stderr)
        return False
    return True


def main() -> int:
    if not _check_dependencies():
        return 1

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("[ERROR] 当前 Python 未包含 tkinter，无法启动图形界面。\n"
              "       可以改用命令行入口:  python run_cli.py --help",
              file=sys.stderr)
        return 1

    from knee_joint.ui.app import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
