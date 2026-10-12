# -*- coding: utf-8 -*-
"""
主应用类 — 膝关节角度实时测量界面
Knee joint angle measurement UI.

Layout: title bar + left-right split + status bar.
Left  (320px): Connection | Joint Angle
Right (expand): 3-axis numeric display + 3-panel live curve canvas

线程模型：JointWorker 在后台线程采集与解算，通过 queue 把消息交给本类的
`_poll_queue` 消费（Tk 只能在主线程更新控件）。
"""

import os
import queue

from enum import Enum

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from ..config import load_config, resolve_path
from ..sources import SerialSource
from ..worker import JointWorker
from . import (DEFAULT_BAUD, JOINT_OPTIONS, CALIB_MODES, CALIB_STATES,
               get_base_dir)
from .dialogs import load_alias_map, open_joint_device_mapping_dialog

# 停止测量后等待线程退出的兜底时长（毫秒）
STOP_WATCHDOG_MS = 8000


# ============================================================
# Style Constants
# ============================================================
# ── Colors ──
BACKGROUND  = "#F3F6FA"
CARD_BG     = "#FFFFFF"
PRIMARY     = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
SUCCESS     = "#22C55E"
SUCCESS_HOVER = "#16A34A"
SUCCESS_MUTED = "#BBF7D0"   # inactive state
WARNING     = "#F59E0B"
DANGER      = "#EF4444"
DANGER_HOVER = "#DC2626"
DANGER_MUTED = "#FECACA"    # inactive state
PRIMARY_MUTED = "#BFDBFE"   # inactive state
TEXT_MAIN    = "#111827"
TEXT_SECONDARY = "#6B7280"
BORDER       = "#E5E7EB"

# ── Badge Colors ──
BADGE_GREEN_BG   = "#DCFCE7"
BADGE_GREEN_TEXT = "#166534"
BADGE_RED_BG     = "#FEE2E2"
BADGE_RED_TEXT   = "#991B1B"
BADGE_ORANGE_BG  = "#FFF7ED"
BADGE_ORANGE_TEXT = "#9A3412"
BADGE_BLUE_BG    = "#DBEAFE"
BADGE_BLUE_TEXT  = "#1E40AF"

# ── Fonts ──
FONT_TITLE      = ("Microsoft YaHei UI", 15, "bold")
FONT_SUBTITLE   = ("Microsoft YaHei UI", 9)
FONT_SECTION    = ("Microsoft YaHei UI", 11, "bold")
FONT_BODY       = ("Microsoft YaHei UI", 10)
FONT_BODY_BOLD  = ("Microsoft YaHei UI", 10, "bold")
FONT_SMALL      = ("Microsoft YaHei UI", 8)
FONT_MONO       = ("Consolas", 9)
FONT_MONO_SM    = ("Consolas", 8)
FONT_BADGE      = ("Microsoft YaHei UI", 9, "bold")

# ── Dimensions ──
LEFT_PANEL_WIDTH = 320


# ============================================================
# State Machine
# ============================================================
class UIState(Enum):
    IDLE      = "idle"
    MEASURING = "measuring"
    STOPPING  = "stopping"


# ============================================================
# Main Application
# ============================================================
class JointAngleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("膝关节角度测量 — Knee Joint Angle Measurement")
        self.root.geometry("1020x780")
        self.root.minsize(880, 600)
        self.root.configure(bg=BACKGROUND)

        self.cfg = load_config()
        self.queue = queue.Queue(maxsize=200)
        self.joint_thread = None
        self.state = UIState.IDLE
        self._watchdog_id = None

        # ── 关节角度状态 ──
        self._joint_calib_state = "uncalibrated"  # uncalibrated|calibrating|calibrated|failed
        self._joint_flexion_var = tk.StringVar(value="--")
        self._joint_abduction_var = tk.StringVar(value="--")
        self._joint_rotation_var = tk.StringVar(value="--")
        self._joint_max_var = tk.StringVar(value="--")
        self._joint_min_var = tk.StringVar(value="--")
        self._joint_rom_var = tk.StringVar(value="--")
        self._kalman_flexion_var = tk.StringVar(value="--")
        self._kalman_abduction_var = tk.StringVar(value="--")
        self._kalman_rotation_var = tk.StringVar(value="--")
        self._kalman_rom_var = tk.StringVar(value="--")
        self._joint_history_t = []
        self._joint_history_flexion = []
        self._joint_history_abduction = []
        self._joint_history_rotation = []
        self._kalman_history_t = []
        self._kalman_history_flexion = []
        self._kalman_history_abduction = []
        self._kalman_history_rotation = []

        # ── 关节段标签 ──
        self._joint_prox_label_var = tk.StringVar(value="大腿 (Thigh)")
        self._joint_dist_label_var = tk.StringVar(value="小腿 (Shank)")
        self._joint_prox_var = tk.StringVar(value="R4")
        self._joint_dist_var = tk.StringVar(value="R5")

        # ── 设备映射 ──
        self._joint_prox_device_var = tk.StringVar(value="—")
        self._joint_dist_device_var = tk.StringVar(value="—")
        self._joint_device_map = {}  # {device_id: alias} 缓存

        # ── 标题栏 ──
        self.title_status_var = tk.StringVar(value="● 就绪")
        self.title_info_var = tk.StringVar(value="")

        self._build_ui()
        # 先加载已保存的设备映射，再触发一次关节切换，
        # 让卡片一打开就能显示 <设备ID → 别名>，而不是清一色的「—」。
        self._load_joint_device_map()
        self.joint_var.set(self._default_joint_key())
        self._init_button_states()
        self._poll_queue()

    # ============================================================
    # UI Helpers
    # ============================================================
    def _default_joint_key(self) -> str:
        key = self.cfg.get("joint_key", "right_knee")
        return key if key in JOINT_OPTIONS else next(iter(JOINT_OPTIONS))

    def _create_card(self, parent, title=None, collapsible=False,
                     start_collapsed=False, **pack_opts):
        """创建白色卡片 Frame，返回 (card, content) —— 子控件一律放进 content。"""
        card = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER,
                        highlightthickness=1, bd=0)
        card.pack(fill="x", pady=(0, 8), **pack_opts)

        content = tk.Frame(card, bg=CARD_BG)

        if title and collapsible:
            title_frame = tk.Frame(card, bg=CARD_BG, cursor="hand2")
            title_frame.pack(fill="x")

            indicator = "▶  " if start_collapsed else "▼  "
            title_lbl = tk.Label(title_frame, text=f"{indicator}{title}",
                                 font=FONT_SECTION, bg=CARD_BG, fg=TEXT_MAIN,
                                 anchor="w")
            title_lbl.pack(fill="x", padx=12, pady=(10, 6))

            card._collapsed = start_collapsed

            def _toggle(e=None):
                if card._collapsed:
                    content.pack(fill="x")
                    title_lbl.config(text=f"▼  {title}")
                    card._collapsed = False
                else:
                    content.pack_forget()
                    title_lbl.config(text=f"▶  {title}")
                    card._collapsed = True

            title_lbl.bind("<Button-1>", _toggle)
            title_frame.bind("<Button-1>", _toggle)

            if not start_collapsed:
                content.pack(fill="x")
        elif title:
            tk.Label(card, text=title, font=FONT_SECTION,
                     bg=CARD_BG, fg=TEXT_MAIN, anchor="w").pack(
                fill="x", padx=12, pady=(10, 6))
            content.pack(fill="x")
        else:
            content.pack(fill="x")

        return card, content

    def _make_action_btn(self, parent, text, command, bg_color, hover_color,
                         width=14):
        """彩色主操作按钮；视觉状态由 _set_action_btn_state 管理。"""
        return tk.Button(parent, text=text, width=width,
                         font=FONT_BODY, bg=bg_color, fg="white",
                         activebackground=hover_color, activeforeground="white",
                         bd=0, padx=12, pady=5,
                         relief="flat", cursor="hand2",
                         command=command)

    def _set_action_btn_state(self, btn, active, active_text, inactive_text,
                              active_bg, muted_bg, muted_fg="#6B7280"):
        """在「可用（实色）」与「不可用（浅色）」之间切换按钮外观。

        刻意不使用 Tk 的 disabled 状态，避免灰化后文字难以辨认。
        """
        if active:
            btn.configure(text=active_text, bg=active_bg, fg="white",
                          activebackground=active_bg, cursor="hand2")
        else:
            btn.configure(text=inactive_text, bg=muted_bg, fg=muted_fg,
                          activebackground=muted_bg, cursor="arrow")

    def _make_secondary_btn(self, parent, text, command, width=12, font=None,
                            state="normal"):
        """白底描边的次要按钮。"""
        if font is None:
            font = FONT_BODY
        return tk.Button(parent, text=text, width=width,
                         font=font, bg=CARD_BG, fg=TEXT_MAIN,
                         activebackground="#F9FAFB",
                         bd=0, padx=8, pady=3,
                         relief="flat", cursor="hand2",
                         highlightbackground=BORDER, highlightthickness=1,
                         command=command, state=state)

    def _make_small_btn(self, parent, text, command, width=10, state="normal"):
        """小号次要按钮。"""
        return self._make_secondary_btn(parent, text, command,
                                        width=width, font=FONT_SMALL, state=state)

    def _update_title_status(self, text, badge_bg, badge_fg):
        self.title_status_var.set(text)
        self.title_status_badge.configure(bg=badge_bg, fg=badge_fg)

    # ============================================================
    # UI Build
    # ============================================================
    def _build_ui(self):
        self._build_title_bar()

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg=BACKGROUND, bd=0, sashwidth=2,
                               sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left_frame = tk.Frame(paned, bg=BACKGROUND, width=LEFT_PANEL_WIDTH,
                              height=100)
        paned.add(left_frame, minsize=280)

        right_frame = tk.Frame(paned, bg=BACKGROUND)
        paned.add(right_frame)

        self._build_connection_card(left_frame)
        self._build_joint_card(left_frame)

        self._build_right_panel(right_frame)

        self._build_status_bar()

    def _build_title_bar(self):
        """顶部标题栏：系统名 + 实时信息 + 状态徽标。"""
        title_bar = tk.Frame(self.root, bg=CARD_BG, highlightbackground=BORDER,
                             highlightthickness=1, bd=0)
        title_bar.pack(fill="x", padx=0, pady=0)

        title_inner = tk.Frame(title_bar, bg=CARD_BG)
        title_inner.pack(fill="x", padx=16, pady=(10, 6))

        title_col = tk.Frame(title_inner, bg=CARD_BG)
        title_col.pack(side="left")
        tk.Label(title_col, text="膝关节角度测量", font=FONT_TITLE,
                 bg=CARD_BG, fg=TEXT_MAIN).pack(anchor="w")
        tk.Label(title_col, text="Knee Joint Angle Measurement System",
                 font=FONT_SUBTITLE, bg=CARD_BG,
                 fg=TEXT_SECONDARY).pack(anchor="w")

        self.title_info_lbl = tk.Label(
            title_inner, textvariable=self.title_info_var,
            font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SECONDARY)
        self.title_info_lbl.pack(side="right", padx=(0, 12))

        self.title_status_badge = tk.Label(
            title_inner, textvariable=self.title_status_var,
            font=FONT_BADGE, bg="#F3F4F6", fg="#374151",
            padx=12, pady=4)
        self.title_status_badge.pack(side="right")

    def _build_connection_card(self, parent):
        """串口连接卡片：端口枚举 + 波特率。"""
        card, content = self._create_card(parent, title="🔌 连接 (Connection)")

        conn_row = tk.Frame(content, bg=CARD_BG)
        conn_row.pack(fill="x", padx=12, pady=(4, 8))

        tk.Label(conn_row, text="串口", font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=4, anchor="w").pack(side="left")

        default_port, available_ports = self._list_ports()

        self.port_var = tk.StringVar(value=default_port)
        self.port_cb = ttk.Combobox(conn_row, textvariable=self.port_var,
                                    values=available_ports, state="readonly",
                                    width=14, font=FONT_MONO)
        self.port_cb.pack(side="left", padx=(2, 2))

        def _refresh_ports():
            _, ports = self._list_ports()
            self.port_cb["values"] = ports
            if ports and self.port_var.get() not in ports:
                self.port_var.set(ports[0])

        self._make_small_btn(conn_row, "↻", _refresh_ports, width=3).pack(
            side="left", padx=(0, 12))

        tk.Label(conn_row, text="波特率", font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=5, anchor="w").pack(side="left")
        self.baud_var = tk.StringVar(
            value=str(self.cfg.get("serial", {}).get("baud") or DEFAULT_BAUD))
        tk.Entry(conn_row, textvariable=self.baud_var, width=8,
                 font=FONT_MONO, bg=CARD_BG, fg=TEXT_MAIN,
                 relief="solid", highlightbackground=BORDER,
                 highlightthickness=1, bd=0).pack(side="left", padx=(2, 0))

    def _list_ports(self):
        """返回 (默认端口, 全部端口列表)。无 pyserial 时返回 ("", [])。"""
        if not serial:
            return "", []
        ports = [p.device for p in serial.tools.list_ports.comports()]
        preferred = ""
        for p in serial.tools.list_ports.comports():
            desc = p.description or ""
            if "CH340" in desc or "CP210" in desc or "USB" in desc:
                preferred = p.device
                break
        return (preferred or (ports[0] if ports else "")), ports

    def _build_joint_card(self, parent):
        """实时关节角度控制卡片。"""
        card, content = self._create_card(parent, title="🦵 实时关节角度 (Joint Angle)",
                                          collapsible=True, start_collapsed=False)

        # ── 关节选择 ──
        sel_row = tk.Frame(content, bg=CARD_BG)
        sel_row.pack(fill="x", padx=12, pady=(4, 4))
        tk.Label(sel_row, text="关节", font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=4, anchor="w").pack(side="left")
        self.joint_var = tk.StringVar(value=self._default_joint_key())
        self.joint_cb = ttk.Combobox(sel_row, textvariable=self.joint_var,
                                     values=list(JOINT_OPTIONS.keys()),
                                     state="readonly", width=18, font=FONT_BODY)
        self.joint_cb.pack(side="left", padx=2)

        def _on_joint_change(*args):
            info = JOINT_OPTIONS.get(self.joint_var.get(), {})
            self._joint_prox_label_var.set(f"{info.get('prox_label', '?')}")
            self._joint_dist_label_var.set(f"{info.get('dist_label', '?')}")
            self._joint_prox_var.set(f"{info.get('proximal', '?')}")
            self._joint_dist_var.set(f"{info.get('distal', '?')}")
            # 重置设备 ID 显示并尝试从已保存映射中解析
            self._joint_prox_device_var.set("—")
            self._joint_dist_device_var.set("—")
            self._update_joint_device_display()
        self.joint_var.trace_add("write", _on_joint_change)

        # ── 本次采集名称（用于原始数据与角度数据的共同会话目录）──
        name_row = tk.Frame(content, bg=CARD_BG)
        name_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(name_row, text="名称", font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=4, anchor="w").pack(side="left")
        self.session_name_var = tk.StringVar(value="")
        self.session_name_entry = tk.Entry(
            name_row, textvariable=self.session_name_var, width=20,
            font=FONT_BODY, bg=CARD_BG, fg=TEXT_MAIN,
            relief="solid", highlightbackground=BORDER,
            highlightthickness=1, bd=0)
        self.session_name_entry.pack(side="left", padx=2, ipady=2)

        # ── 传感器绑定显示（段标签 + 别名 + 设备 ID）──
        prox_row = tk.Frame(content, bg=CARD_BG)
        prox_row.pack(fill="x", padx=12, pady=(2, 2))
        tk.Label(prox_row, textvariable=self._joint_prox_label_var,
                 font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SECONDARY,
                 width=12, anchor="w").pack(side="left")
        tk.Label(prox_row, textvariable=self._joint_prox_var,
                 font=FONT_BODY_BOLD, bg=CARD_BG, fg=PRIMARY).pack(
            side="left", padx=(2, 4))
        tk.Label(prox_row, text="→", font=FONT_SMALL, bg=CARD_BG,
                 fg=TEXT_SECONDARY).pack(side="left", padx=(0, 4))
        tk.Label(prox_row, textvariable=self._joint_prox_device_var,
                 font=FONT_MONO_SM, bg=CARD_BG, fg=TEXT_MAIN).pack(side="left")

        dist_row = tk.Frame(content, bg=CARD_BG)
        dist_row.pack(fill="x", padx=12, pady=(2, 2))
        tk.Label(dist_row, textvariable=self._joint_dist_label_var,
                 font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SECONDARY,
                 width=12, anchor="w").pack(side="left")
        tk.Label(dist_row, textvariable=self._joint_dist_var,
                 font=FONT_BODY_BOLD, bg=CARD_BG, fg=PRIMARY).pack(
            side="left", padx=(2, 4))
        tk.Label(dist_row, text="→", font=FONT_SMALL, bg=CARD_BG,
                 fg=TEXT_SECONDARY).pack(side="left", padx=(0, 4))
        tk.Label(dist_row, textvariable=self._joint_dist_device_var,
                 font=FONT_MONO_SM, bg=CARD_BG, fg=TEXT_MAIN).pack(side="left")

        # ── 设备映射配置 ──
        cfg_row = tk.Frame(content, bg=CARD_BG)
        cfg_row.pack(fill="x", padx=12, pady=(4, 4))
        self.btn_joint_device_map = self._make_small_btn(
            cfg_row, "⚙ 配置设备映射", self._open_joint_device_mapping, width=18)
        self.btn_joint_device_map.pack(side="left")

        # ── 标定模式 ──
        cal_row = tk.Frame(content, bg=CARD_BG)
        cal_row.pack(fill="x", padx=12, pady=(2, 2))
        tk.Label(cal_row, text="校准", font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=4, anchor="w").pack(side="left")
        self.calib_mode_var = tk.StringVar(value="lower_body_standing")
        self.calib_mode_cb = ttk.Combobox(cal_row, textvariable=self.calib_mode_var,
                                          values=list(CALIB_MODES.keys()),
                                          state="readonly", width=16,
                                          font=FONT_BODY)
        self.calib_mode_cb.pack(side="left", padx=2)

        # ── 控制按钮 ──
        btn_row = tk.Frame(content, bg=CARD_BG)
        btn_row.pack(fill="x", padx=12, pady=(4, 2))

        self.btn_joint_start = self._make_action_btn(
            btn_row, "▶  开始测量", self._start_joint, SUCCESS, SUCCESS_HOVER,
            width=11)
        self.btn_joint_start.pack(side="left", padx=(0, 4))

        self.btn_joint_stop = self._make_action_btn(
            btn_row, "■  停止测量", self._stop_joint, DANGER, DANGER_HOVER,
            width=11)
        self.btn_joint_stop.pack(side="left", padx=(0, 4))

        self.btn_joint_calib = self._make_secondary_btn(
            btn_row, "○ 开始校准", self._start_joint_calib, width=10)
        self.btn_joint_calib.pack(side="left")

        # ── 标定状态 ──
        self.joint_calib_status_var = tk.StringVar(value="未标定")
        tk.Label(content, textvariable=self.joint_calib_status_var,
                 font=FONT_SMALL, bg=CARD_BG, fg=WARNING,
                 anchor="w").pack(fill="x", padx=12, pady=(2, 0))

        # ── 测量状态 ──
        self.joint_status_var = tk.StringVar(value="● 未启动")
        tk.Label(content, textvariable=self.joint_status_var,
                 font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SECONDARY,
                 anchor="w", wraplength=LEFT_PANEL_WIDTH - 40,
                 justify="left").pack(fill="x", padx=12, pady=(0, 6))

    def _build_right_panel(self, parent):
        """右侧：三轴数值卡 + 三面板实时曲线。"""
        val_card = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER,
                            highlightthickness=1, bd=0)
        val_card.pack(fill="x", pady=(0, 6))

        axes_info = [
            ("屈曲/伸展\nFlexion", self._joint_flexion_var,
             self._kalman_flexion_var, "#E74C3C"),
            ("外展/内收\nAbduction", self._joint_abduction_var,
             self._kalman_abduction_var, "#2ECC71"),
            ("内旋/外旋\nRotation", self._joint_rotation_var,
             self._kalman_rotation_var, "#3498DB"),
            ("活动范围\nROM", self._joint_rom_var,
             self._kalman_rom_var, PRIMARY),
        ]
        for label, mahony_var, kalman_var, color in axes_info:
            frm = tk.Frame(val_card, bg=CARD_BG)
            frm.pack(side="left", padx=10, pady=6)
            tk.Label(frm, text=label, font=FONT_SMALL, bg=CARD_BG,
                     fg=TEXT_SECONDARY, justify="center").pack()
            for prefix, var, font in (("M", mahony_var, ("Consolas", 17, "bold")),
                                      ("K", kalman_var, ("Consolas", 15, "bold"))):
                val_row = tk.Frame(frm, bg=CARD_BG)
                val_row.pack()
                tk.Label(val_row, text=prefix, font=("Consolas", 8, "bold"),
                         bg=CARD_BG, fg=TEXT_SECONDARY).pack(side="left", padx=(0, 2))
                tk.Label(val_row, textvariable=var, font=font, bg=CARD_BG,
                         fg=color).pack(side="left")
                tk.Label(val_row, text="°", font=("Consolas", 10), bg=CARD_BG,
                         fg=TEXT_SECONDARY).pack(side="left")

        # ── 三面板曲线画布 ──
        curve_outer = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER,
                               highlightthickness=1, bd=0)
        curve_outer.pack(fill="both", expand=True)

        self.joint_canvas_flexion = self._make_curve_canvas(
            curve_outer, "Flexion (屈曲/伸展)  M=实线 / K=EKF虚线")
        self.joint_canvas_abduction = self._make_curve_canvas(
            curve_outer, "Abduction (外展/内收)  M=实线 / K=EKF虚线")
        self.joint_canvas_rotation = self._make_curve_canvas(
            curve_outer, "Rotation (内旋/外旋)  M=实线 / K=EKF虚线")

    def _make_curve_canvas(self, parent, title):
        """创建一块带标题的曲线面板，返回 Canvas。"""
        panel = tk.Frame(parent, bg=CARD_BG)
        panel.pack(fill="both", expand=True, padx=2, pady=(2, 1))
        tk.Label(panel, text=title, font=FONT_SMALL, bg=CARD_BG,
                 fg=TEXT_SECONDARY, anchor="w").pack(padx=6, pady=(2, 0))
        canvas = tk.Canvas(panel, bg="#FAFBFC", height=170, highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        return canvas

    def _build_status_bar(self):
        """底部状态栏。"""
        status_bar = tk.Frame(self.root, bg=CARD_BG, highlightbackground=BORDER,
                              highlightthickness=1, bd=0)
        status_bar.pack(fill="x", side="bottom")

        self.status_bar_var = tk.StringVar(value="就绪 (Ready)")
        tk.Label(status_bar, textvariable=self.status_bar_var,
                 anchor="w", bg=CARD_BG, fg=TEXT_SECONDARY,
                 font=FONT_SMALL, padx=12, pady=3).pack(
            side="left", fill="x", expand=True)

        self.status_progress = ttk.Progressbar(
            status_bar, mode="indeterminate", length=120)
        # 默认隐藏，仅在 STOPPING 时显示

    # ============================================================
    # Curve Drawing
    # ============================================================
    def _draw_one_curve(self, canvas, t_list, deg_list, color,
                        secondary=None, y_label="°"):
        """绘制 Mahony 实线与可选的 EKF 虚线。"""
        if not t_list or not deg_list or len(deg_list) < 2:
            canvas.delete("all")
            canvas.create_text(200, 85, text="等待数据...", font=FONT_SMALL,
                               fill=TEXT_SECONDARY)
            return

        w = canvas.winfo_width() or 600
        h = canvas.winfo_height() or 170
        pad_l, pad_r, pad_t, pad_b = 50, 20, 12, 28
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b
        if pw < 50 or ph < 20:
            return

        t0 = t_list[0]
        t_rel = [t - t0 for t in t_list]
        vals = deg_list
        secondary = secondary or []
        scale_vals = list(vals) + list(secondary)
        y_min = min(scale_vals) - 2
        y_max = max(scale_vals) + 2
        if y_max - y_min < 10:
            mid = (y_max + y_min) / 2
            y_min = mid - 5
            y_max = mid + 5
        t_max = max(t_rel) or 1

        def tx(t):
            return pad_l + (t / t_max) * pw

        def ty(v):
            return pad_t + (1.0 - (v - y_min) / (y_max - y_min)) * ph

        canvas.delete("all")

        # 水平网格
        for i in range(5):
            v = y_min + (y_max - y_min) * i / 4
            y = ty(v)
            canvas.create_line(pad_l, y, w - pad_r, y, fill="#E5E7EB", dash=(2, 3))
            canvas.create_text(pad_l - 4, y, text=f"{v:.0f}", anchor="e",
                               font=("Consolas", 7), fill=TEXT_SECONDARY)
        # 零线
        if y_min < 0 < y_max:
            canvas.create_line(pad_l, ty(0), w - pad_r, ty(0),
                               fill="#9CA3AF", dash=(3, 2))

        # 垂直网格
        for i in range(5):
            t_v = t_max * i / 4
            x = tx(t_v)
            canvas.create_line(x, pad_t, x, h - pad_b, fill="#E5E7EB", dash=(2, 3))
        # X 轴标签
        canvas.create_text(w - pad_r + 4, h - pad_b + 4, text=f"{t_max:.0f}s",
                           anchor="ne", font=("Consolas", 7), fill=TEXT_SECONDARY)

        # 折线
        pts = []
        for j in range(len(vals)):
            pts.extend([tx(t_rel[j]), ty(vals[j])])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=color, width=2, smooth=True)

        if len(secondary) >= 2:
            secondary_pts = []
            count = min(len(t_rel), len(secondary))
            for j in range(count):
                secondary_pts.extend([tx(t_rel[j]), ty(secondary[j])])
            canvas.create_line(*secondary_pts, fill=color, width=2,
                               dash=(6, 3), smooth=True)

        # 图例
        legend = f"M {vals[-1]:.1f}°"
        if secondary:
            legend += f"   K {secondary[-1]:.1f}°"
        canvas.create_text(w - pad_r, pad_t + 2, text=legend,
                           anchor="ne", font=("Consolas", 9, "bold"), fill=color)

    def _redraw_joint_curves(self):
        """重绘三块角度曲线面板。"""
        t = self._joint_history_t
        if not t or len(t) < 2:
            return
        self._draw_one_curve(self.joint_canvas_flexion, t,
                             self._joint_history_flexion, "#E74C3C",
                             self._kalman_history_flexion)
        self._draw_one_curve(self.joint_canvas_abduction, t,
                             self._joint_history_abduction, "#2ECC71",
                             self._kalman_history_abduction)
        self._draw_one_curve(self.joint_canvas_rotation, t,
                             self._joint_history_rotation, "#3498DB",
                             self._kalman_history_rotation)

    # ============================================================
    # State Transitions
    # ============================================================
    def _init_button_states(self):
        """按钮初始外观（IDLE 默认值）。"""
        self._set_action_btn_state(self.btn_joint_start, True,
            "▶  开始测量", "▶  开始测量", SUCCESS, SUCCESS_MUTED)
        self._set_action_btn_state(self.btn_joint_stop, False,
            "■  停止测量", "■  停止测量", DANGER, DANGER_MUTED)
        self.btn_joint_device_map.config(state="normal")

    def _transition_to(self, new_state: UIState):
        self.state = new_state

        if new_state == UIState.IDLE:
            self.title_info_var.set("")
            self.status_progress.stop()
            self.status_progress.pack_forget()
            self._set_action_btn_state(self.btn_joint_start, True,
                "▶  开始测量", "▶  开始测量", SUCCESS, SUCCESS_MUTED)
            self._set_action_btn_state(self.btn_joint_stop, False,
                "■  停止测量", "■  停止测量", DANGER, DANGER_MUTED)
            self.btn_joint_device_map.config(state="normal")
            self.joint_cb.config(state="readonly")
            self.calib_mode_cb.config(state="readonly")
            self.session_name_entry.config(state="normal")

        elif new_state == UIState.MEASURING:
            self._set_action_btn_state(self.btn_joint_start, False,
                "▶  开始测量", "● 测量中", SUCCESS, SUCCESS_MUTED)
            self._set_action_btn_state(self.btn_joint_stop, True,
                "■  停止测量", "■  停止测量", DANGER, DANGER_MUTED)
            self.btn_joint_device_map.config(state="disabled")
            self.joint_cb.config(state="disabled")
            self.calib_mode_cb.config(state="disabled")
            self.session_name_entry.config(state="disabled")
            self._update_title_status("● 测量中", BADGE_BLUE_BG, BADGE_BLUE_TEXT)

        elif new_state == UIState.STOPPING:
            self._set_action_btn_state(self.btn_joint_start, False,
                "▶  开始测量", "● 测量中", SUCCESS, SUCCESS_MUTED)
            self._set_action_btn_state(self.btn_joint_stop, False,
                "■  停止测量", "◌ 停止中", DANGER, DANGER_MUTED)
            self.btn_joint_device_map.config(state="disabled")
            self.session_name_entry.config(state="disabled")
            self._update_title_status("◌ 停止中", BADGE_ORANGE_BG, BADGE_ORANGE_TEXT)
            self.status_progress.pack(side="right", padx=(0, 12), pady=3)
            self.status_progress.start(10)

    # ============================================================
    # Joint Angle Actions
    # ============================================================
    def _start_joint(self):
        """启动实时关节角度测量。"""
        if self.state != UIState.IDLE:
            messagebox.showwarning("操作冲突", "请先停止当前测量再重新启动。")
            return
        session_name = self.session_name_var.get().strip()
        if not session_name:
            messagebox.showwarning("输入错误", "请输入本次采集名称，例如：深蹲测试01")
            self.session_name_entry.focus_set()
            return
        if serial is None:
            messagebox.showerror("缺少依赖",
                                 "未安装 pyserial，无法使用串口采集。\n\n"
                                 "请执行:  pip install pyserial")
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("输入错误", "请选择串口")
            return
        try:
            baud = int(self.baud_var.get().strip())
        except ValueError:
            baud = DEFAULT_BAUD

        joint_key = self.joint_var.get()
        info = JOINT_OPTIONS.get(joint_key) or JOINT_OPTIONS[self._default_joint_key()]

        # 必须先明确映射近端/远端设备，不能按不可预测的上线顺序猜测。
        self._load_joint_device_map()
        self._update_joint_device_display()
        prox_alias = info["proximal"]
        dist_alias = info["distal"]
        prox_ids = [did for did, alias in self._joint_device_map.items()
                    if alias == prox_alias]
        dist_ids = [did for did, alias in self._joint_device_map.items()
                    if alias == dist_alias]
        if len(prox_ids) != 1 or len(dist_ids) != 1 or prox_ids[0] == dist_ids[0]:
            messagebox.showerror(
                "设备映射不完整",
                f"开始 {info['label']} 前，请在「配置设备映射」中分别指定且唯一指定\n"
                f"{prox_alias}（{info['prox_label']}）和 {dist_alias}（{info['dist_label']}）。")
            return

        self._transition_to(UIState.MEASURING)
        self.joint_status_var.set(f"● 连接中... {port}")
        self.status_bar_var.set(f"关节角度测量启动中... 采集名称: {session_name}")
        self._joint_reset_display()
        self._joint_calib_state = "uncalibrated"
        self._update_joint_calib_status()

        calib_dir = os.path.join(get_base_dir(), "temp")
        output_root = resolve_path(self.cfg.get("output_root", "data"))
        cfg_calib = self.cfg.get("calib") or {}
        cfg_sync = self.cfg.get("sync") or {}
        cfg_kalman = self.cfg.get("kalman") or {}

        source = SerialSource(port, baud)
        self.joint_thread = JointWorker(
            source, joint_key, output_root, self.queue,
            device_map=self._joint_device_map,
            calib_dir=calib_dir,
            fs=float(self.cfg.get("fs", 100.0)),
            calib_mode=self.calib_mode_var.get(),
            calib_dur_s=float(cfg_calib.get("dur_s", 3.0)),
            max_sync_dt_ms=float(cfg_sync.get("max_dt_ms", 30.0)),
            save_raw=bool(self.cfg.get("save_raw", True)),
            source_label=f"serial:{port}",
            session_name=session_name,
            kalman_var_gyr=float(cfg_kalman.get("var_gyr", 0.3 ** 2)),
            kalman_var_acc=float(cfg_kalman.get("var_acc", 0.5 ** 2)))
        self.joint_thread.start()

    def _stop_joint(self):
        """请求停止测量，并启动看门狗兜底。"""
        if self.joint_thread is None:
            return
        self.joint_thread.stop()
        self.joint_status_var.set("● 正在停止...")
        self.status_bar_var.set("正在停止关节测量...")
        self._transition_to(UIState.STOPPING)
        self._arm_stop_watchdog()

    def _start_joint_calib(self):
        """请求一次标定（线程必须已在运行）。"""
        if self.joint_thread is None or not self.joint_thread.is_alive():
            messagebox.showwarning("未启动",
                                   "请先启动关节角度测量，等待初始化完成后再校准。")
            self._joint_calib_state = "uncalibrated"
            self._update_joint_calib_status()
            return
        self._joint_calib_state = "calibrating"
        self._update_joint_calib_status()
        mode = self.calib_mode_var.get()
        self.joint_thread.start_calibration(mode)
        self.status_bar_var.set(f"正在标定 ({mode})，请保持静止站立...")

    def _joint_reset_display(self):
        """清空数值与曲线历史。"""
        self._joint_flexion_var.set("--")
        self._joint_abduction_var.set("--")
        self._joint_rotation_var.set("--")
        self._joint_max_var.set("--")
        self._joint_min_var.set("--")
        self._joint_rom_var.set("--")
        self._kalman_flexion_var.set("--")
        self._kalman_abduction_var.set("--")
        self._kalman_rotation_var.set("--")
        self._kalman_rom_var.set("--")
        self._joint_history_t = []
        self._joint_history_flexion = []
        self._joint_history_abduction = []
        self._joint_history_rotation = []
        self._kalman_history_t = []
        self._kalman_history_flexion = []
        self._kalman_history_abduction = []
        self._kalman_history_rotation = []
        self._redraw_joint_curves()
        for canvas in (self.joint_canvas_flexion, self.joint_canvas_abduction,
                       self.joint_canvas_rotation):
            canvas.delete("all")
            canvas.create_text(200, 85, text="等待数据...", font=FONT_SMALL,
                               fill=TEXT_SECONDARY)

    def _update_joint_calib_status(self):
        state = self._joint_calib_state
        color = {"calibrated": SUCCESS, "failed": DANGER}.get(state, WARNING)
        self.joint_calib_status_var.set(CALIB_STATES.get(state, state))
        self.btn_joint_calib.config(
            state="disabled" if state == "calibrating" else "normal")

    # ── 设备映射 ──
    def _joint_calib_dir(self) -> str:
        return os.path.join(get_base_dir(), "temp")

    def _load_joint_device_map(self):
        """从 temp/device_alias_map.json 加载映射并缓存。"""
        self._joint_device_map = load_alias_map(self._joint_calib_dir())
        return self._joint_device_map

    def _update_joint_device_display(self):
        """根据映射反查当前关节的近端/远端设备 ID 并更新显示。"""
        dm = self._joint_device_map
        self._joint_prox_device_var.set(
            next((d for d, a in dm.items() if a == self._joint_prox_var.get()), "—"))
        self._joint_dist_device_var.set(
            next((d for d, a in dm.items() if a == self._joint_dist_var.get()), "—"))

    def _open_joint_device_mapping(self):
        """打开设备映射配置对话框。"""
        if self.state != UIState.IDLE:
            messagebox.showwarning("操作冲突", "请先停止当前测量再配置设备映射。")
            return

        def _on_mapping_saved():
            self._load_joint_device_map()
            self._update_joint_device_display()

        open_joint_device_mapping_dialog(
            self.root, self._joint_calib_dir(), self.joint_var.get(),
            on_save_callback=_on_mapping_saved)

    def _on_joint_done(self):
        """测量线程结束后的收尾。"""
        self._cancel_stop_watchdog()
        session_dir = None
        if self.joint_thread is not None:
            session_dir = getattr(self.joint_thread, "session_dir", None)
        self.joint_thread = None
        self.joint_status_var.set("● 已停止")
        self.status_bar_var.set(
            f"关节角度测量已停止 — 数据: {session_dir}" if session_dir
            else "关节角度测量已停止")
        self._update_title_status("● 就绪", BADGE_GREEN_BG, BADGE_GREEN_TEXT)
        self._transition_to(UIState.IDLE)

    # ── 停止看门狗 ──
    def _arm_stop_watchdog(self, timeout_ms: int = STOP_WATCHDOG_MS):
        """线程迟迟不退出时，兜底把界面恢复可用。"""
        self._cancel_stop_watchdog()
        self._watchdog_id = self.root.after(timeout_ms, self._on_stop_watchdog)

    def _cancel_stop_watchdog(self):
        if self._watchdog_id is not None:
            try:
                self.root.after_cancel(self._watchdog_id)
            except Exception:
                pass
            self._watchdog_id = None

    def _on_stop_watchdog(self):
        self._watchdog_id = None
        if self.joint_thread is not None and self.joint_thread.is_alive():
            messagebox.showwarning(
                "停止超时",
                f"测量线程未在 {STOP_WATCHDOG_MS // 1000} 秒内退出，已强制释放界面。\n"
                f"若串口被占用，请检查设备连接后重试。")
        self._on_joint_done()

    def _on_joint_angle_data(self, data):
        """收到一帧角度数据：更新数值 + 曲线 + 状态。"""
        flex = data.get("flexion_deg", 0)
        abd = data.get("abduction_deg", 0)
        rot = data.get("rotation_deg", 0)
        self._joint_flexion_var.set(f"{flex:.1f}")
        self._joint_abduction_var.set(f"{abd:.1f}")
        self._joint_rotation_var.set(f"{rot:.1f}")
        self._joint_max_var.set(f"{data.get('max_deg', 0):.1f}")
        self._joint_min_var.set(f"{data.get('min_deg', 0):.1f}")
        self._joint_rom_var.set(f"{data.get('rom_deg', 0):.1f}")
        self._kalman_flexion_var.set(
            f"{data.get('kalman_flexion_deg', 0):.1f}")
        self._kalman_abduction_var.set(
            f"{data.get('kalman_abduction_deg', 0):.1f}")
        self._kalman_rotation_var.set(
            f"{data.get('kalman_rotation_deg', 0):.1f}")
        self._kalman_rom_var.set(f"{data.get('kalman_rom_deg', 0):.1f}")

        self._joint_history_t = data.get("history_t", [])
        self._joint_history_flexion = data.get("history_flexion", [])
        self._joint_history_abduction = data.get("history_abduction", [])
        self._joint_history_rotation = data.get("history_rotation", [])
        self._kalman_history_t = data.get("kalman_history_t", [])
        self._kalman_history_flexion = data.get("kalman_history_flexion", [])
        self._kalman_history_abduction = data.get("kalman_history_abduction", [])
        self._kalman_history_rotation = data.get("kalman_history_rotation", [])
        self._redraw_joint_curves()

        cal = "✓" if data.get("calibrated") else "✗"
        init = "✓" if data.get("initialized") else "⏳"
        sync_dt = data.get("sync_dt_ms")
        sync_text = f"  同步差:{sync_dt:.0f}ms" if sync_dt is not None else ""
        k_flex = data.get("kalman_flexion_deg", 0)
        self.joint_status_var.set(
            f"● 测量中 | 校准:{cal} 初始化:{init} | "
            f"屈伸 M:{flex:.1f}° K:{k_flex:.1f}°{sync_text}")

    # ============================================================
    # Queue Polling
    # ============================================================
    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                mt = msg.get("type")

                if mt == "joint_status":
                    state = msg.get("state", "")
                    if state == "connected":
                        self.joint_status_var.set(
                            f"● 已连接  |  {msg.get('message', '')}")
                    elif state == "calibrated":
                        self._joint_calib_state = "calibrated"
                        self._update_joint_calib_status()
                    elif state == "uncalibrated":
                        self._joint_calib_state = "uncalibrated"
                        self._update_joint_calib_status()
                    elif state == "disconnected":
                        self.joint_status_var.set(f"✗ {msg.get('message', '')}")
                        self._on_joint_done()
                    elif state == "device_detected":
                        # 新设备上线 — 刷新映射显示
                        self._load_joint_device_map()
                        self._update_joint_device_display()
                    elif state == "identified":
                        # 传感器已识别 — 用结构化字段直接更新设备 ID
                        prox_id = msg.get("proximal_id", "")
                        dist_id = msg.get("distal_id", "")
                        if prox_id:
                            self._joint_prox_device_var.set(prox_id)
                        if dist_id:
                            self._joint_dist_device_var.set(dist_id)
                    elif state == "waiting_calib":
                        self.joint_status_var.set(f"⚠ {msg.get('message', '')}")
                    elif state == "devices":
                        self._update_joint_device_display()
                    self.status_bar_var.set(msg.get("message", ""))

                elif mt == "joint_angle":
                    self._on_joint_angle_data(msg)

                elif mt == "joint_calib_status":
                    self.joint_status_var.set(f"◌ {msg.get('message', '')}")

                elif mt == "joint_calib_done":
                    self._joint_calib_state = "calibrated"
                    self._joint_reset_display()
                    self._update_joint_calib_status()
                    euler0 = msg.get("euler_rel_0")
                    if euler0 and len(euler0) >= 3:
                        diag = (f"标定完成 ✓  |  Roll(屈伸)={euler0[0]:.1f}°  "
                                f"Pitch(外展)={euler0[1]:.1f}°  Yaw(旋转)={euler0[2]:.1f}°")
                        self.joint_status_var.set(diag)
                    self.status_bar_var.set(msg.get("message", "标定完成"))

                elif mt == "joint_calib_error":
                    self._joint_calib_state = "failed"
                    self._update_joint_calib_status()
                    self.status_bar_var.set(f"标定失败: {msg.get('message', '')}")

                elif mt == "joint_error":
                    self.joint_status_var.set(f"✗ {msg.get('message', '')}")
                    self.status_bar_var.set("关节角度测量错误")
                    tb = msg.get("traceback", "")
                    detail = f"{msg.get('message', '')}\n\n{tb}" if tb else msg.get("message", "")
                    self._on_joint_done()
                    messagebox.showerror("关节测量错误", detail)

        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def on_close(self):
        self._cancel_stop_watchdog()
        if self.joint_thread:
            self.joint_thread.stop()
            self.joint_thread = None
        self.root.destroy()


# ============================================================
# Entry point
# ============================================================
def run():
    root = tk.Tk()
    app = JointAngleApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return app
