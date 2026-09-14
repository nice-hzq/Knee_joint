# -*- coding: utf-8 -*-
"""
端到端回放回归测试
End-to-end replay regression.

用 samples/ 里的真实录制数据（两个 WT901 传感器，100 Hz，20 秒）离线回放，
与提交在 tests/golden/ 下的基准结果逐帧比对。

为什么这是最有价值的测试：它一次性覆盖了
    16 列解析 → 帧同步 → Mahony 解算 → 相对姿态 → 关节角 → 落盘
整条链路，任何一环走样都会让角度序列偏离基准。
"""

import csv
import json
import os
import queue
import unittest

from knee_joint.config import project_root
from knee_joint.sources import CsvReplaySource
from knee_joint.worker import JointWorker


PROJECT_ROOT = project_root()
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples", "joint_angle_20260716_093022")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
CALIB_FILE = os.path.join(TEMP_DIR, "right_knee_pose_calib.json")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
GOLDEN_INFO = os.path.join(GOLDEN_DIR, "replay_right_knee.json")
GOLDEN_ANGLES = os.path.join(GOLDEN_DIR, "replay_right_knee.csv")

# 该录制对应的原始会话（原项目 joint_rawdate/）中的 ROM，
# 用于确认回放结果与当次实拍量级一致。见 tests/golden/replay_right_knee.json。
ROM_BASELINE_TOLERANCE_DEG = 2.0

DEVICE_MAP = {"WT5500010107": "R5", "WT5500010112": "R4"}

# 逐帧角度允许的偏差（度）。录制数据本身完全确定，
# 留一点余量以容忍不同 BLAS/平台在末位浮点上的差异。
ANGLE_ATOL_DEG = 1e-4


def _read_golden_angles(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 表头
        return [[float(v) for v in row] for row in reader]


def _read_angles_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [[float(v) for v in row] for row in reader]


class ReplayRegressionTests(unittest.TestCase):
    """整个 fixture 只跑一次回放，各用例共用结果。"""

    session_dir = None
    session_info = None

    @classmethod
    def setUpClass(cls):
        for path in (SAMPLES_DIR, GOLDEN_INFO, GOLDEN_ANGLES):
            if not os.path.exists(path):
                raise unittest.SkipTest(f"缺少回放夹具: {path}")

        with open(GOLDEN_INFO, encoding="utf-8") as f:
            cls.golden_info = json.load(f)

        import tempfile
        cls._temp_dir = tempfile.TemporaryDirectory()
        output_root = cls._temp_dir.name
        # 测试使用 golden 自带的零位，避免用户最近一次真实标定覆盖
        # temp/right_knee_pose_calib.json 后让回放基准失去确定性。
        cls.calib_file = os.path.join(output_root, "golden_calibration.json")
        with open(cls.calib_file, "w", encoding="utf-8") as f:
            json.dump({
                "joint_name": "right_knee",
                "proximal_sensor": "R4",
                "distal_sensor": "R5",
                "calibration_mode": cls.golden_info["calibration_mode"],
                "q_rel_0": cls.golden_info["calib_q_rel_0"],
            }, f)

        messages = []
        status_queue = queue.Queue()
        source = CsvReplaySource(SAMPLES_DIR, speed=0.0)
        worker = JointWorker(
            source, "right_knee", output_root, status_queue,
            device_map=DEVICE_MAP,
            calib_dir=TEMP_DIR,
            fs=100.0,
            calib_mode="lower_body_standing",
            calib_file=cls.calib_file,
            save_raw=True,
            source_label="test:replay")
        worker.start()
        worker.join(timeout=120)

        while True:
            try:
                messages.append(status_queue.get_nowait())
            except queue.Empty:
                break

        if worker.is_alive():
            worker.stop()
            raise AssertionError("回放线程 120 秒内未结束")

        cls.messages = messages
        cls.session_dir = worker.session_dir

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def _errors(self):
        return [m for m in self.messages if m.get("type") == "joint_error"]

    def _calib_messages(self):
        return [m for m in self.messages
                if m.get("type") in ("joint_calib_done", "joint_calib_error")]

    def _session_info(self):
        """读取本次回放落盘的 session_info.json（各用例独立读取，避免顺序依赖）。"""
        path = os.path.join(self.session_dir, "session_info.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ---- 基本健康 ----

    def test_no_worker_errors(self):
        errors = self._errors()
        self.assertEqual(errors, [], f"回放过程中出现错误: {errors}")

    def test_calibration_file_was_loaded(self):
        self.assertTrue(any("已加载标定文件" in m.get("message", "")
                            for m in self._calib_messages()),
                        "标定文件未被加载，回放结果将不具备可比性")

    def test_calibration_fixture_is_intact(self):
        """
        基准所依赖的标定夹具必须与录制时一致。

        这条断言存在的原因：任何写 <TEMP_DIR>/right_knee_pose_calib.json 的
        代码（例如一次自动标定）都会静默改变所有回放基准的含义。
        """
        with open(self.calib_file, encoding="utf-8") as f:
            actual = json.load(f)["q_rel_0"]
        for got, expected in zip(actual, self.golden_info["calib_q_rel_0"]):
            self.assertAlmostEqual(
                got, expected, places=10,
                msg="temp/right_knee_pose_calib.json 被改写过，回放基准已失效")

    def test_session_was_saved(self):
        self.assertIsNotNone(self.session_dir)
        self.assertTrue(os.path.isfile(
            os.path.join(self.session_dir, "joint_angle_data.csv")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.session_dir, "session_info.json")))

    # ---- 与基准逐项比对 ----

    def test_session_info_matches_golden(self):
        info = self._session_info()

        exact_fields = [
            "joint_key", "proximal_sensor", "distal_sensor",
            "proximal_device_id", "distal_device_id",
            "calibrated", "calibration_mode", "fs",
            "sample_count", "duration_s",
            "sync_matched_count", "sync_dropped_proximal", "sync_dropped_distal",
            "sync_max_dt_ms", "raw_sample_count",
        ]
        for field in exact_fields:
            self.assertEqual(info[field], self.golden_info[field],
                             f"{field} 与基准不一致")

        for field in ("flexion_rom_deg", "flexion_max_deg", "flexion_min_deg"):
            self.assertAlmostEqual(info[field], self.golden_info[field], places=2,
                                   msg=f"{field} 与基准不一致")

        self.assertAlmostEqual(info["sync_avg_dt_ms"],
                               self.golden_info["sync_avg_dt_ms"], places=3)

    def test_angle_series_matches_golden(self):
        actual = _read_angles_csv(
            os.path.join(self.session_dir, "joint_angle_data.csv"))
        expected = _read_golden_angles(GOLDEN_ANGLES)

        self.assertEqual(len(actual), len(expected),
                         "角度序列长度与基准不一致")

        worst = 0.0
        worst_at = None
        for i, (row, ref) in enumerate(zip(actual, expected)):
            for j in range(len(ref)):
                diff = abs(row[j] - ref[j])
                if diff > worst:
                    worst, worst_at = diff, (i, j, row[j], ref[j])

        # 消息只在真的失败时才拼装 —— f-string 会被立即求值，
        # 而 worst_at 在完全一致时仍为 None。
        detail = ""
        if worst_at is not None:
            detail = (f" (行 {worst_at[0]} 列 {worst_at[1]}: "
                      f"实际 {worst_at[2]} vs 基准 {worst_at[3]})")
        self.assertLessEqual(
            worst, ANGLE_ATOL_DEG,
            f"角度序列偏离基准 {worst:.6f}°{detail}")

    def test_rom_is_consistent_with_recorded_session(self):
        """回放得到的 ROM 应与该次实拍的 ROM 量级一致。"""
        rom = self._session_info()["flexion_rom_deg"]
        baseline = self.golden_info["recorded_baseline_rom_deg"]
        self.assertLessEqual(
            abs(rom - baseline), ROM_BASELINE_TOLERANCE_DEG,
            f"ROM {rom}° 偏离实拍基准 {baseline}° 超过 "
            f"{ROM_BASELINE_TOLERANCE_DEG}°")

    # ---- 消息契约 ----

    def test_publishes_expected_message_types(self):
        types = {m.get("type") for m in self.messages}
        for expected in ("joint_status", "joint_angle", "joint_calib_done"):
            self.assertIn(expected, types)

    def test_angle_messages_carry_curve_history(self):
        angles = [m for m in self.messages if m.get("type") == "joint_angle"]
        self.assertTrue(angles)
        last = angles[-1]
        for key in ("history_t", "history_flexion", "history_abduction",
                    "history_rotation"):
            self.assertIn(key, last)
            self.assertTrue(len(last[key]) > 0)

        for key in ("kalman_flexion_deg", "kalman_abduction_deg",
                    "kalman_rotation_deg", "kalman_rom_deg",
                    "kalman_history_t", "kalman_history_flexion",
                    "kalman_history_abduction", "kalman_history_rotation"):
            self.assertIn(key, last)
        self.assertTrue(last["kalman_history_flexion"])
        self.assertEqual(len(last["history_t"]),
                         len(last["kalman_history_t"]))

    def test_saved_csv_contains_kalman_angles(self):
        path = os.path.join(self.session_dir, "joint_angle_data.csv")
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header[-3:], [
            "kalman_flexion_deg", "kalman_abduction_deg",
            "kalman_rotation_deg",
        ])

    def test_reports_disconnected_at_end(self):
        states = [m.get("state") for m in self.messages
                  if m.get("type") == "joint_status"]
        self.assertIn("disconnected", states)


class AutoCalibrationReplayTests(unittest.TestCase):
    """
    回归：不提供标定文件时，回放应能靠自动标定跑通。

    触发点按会话时间判断，但标定需要攒够 calib_dur_s × fs 帧，而配对后的
    有效帧率低于 fs（本录制约 86 帧/秒），所以 3.0 秒时缓冲只有 235 帧。
    早期实现一次性触发、失败即放弃，导致回放永远标不上 —— 现在按秒重试。
    """

    def test_auto_calibration_succeeds_and_matches_stored_calibration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work_dir:
            # calib_dir 必须指向临时目录：标定成功后 worker 会把结果写回
            # <calib_dir>/<joint>_pose_calib.json，用项目 temp/ 会覆盖掉
            # 回放基准所依赖的标定夹具。
            status_queue = queue.Queue()
            worker = JointWorker(
                CsvReplaySource(SAMPLES_DIR, speed=0.0),
                "right_knee", work_dir, status_queue,
                device_map=DEVICE_MAP,
                calib_dir=work_dir,
                fs=100.0,
                calib_mode="lower_body_standing",
                calib_file=None,          # 强制走自动标定
                auto_calib_at_s=3.0,
                save_raw=False,
                source_label="test:autocalib")
            worker.start()
            worker.join(timeout=120)
            self.assertFalse(worker.is_alive(), "回放线程未结束")

            messages = []
            while True:
                try:
                    messages.append(status_queue.get_nowait())
                except queue.Empty:
                    break

        self.assertEqual([m for m in messages if m.get("type") == "joint_error"],
                         [], "自动标定回放过程中出现错误")

        done = [m for m in messages if m.get("type") == "joint_calib_done"
                and m.get("euler_rel_0")]
        self.assertTrue(done, "自动标定最终没有成功")

        # 自动标定算出的零点应与人工标定的存档高度一致
        with open(GOLDEN_INFO, encoding="utf-8") as f:
            stored = json.load(f)["calib_q_rel_0"]
        for got, expected in zip(done[-1]["q_rel_0"], stored):
            self.assertAlmostEqual(got, expected, places=2)

        self.assertIsNotNone(worker.session_dir)


if __name__ == "__main__":
    unittest.main()
