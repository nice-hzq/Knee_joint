# -*- coding: utf-8 -*-
"""
双 IMU 帧同步测试
FrameSynchronizer / SessionClock / SessionRecorder 的行为契约。

迁移自原项目的 tests/test_joint_realtime.py（改用新包路径与新组件 API），
并补充了「匹配帧必须被消费，不得留在缓冲区」的回归用例。
"""

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from knee_joint.models import JointAngleState, JointCalibration
from knee_joint.protocol import Frame, abs_u32_delta_ms, signed_u32_delta_ms
from knee_joint.recorder import SessionRecorder
from knee_joint.sync import FrameSynchronizer


PROX_ID = "P"
DIST_ID = "D"


def _imu(marker):
    value = np.zeros(9, dtype=float)
    value[0] = marker
    return value


def _frame(device_id, timestamp_ms, marker=0.0, raw_row=None):
    return Frame(device_id=device_id, esp32_rx_ms=timestamp_ms,
                 imu9=_imu(marker), raw_row=raw_row)


def _sync():
    """构造一个已绑定近端/远端的同步器。"""
    synchronizer = FrameSynchronizer(max_dt_ms=30.0)
    synchronizer.set_roles(PROX_ID, DIST_ID)
    # 让两个设备出现在 buffers 中（set_roles 不预建缓冲区）
    synchronizer.feed(_frame(PROX_ID, 0))
    synchronizer.feed(_frame(DIST_ID, 0))
    synchronizer.buffers[PROX_ID].clear()
    synchronizer.buffers[DIST_ID].clear()
    return synchronizer


class FrameSyncTests(unittest.TestCase):
    def test_pairs_are_consumed_once(self):
        synchronizer = _sync()
        for marker, ts in ((1, 100), (2, 110)):
            synchronizer.feed(_frame(PROX_ID, ts, marker))
        for marker, ts in ((3, 102), (4, 112)):
            synchronizer.feed(_frame(DIST_ID, ts, marker))

        first = synchronizer.next_pair()
        second = synchronizer.next_pair()
        third = synchronizer.next_pair()

        self.assertEqual(first.dt_ms, 2)
        self.assertEqual(second.dt_ms, 2)
        self.assertEqual(first.t_s, 0.0)
        self.assertAlmostEqual(second.t_s, 0.010)
        self.assertIsNone(third)
        self.assertEqual(synchronizer.matched_count, 2)
        self.assertEqual(list(synchronizer.buffers[PROX_ID]), [])
        self.assertEqual(list(synchronizer.buffers[DIST_ID]), [])

    def test_matched_frame_is_removed_from_buffer(self):
        """
        回归：一对一近邻匹配时，命中的那一帧本身也必须出队。

        早期实现只弹出了 best_idx 个更早的帧，把命中帧留在队首，
        导致它被下一个远端帧重复匹配（曾出现 1730 帧报出 1728 次丢弃）。
        """
        synchronizer = _sync()
        for ts in (100, 110, 120, 130, 140):
            synchronizer.feed(_frame(PROX_ID, ts))
        synchronizer.feed(_frame(DIST_ID, 130))

        pair = synchronizer.next_pair()

        self.assertIsNotNone(pair)
        self.assertEqual(pair.dt_ms, 0)
        # 只有比命中帧更晚的近端帧留下
        self.assertEqual([f.esp32_rx_ms for f in synchronizer.buffers[PROX_ID]], [140])
        self.assertEqual(synchronizer.drop_prox_count, 3)   # 100/110/120 被丢弃
        self.assertEqual(list(synchronizer.buffers[DIST_ID]), [])

    def test_sync_tolerance_boundary(self):
        synchronizer = _sync()
        synchronizer.feed(_frame(PROX_ID, 130))
        synchronizer.feed(_frame(DIST_ID, 100))
        self.assertIsNotNone(synchronizer.next_pair())

        synchronizer = _sync()
        synchronizer.feed(_frame(PROX_ID, 131))
        synchronizer.feed(_frame(DIST_ID, 100))
        self.assertIsNone(synchronizer.next_pair())
        self.assertEqual(synchronizer.drop_dist_count, 1)

    def test_uint32_timestamp_wrap(self):
        self.assertEqual(signed_u32_delta_ms(5, 0xFFFFFFF0), 21)
        self.assertEqual(abs_u32_delta_ms(0xFFFFFFF5, 5), 16)

        synchronizer = _sync()
        self.assertEqual(synchronizer.clock.unwrap(0xFFFFFFF0), 0.0)
        self.assertAlmostEqual(synchronizer.clock.unwrap(5), 0.021)

    def test_matched_pair_keeps_corresponding_raw_rows(self):
        synchronizer = _sync()
        raw_p = [PROX_ID] + [str(i) for i in range(1, 16)]
        raw_d = [DIST_ID] + [str(i) for i in range(1, 16)]
        synchronizer.feed(_frame(PROX_ID, 100, raw_row=raw_p))
        synchronizer.feed(_frame(DIST_ID, 102, raw_row=raw_d))

        matched = synchronizer.next_pair()

        self.assertEqual(matched.raw_proximal, raw_p)
        self.assertEqual(matched.raw_distal, raw_d)

    def test_stats_shape(self):
        synchronizer = _sync()
        synchronizer.feed(_frame(PROX_ID, 100))
        synchronizer.feed(_frame(DIST_ID, 102))
        synchronizer.next_pair()

        self.assertEqual(synchronizer.stats(), {
            "sync_matched_count": 1,
            "sync_dropped_proximal": 0,
            "sync_dropped_distal": 0,
            "sync_avg_dt_ms": 2.0,
            "sync_max_dt_ms": 2.0,
        })

    def test_no_roles_means_no_pairing(self):
        synchronizer = FrameSynchronizer()
        synchronizer.feed(_frame(PROX_ID, 100))
        synchronizer.feed(_frame(DIST_ID, 100))
        self.assertIsNone(synchronizer.next_pair())


class RawDataSaveTests(unittest.TestCase):
    def test_raw_rows_are_saved_by_device(self):
        row = [
            "WT/TEST", "2026-07-15 12:00:00.000", "100", "110",
            "0.1", "0.2", "1.0", "1", "2", "3",
            "4", "5", "6", "7", "8", "9",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SessionRecorder(temp_dir, session_timestamp="20260715_120000")
            recorder.record_raw_row(row)
            recorder.close_raw()

            expected_dir = os.path.join(temp_dir, "raw", "joint_angle_20260715_120000")
            expected_file = os.path.join(expected_dir, "WT_TEST.csv")
            self.assertEqual(recorder.raw_dir, expected_dir)
            self.assertTrue(os.path.isfile(expected_file))

            with open(expected_file, newline="", encoding="utf-8") as f:
                saved = list(csv.reader(f))
            self.assertEqual(saved[0][0:4], [
                "device_id", "sensor_timestamp", "sensor_ms", "esp32_rx_ms"])
            self.assertEqual(saved[1], row)
            self.assertEqual(recorder.raw_total_rows, 1)

    def test_session_writes_angles_and_meta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SessionRecorder(temp_dir, session_timestamp="20260715_120000")
            state = JointAngleState(joint_name="left_knee")
            for i in range(5):
                state.update(float(i), t=float(i) * 0.01)

            session_dir = recorder.save_session(state, {
                "joint_key": "left_knee", "fs": 100,
                "sync": {"sync_matched_count": 5},
            })

            self.assertTrue(os.path.isfile(
                os.path.join(session_dir, "joint_angle_data.csv")))
            with open(os.path.join(session_dir, "session_info.json"),
                      encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(info["sample_count"], 5)
            self.assertEqual(info["sync_matched_count"], 5)
            self.assertEqual(info["flexion_rom_deg"], 4.0)

    def test_insufficient_data_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SessionRecorder(temp_dir)
            self.assertIsNone(
                recorder.save_session(JointAngleState(), {"joint_key": "left_knee"}))


class JointStateRegressionTests(unittest.TestCase):
    def test_first_sample_does_not_include_zero_in_rom(self):
        state = JointAngleState()
        state.update(-133.0, t=0.0)
        self.assertEqual(state.max_flexion_deg, -133.0)
        self.assertEqual(state.min_flexion_deg, -133.0)
        self.assertEqual(state.rom_deg, 0.0)

    def test_reset_removes_precalibration_history(self):
        state = JointAngleState()
        state.update(-133.0, t=0.0)
        state.reset()
        state.update(0.2, t=1.0)
        self.assertEqual(state.history_flexion, [0.2])
        self.assertEqual(state.rom_deg, 0.0)

    def test_calibration_validity_is_json_serializable_bool(self):
        calibration = JointCalibration(
            joint_name="left_knee", proximal_sensor="L4",
            distal_sensor="L5", calibration_mode="standing")
        self.assertIs(type(calibration.is_valid()), bool)
        self.assertEqual(json.loads(json.dumps({"valid": calibration.is_valid()})),
                         {"valid": True})


if __name__ == "__main__":
    unittest.main()
