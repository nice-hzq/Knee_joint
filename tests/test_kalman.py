# -*- coding: utf-8 -*-
"""Quaternion EKF 与双姿态输出的行为契约。"""

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from knee_joint.engine import JointAngleEngine
from knee_joint.kalman import KalmanOrientationNode
from knee_joint.models import JointAngleState, JointBinding, JointCalibration
from knee_joint.recorder import SessionRecorder


def _static_imu(count=220):
    data = np.zeros((count, 9), dtype=float)
    data[:, 2] = 1.0
    return data


class KalmanOrientationTests(unittest.TestCase):
    def test_static_updates_return_unit_quaternions(self):
        imu = _static_imu()
        node = KalmanOrientationNode("test", fs=100.0)
        node.init_from_static(
            imu, n_first=0, n_init=200, estimate_gyro_bias=True)

        quaternions = node.run_batch(imu[200:])

        self.assertTrue(np.all(np.isfinite(quaternions)))
        np.testing.assert_allclose(
            np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-8)

    def test_engine_keeps_both_histories_in_lockstep(self):
        sample = _static_imu(1)[0]
        engine = JointAngleEngine(JointBinding("right_knee", "R4", "R5"))
        engine.load_calibration(JointCalibration(
            "right_knee", "R4", "R5", "standing"))

        for i in range(205):
            engine.update(sample, sample, timestamp_s=i / 100.0)

        mahony = engine.get_state()
        kalman = engine.get_kalman_state()
        self.assertTrue(engine.is_ready())
        self.assertEqual(len(mahony.history_t), 5)
        self.assertEqual(mahony.history_t, kalman.history_t)
        self.assertAlmostEqual(kalman.flexion_deg, 0.0, places=6)


class DualOutputCompatibilityTests(unittest.TestCase):
    def test_old_calibration_falls_back_to_mahony_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "old.json")
            q_rel = [0.7, 0.7, 0.0, 0.0]
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "joint_name": "right_knee",
                    "proximal_sensor": "R4",
                    "distal_sensor": "R5",
                    "calibration_mode": "standing",
                    "q_rel_0": q_rel,
                }, f)

            calibration = JointCalibration.load(path)

        self.assertEqual(calibration.kalman_q_rel_0(), q_rel)

    def test_recorder_writes_both_angle_sets(self):
        mahony = JointAngleState("right_knee")
        kalman = JointAngleState("right_knee")
        for i in range(3):
            mahony.update(i, i + 1, i + 2, t=i / 100.0)
            kalman.update(i + 0.1, i + 1.1, i + 2.1, t=i / 100.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SessionRecorder(temp_dir)
            session_dir = recorder.save_session(
                mahony, {"joint_key": "right_knee"}, kalman_state=kalman)
            csv_path = os.path.join(session_dir, "joint_angle_data.csv")
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0], [
            "time_s", "flexion_deg", "abduction_deg", "rotation_deg",
            "kalman_flexion_deg", "kalman_abduction_deg",
            "kalman_rotation_deg",
        ])
        self.assertEqual(len(rows[1]), 7)


if __name__ == "__main__":
    unittest.main()
