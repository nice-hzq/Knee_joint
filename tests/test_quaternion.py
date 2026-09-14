# -*- coding: utf-8 -*-
"""knee_joint/quaternion.py 单元测试"""
import unittest
import numpy as np

from knee_joint.quaternion import (
    quat_normalize, quat_conj, quat_inv, quat_mul, quat_rotate,
    quat_to_rotmat, quat_relative, average_quaternions,
    quat_align_hemisphere, enforce_quat_continuity,
    quat_to_euler, euler_to_quat, wrap_to_pi, wrap_to_180, R_to_quat,
)

Q_ID = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
Q_NEG = np.array([-1.0, 0.0, 0.0, 0.0], dtype=float)
EPS = 1e-10


class TestQuatNormalize(unittest.TestCase):
    def test_identity(self):
        q = quat_normalize(Q_ID)
        np.testing.assert_allclose(q, Q_ID, atol=EPS)

    def test_already_normalized(self):
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        result = quat_normalize(q)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)

    def test_unnormalized(self):
        q = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
        result = quat_normalize(q)
        np.testing.assert_allclose(result, Q_ID, atol=EPS)

    def test_near_zero(self):
        q = np.array([1e-15, 1e-15, 1e-15, 1e-15], dtype=float)
        result = quat_normalize(q)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_negative(self):
        q = np.array([-3.0, 0.0, 0.0, 0.0], dtype=float)
        result = quat_normalize(q)
        np.testing.assert_allclose(result, Q_NEG, atol=EPS)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)

    def test_batch_2d(self):
        qs = np.array([[2.0, 0.0, 0.0, 0.0],
                       [0.0, 3.0, 0.0, 0.0]], dtype=float)
        result = quat_normalize(qs)
        for r in result:
            self.assertAlmostEqual(np.linalg.norm(r), 1.0, places=10)


class TestQuatConj(unittest.TestCase):
    def test_identity(self):
        c = quat_conj(Q_ID)
        np.testing.assert_allclose(c, Q_ID, atol=EPS)

    def test_conjugate(self):
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        c = quat_conj(q)
        self.assertEqual(c[0], q[0])
        np.testing.assert_allclose(c[1:], -q[1:], atol=EPS)

    def test_double_conj(self):
        q = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        c2 = quat_conj(quat_conj(q))
        np.testing.assert_allclose(c2, q, atol=EPS)

    def test_q_mul_qconj_is_identity(self):
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
        result = quat_mul(q, quat_conj(q))
        np.testing.assert_allclose(result, Q_ID, atol=EPS)


class TestQuatInv(unittest.TestCase):
    def test_identity(self):
        inv = quat_inv(Q_ID)
        np.testing.assert_allclose(inv, Q_ID, atol=EPS)

    def test_q_mul_qinv(self):
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
        result = quat_mul(q, quat_inv(q))
        np.testing.assert_allclose(result, Q_ID, atol=EPS)

    def test_inv_of_inv(self):
        q = quat_normalize(np.array([0.7, 0.3, 0.2, 0.6], dtype=float))
        inv2 = quat_inv(quat_inv(q))
        np.testing.assert_allclose(inv2, q, atol=EPS)


class TestQuatMul(unittest.TestCase):
    def test_identity_left(self):
        q = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        result = quat_mul(Q_ID, q)
        np.testing.assert_allclose(result, q, atol=EPS)

    def test_identity_right(self):
        q = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        result = quat_mul(q, Q_ID)
        np.testing.assert_allclose(result, q, atol=EPS)

    def test_known_product(self):
        q1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        q2 = np.array([0.5, -0.5, -0.5, -0.5], dtype=float)
        result = quat_mul(q1, q2)
        np.testing.assert_allclose(result, Q_ID, atol=EPS)

    def test_non_unit(self):
        q1 = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
        q2 = np.array([0.0, 3.0, 0.0, 0.0], dtype=float)
        result = quat_mul(q1, q2)
        np.testing.assert_allclose(result, [0.0, 6.0, 0.0, 0.0], atol=EPS)


class TestQuatRotate(unittest.TestCase):
    def test_identity_does_nothing(self):
        v = np.array([1.0, 2.0, 3.0])
        r = quat_rotate(Q_ID, v)
        np.testing.assert_allclose(r, v, atol=EPS)

    def test_180_around_z(self):
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        v = np.array([1.0, 0.0, 0.0])
        r = quat_rotate(q, v)
        np.testing.assert_allclose(r, [-1.0, 0.0, 0.0], atol=EPS)

    def test_90_around_z(self):
        angle = np.pi / 4
        q = np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)], dtype=float)
        v = np.array([1.0, 0.0, 0.0])
        r = quat_rotate(q, v)
        np.testing.assert_allclose(r, [0.0, 1.0, 0.0], atol=EPS)

    def test_roundtrip(self):
        q = quat_normalize(np.array([0.3, 0.6, 0.2, 0.7], dtype=float))
        v = np.array([1.5, -0.8, 2.2])
        r = quat_rotate(q, v)
        r_back = quat_rotate(quat_conj(q), r)
        np.testing.assert_allclose(r_back, v, atol=EPS)


class TestQuatToRotmat(unittest.TestCase):
    def test_identity(self):
        R = quat_to_rotmat(Q_ID)
        np.testing.assert_allclose(R, np.eye(3), atol=EPS)

    def test_orthogonal(self):
        q = quat_normalize(np.array([0.4, 0.5, 0.6, 0.3], dtype=float))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=EPS)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=8)

    def test_180_z(self):
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        R = quat_to_rotmat(q)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=EPS)


class TestQuatRelative(unittest.TestCase):
    def test_same_quat(self):
        q = quat_normalize(np.array([0.3, 0.4, 0.5, 0.7], dtype=float))
        rel = quat_relative(q, q)
        np.testing.assert_allclose(np.abs(rel), Q_ID, atol=EPS)

    def test_child_from_parent(self):
        q_parent = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        q_child = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        rel = quat_relative(q_parent, q_child)
        np.testing.assert_allclose(np.abs(rel), np.abs(q_parent), atol=EPS)


class TestAverageQuaternions(unittest.TestCase):
    def test_single(self):
        q = quat_normalize(np.array([0.2, 0.3, 0.4, 0.8], dtype=float))
        avg = average_quaternions(np.array([q]))
        np.testing.assert_allclose(avg, q, atol=EPS)

    def test_identical(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        qs = np.array([q, q, q])
        avg = average_quaternions(qs)
        np.testing.assert_allclose(avg, q, atol=EPS)

    def test_near_identical(self):
        q = Q_ID
        qs = np.array([q, q + 0.001, q - 0.001])
        avg = average_quaternions(qs)
        self.assertAlmostEqual(np.linalg.norm(avg), 1.0, places=10)

    def test_hemisphere_flip(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        qs = np.array([q, -q])
        avg = average_quaternions(qs)
        self.assertAlmostEqual(np.linalg.norm(avg), 1.0, places=10)


class TestQuatAlignHemisphere(unittest.TestCase):
    def test_same_hemisphere(self):
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        ref = np.array([0.6, 0.4, 0.4, 0.4], dtype=float)
        aligned = quat_align_hemisphere(q, ref)
        np.testing.assert_allclose(aligned, q, atol=EPS)

    def test_opposite_hemisphere(self):
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        ref = np.array([-0.6, -0.4, -0.4, -0.4], dtype=float)
        aligned = quat_align_hemisphere(q, ref)
        np.testing.assert_allclose(aligned, -q, atol=EPS)


class TestEnforceQuatContinuity(unittest.TestCase):
    def test_no_flip_needed(self):
        qs = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.999, 0.01, 0.01, 0.01],
                       [0.998, 0.02, 0.02, 0.02]])
        result = enforce_quat_continuity(qs)
        np.testing.assert_allclose(result, qs, atol=EPS)

    def test_flip_correction(self):
        qs = np.array([[1.0, 0.0, 0.0, 0.0],
                       [-0.999, -0.01, -0.01, -0.01]])
        result = enforce_quat_continuity(qs)
        self.assertGreater(np.dot(result[0], result[1]), 0)

    def test_alternating_flips(self):
        qs = np.array([[1.0, 0.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0]])
        result = enforce_quat_continuity(qs)
        self.assertTrue(np.all(np.dot(result[1:], result[:-1].T) >= 0))


class TestQuatToEuler(unittest.TestCase):
    def test_identity_radians(self):
        euler = quat_to_euler(Q_ID, degrees=False)
        np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=EPS)

    def test_identity_degrees(self):
        euler = quat_to_euler(Q_ID, degrees=True)
        np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=EPS)

    def test_90_yaw(self):
        q = euler_to_quat(0.0, 0.0, np.pi / 2)
        euler = quat_to_euler(q, degrees=True)
        np.testing.assert_allclose(euler, [0.0, 0.0, 90.0], atol=1e-8)

    def test_batch(self):
        qs = np.array([Q_ID, Q_ID])
        eulers = quat_to_euler(qs, degrees=True)
        np.testing.assert_allclose(eulers, [[0, 0, 0], [0, 0, 0]], atol=EPS)

    def test_non_unit_input(self):
        q = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
        euler = quat_to_euler(q, degrees=True)
        self.assertEqual(euler.shape, (3,))


class TestEulerToQuat(unittest.TestCase):
    def test_zero(self):
        q = euler_to_quat(0.0, 0.0, 0.0)
        np.testing.assert_allclose(q, Q_ID, atol=EPS)

    def test_90_pitch(self):
        q = euler_to_quat(0.0, np.pi / 2, 0.0)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=10)

    def test_known_roll_90(self):
        q = euler_to_quat(np.pi / 2, 0.0, 0.0)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=10)

    def test_roundtrip(self):
        roll, pitch, yaw = 0.3, -0.5, 1.2
        q = euler_to_quat(roll, pitch, yaw)
        euler = quat_to_euler(q)
        np.testing.assert_allclose(euler, [roll, pitch, yaw], atol=1e-8)


class TestWrapToPi(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(wrap_to_pi(0.0), 0.0)

    def test_within_range(self):
        self.assertAlmostEqual(wrap_to_pi(1.0), 1.0, places=10)

    def test_above_pi(self):
        self.assertAlmostEqual(wrap_to_pi(4.0), 4.0 - 2 * np.pi, places=10)

    def test_below_minus_pi(self):
        self.assertAlmostEqual(wrap_to_pi(-4.0), -4.0 + 2 * np.pi, places=10)

    def test_array(self):
        arr = np.array([0.0, 4.0, -4.0])
        result = wrap_to_pi(arr)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(np.abs(result) <= np.pi))


class TestWrapTo180(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(wrap_to_180(0.0), 0.0)

    def test_within_range(self):
        self.assertAlmostEqual(wrap_to_180(90.0), 90.0, places=10)

    def test_above_180(self):
        self.assertAlmostEqual(wrap_to_180(200.0), -160.0, places=10)

    def test_below_minus_180(self):
        self.assertAlmostEqual(wrap_to_180(-200.0), 160.0, places=10)

    def test_array(self):
        arr = np.array([0.0, 200.0, -200.0, 350.0, -350.0])
        result = wrap_to_180(arr)
        self.assertTrue(np.all(np.abs(result) <= 180.0))


class TestRToQuat(unittest.TestCase):
    def test_identity(self):
        R = np.eye(3)
        q = R_to_quat(R)
        np.testing.assert_allclose(np.abs(q), Q_ID, atol=EPS)

    def test_180_around_z(self):
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        q = R_to_quat(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=10)

    def test_roundtrip(self):
        q_orig = quat_normalize(np.array([0.3, 0.6, 0.2, 0.7], dtype=float))
        R = quat_to_rotmat(q_orig)
        q_back = R_to_quat(R)
        np.testing.assert_allclose(q_orig, q_back, atol=1e-8)
        self.assertAlmostEqual(np.dot(q_orig, q_back), 1.0, places=8)

    def test_90_x(self):
        c = np.cos(np.pi / 2)
        s = np.sin(np.pi / 2)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
        q = R_to_quat(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
