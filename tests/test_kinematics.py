import unittest
from unittest.mock import Mock
import numpy as np

from mathmetics import ArmKinematics
from control_arm import solve_joystick_step


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.arm = ArmKinematics()

    def test_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            q = rng.uniform(self.arm.q_min, self.arm.q_max)
            numerical = np.column_stack([
                (self.arm.position(q + np.eye(5)[i] * 1e-6) -
                 self.arm.position(q - np.eye(5)[i] * 1e-6)) / 2e-6
                for i in range(5)
            ])
            np.testing.assert_allclose(self.arm.jacobian(q), numerical, atol=1e-9)

    def test_reachable_target_with_shoulder_at_upper_limit(self):
        seed = [-0.6623687314, 1.74533, 0.6278125076, -0.6735819851, 2.5555784522]
        goal_q = [-0.6565421790, 1.74533, 0.6395866265, -0.6691911184, 2.5730177974]
        result = self.arm.solve_position_IK(self.arm.position(goal_q), seed, tol=5e-5)
        self.assertTrue(result['success'])
        self.assertLess(result['error'], 5e-5)
        self.assertTrue(np.all(result['q'] <= self.arm.q_max))
        self.assertTrue(np.all(result['q'] >= self.arm.q_min))

    def test_success_on_final_iteration(self):
        arm = self.arm
        seed = np.zeros(5)
        target = arm.position([0.01, 0, 0, 0, 0])
        one_step = arm.solve_position_IK(target, seed, max_iter=1, tol=1e-12)
        tolerance = (np.linalg.norm(target - arm.position(seed)) + one_step['error']) / 2
        result = arm.solve_position_IK(target, seed, max_iter=1, tol=tolerance)
        self.assertTrue(result['success'])

    def test_unreachable_goal_not_reported_as_success(self):
        result = self.arm.solve_position_IK([10, 10, 10], np.zeros(5), max_iter=30)
        self.assertFalse(result['success'])

    def test_joystick_reduces_step_in_same_direction(self):
        arm = Mock()
        arm.solve_position_IK.side_effect = [
            dict(success=False, q=np.zeros(5), error=0.001, iterations=1),
            dict(success=True, q=np.zeros(5), error=0.0, iterations=1),
        ]
        origin = np.array([0.1, 0.2, 0.3])
        target = origin + np.array([0.001, -0.002, 0])
        result, accepted = solve_joystick_step(arm, np.zeros(5), origin, target)
        self.assertTrue(result['success'])
        np.testing.assert_allclose(accepted, (origin + target) / 2)

    def test_extended_pose_step_respects_servo_speed(self):
        seed = np.deg2rad([-5.2734375, 83.3203125, -79.27734375, 15.73242188, -0.96679687])
        origin = self.arm.position(seed)
        target = origin + np.array([0.001, 0, 0])
        limit = 800 * (2 * np.pi / 4096) * 0.02
        result, accepted = solve_joystick_step(self.arm, seed, origin, target, max_joint_step=limit)
        self.assertLessEqual(np.max(np.abs(result['q'] - seed)), limit + 1e-12)
        np.testing.assert_allclose(accepted, self.arm.position(result['q']))
        self.assertGreater(np.linalg.norm(accepted - origin), 0)
        self.assertGreater(np.linalg.norm(accepted - target), 1e-5)


if __name__ == '__main__':
    unittest.main()
