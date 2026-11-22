import unittest
from unittest.mock import Mock
import numpy as np

from mathmetics import ArmKinematics
from control_arm import (
    IK_POSITION_TOLERANCE_M, joystick_direction, solve_joystick_step,
    update_filtered_direction,
)
from controller_input import Input


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
        shoulder_max = self.arm.q_max[1]
        seed = [-0.6624, shoulder_max, -0.628, -0.674, 1.0]
        goal_q = [-0.6565, shoulder_max, -0.640, -0.669, 1.017]
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
        limit = 0.002
        result, accepted = solve_joystick_step(self.arm, seed, origin, target, max_joint_step=limit)
        self.assertLessEqual(np.max(np.abs(result['q'] - seed)), limit + 1e-12)
        np.testing.assert_array_equal(accepted[1:], origin[1:])
        self.assertLess(np.linalg.norm(accepted - self.arm.position(result['q'])),
                        IK_POSITION_TOLERANCE_M)
        self.assertLess(origin[0], accepted[0])
        self.assertLess(accepted[0], target[0])

    def test_repeated_positive_x_steps_do_not_accumulate_cross_axis_error(self):
        q = np.zeros(self.arm.dof)
        start = self.arm.position(q)
        origin = start.copy()
        limit = 800 * (2 * np.pi / 4096) * 0.02
        for _ in range(40):
            result, origin = solve_joystick_step(
                self.arm, q, origin, origin + [0.001, 0, 0], max_joint_step=limit,
            )
            self.assertLessEqual(np.max(np.abs(result['q'] - q)), limit)
            q = result['q']
            np.testing.assert_array_equal(origin[1:], start[1:])
            self.assertLess(np.linalg.norm(self.arm.position(q)[1:] - start[1:]),
                            IK_POSITION_TOLERANCE_M)
        self.assertGreater(origin[0] - start[0], 0.01)

    def test_joystick_rejects_step_when_no_candidate_meets_joint_speed(self):
        arm = Mock()
        arm.solve_position_IK.return_value = dict(
            success=True, q=np.full(5, 0.1), error=0.0, iterations=1,
        )
        origin = np.zeros(3)
        with self.assertRaisesRegex(ValueError, "joint speed limit"):
            solve_joystick_step(arm, np.zeros(5), origin, np.array([0.001, 0, 0]),
                                max_joint_step=0.02)
        arm.position.assert_not_called()

    def test_stick_axis_switch_and_intentional_diagonal(self):
        direction = joystick_direction(Input(leftX=0.8, leftY=0.15, rightY=0.08))
        np.testing.assert_array_equal(direction, [0.8, 0.0, 0.0])
        filtered = direction.copy()
        direction = joystick_direction(Input(leftX=0.12, leftY=-0.8, rightY=-0.09))
        update_filtered_direction(filtered, direction, 0.02)
        np.testing.assert_array_equal(filtered[[0, 2]], [0.0, 0.0])
        self.assertLess(filtered[1], 0.0)
        update_filtered_direction(filtered, joystick_direction(Input()), 0.02)
        np.testing.assert_array_equal(filtered, [0.0, 0.0, 0.0])
        diagonal = joystick_direction(Input(leftX=-0.7, leftY=0.6, rightY=0.4))
        np.testing.assert_array_equal(diagonal, [-0.7, 0.6, 0.4])


if __name__ == '__main__':
    unittest.main()
