import unittest

import numpy as np

from mathmetics import ArmKinematics


class TcpTests(unittest.TestCase):
    def test_fixed_jaw_tip_is_default_in_multiple_poses(self):
        arm = ArmKinematics()
        flange = ArmKinematics(tcp_xyz=(0, 0, 0))
        tip = np.array([0.0098493, -0.106375, 0.0, 1.0])
        for q in (np.zeros(5), [0.3, 1.0, -0.8, 0.2, 0.7]):
            expected = flange.fk(q) @ tip
            np.testing.assert_allclose(arm.position(q), expected[:3], atol=1e-12)
            self.assertGreater(np.linalg.norm(arm.position(q) - flange.position(q)), 0.1)

    def test_tip_jacobian_matches_numerical_derivative(self):
        arm = ArmKinematics()
        q = np.array([0.3, 1.0, -0.8, 0.2, 0.7])
        h = 1e-6
        numerical = np.column_stack([
            (arm.position(q + step) - arm.position(q - step)) / (2 * h)
            for step in np.eye(5) * h
        ])
        np.testing.assert_allclose(arm.jacobian(q), numerical, atol=1e-9)

    def test_ik_targets_tip(self):
        arm = ArmKinematics()
        seed = np.array([0.3, 1.0, -0.8, 0.2, 0.7])
        target = arm.position(seed + [0.01, -0.01, 0.01, 0, 0])
        result = arm.solve_position_IK(target, seed, tol=5e-5)
        self.assertTrue(result['success'])
        self.assertLess(np.linalg.norm(arm.position(result['q']) - target), 5e-5)


if __name__ == '__main__':
    unittest.main()
