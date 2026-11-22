import unittest

import numpy as np

from control_arm import ArmKinematics, measured_control_q, SERVO_IDS


class MeasuredLimitsTests(unittest.TestCase):
    def test_encoder_drift_at_calibrated_zero_is_accepted(self):
        arm = ArmKinematics()
        q = np.array([0, -2, 2, 0, 0]) * (2 * np.pi / 4096)
        np.testing.assert_allclose(measured_control_q(arm, q, SERVO_IDS), np.zeros(5))

    def test_large_deviation_reports_joint_and_limits(self):
        arm = ArmKinematics()
        with self.assertRaisesRegex(ValueError, r'ID 2 \(shoulder_lift\): -3.00 deg'):
            measured_control_q(arm, np.deg2rad([0, -3, 0, 0, 0]), SERVO_IDS)

    def test_valid_measurements_are_preserved(self):
        arm = ArmKinematics()
        q = (arm.q_min + arm.q_max) / 2
        np.testing.assert_array_equal(measured_control_q(arm, q, SERVO_IDS), q)

    def test_nonfinite_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            measured_control_q(ArmKinematics(), [0, float('nan'), 0, 0, 0], SERVO_IDS)


if __name__ == '__main__':
    unittest.main()
