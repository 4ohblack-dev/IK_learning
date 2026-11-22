import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import control_arm as control
from servo_com import handle_STS3215, COMM_RX_CORRUPT


class ConnectionTests(unittest.TestCase):
    def test_placeholder_requires_calibration_and_measured_pose_reloads(self):
        servo = self.servo()
        servo.zero_positions = {}
        servo.directions = {}
        with tempfile.TemporaryDirectory() as folder:
            servo.calibration_path = Path(folder) / 'calibration.json'
            servo.calibration_path.write_text(json.dumps({
                'kinematics_model': control.KINEMATICS_MODEL_ID,
                'physical_calibration_verified': False,
                'zero_positions': {'1': 2048},
                'directions': {'1': -1},
            }))
            self.assertFalse(servo.load_calibration([1]))
            self.assertEqual(servo.zero_positions, {})
            servo.st3215.ReadPosition.return_value = 3100
            servo.calibrate_and_save([1], [0.0], [-1])
            saved = json.loads(servo.calibration_path.read_text())
            self.assertIs(saved['physical_calibration_verified'], True)
            servo.zero_positions = {}
            self.assertTrue(servo.load_calibration([1]))
            self.assertEqual(servo.toAngle([1], [3100]).tolist(), [0.0])
            servo.st3215.WritePosition.assert_not_called()

    def servo(self):
        servo = handle_STS3215.__new__(handle_STS3215)
        servo.st3215 = Mock()
        return servo

    @patch('servo_com.time.sleep')
    def test_corrupt_ping_recovers(self, sleep):
        servo = self.servo()
        servo.st3215.ping.side_effect = [(0, COMM_RX_CORRUPT, 0), (3215, 0, 0)]
        servo.setup([1])
        self.assertEqual(servo.st3215.ping.call_count, 2)
        servo.st3215.portHandler.ser.reset_input_buffer.assert_called_once()

    @patch('servo_com.time.sleep')
    def test_persistent_corruption_reports_id_and_code(self, sleep):
        servo = self.servo()
        servo.st3215.ping.return_value = (0, COMM_RX_CORRUPT, 0)
        with self.assertRaisesRegex(ConnectionError, 'ID 1:.*comm=-7'):
            servo.setup([1])
        self.assertEqual(servo.st3215.ping.call_count, 3)

    def test_hardware_error_not_retried(self):
        servo = self.servo()
        servo.st3215.ping.return_value = (3215, 0, 32)
        with self.assertRaises(ConnectionError):
            servo.setup([1])
        servo.st3215.ping.assert_called_once()

    @patch('servo_com.time.sleep')
    def test_torque_read_retried_without_resending_stop(self, sleep):
        servo = self.servo()
        servo.st3215.read1ByteTxRx.side_effect = [(0, -7, 0), (0, 0, 0)]
        servo.release_torque([1])
        servo.st3215.StopServo.assert_called_once_with(1)

    def test_cleanup_preserves_setup_error_and_closes_resources(self):
        servo = Mock()
        primary = ConnectionError('ID 1 setup failed')
        servo.setup.side_effect = primary
        servo.release_torque.side_effect = ConnectionError('Torque OFF unverified')
        controller = Mock()
        with (patch.object(control, 'Controller', return_value=controller),
              patch.object(control, 'handle_STS3215', return_value=servo),
              patch.object(control, 'ArmKinematics', return_value=SimpleNamespace(dof=5))):
            with self.assertRaises(ConnectionError) as caught:
                control.run('mock', [1, 2, 3, 4, 5], None, None)
        self.assertIs(caught.exception, primary)
        self.assertIn('Torque OFF unverified', primary.__notes__[0])
        servo.close.assert_called_once()
        controller.close.assert_called_once()

    def test_speed_lost_ack_accepted_only_with_matching_readback(self):
        servo = self.servo()
        servo.st3215.write2ByteTxRx.return_value = (-7, 0)
        servo.st3215.read2ByteTxRx.return_value = (800, 0, 0)
        control.write_verified_setting(servo, 3, control.STS_GOAL_SPEED_L, 800,
                                       width=2, name='speed')
        servo.st3215.write2ByteTxRx.assert_called_once()

    def test_speed_mismatch_stops(self):
        servo = self.servo()
        servo.st3215.write2ByteTxRx.return_value = (-6, 0)
        servo.st3215.read2ByteTxRx.return_value = (200, 0, 0)
        with self.assertRaisesRegex(ConnectionError, 'expected=800, actual=200'):
            control.write_verified_setting(servo, 3, control.STS_GOAL_SPEED_L, 800,
                                           width=2, name='speed')

    def test_speed_hardware_error_not_ignored(self):
        servo = self.servo()
        servo.st3215.write2ByteTxRx.return_value = (0, 32)
        with self.assertRaisesRegex(ConnectionError, 'error=32'):
            control.write_verified_setting(servo, 3, control.STS_GOAL_SPEED_L, 800,
                                           width=2, name='speed')
        servo.st3215.read2ByteTxRx.assert_not_called()


if __name__ == '__main__':
    unittest.main()
