import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import control_arm as control
from controller_input import Input
from servo_com import handle_STS3215


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeServo(handle_STS3215):
    """Real angle conversion and control code, without opening a serial port."""
    def __init__(self, clock, drift=False, moving=True):
        self.clock = clock
        self.drift = drift
        self.moving = moving
        calibration = json.loads(Path('calibration.json').read_text())
        self.zero_positions = {int(k): v for k, v in calibration['zero_positions'].items()}
        self.directions = {int(k): v for k, v in calibration['directions'].items()}
        self.positions = dict(zip(control.SERVO_IDS, [1988, 1100, 2949, 2227, 2037]))
        self.st3215 = self
        self.groupSyncWrite = self
        self.params = {}
        self.settings = {}
        self.sent = []
        self.mode = 0
        self.torque = 0
        self.released = False
        self.closed = False
        self.ignore_sync = False
        self.portHandler = SimpleNamespace(ser=SimpleNamespace(
            flush=lambda: None, reset_input_buffer=lambda: None,
        ))

    def setup(self, ids):
        pass

    def load_calibration(self, ids):
        return True

    def read_positions(self, ids):
        result = [self.positions[i] for i in ids]
        if self.drift and 0.4 <= self.clock.now < 0.6:
            result[0] += 100
        return result

    def MoveTo(self, servo_id, position, **kwargs):
        self.positions[servo_id] = position
        return True

    def IsMoving(self, servo_id):
        return self.moving

    def SetMode(self, servo_id, mode):
        self.mode = mode
        return 0, 0

    def WritePosition(self, servo_id, position):
        if self.torque and self.positions[servo_id] != position:
            self.sent.append(self.clock.now)
        self.positions[servo_id] = position
        return True

    def SetAcceleration(self, *args):
        return True

    def SetSpeed(self, *args):
        return True

    def StartServo(self, servo_id):
        self.torque = 1
        return 0, 0

    def read1ByteTxRx(self, servo_id, address):
        if address == control.STS_ACC:
            return self.settings.get((servo_id, address), 0), 0, 0
        return (self.mode if address == control.STS_MODE else self.torque), 0, 0

    def read2ByteTxRx(self, servo_id, address):
        if address == control.STS_GOAL_SPEED_L:
            return self.settings.get((servo_id, address), 0), 0, 0
        return self.positions[servo_id], 0, 0

    def write1ByteTxRx(self, servo_id, address, value):
        self.settings[servo_id, address] = value
        return 0, 0

    write2ByteTxRx = write1ByteTxRx

    def clearParam(self):
        self.params.clear()

    def addParam(self, servo_id, data):
        self.params[servo_id] = data
        return True

    def txPacket(self):
        self.sent.append(self.clock.now)
        if not self.ignore_sync:
            for servo_id, data in self.params.items():
                self.positions[servo_id] = data[1] + (data[2] << 8)
        return 0

    @staticmethod
    def sts_lobyte(value):
        return value & 255

    @staticmethod
    def sts_hibyte(value):
        return (value >> 8) & 255

    def release_torque(self, ids):
        self.released = True

    def close(self):
        self.closed = True


class PlaybackTests(unittest.TestCase):
    def exercise(self, drift=False, playback=True, moving=True, ignore_sync=False):
        clock = Clock()
        servo = FakeServo(clock, drift, moving)
        servo.ignore_sync = ignore_sync
        class Controller:
            calls = 0
            def getController(self):
                return True
            def read(self):
                self.calls += 1
                if clock.now > 2.2 or (servo.sent and not ignore_sync):
                    raise KeyboardInterrupt
                if self.calls <= 3:
                    return Input()
                return Input(leftX=0.2, circle_button=int(playback and self.calls == 4))
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            q = servo.toAngle(control.SERVO_IDS, servo.read_positions(control.SERVO_IDS))
            (path / 'recorded_poses.jsonl').write_text(json.dumps({
                'kinematics_model': control.KINEMATICS_MODEL_ID,
                'servo_ids': control.SERVO_IDS,
                'positions': servo.read_positions(control.SERVO_IDS),
                'joint_angles_rad': q.tolist(),
            }) + '\n')
            with (patch.object(control, 'Controller', Controller),
                  patch.object(control, 'handle_STS3215', return_value=servo),
                  patch.object(control, 'time', clock),
                  patch.object(control, '__file__', str(path / 'control_arm.py')),
                  contextlib.redirect_stdout(io.StringIO())):
                control.run('mock', control.SERVO_IDS, control.REFERENCE_Q, control.DIRECTIONS)
        self.assertTrue(servo.released and servo.closed)
        return servo

    def test_stale_moving_flag_does_not_block_resume(self):
        servo = self.exercise()
        self.assertTrue(servo.sent)
        self.assertGreaterEqual(servo.sent[0], 1.0)
        self.assertLess(servo.sent[0], 1.1)

    def test_drift_restarts_one_second_hold(self):
        servo = self.exercise(drift=True)
        self.assertGreaterEqual(servo.sent[0], 1.6)
        self.assertLess(servo.sent[0], 1.7)

    def test_normal_completion(self):
        self.assertTrue(self.exercise(moving=False).sent)

    def test_playback_restores_joystick_speed_and_uses_measured_seed(self):
        with patch.object(control, 'solve_joystick_step', wraps=control.solve_joystick_step) as solve:
            servo = self.exercise()
        for servo_id in control.SERVO_IDS:
            self.assertEqual(servo.settings[servo_id, control.STS_GOAL_SPEED_L], control.CONTROLLER_SPEED)
            self.assertEqual(servo.settings[servo_id, control.STS_ACC], control.CONTROLLER_ACCELERATION)
        expected = servo.toAngle(control.SERVO_IDS, [1988, 1100, 2949, 2227, 2037])
        np.testing.assert_allclose(solve.call_args.args[1], expected)
        self.assertGreater(solve.call_args.kwargs['max_joint_step'], 0)


    def test_joystick_without_playback(self):
        self.assertLess(self.exercise(playback=False).sent[0], 0.1)

    def test_next_axis_waits_for_encoder_tracking_after_neutral(self):
        clock = Clock()

        class LaggingServo(FakeServo):
            def __init__(self):
                super().__init__(clock)
                self.positions[1] = 1320
                self.pending = self.positions.copy()

            def WritePosition(self, servo_id, position):
                if self.torque:
                    self.sent.append(clock.now)
                self.pending[servo_id] = position
                return True

        servo = LaggingServo()

        class Controller:
            def getController(self):
                return True

            def read(self):
                if clock.now >= 0.40:
                    raise KeyboardInterrupt
                if clock.now >= 0.22:
                    servo.positions.update(servo.pending)
                if clock.now < 0.10:
                    return Input(leftY=1.0)
                if clock.now < 0.14:
                    return Input()
                return Input(leftX=-1.0)

            def close(self):
                pass

        with (patch.object(control, 'Controller', Controller),
              patch.object(control, 'handle_STS3215', return_value=servo),
              patch.object(control, 'time', clock),
              contextlib.redirect_stdout(io.StringIO())):
            control.run('mock', control.SERVO_IDS, control.REFERENCE_Q, control.DIRECTIONS)

        self.assertTrue(any(t < 0.10 for t in servo.sent))
        self.assertFalse(any(0.14 <= t < 0.22 for t in servo.sent))
        self.assertTrue(any(t >= 0.22 for t in servo.sent))

    def test_y_motion_uses_other_joints_at_wrist_encoder_limit(self):
        servo = FakeServo(Clock())
        ids = control.SERVO_IDS
        arm = control.ArmKinematics()
        q = servo.toAngle(ids, [2930, 2011, 3664, 1359, 4095])
        origin = arm.position(q)
        limits = servo.command_angle_limits(ids, arm.q_min, arm.q_max)
        result, waypoint = control.solve_joystick_step(
            arm, q, origin, origin + [0.0, -0.001, 0.0],
            max_joint_step=16 * control.RADIANS_PER_TICK, joint_limits=limits,
        )
        np.testing.assert_array_equal(waypoint[[0, 2]], origin[[0, 2]])
        self.assertGreaterEqual(origin[1] - waypoint[1], 0.0005)
        counts = servo.toPosition(ids, result['q'], *limits)
        self.assertEqual(counts[-1], 4095)

    def test_joystick_works_when_broadcast_is_ignored(self):
        servo = self.exercise(playback=False, ignore_sync=True)
        self.assertTrue(servo.sent)

    def test_acknowledged_writes_deliver_goals_when_broadcast_is_ignored(self):
        servo = FakeServo(Clock())
        servo.ignore_sync = True
        control.sync_write_positions(servo, control.SERVO_IDS, [1991, 2046, 1965, 2225, 2037], 800, 50)
        self.assertEqual(servo.read_positions(control.SERVO_IDS), [1988, 1100, 2949, 2227, 2037])
        control.write_position_targets(servo, control.SERVO_IDS, [1991, 2046, 1965, 2225, 2037])
        self.assertEqual(servo.read_positions(control.SERVO_IDS), [1991, 2046, 1965, 2225, 2037])

    def test_one_joystick_step_exceeds_ik_tolerance_and_moves_encoder(self):
        servo = FakeServo(Clock())
        initial_positions = [1990, 2045, 2048, 2224, 2036]
        servo.positions.update(dict(zip(control.SERVO_IDS, initial_positions)))
        arm = control.ArmKinematics()
        q = servo.toAngle(control.SERVO_IDS, servo.read_positions(control.SERVO_IDS))
        position = arm.position(q)
        target = position + np.array([0.0, 0.0002, 0.0])
        old_tolerance = arm.solve_position_IK(
            target, seed=q, tol=0.0005, max_iter=control.CONTROLLER_IK_MAX_ITERATIONS,
        )
        new_tolerance = arm.solve_position_IK(
            target, seed=q, tol=control.IK_POSITION_TOLERANCE_M,
            max_iter=control.CONTROLLER_IK_MAX_ITERATIONS,
        )
        old_positions = servo.toPosition(
            control.SERVO_IDS, old_tolerance['q'], arm.q_min, arm.q_max,
        )
        new_positions = servo.toPosition(
            control.SERVO_IDS, new_tolerance['q'], arm.q_min, arm.q_max,
        )
        self.assertEqual(old_positions, initial_positions)
        self.assertNotEqual(new_positions, old_positions)

    def test_disabled_torque_and_wrong_mode_are_detected(self):
        servo = FakeServo(Clock())
        for mode, torque in [(0, 0), (1, 1)]:
            servo.mode, servo.torque = mode, torque
            with self.assertRaises(ConnectionError):
                control.read_servo_control_state(servo, control.SERVO_IDS)

    def test_corrupt_read_is_retried_without_resending_motion(self):
        servo = FakeServo(Clock())
        servo.torque = 1
        real_reader = servo.read1ByteTxRx
        calls = []
        def reader(servo_id, address):
            calls.append((servo_id, address))
            if len(calls) == 1:
                return 0, control.COMM_RX_CORRUPT, 0
            return real_reader(servo_id, address)
        servo.read1ByteTxRx = reader
        with patch.object(servo.portHandler.ser, 'reset_input_buffer') as reset:
            states = control.read_servo_control_state(servo, [1])
        self.assertEqual(states[0]['mode'], 0)
        self.assertEqual(calls[:2], [(1, control.STS_MODE)] * 2)
        reset.assert_called_once()
        self.assertFalse(servo.sent)

    def test_persistent_read_error_still_stops(self):
        servo = FakeServo(Clock())
        with patch.object(servo, 'read1ByteTxRx', return_value=(0, -7, 0)) as read:
            with self.assertRaises(ConnectionError):
                control.read_servo_control_state(servo, [1])
        self.assertEqual(read.call_count, control.SERVO_READ_ATTEMPTS)

    def test_servo_hardware_error_is_not_retried(self):
        servo = FakeServo(Clock())
        with patch.object(servo, 'read1ByteTxRx', return_value=(0, 0, 32)) as read:
            with self.assertRaises(ConnectionError):
                control.read_servo_control_state(servo, [1])
        read.assert_called_once()

    def test_broadcast_finishes_before_next_read(self):
        servo = FakeServo(Clock())
        events = []
        with (patch.object(servo.portHandler.ser, 'flush', side_effect=lambda: events.append('flush')),
              patch.object(servo.portHandler.ser, 'reset_input_buffer', side_effect=lambda: events.append('clear')),
              patch.object(control.time, 'sleep', side_effect=lambda delay: events.append(delay))):
            control.sync_write_positions(servo, [1], [1988], 800, 50)
        self.assertEqual(events, ['flush', control.SYNC_WRITE_SETTLE_S, 'clear'])

    def test_unacknowledged_goal_stops_writing(self):
        servo = FakeServo(Clock())
        with patch.object(servo, 'WritePosition', return_value=None) as write:
            with self.assertRaises(ConnectionError):
                control.write_position_targets(servo, [1, 2], [2000, 2000])
        write.assert_called_once_with(1, 2000)

    def test_joystick_changes_encoder_without_broadcast(self):
        with patch.object(control, 'sync_write_positions') as broadcast:
            servo = self.exercise(playback=False, ignore_sync=True)
        broadcast.assert_not_called()
        self.assertNotEqual(servo.read_positions(control.SERVO_IDS),
                            [1988, 1100, 2949, 2227, 2037])


if __name__ == '__main__':
    unittest.main()
