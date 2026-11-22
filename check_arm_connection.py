"""Read-only preflight: never enables torque or sends movement commands."""
import json
import numpy as np

import control_arm as control
from st3215.values import STS_MODE, STS_TORQUE_ENABLE, STS_STATUS


def main():
    arm = control.ArmKinematics()
    controller = control.Controller()
    servo = None
    try:
        connected = controller.getController()
        print('controller_connected:', connected, flush=True)
        if connected:
            print('controller_input:', controller.read(), flush=True)
        servo = control.handle_STS3215(control.PORT)
        print('port:', control.PORT, 'baud:', servo.st3215.portHandler.getBaudRate(), flush=True)
        for servo_id in [*control.SERVO_IDS, 6]:
            ping = servo._read_with_retry(servo.st3215.ping, servo_id)
            print('ping:', servo_id, ping, flush=True)
            if ping[1:] != (0, 0):
                continue
            state = {}
            for name, address in [('mode', STS_MODE), ('torque', STS_TORQUE_ENABLE), ('status', STS_STATUS)]:
                state[name] = servo._read_with_retry(servo.st3215.read1ByteTxRx, servo_id, address)
            print('state:', servo_id, json.dumps(state), flush=True)
        servo.setup(control.SERVO_IDS)
        print('arm_setup: PASS', flush=True)
        if not servo.load_calibration(control.SERVO_IDS):
            raise RuntimeError('Calibration missing')
        for sample in range(3):
            positions = servo.read_positions(control.SERVO_IDS)
            q = servo.toAngle(control.SERVO_IDS, positions)
            print('sample:', sample, 'positions:', positions, 'angles_deg:', np.degrees(q).round(2).tolist(), flush=True)
        valid = (q >= arm.q_min) & (q <= arm.q_max)
        print('joint_limits_ok:', valid.tolist(), flush=True)
        if not np.all(valid):
            for i in np.flatnonzero(~valid):
                print('out_of_range:', control.SERVO_IDS[i], arm.joint_names[i],
                      np.degrees([q[i], arm.q_min[i], arm.q_max[i]]).tolist(), flush=True)
            raise RuntimeError('Current pose is outside model limits')
        print('roundtrip_positions:', servo.toPosition(control.SERVO_IDS, q, arm.q_min, arm.q_max), flush=True)
        xyz = arm.position(q)
        for axis in range(3):
            for sign in (-1, 1):
                target = xyz.copy()
                target[axis] += sign * 0.0002
                result = arm.solve_position_IK(target, q, tol=control.IK_POSITION_TOLERANCE_M,
                                               max_iter=control.CONTROLLER_IK_MAX_ITERATIONS)
                print('IK_0.2mm:', axis, sign, 'success:', result['success'], 'error_m:', result['error'], flush=True)
        print('read_only_preflight: PASS (motion and write acknowledgements not tested)', flush=True)
    finally:
        if servo is not None:
            servo.close()
        controller.close()


if __name__ == '__main__':
    main()
