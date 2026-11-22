"""Configure the single connected gripper without moving it."""
import json
import time
from datetime import datetime
from pathlib import Path

from servo_com import handle_STS3215
from st3215.values import STS_ID, STS_BAUD_RATE, STS_MODE, STS_LOCK, STS_TORQUE_ENABLE


def setup(port='COM9', target_id=6):
    servo = handle_STS3215(port)
    bus = servo.st3215
    try:
        ids = [i for i in range(254) if bus.PingServo(i)]
        if len(ids) != 1:
            raise RuntimeError(f'Connect exactly one servo; detected {ids}')
        old_id = ids[0]

        def read(servo_id, address):
            value, comm, error = servo._read_with_retry(bus.read1ByteTxRx, servo_id, address)
            if comm or error:
                raise ConnectionError((servo_id, address, comm, error))
            return value

        registers = {'id': STS_ID, 'baud_code': STS_BAUD_RATE, 'mode': STS_MODE,
                     'lock': STS_LOCK, 'torque': STS_TORQUE_ENABLE}
        before = {name: read(old_id, address) for name, address in registers.items()}
        path = Path(__file__).parent / ('gripper_setup_before_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        path.write_text(json.dumps({'port': port, 'baud': bus.portHandler.getBaudRate(),
                                    'registers': before, 'position': bus.ReadPosition(old_id)}, indent=2))
        print('Backup:', path, flush=True)
        servo.release_torque([old_id])
        if bus.write1ByteTxRx(old_id, STS_LOCK, 0) != (0, 0):
            raise ConnectionError('EEPROM unlock failed')
        time.sleep(0.02)
        try:
            if read(old_id, STS_LOCK) != 0:
                raise ConnectionError('EEPROM remains locked')
            if bus.SetMode(old_id, 0) != (0, 0):
                raise ConnectionError('Position mode setup failed')
            if old_id != target_id:
                # An ID change can reply using either address. Readback is authoritative.
                reply = bus.write1ByteTxRx(old_id, STS_ID, target_id)
                print('ID write reply:', reply, flush=True)
                bus.portHandler.ser.flush()
                time.sleep(0.05)
                bus.portHandler.ser.reset_input_buffer()
            if read(target_id, STS_ID) != target_id:
                raise ConnectionError('New ID verification failed')
        finally:
            # Lock the NEW ID (the library ChangeId incorrectly locks the old ID).
            for servo_id in set((old_id, target_id)):
                if bus.PingServo(servo_id):
                    bus.write1ByteTxRx(servo_id, STS_LOCK, 1)
                    time.sleep(0.02)
            bus.portHandler.ser.reset_input_buffer()
        after = {name: read(target_id, address) for name, address in registers.items()}
        if after != dict(before, id=target_id, mode=0, lock=1, torque=0):
            raise ConnectionError(f'Unexpected settings: {after}')
        print('Verified:', after, 'baud=', bus.portHandler.getBaudRate(), flush=True)
        print('Position:', bus.ReadPosition(target_id), flush=True)
    finally:
        servo.close()


if __name__ == '__main__':
    setup()
