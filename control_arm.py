import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import json
from datetime import datetime

from controller_input import Controller
from mathmetics import ArmKinematics, Joint
from servo_com import ST3215
'''
通信を開く
↓
必要なサーボIDがすべて応答するか確認
↓
ReadPosition(id)で各関節の現在位置を取得
↓
校正情報を使って位置カウント → 関節角度[rad]に変換
↓
solve_position_IK(target, seed=現在の関節角度)
↓
result["success"] を確認（失敗したら送信しない）
↓
目標関節角度 → 位置カウントに変換し、範囲を確認
↓
位置モード・速度を設定してトルクを有効化
↓
IKで得た全サーボの目標位置を同期書き込み
'''

import time

from controller_input import Controller
from servo_com import handle_STS3215

PORT = "COM9"
SERVO_IDS = [1, 2, 3, 4, 5]
REFERENCE_Q = [0.0, 0.0, 0.0, 0.0, 0.0]
DIRECTIONS = [1,-1,-1,1,1]
CONTROLLER_SPEED = 800
CONTROLLER_ACCELERATION = 50
IK_POSITION_TOLERANCE_M = 0.0005
CONTROL_POLL_INTERVAL_S = 0.02
CONTROLLER_VELOCITY_M_S = 0.05
INPUT_SMOOTHING_S = 0.05

def run(port, servo_ids, reference_q, directions):
    arm = ArmKinematics(base_link="base_link")
    user_zero = UserZero(arm)
    if len(servo_ids) != arm.dof:
        raise ValueError("servo_ids error : unmatch with arm.dof")
    controller = Controller()
    servo = None
    commanded_xyz: NDArray[np.float64] | None = None
    commanded_q: NDArray[np.float64] | None = None
    filtered_direction = np.zeros(3, dtype=float)
    last_update = time.monotonic()
    last_error_print = 0.0
    moving_pose_q: NDArray[np.float64] | None = None
    pose_tolerance = np.deg2rad(2.0)
    deadline = 0.0

    try:
        if not controller.getController():
            raise RuntimeError("controller not found")

        servo = handle_STS3215(port)
        servo.setup(servo_ids)

        calibrated = servo.load_calibration(servo_ids)
        if calibrated:
            print("calibrated. load initial position & set user zero if press △")
        else:
            if reference_q is None or directions is None:
                raise ValueError("set reference_q and directions")
            user_zero.validate_model_q(reference_q)
            print("set initial position & press △. follow the picture")

        previous_button = bool(controller.read().tri_button)
        print(f"XYZ速度 = [+左Y, -左X, +右Y] × {CONTROLLER_VELOCITY_M_S * 1000:g}mm/s × スティック値。")
        print(f"目標更新周期: {CONTROL_POLL_INTERVAL_S * 1000:g}ms。入力を平滑化して連続送信します。")
        print("終了時は全軸のトルクを解除します。終了前にアームを支えてください。")
        print("△: 初期校正 / ユーザーゼロ設定。終了: Ctrl+C")
        print("ユーザーゼロを変更しても、目標座標の原点は変わりません。")

        record_path = Path(__file__).resolve().parent / "recorded_poses.jsonl"
        previous_rec_button = bool(controller.read().rec_button)
        previous_cir_button = bool(controller.read().circle_button)

        while True:
            inputs = controller.read()
            now = time.monotonic()
            dt = min(max(now - last_update, 0.0), 0.05)
            last_update = now
            current_rec_button = bool(inputs.rec_button)
            rec_pressed = current_rec_button and not previous_rec_button
            previous_rec_button = current_rec_button
            if rec_pressed:
                if not calibrated:
                    print("press △ first")
                else:
                    try:
                        positions = servo.read_positions(servo_ids)
                        current_q = servo.toAngle(servo_ids,positions)
                        pose = {
                            "recorded_at": datetime.now().astimezone().isoformat(),
                            "servo_ids": list(servo_ids),
                            "positions": [int(p) for p in positions],
                            "joint_angles_rad": current_q.tolist(),
                        }
                        line = json.dumps(pose, ensure_ascii=False, allow_nan=False)
                        with record_path.open("a", encoding="utf-8") as file:
                            file.write(line + "\n")
                    except (OSError, ValueError) as error:
                        print(f"failed to get position: {error}")
                    else:
                        print("success to save position")


            current_cir_button = bool(inputs.circle_button)
            cir_pressed = current_cir_button and not previous_cir_button
            previous_cir_button = current_cir_button
            if cir_pressed:
                if not calibrated:
                    print("press △ first")
                elif not record_path.exists():
                    print("no saved points")
                elif moving_pose_q is not None:
                    print("press after finishing moving")
                else:
                    try:
                        current_positions = servo.read_positions(servo_ids)
                        current_q = servo.toAngle(servo_ids,current_positions)
                        poses = []
                        with record_path.open("r",encoding="utf-8") as file:
                            for line_number, line in enumerate(file,start=1):
                                if not line.strip():
                                    continue
                                try:
                                    poses.append(json.loads(line))
                                except (OSError,ValueError) as error:
                                    raise ValueError(f"line {line_number} : data is broken") from error
                        if not poses:
                            raise ValueError("no position data")
                        moving_pose_q = move_to_pose(arm, servo, servo_ids, poses[-1])
                        deadline = time.monotonic() + 30.0
                    except ConnectionError:
                        raise
                    except (OSError,ValueError) as e:
                        print(f"failed to move to saved position: {e}")
                    else:
                        print("move to saved position")




            current_button = bool(inputs.tri_button)
            pressed = current_button and not previous_button
            previous_button = current_button

            if pressed:
                if moving_pose_q is not None:
                    print("移動中です。停止後にもう一度△を押してください")
                else:
                    if not calibrated:
                        servo.calibrate_and_save(servo_ids,reference_q,directions)
                        calibrated = True
                        print("save initial calibration")

                    positions = servo.read_positions(servo_ids)
                    current_q = servo.toAngle(servo_ids, positions)
                    user_zero.reset(current_q)
                    print("set user zero : current position")

            if moving_pose_q is not None:
                moving = [servo.st3215.IsMoving(servo_id) for servo_id in servo_ids]
                if any(value is None for value in moving):
                    raise ConnectionError("can't get status")
                reached = False
                if not any(moving):
                    actual_q = servo.toAngle(servo_ids, servo.read_positions(servo_ids))
                    actual_xyz = arm.position(actual_q)
                    angle_error = np.abs(actual_q - moving_pose_q)
                    reached = bool(np.all(angle_error <= pose_tolerance))
                if reached:
                    moving_pose_q = None
                    # 再生前のスティック目標を破棄し、次回は実測姿勢から開始する。
                    commanded_xyz = None
                    commanded_q = None
                    filtered_direction.fill(0.0)
                elif time.monotonic() >= deadline:
                    raise TimeoutError("移動完了を確認できませんでした")
            elif calibrated:
                direction = np.array([
                    inputs.leftY, -inputs.leftX, inputs.rightY,
                ], dtype=float)
                smoothing = dt / (INPUT_SMOOTHING_S + dt) if dt > 0 else 0.0
                filtered_direction += smoothing * (direction - filtered_direction)

                if np.any(direction) and commanded_xyz is None:
                    positions = servo.read_positions(servo_ids)
                    commanded_q = servo.toAngle(servo_ids, positions)
                    if np.any(commanded_q < arm.q_min) or np.any(commanded_q > arm.q_max):
                        raise ValueError("current angle is out of range before joystick control")
                    commanded_xyz = arm.position(commanded_q)
                    enable_position_control(
                        servo, servo_ids,
                        speed=CONTROLLER_SPEED,
                        acc=CONTROLLER_ACCELERATION,
                    )
                if commanded_xyz is not None and np.any(np.abs(filtered_direction) > 1e-4):
                    target = commanded_xyz + filtered_direction * CONTROLLER_VELOCITY_M_S * dt
                    try:
                        result = arm.solve_position_IK(
                            target=target,
                            seed=commanded_q,
                            tol=IK_POSITION_TOLERANCE_M,
                        )
                        if not result["success"]:
                            raise ValueError(f"IK unsuccessful. error: {result['error']:.6f} m")
                        target_positions = servo.toPosition(
                            servo_ids, result["q"], arm.q_min, arm.q_max,
                        )
                        sync_write_positions(
                            servo, servo_ids, target_positions,
                            speed=CONTROLLER_SPEED,
                            acc=CONTROLLER_ACCELERATION,
                        )
                        commanded_xyz = target
                        commanded_q = result["q"].copy()
                    except ValueError as error:
                        if now - last_error_print >= 1.0:
                            print(error)
                            last_error_print = now

            time.sleep(CONTROL_POLL_INTERVAL_S)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if servo is not None:
                try:
                    servo.release_torque(servo_ids)
                finally:
                    servo.close()
        finally:
            controller.close()


def move_to_xyz(
    arm, servo, servo_ids, target, speed=200, acc=10,
    position_tolerance=0.002, enable_before_move=False, before_send=None,
):
    servo_ids = tuple(servo_ids)
    if len(servo_ids) != arm.dof:
        raise ValueError("the number error")
    if len(set(servo_ids)) != len(servo_ids):
        raise ValueError("servo_ids must be identical")
    target = arm.validate_vector(target,3,"target")
    current_positions = servo.read_positions(servo_ids)
    current_q = servo.toAngle(servo_ids,current_positions)
    if np.any(current_q < arm.q_min) or np.any(current_q > arm.q_max):
        details = []
        for i, servo_id in enumerate(servo_ids):
            if current_q[i] < arm.q_min[i] or current_q[i] > arm.q_max[i]:
                details.append(
                    f"ID {servo_id} ({arm.joint_names[i]}): "
                    f"{np.degrees(current_q[i]):.2f} deg, "
                    f"limits [{np.degrees(arm.q_min[i]):.2f}, "
                    f"{np.degrees(arm.q_max[i]):.2f}] deg, "
                    f"raw={current_positions[i]}, zero={servo.zero_positions[servo_id]:.1f}"
                )
        raise ValueError("current angle is out of range: " + "; ".join(details))


    result = arm.solve_position_IK(
        target=target,
        seed=current_q,
        tol=position_tolerance,
    )
    if not result["success"]:
        raise ValueError(f"IK unsuccessed. err : {result['error']:.6f} m")

    target_positions = servo.toPosition(servo_ids,result["q"],arm.q_min,arm.q_max)

    if before_send is not None:
        before_send()
    if enable_before_move:
        enable_position_control(servo, servo_ids)

    for servo_id,position in zip(servo_ids,target_positions):
        success = servo.st3215.MoveTo(servo_id,position,speed=speed,acc=acc,wait=False)
        if success is not True:
            raise ConnectionError(f"moving ID {servo_id} is failed")
    return result

def move_to_pose(arm, servo, servo_ids, pose, speed=200, acc=10):
    servo_ids = list(servo_ids)
    if not isinstance(pose,dict):
        raise ValueError("pose must be dict")
    if pose.get("joint_angles_rad") is None or pose.get("servo_ids") is None:
            raise ValueError("data fomat is incorrect")
    if pose["servo_ids"] != servo_ids:
        raise ValueError("servoIds don't match with data")
    q = arm.validate_q(pose["joint_angles_rad"])
    positions = servo.toPosition(servo_ids,q,arm.q_min,arm.q_max)
    enable_position_control(servo,servo_ids,speed=speed,acc=acc)
    for servo_id, position in zip(servo_ids,positions):
        success = servo.st3215.MoveTo(servo_id,position,speed=speed,acc=acc,wait=False)
        if success is not True:
            raise ConnectionError(f"ID {servo_id} : failed to move")
    return q.copy()

def enable_position_control(servo,servo_ids,speed=800,acc=50):
    positions = servo.read_positions(servo_ids)
    for servo_id,position in zip(servo_ids,positions):
        comm, error = servo.st3215.SetMode(servo_id,0)
        if comm != 0 or error != 0:
            raise ConnectionError(f"setting ID {servo_id} is failed")
        if servo.st3215.WritePosition(servo_id,position) is not True:
            raise ConnectionError(f"setting static angle to ID {servo_id} is failed")
        if servo.st3215.SetAcceleration(servo_id,acc) is not True:
            raise ConnectionError(f"setting acceleration for ID {servo_id} failed")
        if servo.st3215.SetSpeed(servo_id,speed) is not True:
            raise ConnectionError(f"setting speed for ID {servo_id} failed")
    for servo_id in servo_ids:
        comm, error = servo.st3215.StartServo(servo_id)
        if comm != 0 or error != 0:
            raise ConnectionError(f"ID {servo_id} : failed to start servo")

def sync_write_positions(servo,servo_ids,positions,speed,acc):
    """全サーボの位置目標を1パケットで同時更新する。"""
    if len(servo_ids) != len(positions):
        raise ValueError("servo_ids and positions don't match")

    writer = servo.st3215.groupSyncWrite
    writer.clearParam()
    for servo_id,position in zip(servo_ids,positions):
        data = [
            acc,
            servo.st3215.sts_lobyte(position),
            servo.st3215.sts_hibyte(position),
            0, 0,
            servo.st3215.sts_lobyte(speed),
            servo.st3215.sts_hibyte(speed),
        ]
        if not writer.addParam(servo_id,data):
            writer.clearParam()
            raise ValueError(f"could not prepare sync write for ID {servo_id}")

    result = writer.txPacket()
    writer.clearParam()
    if result != 0:
        raise ConnectionError(f"sync position command failed: comm={result}")

class UserZero:
    def __init__(self, arm):
        self.arm = arm
        self.offset = np.zeros(arm.dof)

    def validate_model_q(self, q):
        q = self.arm.validate_q(q)

        if np.any(q < self.arm.q_min) or np.any(q > self.arm.q_max):
            raise ValueError("out of URDF range")

        return q

    def reset(self, current_q):
        self.offset = self.validate_model_q(current_q).copy()

    def to_user(self, model_q):
        return self.validate_model_q(model_q) - self.offset

    def to_model(self, user_q):
        model_q = self.arm.validate_q(user_q) + self.offset
        return self.validate_model_q(model_q)

    def user_limits(self):
        return (self.arm.q_min - self.offset,self.arm.q_max - self.offset,)

if __name__ == "__main__":
    run(port=PORT,
        servo_ids=SERVO_IDS,
        reference_q=REFERENCE_Q,
        directions=DIRECTIONS,)
