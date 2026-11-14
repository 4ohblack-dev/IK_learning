import numpy as np
from numpy.typing import NDArray

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
位置モード・保持目標を整えてトルクを有効化
↓
MoveTo()で各サーボへ目標を送信
'''

import time

from controller_input import Controller
from servo_com import handle_STS3215

PORT = "COM9"
SERVO_IDS = [1, 2, 3, 4, 5]
REFERENCE_Q = [0.0, 0.0, 0.0, 0.0, 0.0]
DIRECTIONS = [1,-1,-1,1,1]
CONTROLLER_STEP_M = 0.002  # 各軸の最大移動量 [m/指令]

def run(port, servo_ids, reference_q, directions):
    arm = ArmKinematics(base_link="base_link")
    user_zero = UserZero(arm)
    if len(servo_ids) != arm.dof:
        raise ValueError("servo_ids error : unmatch with arm.dof")
    controller = Controller()
    servo = None
    moving_target: NDArray[np.float64] | None = None
    deadline = 0.0
    enabled = False
    motion_started = False

    def mark_motion_started():
        nonlocal motion_started
        motion_started = True

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
        print("左スティック: 横→-Y、縦→X。右スティック上下: Z。")
        print("終了時は全軸のトルクを解除します。終了前にアームを支えてください。")
        print("△: 初期校正 / ユーザーゼロ設定。終了: Ctrl+C")
        print("ユーザーゼロを変更しても、目標座標の原点は変わりません。")

        print(f"左スティック: X/Y、右スティック上下: Z。各軸は1回最大{CONTROLLER_STEP_M * 1000:g}mm移動します。")

        while True:
            inputs = controller.read()
            current_button = bool(inputs.tri_button)
            pressed = current_button and not previous_button
            previous_button = current_button

            if pressed:
                if moving_target is not None:
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

            if moving_target is not None:
                moving = [servo.st3215.IsMoving(servo_id) for servo_id in servo_ids]
                if any(value is None for value in moving):
                    raise ConnectionError("can't get status")
                if not any(moving):
                    actual_q = servo.toAngle(servo_ids, servo.read_positions(servo_ids))
                    actual_xyz = arm.position(actual_q)
                    print("実測モデル位置 [m]:", actual_xyz)
                    print("目標との距離 [m]:", np.linalg.norm(moving_target - actual_xyz))
                    print("ユーザー角度 [rad]:", user_zero.to_user(actual_q))
                    moving_target = None
                elif time.monotonic() >= deadline:
                    raise TimeoutError("移動完了を確認できませんでした")
            elif calibrated:
                # read()済みの入力を使い、現在の手先位置から相対移動する。
                # 実機確認結果に合わせ、入力X/Yを入れ替えて符号を補正する。
                direction = np.array([
                    inputs.leftY, -inputs.leftX, inputs.rightY,
                ], dtype=float)
                if np.any(direction):
                    try:
                        positions = servo.read_positions(servo_ids)
                        current_q = servo.toAngle(servo_ids, positions)
                        current_xyz = arm.position(current_q)
                        target = current_xyz + direction * CONTROLLER_STEP_M

                        move_to_xyz(
                            arm, servo, servo_ids, target,
                            enable_before_move=not enabled,
                            before_send=mark_motion_started,
                        )
                        enabled = True
                        moving_target = target
                        deadline = time.monotonic() + 30.0
                    except ValueError as error:
                        print(error)

            time.sleep(0.05)

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


def move_to_xyz(arm,servo,servo_ids,target,speed=200,acc=10,enable_before_move=False,before_send=None):
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


    result = arm.solve_position_IK(target=target,seed=current_q)
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

def enable_position_control(servo,servo_ids):
    positions = servo.read_positions(servo_ids)
    for servo_id,position in zip(servo_ids,positions):
        comm, error = servo.st3215.SetMode(servo_id,0)
        if comm != 0 or error != 0:
            raise ConnectionError(f"setting ID {servo_id} is failed")
        if servo.st3215.WritePosition(servo_id,position) is not True:
            raise ConnectionError(f"setting static angle to ID {servo_id} is failed")
    for servo_id in servo_ids:
        comm, error = servo.st3215.StartServo(servo_id)
        if comm != 0 or error != 0:
            raise ConnectionError(f"ID {servo_id} : failed to start servo")

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
