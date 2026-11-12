import numpy as np

from controller_input import Controller
from mathmetics import ArmKinematics, Joint
from servo_com import ST3215


'''
初期状態の決定
pingservo(id)でservoがあるかチェック
listservos()→5個独立にあるかチェック
stopservo(id)→あるなら確実に止めておく
setmode(id,mode)→適切なモードに設定
setacc(id,acc)→適切に設定
setspeed(id,speed)→適切に設定
startservo(id)→サーボが使えるようにする
definemiddle(id)

今の状態を把握
readstatus→angle:返ってくる配列[4]
↓
目標位置の入力
angle→qに代入
solve_position_IK
返ってきたものの"q"が角度出力
↓
servoへの出力
moveto→(id,position),speed,accもいじれる
'''

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

PORT = "COM7"
SERVO_IDS = [1, 2, 3, 4, 5]
REFERENCE_Q = [0.0, 0.0, 0.0, 0.0, 0.0]
DIRECTIONS = [1,1,1,1,1]

def run(port, servo_ids, reference_q, directions):
    arm = ArmKinematics(base_link="base_link")
    user_zero = UserZero(arm)
    if len(servo_ids) != arm.dof:
        raise ValueError("servo_ids error : unmatch with arm.dof")
    controller = Controller()
    servo = None

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

        while True:
            inputs = controller.read()
            current_button = bool(inputs.tri_button)
            pressed = current_button and not previous_button
            previous_button = current_button

            if pressed:
                if not calibrated:
                    servo.calibrate_and_save(servo_ids,reference_q,directions)
                    calibrated = True
                    print("save initial calibration")

                positions = servo.read_positions(servo_ids)
                current_q = servo.toAngle(servo_ids, positions)
                user_zero.reset(current_q)
                print("set user zero : current position")

            if calibrated:
                positions = servo.read_positions(servo_ids)
                current_q = servo.toAngle(servo_ids, positions)
                user_q = user_zero.to_user(current_q)
                print("user angle [rad] : ",user_q)
                # IKにはユーザー角度ではなくcurrent_qを渡す
                # result = arm.solve_position_IK(target, seed=current_q)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if servo is not None:
                servo.close()
        finally:
            controller.close()

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
