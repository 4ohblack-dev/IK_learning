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
REFERENCE_Q = [0.0, 1.75, -1.57, -0.5, 0.0]
DIRECTIONS = [1,1,1,1,1]

def run(port, servo_ids, reference_q, directions):
    controller = Controller()
    servo = None

    try:
        if not controller.getController():
            raise RuntimeError("コントローラがありません")

        servo = handle_STS3215(port)
        servo.setup(servo_ids)

        calibrated = servo.load_calibration(servo_ids)
        print("校正済み" if calibrated else "未校正です")
        print("校正姿勢に合わせて、△ボタンを押してください")

        # 起動時から押されている場合は、いったん離してから受け付ける
        previous_button = bool(controller.read().tri_button)

        while True:
            inputs = controller.read()
            current_button = bool(inputs.tri_button)

            # 押した瞬間だけ校正する
            if current_button and not previous_button:
                calibrated = False

                servo.calibrate_and_save(
                    servo_ids,
                    reference_q,
                    directions,
                )

                calibrated = True
                print("校正値を保存しました")

            previous_button = current_button

            if calibrated:
                positions = servo.read_positions(servo_ids)
                current_q = servo.toAngle(servo_ids, positions)

                print("関節角度 [rad]:", current_q)
                # IKには、このcurrent_qをseedとして渡す
                # result = arm.solve_position_IK(target, current_q)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        if servo is not None:
            servo.close()
        controller.close()



if __name__ == "__main__":
    run(port=PORT,
        servo_ids=SERVO_IDS,
        reference_q=REFERENCE_Q,
        directions=DIRECTIONS,)
