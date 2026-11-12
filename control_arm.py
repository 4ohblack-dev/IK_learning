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

def main():
    arm = ArmKinematics()
    target = np.array([0.15, 0.0, 0.20])
    seed = np.array([0.0, 1.75, -1.57, -0.5, 0.0])#今のsts3215の角度
    result = arm.solve_position_IK(target=target,seed=seed,)

    print("goal [m]:", target)
    print("result:", result)
    print("position from FK [m]:", arm.position(result["q"]))


if __name__ == "__main__":
    main()
