import numpy as np
from numpy.typing import NDArray

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
from queue import Empty, Queue
from threading import Event, Thread

from controller_input import Controller
from servo_com import handle_STS3215

PORT = "COM6"
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
    targets: Queue[NDArray[np.float64] | None] = Queue(maxsize=1)
    input_ready = Event()
    input_stop = Event()
    input_thread = None
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
        print("目標位置はベース基準の x y z [m]。対象点は最終リンクの原点です。")
        print("△: 初期校正 / ユーザーゼロ設定。終了: Ctrl+C または q")
        print("ユーザーゼロを変更しても、目標座標の原点は変わりません。")

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

            if calibrated and input_thread is None:
                # input()だけを別スレッドへ分離。機器操作はrun()で行う。
                input_ready.set()
                input_thread = Thread(
                    target=coordinate_input_loop,
                    args=(targets, input_ready, input_stop),
                    daemon=True,
                )
                input_thread.start()

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
                    input_ready.set()
                elif time.monotonic() >= deadline:
                    raise TimeoutError("移動完了を確認できませんでした")
            elif calibrated:
                try:
                    target = targets.get_nowait()
                except Empty:
                    time.sleep(0.05)
                    continue
                if target is None:
                    break
                if not isinstance(target, str):
                    try:
                        # 全目標の検証後、初回の移動時だけ保持を有効化する。
                        move_to_xyz(
                            arm, servo, servo_ids, target,
                            enable_before_move=not enabled,
                            before_send=mark_motion_started,
                        )
                        enabled = True
                        moving_target = target
                        deadline = time.monotonic() + 30.0
                        print("移動指令を送信しました")
                    except ValueError as error:
                        print(error)
                        input_ready.set()

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        input_stop.set()
        input_ready.set()
        try:
            if servo is not None:
                try:
                    if motion_started:
                        # 通信を閉じるだけでは動作は停止しないため、現在位置を目標にする。
                        # トルク解除は自重落下を招くため、ここでは行わない。
                        for servo_id in servo_ids:
                            try:
                                position = servo.st3215.ReadPosition(servo_id)
                                if position is None or servo.st3215.WritePosition(servo_id, position) is not True:
                                    raise ConnectionError("hold failed")
                            except Exception as error:
                                print(f"ID {servo_id}: 保持を確認できません: {error}")
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
        raise ValueError("current angle is out of range")


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

def coordinate_input_loop(
    targets: Queue[NDArray[np.float64] | None],
    input_ready: Event,
    input_stop: Event,
):
    """目標座標の入力だけを担当し、サーボにはアクセスしない。"""
    while not input_stop.is_set():
        input_ready.wait()
        if input_stop.is_set():
            return
        input_ready.clear()
        try:
            text = input("xyz [m] > ").strip()
        except EOFError:
            text = "q"
        if input_stop.is_set():
            return
        if text.lower() == "q":
            targets.put(None)
            return
        try:
            target = np.array([float(value) for value in text.split()], dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("有限の数値を3個、x y z の順に入力してください")
        except ValueError as error:
            print(error)
            input_ready.set()
            continue
        targets.put(target)

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
