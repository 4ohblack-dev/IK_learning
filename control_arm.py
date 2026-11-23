import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import json
import sys
from datetime import datetime

from controller_input import Controller
from mathmetics import ArmKinematics, IKResult, KINEMATICS_MODEL_ID
from servo_com import CALIBRATION_REFERENCE, RADIANS_PER_TICK, ST3215
from st3215.values import (
    STS_MODE, STS_TORQUE_ENABLE, STS_GOAL_POSITION_L, STS_ACC, STS_GOAL_SPEED_L,
    COMM_RX_TIMEOUT, COMM_RX_CORRUPT,
)
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
# Native URDF angles for upright upper-arm and forward horizontal forearm/wrist.
REFERENCE_Q = [0.0, 1.55621943, -1.28847511, -0.60932432, 0.0]
DIRECTIONS = [-1,-1,-1,1,1]
CONTROLLER_SPEED = 800
CONTROLLER_ACCELERATION = 50
IK_POSITION_TOLERANCE_M = 0.00005
CONTROLLER_IK_MAX_ITERATIONS = 20
CONTROLLER_IK_DAMPING = 0.02
CONTROL_POLL_INTERVAL_S = 0.02
CONTROLLER_VELOCITY_M_S = 0.05
TRACKING_TOLERANCE_M = 0.0015
TRACKING_TOLERANCE_TICKS = 6
INPUT_SMOOTHING_S = 0.05
JOYSTICK_DEADZONE = 0.10
LEFT_STICK_CROSS_AXIS_RATIO = 0.25
JOYSTICK_RESUME_DELAY_S = 1.0
SERVO_READ_ATTEMPTS = 3
SERVO_READ_RETRY_DELAY_S = 0.005
SYNC_WRITE_SETTLE_S = 0.002
MEASURED_LIMIT_TOLERANCE_RAD = np.deg2rad(2.0)


def measured_control_q(arm, q, servo_ids):
    """Allow small measured boundary deviations; command limits stay strict."""
    q = arm.validate_q(q)
    bounded = np.clip(q, arm.q_min, arm.q_max)
    outside = np.abs(q - bounded) > MEASURED_LIMIT_TOLERANCE_RAD
    if np.any(outside):
        details = [
            f"ID {servo_ids[i]} ({arm.joint_names[i]}): {np.degrees(q[i]):.2f} deg, "
            f"limits [{np.degrees(arm.q_min[i]):.2f}, {np.degrees(arm.q_max[i]):.2f}] deg"
            for i in np.flatnonzero(outside)
        ]
        raise ValueError("操作開始を保留: " + "; ".join(details)
                         + "。姿勢・校正値を確認してください（許容ずれ2度）。")
    return bounded


def joystick_direction(inputs):
    """Filter stick noise in base XYZ, preserving deliberate diagonal input."""
    direction = np.array([inputs.leftX, inputs.leftY, inputs.rightY], dtype=float)
    direction[np.abs(direction) < JOYSTICK_DEADZONE] = 0.0
    left_magnitudes = np.abs(direction[:2])
    direction[:2] = np.where(
        left_magnitudes < LEFT_STICK_CROSS_AXIS_RATIO * left_magnitudes.max(),
        0.0,
        direction[:2],
    )
    return direction


def update_filtered_direction(filtered, direction, dt):
    """Smooth speed while keeping the current stick direction."""
    magnitude = np.linalg.norm(direction)
    if magnitude == 0.0:
        filtered.fill(0.0)
        return
    speed = np.linalg.norm(filtered)
    speed += (dt / (INPUT_SMOOTHING_S + dt)) * (min(magnitude, 1.0) - speed)
    filtered[:] = direction * (speed / magnitude)


def run(port, servo_ids, reference_q, directions):
    arm = ArmKinematics()
    user_zero = UserZero(arm)
    if len(servo_ids) != arm.dof:
        raise ValueError("servo_ids error : unmatch with arm.dof")
    controller = Controller()
    servo = None
    commanded_xyz: NDArray[np.float64] | None = None
    goal: tuple[NDArray[np.int64], NDArray[np.float64]] | None = None
    command_limits = None
    joystick_ready = False
    wait_for_neutral = False
    filtered_direction = np.zeros(3, dtype=float)
    last_update = time.monotonic()
    last_error_print = 0.0
    moving_pose_q: NDArray[np.float64] | None = None
    pose_tolerance = np.deg2rad(2.0)
    deadline = 0.0
    joystick_resume_at: float | None = None
    last_diagnostic = 0.0

    try:
        if not controller.getController():
            raise RuntimeError("controller not found")

        servo = handle_STS3215(port)
        servo.setup(servo_ids)

        calibrated = servo.load_calibration(servo_ids)
        if calibrated:
            command_limits = servo.command_angle_limits(servo_ids, arm.q_min, arm.q_max)
            print("calibrated. load initial position & set user zero if press △")
        else:
            if reference_q is None or directions is None:
                raise ValueError("set reference_q and directions")
            user_zero.validate_model_q(reference_q)
            print("新しい基準姿勢の校正が必要です。IMPORTS.md の手順で姿勢を合わせてから △ を押してください。")

        previous_button = bool(controller.read().tri_button)
        print(f"XYZ速度 = [+左X, +左Y, +右Y] × {CONTROLLER_VELOCITY_M_S * 1000:g}mm/s × スティック値。")
        print("左スティック上下: X方向、左右: Y方向、右スティック上下: Z方向")
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
            dt = min(max(now - last_update, 0.0), CONTROL_POLL_INTERVAL_S)
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
                            "kinematics_model": KINEMATICS_MODEL_ID,
                            "reference_pose": CALIBRATION_REFERENCE,
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
                elif joystick_resume_at is not None:
                    print("ジョイスティック操作の再開待ちです")
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
                                    pose = json.loads(line)
                                except json.JSONDecodeError as error:
                                    raise ValueError(f"line {line_number} : data is broken") from error
                                if pose.get("servo_ids") == list(servo_ids) and pose.get("positions") is not None:
                                    pose["joint_angles_rad"] = servo.toAngle(
                                        servo_ids, pose["positions"],
                                    ).tolist()
                                    pose["kinematics_model"] = KINEMATICS_MODEL_ID
                                elif (pose.get("kinematics_model") != KINEMATICS_MODEL_ID
                                      or pose.get("reference_pose") != CALIBRATION_REFERENCE):
                                    raise ValueError(f"line {line_number}: saved pose needs encoder positions")
                                poses.append(pose)
                        if not poses:
                            raise ValueError("no position data")
                        moving_pose_q = move_to_pose(arm, servo, servo_ids, poses[-1])
                        commanded_xyz = None
                        goal = None
                        joystick_ready = False
                        filtered_direction.fill(0.0)
                        joystick_resume_at = None
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
                        command_limits = servo.command_angle_limits(servo_ids, arm.q_min, arm.q_max)
                        calibrated = True
                        wait_for_neutral = True
                        print("校正を保存しました。スティックを中央に戻すと操作できます。")

                    positions = servo.read_positions(servo_ids)
                    current_q = servo.toAngle(servo_ids, positions)
                    try:
                        user_zero.reset(measured_control_q(arm, current_q, servo_ids))
                    except ValueError as error:
                        print(error)
                    else:
                        print("set user zero : current position")

            if moving_pose_q is not None:
                moving = [servo.st3215.IsMoving(servo_id) for servo_id in servo_ids]
                if any(value is None for value in moving):
                    raise ConnectionError("can't get status")
                actual_q = servo.toAngle(servo_ids, servo.read_positions(servo_ids))
                angle_error = np.abs(actual_q - moving_pose_q)
                in_tolerance = bool(np.all(angle_error <= pose_tolerance))
                checked_at = time.monotonic()
                # 移動中フラグではなく、実測角度が1秒間許容差内にあることを確認する。
                if in_tolerance:
                    if joystick_resume_at is None:
                        joystick_resume_at = checked_at + JOYSTICK_RESUME_DELAY_S
                        print("目標姿勢に到着しました。1秒間の保持を確認します")
                    elif checked_at >= joystick_resume_at:
                        moving_pose_q = None
                        joystick_resume_at = None
                        commanded_xyz = None
                        goal = None
                        filtered_direction.fill(0.0)
                        print("ジョイスティック操作を再開しました")
                else:
                    joystick_resume_at = None
                if moving_pose_q is not None:
                    if checked_at - last_diagnostic >= 1.0:
                        print("再生待ち: 誤差[deg]=", np.round(np.degrees(angle_error), 2),
                              "moving=", moving)
                        last_diagnostic = checked_at
                    if checked_at >= deadline:
                        raise TimeoutError(
                            f"姿勢に到達できません: 誤差[deg]={np.degrees(angle_error)}, moving={moving}"
                        )
            elif calibrated:
                direction = joystick_direction(inputs)
                update_filtered_direction(filtered_direction, direction, dt)

                if not np.any(direction):
                    # Re-anchor the next stroke only after the pending goal has settled.
                    commanded_xyz = None
                    wait_for_neutral = False
                elif not wait_for_neutral and dt > 0 and command_limits is not None:
                    try:
                        positions = np.asarray(servo.read_positions(servo_ids))
                        measured_q = servo.toAngle(servo_ids, positions)
                        measured_xyz = arm.position(measured_q)
                        previous_goal = positions
                        if goal is not None:
                            goal_positions, goal_xyz = goal
                            previous_goal = goal_positions
                            lag_ticks = np.max(np.abs(positions - goal_positions))
                            lag_m = np.linalg.norm(measured_xyz - goal_xyz)
                            if lag_ticks > TRACKING_TOLERANCE_TICKS or lag_m > TRACKING_TOLERANCE_M:
                                raise ValueError(f"爪先の追従待ち: {lag_m * 1000:.2f} mm, {lag_ticks} ticks")

                        seed = np.clip(measured_control_q(arm, measured_q, servo_ids), *command_limits)
                        max_joint_step = (CONTROLLER_SPEED * RADIANS_PER_TICK * dt
                                          - np.max(np.abs(seed - measured_q)))
                        if max_joint_step <= 0:
                            raise ValueError("実測角度から関節範囲に戻るには速度上限を超えます。")
                        if commanded_xyz is None:
                            bounded_positions = servo.toPosition(servo_ids, seed, *command_limits)
                            if np.any(np.abs(positions - bounded_positions) * RADIANS_PER_TICK
                                      > MEASURED_LIMIT_TOLERANCE_RAD):
                                raise ValueError("実測角度が校正済みのエンコーダ範囲外です。")
                            if not joystick_ready:
                                enable_position_control(
                                    servo, servo_ids, speed=CONTROLLER_SPEED, acc=CONTROLLER_ACCELERATION,
                                )
                                states = read_servo_control_state(servo, servo_ids)
                                print("ジョイスティック制御開始: サーボ状態", states)
                                joystick_ready = True
                                # Torque setup may move the joints. Seed the line on the next read.
                                time.sleep(CONTROL_POLL_INTERVAL_S)
                                continue
                            commanded_xyz = measured_xyz

                        target = commanded_xyz + filtered_direction * CONTROLLER_VELOCITY_M_S * dt
                        result, target = solve_joystick_step(
                            arm, seed, commanded_xyz, target,
                            max_joint_step=max_joint_step,
                            joint_limits=command_limits,
                        )
                        target_positions = servo.toPosition(servo_ids, result["q"], *command_limits)
                        # Broadcast success only confirms transmission. Use acknowledged
                        # goal writes so a rejected command cannot silently advance IK.
                        if not np.array_equal(target_positions, previous_goal):
                            write_position_targets(servo, servo_ids, target_positions)
                            goal_positions = np.asarray(target_positions)
                            goal = goal_positions, arm.position(servo.toAngle(servo_ids, goal_positions))
                        commanded_xyz = target
                        if now - last_diagnostic >= 1.0:
                            print("スティック=", direction,
                                  "爪先 実測FK[mm]=", np.round(1000 * measured_xyz, 1),
                                  "目標[mm]=", np.round(1000 * target, 1))
                            last_diagnostic = now
                    except ValueError as error:
                        if now - last_error_print >= 1.0:
                            print(error)
                            last_error_print = now

            time.sleep(CONTROL_POLL_INTERVAL_S)

    except KeyboardInterrupt:
        pass
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        actions = []
        if servo is not None:
            actions.extend([lambda: servo.release_torque(servo_ids), servo.close])
        actions.append(controller.close)
        for action in actions:
            try:
                action()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(f"終了処理でもエラー: {error}")
            else:
                for error in cleanup_errors[1:]:
                    cleanup_errors[0].add_note(str(error))
                raise cleanup_errors[0]


def solve_joystick_step(
    arm: ArmKinematics, seed, origin, target, max_joint_step=None, *, joint_limits=None,
) -> tuple[IKResult, NDArray[np.float64]]:
    """Solve on the Cartesian segment under encoder and per-cycle joint limits."""
    if max_joint_step is not None and (not np.isfinite(max_joint_step) or max_joint_step < 0):
        raise ValueError("max_joint_step must be finite and nonnegative")
    if max_joint_step is not None:
        lower, upper = (arm.q_min, arm.q_max) if joint_limits is None else joint_limits
        joint_limits = np.maximum(lower, seed - max_joint_step), np.minimum(upper, seed + max_joint_step)
    delta = target - origin
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = origin + scale * delta
        result = arm.solve_position_IK(
            target=candidate, seed=seed, tol=IK_POSITION_TOLERANCE_M,
            max_iter=CONTROLLER_IK_MAX_ITERATIONS, damping=CONTROLLER_IK_DAMPING,
            joint_limits=joint_limits,
        )
        if result["success"]:
            # Preserve the ideal line while measured FK lags or differs by rounding.
            return result, candidate
    raise ValueError(f"IK unsuccessful (workspace, encoder or joint speed limit). error: {result['error']:.6f} m")


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
        write_verified_setting(servo, servo_id, STS_ACC, acc, width=1, name="acceleration")
        write_verified_setting(servo, servo_id, STS_GOAL_SPEED_L, speed, width=2, name="speed")
    for servo_id in servo_ids:
        comm, error = servo.st3215.StartServo(servo_id)
        if comm != 0 or error != 0:
            raise ConnectionError(f"ID {servo_id} : failed to start servo")

def write_verified_setting(servo, servo_id, address, value, *, width, name):
    """Recover a lost write reply only when readback proves the requested value."""
    writer = servo.st3215.write1ByteTxRx if width == 1 else servo.st3215.write2ByteTxRx
    reader = servo.st3215.read1ByteTxRx if width == 1 else servo.st3215.read2ByteTxRx
    comm, error = writer(servo_id, address, value)
    if error or comm not in (0, COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
        raise ConnectionError(f"ID {servo_id}: {name} write failed comm={comm}, error={error}")
    if comm != 0:
        time.sleep(SERVO_READ_RETRY_DELAY_S)
        servo.st3215.portHandler.ser.reset_input_buffer()
    actual, read_comm, read_error = servo._read_with_retry(reader, servo_id, address)
    if read_comm or read_error or actual != value:
        raise ConnectionError(
            f"ID {servo_id}: {name} verification failed expected={value}, actual={actual}, "
            f"write_comm={comm}, read_comm={read_comm}, error={read_error}"
        )
    if comm != 0:
        print(f"ID {servo_id}: {name} 応答エラー({comm})。読み戻しで設定値 {value} を確認しました")


def read_servo_control_state(servo, servo_ids):
    """制御レジスタを読み戻し、通信成功と位置制御・トルクONを確認する。"""
    states = []
    for servo_id in servo_ids:
        state = {"id": servo_id}
        for name, address, reader in (
            ("mode", STS_MODE, servo.st3215.read1ByteTxRx),
            ("torque", STS_TORQUE_ENABLE, servo.st3215.read1ByteTxRx),
            ("goal", STS_GOAL_POSITION_L, servo.st3215.read2ByteTxRx),
        ):
            value, comm, error = reader(servo_id, address)
            for _ in range(SERVO_READ_ATTEMPTS - 1):
                if comm not in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
                    break
                # 読み取りだけを再試行する。遅延した応答・破損した残りを除去する。
                time.sleep(SERVO_READ_RETRY_DELAY_S)
                servo.st3215.portHandler.ser.reset_input_buffer()
                value, comm, error = reader(servo_id, address)
            if comm != 0 or error != 0:
                raise ConnectionError(f"ID {servo_id}: {name} 読み取り失敗 comm={comm}, error={error}")
            state[name] = value
        if state["mode"] != 0 or state["torque"] != 1:
            raise ConnectionError(f"位置制御が有効ではありません: {state}")
        states.append(state)
    return states


def write_position_targets(servo, servo_ids, positions):
    """Write goals with per-servo acknowledgement; speed/acc are set on enable."""
    if len(servo_ids) != len(positions):
        raise ValueError("servo_ids and positions don't match")
    for servo_id, position in zip(servo_ids, positions):
        if servo.st3215.WritePosition(servo_id, position) is not True:
            raise ConnectionError(f"ID {servo_id}: position command was not acknowledged")


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
    # Broadcastには応答がない。実送信完了と半二重バスの切替を待ってから
    # 次の読み取りを行い、送信エコーなどを応答パケットと取り違えないようにする。
    serial_port = servo.st3215.portHandler.ser
    serial_port.flush()
    time.sleep(SYNC_WRITE_SETTLE_S)
    serial_port.reset_input_buffer()

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
