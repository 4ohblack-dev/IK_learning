import sys
from pathlib import Path
import json
import numpy as np

ST3215_SOURCE = Path(__file__).resolve().parent / "python-st3215"
if str(ST3215_SOURCE) not in sys.path:
    sys.path.insert(0, str(ST3215_SOURCE))

from st3215 import ST3215
from st3215.port_handler import PortHandler

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
class handle_STS3215:
    def __init__(self,port:str):
        self.st3215 = ST3215(port)
        self.zero_positions = {}
        self.directions = {}
        self.calibration_path = (Path(__file__).resolve().parent / "calibration.json")

    def setup(self,servo_ids):
        missing = [servo_id for servo_id in servo_ids if not self.st3215.PingServo(servo_id)]
        if missing:
            raise ConnectionError(f"not found id:{missing}")

    def save_calibration(self):
        data = {
            "zero_positions": {str(key): float(value) for key, value in self.zero_positions.items()},
            "directions": {str(key): int(value) for key, value in self.directions.items()},
        }
        temporary = self.calibration_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(self.calibration_path)

    def load_calibration(self,servo_ids):
        if not self.calibration_path.exists():
            return False

        data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        zeros = {int(key): float(value) for key, value in data["zero_positions"].items()}
        directions = {int(key): int(value) for key, value in data["directions"].items()}

        for servo_id in servo_ids:
            if servo_id not in zeros or servo_id not in directions:
                raise ValueError(f"calibration data of ({servo_id}) not found")
            if not np.isfinite(zeros[servo_id]):
                raise ValueError(f"zero of ({servo_id}) is error")
            if directions[servo_id] not in (-1, 1):
                raise ValueError(f"direction of ({servo_id}) is error")

        self.zero_positions = zeros
        self.directions = directions
        return True

    def calibrate(self,servo_ids,reference_q,directions):
        """
        reference_q:
            校正姿勢の関節角度 [rad]。URDF基準。
        directions:
            サーボ位置の増加がURDFの正方向なら +1、逆なら -1。
        """
        servo_ids = tuple(servo_ids)
        reference_q = np.asarray(reference_q,dtype=float)
        directions = np.asarray(directions,dtype=float)
        count = len(servo_ids)
        if count != len(set(servo_ids)) or count == 0:
            raise ValueError("servo id is not correct")
        if reference_q.shape != (count,) or directions.shape != (count,):
            raise ValueError("correct reference_q or directions")
        if not np.all(np.isfinite(reference_q)):
            raise ValueError("reference_q must be finite")
        if not np.all(np.isin(directions,(-1,1))):
            raise ValueError("directions must be -1 or 1")

        positions = np.asarray(self.read_positions(servo_ids),dtype=float)
        zeros = positions - directions*reference_q*4096/(2*np.pi)
        self.zero_positions = dict(zip(servo_ids,zeros))
        self.directions = dict(zip(servo_ids,directions))

    def calibrate_and_save(self,servo_ids,reference_q,directions):
        self.calibrate(servo_ids,reference_q,directions)
        self.save_calibration()

    
    def read_positions(self,servo_ids):
        positions = []
        for servo_id in servo_ids:
            position = self.st3215.ReadPosition(servo_id)
            if position is None:
                raise ConnectionError(f"can't find {servo_id} position")
            positions.append(position)
        return positions

    def toAngle(self,servo_ids,angle):
        servo_ids = tuple(servo_ids)
        positions = np.asarray(angle,dtype=float)
        if positions.shape != (len(servo_ids),):
            raise ValueError("servo_ids and positions don't match")
        if not np.all(np.isfinite(positions),):
            raise ValueError("positions must be finite")
        q = []
        for servo_id,position in zip(servo_ids,positions):
            zero = self.zero_positions[servo_id]
            direction = self.directions[servo_id]
            radians = direction*(position - zero)*(2*np.pi/4096)
            q.append(radians)
        return np.array(q,dtype=float)

    def close(self):
        if self.st3215.portHandler.is_open:
            self.st3215.portHandler.closePort()




def main():
    servo_ids = [1, 2, 3, 4, 5]
    servo = handle_STS3215(port="COM7")

    try:
        servo.setup(servo_ids)
        positions = servo.read_positions(servo_ids)
        print("現在位置:", positions)
    finally:
        servo.close() 


if __name__ == "__main__":
    main()
