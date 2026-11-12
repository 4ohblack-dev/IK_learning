import sys
from pathlib import Path

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



def main():
    pass


if __name__ == "__main__":
    main()
