# Pythonライブラリの使い方

このプロジェクトは `python control_arm.py` のように各ファイルを直接実行します。
同じフォルダのモジュールは、先頭に `.` を付けずにimportしてください。

```python
from controller_input import Controller, Input
from mathmetics import ArmKinematics, Joint
from servo_com import ST3215, PortHandler
```

cloneした `python-st3215` の検索先設定は `servo_com.py` に集約しています。
他のファイルで `sys.path` を変更する必要はありません。
`ST3215` は内部で `PortHandler` を作るため、通常は `ST3215` だけで十分です。

外部依存は、実行に使うPython環境にインストールしてください。

```powershell
python -m pip install -r requirements.txt
```

配布名は `pyserial` ですが、直接使う際のimport名は `serial` です。
`st3215` 自体のpipインストールは不要です。

VS Codeではこのフォルダを開き、`Python: Select Interpreter` で上記と同じ環境を選択してください。
`pyrightconfig.json` はローカルの `st3215` を補完・解析するための検索先設定です。

## SO-100 の校正と座標

`control_arm.py` は台座に固定した座標で動きます。+X は正面、+Y は左、+Z は上です。TCP は固定爪の先端の縁の中点です。

初回はアームを支えて、次の姿勢に合わせてから △ を押します。姿勢合わせでは**関節軸の中心を結んだ線**を基準にします。ケースの縁や `image.png` は姿勢の目安です。

1. 肩の旋回を台座正面に合わせます。
2. 肩の上下軸から肘の軸が真上に来るようにします。
3. 肘軸から手首の曲げ軸までと、手首の曲げ軸からロール軸までを、どちらも正面に水平に向けます。
4. 爪のロールをそろえ、固定爪が下、可動爪が上になるようにします。

この姿勢の URDF 関節角は、旋回・肩・肘・手首曲げ・ロールの順に `0°, 89.165°, −73.824°, −34.912°, 0°` です。新しい校正は、この作業フォルダーの `calibration.so100_upright.json` に保存されます。旧 `calibration.json` には参照姿勢が記録されていないため、新しい校正は手動で行います。記録済みの姿勢にエンコーダ値があれば、再生時に現在の校正で角度を変換します。

### 公式 CAD の寸法（mm）

| 部品 | STL のローカル XYZ 外形 |
| --- | --- |
| 台座 | 110.925 × 95.900 × 83.496 |
| 肩旋回部 | 52.400 × 86.300 × 62.800 |
| 上腕 | 70.000 × 142.082 × 28.000 |
| 前腕 | 61.200 × 36.900 × 130.700 |
| 手首 | 63.275 × 74.400 × 32.000 |
| 固定爪 | 65.200 × 106.375 × 48.000 |
| 可動爪 | 22.300 × 91.955 × 48.000 |

6つのモーター STL は軸順をそろえるとすべて 24.8 × 39.6 × 45.4 mm です。上表は CAD の外接寸法です。IK で使う中心間距離は、肩の上下軸→肘が 116.000 mm、肘→手首の曲げ軸が 135.000 mm、そこからロール軸が 60.100 mm、ロール軸→固定爪 TCP が 106.830 mm です。固定爪 TCP は爪リンク座標で `[9.8493, −106.375, 0] mm` です。[公式 SO-100 STL](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO100/assets) [公式 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO100/so100.urdf)
