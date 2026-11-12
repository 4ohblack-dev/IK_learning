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
