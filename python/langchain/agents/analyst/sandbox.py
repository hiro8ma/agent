"""生成されたコードを実行する環境。

エージェントが書いたコードをそのまま実行するため、
何を許すかを明示的に決める。教材の「環境（コード実行 / 行動）」にあたる。

## 2 つの実行先

`ANALYST_SANDBOX` で切り替える。

| 値 | 実行先 | 隔離 |
|---|---|---|
| 既定 | ローカルの別プロセス | **無い** |
| `e2b` | E2B の microVM | **ある** |

### ローカル別プロセスの限界

制限できるのは 3 つだけになる。

  持ち込めるもの   許可した名前だけを globals に入れる
  時間            別プロセスにして上限で打ち切る
  出力            標準出力と生成した値だけを回収する

**ネットワークとファイル書き込みは塞げない。**
別プロセスにしても同じ利用者の権限で動くため、
`open("~/.ssh/id_rsa").read()` も `requests.post(...)` も通る。
信用できないコードを走らせる用途には使えない。

鍵が要らないので既定にしているが、**これは開発用の妥協になる。**

### E2B

Firecracker の microVM でクラウド上に隔離環境を立てる。
Ubuntu が起動し、UNIX コマンドもファイルシステムも使える。
壊れても手元には影響しない。

既定で 5 分で落ちる。`Sandbox(timeout=...)` で伸ばせるが、
Hobby プランはセッション最大 1 時間・同時 20 個までになる。

    pip install e2b-code-interpreter
    export E2B_API_KEY=...
    export ANALYST_SANDBOX=e2b

**他の選択肢**

  Azure Container Apps Dynamic Sessions   サーバーレスの隔離実行
  Jupyter カーネルゲートウェイ              REST / WebSocket でカーネルを叩く。CodeAct が使う
  AutoGen DockerCommandLineCodeExecutor   コードをファイルに書いて Docker で実行
  Container Use                           エージェントごとに独立したコンテナ。MCP サーバーとして提供
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecResult:
    """コード実行の結果。

    失敗も値として返す。例外を投げると、
    エージェントが「なぜ失敗したか」を読めない。
    """

    ok: bool
    stdout: str
    error: str
    # result は最後の式の値を文字列にしたもの。
    result: str


def _worker(code: str, data: dict[str, Any], queue: mp.Queue) -> None:
    buf = io.StringIO()
    # 制御用の鍵はコードへ渡さない。名前が衝突するうえ、
    # エージェントがパスを見てファイルを直接読みにいく。
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    env: dict[str, Any] = {"__builtins__": __builtins__, **payload}
    try:
        with redirect_stdout(buf):
            exec(code, env)  # noqa: S102 — 実行することが目的の関数
        result = env.get("result", "")
        queue.put(ExecResult(True, buf.getvalue(), "", str(result)[:4000]))
    except Exception:
        queue.put(ExecResult(False, buf.getvalue(), traceback.format_exc()[-2000:], ""))


def run_code(code: str, data: dict[str, Any] | None = None, timeout: float = 20.0) -> ExecResult:
    """コードを実行する。実行先は環境変数で切り替える。"""

    if os.environ.get("ANALYST_SANDBOX") == "e2b":
        return _run_in_e2b(code, data or {}, timeout)
    return _run_locally(code, data or {}, timeout)


def _run_in_e2b(code: str, data: dict[str, Any], timeout: float) -> ExecResult:
    """E2B の microVM で実行する。

    ローカルと違い、壊れても手元には影響しない。
    代わりにデータを送る手間と、起動の待ち時間が乗る。

    **データは変数ではなくファイルで送る。**
    ローカル実行では DataFrame を globals に直接入れられるが、
    リモートには送れない。JSON へ落とすと型と欠損の扱いが変わる。
    CSV を書き込んで向こうで `pd.read_csv` させるのが、
    型を保ったまま渡す最も素直な方法になる。

    同じ制約はローカルでも出た。
    `load_frames` にモジュールを入れて `cannot pickle 'module' object` で落ちている。
    プロセスを跨ぐ時点で「送れるもの」が決まる。
    """

    try:
        from e2b_code_interpreter import Sandbox  # type: ignore[import-not-found]
    except ImportError:
        return ExecResult(
            False, "", "e2b-code-interpreter が入っていない。pip install で追加する", ""
        )

    csv_path = data.get("_csv_path")
    if not csv_path:
        return ExecResult(
            False, "", "リモート実行には _csv_path が要る。DataFrame は直接送れない", ""
        )

    try:
        with Sandbox(timeout=int(timeout)) as sandbox:
            remote = "/home/user/data.csv"
            with open(csv_path, "rb") as fi:
                sandbox.files.write(remote, fi)
            # 向こうで読ませる。ローカル実行と同じ `df` という名前に揃える。
            setup = f"import pandas as pd; df = pd.read_csv('{remote}')"
            sandbox.run_code(setup, timeout=int(timeout))
            execution = sandbox.run_code(code, timeout=int(timeout))
            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)
            if execution.error:
                return ExecResult(False, stdout, str(execution.error)[-2000:], "")
            # 画像が生成されていれば件数だけ残す。本体は保存先から取る。
            images = sum(1 for r in execution.results if getattr(r, "png", None))
            text = execution.text or ""
            if images:
                stdout += f"\n(画像 {images} 件を生成)"
            return ExecResult(True, stdout, stderr, str(text)[:4000])
    except Exception as exc:
        return ExecResult(False, "", f"E2B の実行に失敗した: {exc}", "")


def _run_locally(code: str, data: dict[str, Any], timeout: float) -> ExecResult:
    """手元の別プロセスで実行する。

    上限を超えたら打ち切る。
    生成されたコードは無限ループを書きうるため、
    同じプロセスで実行するとエージェントごと止まる。

    隔離はされていない。開発用の妥協になる。
    """

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, data, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExecResult(False, "", f"実行が {timeout} 秒を超えたため打ち切った", "")

    if queue.empty():
        return ExecResult(False, "", "実行結果を取得できなかった", "")
    return queue.get()
