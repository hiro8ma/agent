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
import json
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
    env: dict[str, Any] = {"__builtins__": __builtins__, **data}
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


def _serialize(data: dict[str, Any]) -> str:
    """持ち込む変数を、リモートで復元できる形にする。

    ローカル実行は globals に直接入れられるが、
    リモートでは値を送る必要がある。
    JSON で送れないものは持ち込めない、という制約がここで出る。
    """

    lines = [
        f"{k} = json.loads({json.dumps(json.dumps(v, ensure_ascii=False))})"
        for k, v in data.items()
    ]
    return "import json\n" + "\n".join(lines)


def _run_in_e2b(code: str, data: dict[str, Any], timeout: float) -> ExecResult:
    """E2B の microVM で実行する。

    ローカルと違い、壊れても手元には影響しない。
    代わりに変数を送る手間と、起動の待ち時間が乗る。
    """

    try:
        from e2b_code_interpreter import Sandbox  # type: ignore[import-not-found]
    except ImportError:
        return ExecResult(
            False, "", "e2b-code-interpreter が入っていない。pip install で追加する", ""
        )

    try:
        with Sandbox(timeout=int(timeout)) as sandbox:
            execution = sandbox.run_code(_serialize(data) + "\n" + code)
            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)
            if execution.error:
                return ExecResult(False, stdout, str(execution.error)[-2000:], "")
            # result は最後の式の値。ローカル実行の約束に合わせる。
            text = execution.text or ""
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
