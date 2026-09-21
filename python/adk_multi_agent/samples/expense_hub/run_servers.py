"""2 つの Spoke（経費精算 8001、承認 8002）を子プロセスで起動し、Ctrl+C でまとめて止める。

プロジェクトの直下（adk_multi_agent/）で python -m samples.expense_hub.run_servers と実行する。
ポートは EXPENSE_PORT / APPROVAL_PORT で変えられる。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

SERVERS = {
    "経費精算エージェント": (
        "samples.expense_hub.expense_server.agent:create_app",
        "EXPENSE_PORT",
        8001,
        "EXPENSE_AGENT_URL",
    ),
    "承認エージェント": (
        "samples.expense_hub.approval_server.agent:create_app",
        "APPROVAL_PORT",
        8002,
        "APPROVAL_AGENT_URL",
    ),
}
STOP_TIMEOUT = 10


def start() -> dict[str, subprocess.Popen]:
    procs: dict[str, subprocess.Popen] = {}
    for name, (target, port_env, default_port, url_env) in SERVERS.items():
        port = int(os.environ.get(port_env, default_port))
        env = os.environ | {
            url_env: os.environ.get(url_env, f"http://localhost:{port}")
        }
        procs[name] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--factory",
                target,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
        )
        print(
            f"[INFO] {name}（PID: {procs[name].pid}、ポート {port}）を起動しました",
            flush=True,
        )
    return procs


def stop(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + STOP_TIMEOUT
    for name, proc in procs.items():
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(
            f"[INFO] {name} を停止しました（終了コード {proc.returncode}）", flush=True
        )


def main() -> int:
    procs = start()
    stopping = False

    def on_signal(signum, frame) -> None:
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        while not stopping:
            # 片方が落ちたら、もう片方も止める。片方だけ動いていると、オーケストレーターの失敗の原因が分かりにくい。
            exited = [n for n, p in procs.items() if p.poll() is not None]
            if exited:
                print(
                    f"[ERROR] {'、'.join(exited)} が終了したので、すべて止めます",
                    flush=True,
                )
                return 1
            time.sleep(0.2)
    finally:
        stop(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
