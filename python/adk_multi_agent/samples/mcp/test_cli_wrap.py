"""教材の gcloud の包み方に、モデルの出しうる入力を渡す。本物の gcloud は呼ばず、引数を記録する偽物を使う。"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest


def run_gcloud_command(command: str) -> dict:
    """教材のコードそのまま。"""
    args = ["gcloud", *shlex.split(command), "--format=json"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "タイムアウト", "return_code": -1}


@pytest.fixture
def fake_gcloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """受け取った引数を JSON で返す gcloud を PATH の先頭に置く。"""
    script = tmp_path / "gcloud"
    script.write_text('#!/bin/sh\npython3 -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return script


def argv(result: dict) -> list[str]:
    return json.loads(result["stdout"])


def test_shell_metacharacters_are_not_interpreted(fake_gcloud: Path) -> None:
    """shlex.split と引数の配列で渡すので、シェルを通らない。; や $() はただの文字列になる。"""
    got = argv(run_gcloud_command('compute instances list "; rm -rf ~" $(whoami)'))
    assert got == ["compute", "instances", "list", "; rm -rf ~", "$(whoami)", "--format=json"]


@pytest.mark.parametrize(
    "command",
    [
        "auth print-access-token",
        "compute instances delete web-1 --quiet",
        "compute instances list --impersonate-service-account=admin@example.iam.gserviceaccount.com",
        "config set project another-project",
    ],
    ids=[
        "アクセストークンの表示",
        "インスタンスの削除",
        "別のサービスアカウントへのなりすまし",
        "gcloud の設定を書き換える",
    ],
)
def test_any_subcommand_and_flag_reaches_gcloud(fake_gcloud: Path, command: str) -> None:
    """シェルを通らなくても、gcloud に渡る中身は何も絞られていない。"""
    got = argv(run_gcloud_command(command))
    assert got[:-1] == shlex.split(command)
