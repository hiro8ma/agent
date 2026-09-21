"""教材のツールのガバナンスとインフラ監視の前提を確かめる。本物の kubectl と BigQuery は呼ばない。"""

from __future__ import annotations

import re
import stat
import subprocess
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

# 教材の検査。英数字、ハイフン、アンダースコアだけを許す。
MATERIAL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# Kubernetes の名前の規則。Pod や Node の名前は DNS-1123 のサブドメイン、名前空間はラベル。
DNS1123_SUBDOMAIN = re.compile(
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
)


@pytest.mark.parametrize(
    ("name", "k8s_valid"),
    [
        ("payments-service-ghl55", True),
        ("ip-10-0-1-2.ec2.internal", True),
        ("gke-prod-pool-1-a1b2c3.asia-northeast1-a", True),
        ("My_Pod", False),
        ("payments_service", False),
    ],
    ids=[
        "通常の Pod 名",
        "ドットを含む Node 名",
        "ゾーンを含む Node 名",
        "大文字とアンダースコア",
        "アンダースコア",
    ],
)
def test_material_validation_disagrees_with_kubernetes_names(
    name: str, k8s_valid: bool
) -> None:
    """教材の検査は、実在する Node 名を拒み、Kubernetes が受け付けない名前を通す。"""
    assert bool(DNS1123_SUBDOMAIN.fullmatch(name)) is k8s_valid
    material_ok = bool(MATERIAL_PATTERN.fullmatch(name))
    if name in ("ip-10-0-1-2.ec2.internal", "gke-prod-pool-1-a1b2c3.asia-northeast1-a"):
        assert material_ok is False
    if name in ("My_Pod", "payments_service"):
        assert material_ok is True


def test_subprocess_timeout_leaves_grandchildren_running(tmp_path: Path) -> None:
    """timeout で止まるのは親だけで、孫のプロセスは残って動き続ける。"""
    marker = "4.3217"
    script = tmp_path / "kubectl"
    script.write_text(f"#!/bin/sh\nsleep {marker} & sleep {marker}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [str(script)], capture_output=True, text=True, timeout=1, check=False
        )
    assert time.monotonic() - start < 2

    left = subprocess.run(
        ["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True, check=False
    )
    pids = left.stdout.split()
    try:
        assert pids, "孫のプロセスが残っていない"
    finally:
        for pid in pids:
            subprocess.run(["kill", pid], check=False)


class Calls(BaseLlm):
    """1 回目に kubectl_delete を呼び、応答を受けたら答える。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        last = llm_request.contents[-1]
        if any(p.function_response for p in last.parts or []):
            part = types.Part(text="終わりました")
        else:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="kubectl_delete", args={"name": "web-1"}
                )
            )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def kubectl_delete(name: str) -> dict:
    """Pod を消す。"""
    return {"deleted": name}


async def run(agent: Agent, plugins: list | None = None) -> None:
    runner = InMemoryRunner(agent=agent, app_name="infra", plugins=plugins)
    await runner.session_service.create_session(
        app_name="infra", user_id="u", session_id="s"
    )
    message = types.Content(role="user", parts=[types.Part(text="web-1 を消して")])
    async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
        pass


def guard(tool, args, tool_context):
    return {"error": "この操作は許可されていない"}


async def test_audit_after_a_guard_misses_blocked_calls() -> None:
    """コールバックは並べた順に走り、最初に結果を返したところで止まる。監査が後ろだと、止めた呼び出しを記録しない。"""
    logged: list[str] = []

    def audit(tool, args, tool_context):
        logged.append(tool.name)

    agent = Agent(
        name="infra",
        model=Calls(model="gemini-3.8-flash"),
        instruction="x",
        tools=[kubectl_delete],
        before_tool_callback=[guard, audit],
    )
    await run(agent)
    assert logged == []


async def test_audit_as_plugin_sees_blocked_calls() -> None:
    """プラグインのコールバックはエージェントのコールバックより先に走るので、止めた呼び出しも記録できる。"""
    logged: list[str] = []

    class Audit(BasePlugin):
        async def before_tool_callback(self, *, tool, tool_args, tool_context):
            logged.append(tool.name)

    agent = Agent(
        name="infra",
        model=Calls(model="gemini-3.8-flash"),
        instruction="x",
        tools=[kubectl_delete],
        before_tool_callback=guard,
    )
    with pytest.warns(DeprecationWarning):
        await run(agent, plugins=[Audit(name="audit")])
    assert logged == ["kubectl_delete"]
