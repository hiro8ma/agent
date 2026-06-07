"""Shared tool catalog for the three tool-selection strategies.

Six fictional tools across three groups (Computation / Automation / Communication).
Every tool is a fake, offline implementation: no real network call is made. Any
credential a real version would need is read from ``os.getenv`` and only echoed as a
boolean presence flag, never embedded in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ToolGroup:
    """A named bucket of tools used by the hierarchical router."""

    name: str
    description: str
    tool_names: tuple[str, ...]


# --- Computation -------------------------------------------------------------


class SolveEquationArgs(BaseModel):
    expression: str = Field(description="An equation or expression to solve, e.g. '2x+3=7'.")


def solve_equation(expression: str) -> str:
    """Solve a math equation or evaluate a symbolic expression (compute engine)."""

    return f"[compute] solved '{expression}' (fake symbolic engine)"


class ConvertUnitsArgs(BaseModel):
    quantity: str = Field(description="Value and unit to convert, e.g. '10 km to miles'.")


def convert_units(quantity: str) -> str:
    """Convert between physical units (compute engine)."""

    return f"[compute] converted '{quantity}' (fake unit table)"


# --- Automation --------------------------------------------------------------


class RunWebhookArgs(BaseModel):
    workflow: str = Field(description="Automation workflow / webhook name to trigger.")
    payload: str = Field(default="", description="JSON-ish payload to pass to the workflow.")


def run_webhook(workflow: str, payload: str = "") -> str:
    """Trigger an automation workflow via webhook (Zapier-style)."""

    has_token = bool(os.getenv("AUTOMATION_WEBHOOK_TOKEN"))
    return f"[automation] triggered '{workflow}' payload={payload!r} token_present={has_token}"


class ScheduleJobArgs(BaseModel):
    job: str = Field(description="Name of the recurring job to schedule.")
    cron: str = Field(default="0 9 * * *", description="Cron expression for the schedule.")


def schedule_job(job: str, cron: str = "0 9 * * *") -> str:
    """Schedule a recurring background job (automation)."""

    return f"[automation] scheduled '{job}' at cron='{cron}'"


# --- Communication -----------------------------------------------------------


class SendChatMessageArgs(BaseModel):
    channel: str = Field(description="Chat channel, e.g. '#general'.")
    text: str = Field(description="Message body to post.")


def send_chat_message(channel: str, text: str) -> str:
    """Post a message to a team chat channel (Slack-style)."""

    has_token = bool(os.getenv("CHAT_BOT_TOKEN"))
    return f"[comm] posted to {channel}: {text!r} token_present={has_token}"


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(default="", description="Email body.")


def send_email(to: str, subject: str, body: str = "") -> str:
    """Send an email to a recipient (communication)."""

    has_key = bool(os.getenv("EMAIL_API_KEY"))
    return f"[comm] emailed {to} subj={subject!r} key_present={has_key}"


def _tool(func: object, name: str, schema: type[BaseModel], description: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,  # type: ignore[arg-type]
        name=name,
        description=description,
        args_schema=schema,
    )


SOLVE_EQUATION = _tool(
    solve_equation,
    "solve_equation",
    SolveEquationArgs,
    "Solve a math equation or evaluate a symbolic/numeric expression.",
)
CONVERT_UNITS = _tool(
    convert_units,
    "convert_units",
    ConvertUnitsArgs,
    "Convert a quantity between physical units (length, mass, temperature).",
)
RUN_WEBHOOK = _tool(
    run_webhook,
    "run_webhook",
    RunWebhookArgs,
    "Trigger an external automation workflow or webhook with a payload.",
)
SCHEDULE_JOB = _tool(
    schedule_job,
    "schedule_job",
    ScheduleJobArgs,
    "Schedule a recurring background job on a cron expression.",
)
SEND_CHAT_MESSAGE = _tool(
    send_chat_message,
    "send_chat_message",
    SendChatMessageArgs,
    "Post a message to a team chat channel to notify people.",
)
SEND_EMAIL = _tool(
    send_email,
    "send_email",
    SendEmailArgs,
    "Send an email with a subject and body to a recipient.",
)


ALL_TOOLS: list[StructuredTool] = [
    SOLVE_EQUATION,
    CONVERT_UNITS,
    RUN_WEBHOOK,
    SCHEDULE_JOB,
    SEND_CHAT_MESSAGE,
    SEND_EMAIL,
]

TOOLS_BY_NAME: dict[str, StructuredTool] = {t.name: t for t in ALL_TOOLS}

GROUPS: list[ToolGroup] = [
    ToolGroup(
        name="Computation",
        description="Math, equations, unit conversion, numeric evaluation.",
        tool_names=("solve_equation", "convert_units"),
    ),
    ToolGroup(
        name="Automation",
        description="Trigger workflows, webhooks, schedule recurring jobs.",
        tool_names=("run_webhook", "schedule_job"),
    ),
    ToolGroup(
        name="Communication",
        description="Notify people via chat messages or email.",
        tool_names=("send_chat_message", "send_email"),
    ),
]

GROUPS_BY_NAME: dict[str, ToolGroup] = {g.name: g for g in GROUPS}


def run_tool(name: str, args: dict[str, object]) -> str:
    """Execute a catalog tool by name with keyword args (fake, offline)."""

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"(no such tool: {name})"
    return str(tool.invoke(args))
