from __future__ import annotations

from agents.inventory import ReorderDecision, run


def test_reorders_when_at_or_below_threshold() -> None:
    decision, is_fake = run("widget-a", use_fake=True)
    assert is_fake
    assert isinstance(decision, ReorderDecision)
    assert decision.item == "widget-a"
    assert decision.reorder is True
    assert decision.quantity > 0


def test_no_reorder_when_above_threshold() -> None:
    decision, _ = run("widget-b", use_fake=True)
    assert decision.reorder is False
    assert decision.quantity == 0


def test_reorders_at_exact_threshold() -> None:
    decision, _ = run("gizmo-x", use_fake=True)
    assert decision.reorder is True
