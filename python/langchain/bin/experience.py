from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/experience.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.experience import (  # noqa: E402
    Observation,
    RuleBook,
    Tier,
    distill_insight,
    extract_insight,
    promote_or_demote,
    reflect,
    select_model,
)

# Five finished runs: a report plus whether it hit its KPI target. The cart and
# order-value pairs intentionally point in opposite directions so distillation
# has to REMOVE a contradicted rule, not just accumulate.
DEMO_REPORTS: list[Observation] = [
    Observation("Doubled website traffic via SEO; sessions +40% as targeted.", success=True),
    Observation("New email subject lines; open rate fell below the 25% target.", success=False),
    Observation("Cart abandonment cut with one-click checkout; hit the target.", success=True),
    Observation("Added shipping fees; cart abandonment rose past the target.", success=False),
    Observation("Bundle pricing lifted average order value above target.", success=True),
]


def _show(book: RuleBook) -> None:
    print("\n=== rule book ===")
    print(book.render())
    for tier in (Tier.PROMOTED, Tier.ACTIVE, Tier.DEMOTED):
        ids = [r.id for r in book.by_tier(tier)]
        print(f"  {tier.value}: {ids}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="experience",
        description=(
            "Learn from past runs: extract one insight per report, distil it into a "
            "small rule list via AGREE/REMOVE/EDIT/ADD, promote/demote on the KPI "
            "outcome, then reflect on the survivors. Runs offline with a fake model."
        ),
    )
    parser.add_argument(
        "--edit",
        metavar="ID=TEXT",
        help="Human-in-the-loop override: reword rule ID after learning.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline fake model even when a key is set.",
    )
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake model.\n", file=sys.stderr)

    model, _ = select_model(use_fake)
    book = RuleBook()

    print("=== learning from reports ===")
    for obs in DEMO_REPORTS:
        insight = extract_insight(model, obs)
        result = distill_insight(model, book, insight)
        rule = promote_or_demote(book, result.rule, obs.success)
        flag = "SUCCESS" if obs.success else "FAILURE"
        tier = rule.tier.value if rule else "-"
        print(f"\n[{flag}] {obs.report}")
        print(f"  insight : {insight}")
        print(f"  op      : {result.op} ({result.note})")
        print(f"  outcome : {'promote' if obs.success else 'demote'} -> tier={tier}")

    _show(book)

    if args.edit:
        rule_id_str, _, text = args.edit.partition("=")
        rule = book.edit(int(rule_id_str), text.strip())
        print(f"\n=== HITL edit ===\nrule {rule_id_str} -> {rule.text if rule else 'not found'}")
        _show(book)

    print(f"\n=== reflect ===\n{reflect(model, book)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
