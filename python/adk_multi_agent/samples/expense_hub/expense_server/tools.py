"""経費の登録と照会。A2A には依存しない。"""

from __future__ import annotations

from datetime import date

from samples.expense_hub import store

CATEGORIES = ("交通費", "宿泊費", "会議費", "消耗品費")
MAX_AMOUNT = 1_000_000_000


def _error(message: str) -> dict:
    return {"status": "error", "error_message": message}


def register_expense(
    expense_date: str, category: str, amount: int, description: str
) -> dict:
    """経費を 1 件登録する。

    Args:
        expense_date: 使った日。YYYY-MM-DD
        category: 交通費 / 宿泊費 / 会議費 / 消耗品費 のどれか
        amount: 金額（円）。1 以上
        description: 用途の説明

    Returns:
        登録した経費の ID と金額
    """
    try:
        date.fromisoformat(expense_date)
    except ValueError:
        return _error(f"日付は YYYY-MM-DD で指定する: {expense_date}")
    if category not in CATEGORIES:
        return _error(f"カテゴリは {' / '.join(CATEGORIES)} のどれか: {category}")
    if not 0 < amount <= MAX_AMOUNT:
        return _error(f"金額は 1 以上 {MAX_AMOUNT} 以下: {amount}")
    expense = store.add(expense_date, category, amount, description)
    return {"status": "success", "expense_id": expense.id, "amount": expense.amount}


def list_expenses(start_date: str, end_date: str) -> dict:
    """期間内の経費を一覧する。

    Args:
        start_date: 期間の初日。YYYY-MM-DD
        end_date: 期間の末日。YYYY-MM-DD

    Returns:
        経費の一覧と合計
    """
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return _error("日付は YYYY-MM-DD で指定する")
    if start > end:
        return _error("初日が末日より後になっている")
    items = store.between(start_date, end_date)
    return {
        "status": "success",
        "expenses": [e.__dict__ for e in items],
        "total": sum(e.amount for e in items),
    }
