"""承認の申請と判定。A2A には依存しない。

金額は経費の保存先から引く。申請の文面の金額は使わない。
上長と部長の承認は、入力待ち（input-required）で呼び出し元に尋ね、その返答を判断として記録する。
呼び出し元はオーケストレーター、つまり申請者の側なので、この作りでは申請者が自分で承認できる。
本番では承認を対話の外に置き、承認者を認証して判断させる。
"""

from __future__ import annotations

from samples.expense_hub import store

AUTO_LIMIT = 5_000
MANAGER_LIMIT = 50_000


def approval_route(amount: int) -> str:
    """金額から承認の経路を決める。5,000 円以下は自動、50,000 円以下は上長、それを超えると部長。"""
    if amount <= 0:
        raise ValueError(f"金額は 1 以上: {amount}")
    if amount <= AUTO_LIMIT:
        return "auto"
    if amount <= MANAGER_LIMIT:
        return "manager"
    return "director"


APPROVERS = {"manager": "上長", "director": "部長"}


def request_approval(expense_id: str, reason: str) -> dict:
    """経費の承認を申請する。

    Args:
        expense_id: 経費 ID（EXP-0001 の形）
        reason: 申請の理由

    Returns:
        自動承認なら approved、上長か部長の承認が要るなら pending と承認者
    """
    expense = store.get(expense_id)
    if expense is None:
        return {"status": "error", "error_message": f"経費が見つからない: {expense_id}"}
    if not reason.strip():
        return {"status": "error", "error_message": "申請の理由が空"}
    route = approval_route(expense.amount)
    if route == "auto":
        store.set_approval(expense_id, "approved")
        return {"status": "approved", "route": route, "amount": expense.amount}
    store.set_approval(expense_id, "pending")
    return {
        "status": "pending",
        "route": route,
        "approver": APPROVERS[route],
        "amount": expense.amount,
    }


def record_decision(expense_id: str, approved: bool) -> dict:
    """承認者の判断を記録する。

    Args:
        expense_id: 経費 ID
        approved: 承認なら true、却下なら false

    Returns:
        記録した結果
    """
    expense = store.get(expense_id)
    if expense is None or expense.approval != "pending":
        return {
            "status": "error",
            "error_message": f"承認待ちの経費ではない: {expense_id}",
        }
    result = "approved" if approved else "rejected"
    store.set_approval(expense_id, result)
    return {"status": result, "expense_id": expense_id}
