"""経費と承認の保存先。2 つの Spoke が同じ SQLite のファイルを読み書きする。

承認の Spoke は金額をここから引く。オーケストレーターが文面で渡す金額は、モデルが書いたものなので使わない。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass

DEFAULT_PATH = "expense_hub.sqlite3"


@dataclass(frozen=True)
class Expense:
    id: str
    date: str
    category: str
    amount: int
    description: str
    approval: str


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(os.environ.get("EXPENSE_HUB_DB", DEFAULT_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS expenses ("
        " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
        " date TEXT NOT NULL, category TEXT NOT NULL, amount INTEGER NOT NULL,"
        " description TEXT NOT NULL, approval TEXT NOT NULL DEFAULT 'none')"
    )
    return conn


def _row(row: tuple) -> Expense:
    seq, date, category, amount, description, approval = row
    return Expense(f"EXP-{seq:04d}", date, category, amount, description, approval)


def add(date: str, category: str, amount: int, description: str) -> Expense:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
            (date, category, amount, description),
        )
        seq = cur.lastrowid
    return Expense(f"EXP-{seq:04d}", date, category, amount, description, "none")


def get(expense_id: str) -> Expense | None:
    prefix, _, number = expense_id.partition("-")
    if prefix != "EXP" or not number.isdigit():
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT seq, date, category, amount, description, approval"
            " FROM expenses WHERE seq = ?",
            (int(number),),
        ).fetchone()
    return _row(row) if row else None


def between(start: str, end: str) -> list[Expense]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT seq, date, category, amount, description, approval"
            " FROM expenses WHERE date BETWEEN ? AND ? ORDER BY date, seq",
            (start, end),
        ).fetchall()
    return [_row(r) for r in rows]


def set_approval(expense_id: str, approval: str) -> None:
    expense = get(expense_id)
    if expense is None:
        raise KeyError(expense_id)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE expenses SET approval = ? WHERE seq = ?",
            (approval, int(expense_id.split("-")[1])),
        )
