"""Grammar-as-FSM: tokens constrained by a finite state machine instead of BNF.

これは BNF 文法制約の最小モデルである: BNF / EBNF が表す言語は有限状態 or 文脈自由文法で、
本デモはパイプ区切りレコード ``author | title | year`` を有限状態機械（FSM）として表し、
各状態で「次に来てよいトークン集合」を返す。本番の transformers-cfg はこの遷移を
GBNF / EBNF から自動構築し、その許可集合外のロジットを -inf にマスクしている。

文法（year は 4 桁数字または NULL）:
    record := WORD+ "|" WORD+ "|" (YEAR | "NULL")
状態:
    AUTHOR -> (WORD で滞留) -> SEP1("|") -> TITLE -> (WORD で滞留) -> SEP2("|") -> YEAR -> DONE
"""

from __future__ import annotations

from collections.abc import Sequence

SEP = "|"
NULL = "NULL"

# 状態名
AUTHOR = "AUTHOR"
_SEP1 = "_SEP1"  # author 列の '|' を読んだ直後（title 開始待ち）
TITLE = "TITLE"
_SEP2 = "_SEP2"  # title 列の '|' を読んだ直後（year 開始待ち）
YEAR = "YEAR"
DONE = "DONE"


def _is_word(tok: str) -> bool:
    return tok.isalpha()


def _is_year(tok: str) -> bool:
    return len(tok) == 4 and tok.isdigit()


class FsmGrammar:
    """状態遷移でトークンを制約する有限状態機械。

    ``allowed_next`` は現在状態と語彙から「次に許可するトークン集合」を返す。
    AllowedTokens は固定の許可集合、FsmGrammar は状態遷移で許可集合が動く（発展形）。
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = list(vocab)
        self.words = {t for t in self.vocab if _is_word(t) and t != NULL}
        self.years = {t for t in self.vocab if _is_year(t)}

    def initial_state(self) -> str:
        return AUTHOR

    def step(self, state: str, token: str) -> str:
        """1 トークン消費して次状態へ。文法違反トークンは状態を進めない。"""
        if state == AUTHOR:
            return _SEP1 if token == SEP else AUTHOR
        if state == _SEP1:
            return TITLE if token in self.words else state
        if state == TITLE:
            return _SEP2 if token == SEP else TITLE
        if state == _SEP2:
            return YEAR if (token in self.years or token == NULL) else state
        if state == YEAR:
            return DONE
        return state

    def allowed_next(self, state: str, history: Sequence[str]) -> set[str]:
        """現在状態で許可するトークン集合（許可外はロジット -inf 対象）。"""
        if state == AUTHOR:
            # author の WORD 群。直前が WORD なら区切り '|' へ進んでよい。
            allowed = set(self.words)
            if history and history[-1] in self.words:
                allowed.add(SEP)
            return allowed
        if state == _SEP1:
            return set(self.words)  # 区切り直後は title の WORD のみ
        if state == TITLE:
            allowed = set(self.words)
            if history and history[-1] in self.words:
                allowed.add(SEP)
            return allowed
        if state == _SEP2:
            return self.years | {NULL}  # 区切り直後は YEAR か NULL のみ
        return set()
