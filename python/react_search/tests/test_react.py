from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import pytest

from llm import Generation, Request
from metrics import answer_score, evidence_recall
from questions import Question
from react import MALFORMED_OBSERVATION, Fabrication, run_rag, run_react
from retrieval import BM25, containment, load_articles
from tools import Environment

ARTICLES = load_articles()
INDEX = BM25(ARTICLES)
QUESTION = Question("x", "single", "参議院議員の任期は何年か", (("六年",),), ((46,),))


def make_env() -> Environment:
    return Environment(INDEX.search, ARTICLES)


class ScriptedLLM:
    def __init__(self, script: Sequence[str], repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: dict[str, int] = defaultdict(int)
        self.requests: list[Request] = []

    def generate(
        self, requests: Sequence[Request], max_tokens: int, early_stop: Sequence[str] = ()
    ) -> list[Generation]:
        out = []
        for r in requests:
            self.requests.append(r)
            i = self.calls[r.user]
            self.calls[r.user] += 1
            text = self.script[min(i, len(self.script) - 1)] if self.repeat_last else self.script[i]
            out.append(Generation(text, 1, 0.5))
        return out


@pytest.mark.parametrize("max_steps", [2, 5])
def test_react_stops_at_step_limit(max_steps: int) -> None:
    llm = ScriptedLLM([" 探す\n行動: search[参議院]"], repeat_last=True)
    (o,) = run_react([QUESTION], make_env, llm, max_steps=max_steps)
    assert o.forced
    assert o.answer == ""
    assert o.calls == max_steps
    assert o.searches == max_steps


def test_react_with_stop_discards_written_observation() -> None:
    llm = ScriptedLLM(
        [
            " 任期を探す\n行動 1: search[参議院議員 任期]\n観測 1: 第九十九条 任期は百年\n思考 2: 百年\n行動 2: finish[百年]",
            " 第四十六条に六年とある\n行動 2: finish[六年]",
        ]
    )
    (o,) = run_react([QUESTION], make_env, llm, max_steps=5)
    assert o.answer == "六年"
    assert not o.forced
    assert o.calls == 2
    assert o.searches == 1
    assert 46 in o.retrieved
    assert "百年" not in o.transcript
    assert "観測 1: 第四十六条" in o.transcript
    assert llm.requests[1].prefix.endswith("思考 2:")


def test_react_without_stop_takes_finish_from_written_chain() -> None:
    llm = ScriptedLLM(
        [
            " 任期を探す\n行動 1: search[参議院議員 任期]\n観測 1: 第九十九条 任期は百年\n思考 2: 百年\n行動 2: finish[百年]"
        ]
    )
    (o,) = run_react([QUESTION], make_env, llm, max_steps=5, use_stop=False)
    assert o.answer == "百年"
    assert o.calls == 1
    assert o.searches == 0
    assert len(o.fabrications) == 1
    fab = o.fabrications[0]
    assert fab.written == "第九十九条 任期は百年"
    assert "第四十六条" in fab.actual
    assert fab.mismatch


def test_react_counts_malformed_action_and_continues() -> None:
    llm = ScriptedLLM([" 答えは六年だと思う", " 行動 2: finish[六年]"])
    (o,) = run_react([QUESTION], make_env, llm, max_steps=5)
    assert o.malformed == 1
    assert MALFORMED_OBSERVATION in o.transcript
    assert o.answer == "六年"


@pytest.mark.parametrize(
    ("written", "mismatch"),
    [
        pytest.param("第四十六条　参議院議員の任期は、六年とし", False, id="copies real article"),
        pytest.param(
            "第百十三条　参議院議員の任期は、六年とし", True, id="cites article not returned"
        ),
        pytest.param("国会は、国民の主権を代表する議会である", True, id="little overlap"),
    ],
)
def test_fabrication_mismatch(written: str, mismatch: bool) -> None:
    actual = make_env().search("参議院議員の任期")
    assert Fabrication(None, written, actual, containment(written, actual)).mismatch is mismatch


def test_rag_searches_question_once() -> None:
    llm = ScriptedLLM(["六年\n（参考）第四十五条"])
    (o,) = run_rag([QUESTION], make_env, llm)
    assert o.searches == 1
    assert 46 in o.retrieved
    assert "文書1: 第四十六条" in llm.requests[0].user
    assert llm.requests[0].user.endswith("問い: 参議院議員の任期は何年か")
    assert o.answer == "六年"


@pytest.mark.parametrize(
    ("answer", "want"),
    [
        pytest.param("勤労、納税、教育", 1.0, id="all items"),
        pytest.param("納税の義務", 1 / 3, id="one of three"),
        pytest.param("兵役", 0.0, id="none"),
    ],
)
def test_answer_score_is_item_recall(answer: str, want: float) -> None:
    q = Question("l", "list", "義務", (("教育",), ("勤労",), ("納税",)), ((26,), (27,), (30,)))
    assert answer_score(answer, q) == pytest.approx(want)


def test_evidence_recall_accepts_alternatives() -> None:
    q = Question("t", "twohop", "", (("総意",),), ((7, 96), (1,)))
    assert evidence_recall({96}, q) == 0.5
    assert evidence_recall({7, 1}, q) == 1.0
