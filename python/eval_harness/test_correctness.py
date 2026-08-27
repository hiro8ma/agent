"""DeepEval を pytest に載せ、生成 AI の応答品質を回帰テストにする。

書籍やネット記事のサンプルは `from deepeval.metrics import CorrectnessMetric` と
書いていることがあるが、このクラスは存在しない。実行すると ImportError になる。
正確性は GEval に採点手順を渡して表現する。同じ記事の本文が
「GEval 方式を取り入れ」と書いていても、サンプルコードだけ古いことがある。

同じ理由で evaluation_params には SingleTurnParams を渡す。
LLMTestCaseParams は非推奨で、実行すると DeprecationWarning が出る。

    pytest test_correctness.py -v

GEMINI_API_KEY が無ければ skip する。CI では EVAL_REQUIRE=1 を立てる。
無料枠は 1 分あたり 5 回・1 日 20 回の上限があるため、CI で常時回すなら有料枠が要る。
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def correctness_metric(judge) -> GEval:
    """正確性の採点基準。

    表記の違いで落とさないよう、意味の一致を見るよう手順で明示している。
    ここを書かないと「秒速約 30 万 km」と「毎秒 299,792,458 メートル」が
    不一致と判定される。文字列一致ではなく LLM に判定させる利点が消える。

    threshold は「これ以上なら合格」の下限。DeepEval 4.x で全指標が
    この向きに統一された。旧仕様では Bias / Toxicity だけ上限だったため、
    古い記事の threshold をそのまま持ち込むと判定が反転する。
    """
    return GEval(
        name="正確性",
        criteria="実際の出力が、期待される出力と事実として一致しているかを判定する",
        evaluation_steps=[
            "実際の出力から事実にあたる主張を取り出す",
            "期待される出力と照合し、事実として矛盾がないか確かめる",
            "言い回し・単位・表記の違いは減点しない。意味が同じなら一致とみなす",
            "事実が誤っている場合のみ減点する",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
        model=judge,
    )


@pytest.mark.parametrize(
    "input_text,actual_output,expected_output",
    [
        pytest.param(
            "世界で最も高い山は何ですか",
            "エベレストです",
            "エベレスト山",
            id="事実として正しい",
        ),
        pytest.param(
            "光の速さは",
            "光速は真空中で毎秒およそ 299,792,458 メートルです",
            "秒速約 30 万 km",
            id="表記は違うが意味は同じ",
        ),
    ],
)
def test_correctness_passes(judge, input_text, actual_output, expected_output):
    """正しい応答が合格することを確かめる。"""
    assert_test(
        LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
        ),
        [correctness_metric(judge)],
    )


def test_correctness_catches_wrong_answer(judge):
    """誤った応答を実際に落とすことを確かめる。

    合格するケースだけ並べても、指標が機能している証拠にはならない。
    常に満点を返す壊れ方は、合格ケースでは合格に見える。
    落ちるべきものが落ちて初めて、回帰テストとして意味を持つ。
    """
    metric = correctness_metric(judge)
    metric.measure(
        LLMTestCase(
            input="Python の生みの親は誰ですか",
            actual_output="Python は Google によって開発されました",
            expected_output="グイド・ヴァン・ロッサム",
        )
    )

    assert metric.score < metric.threshold, (
        f"事実誤認が合格した。スコア {metric.score} 閾値 {metric.threshold}\n"
        f"理由: {metric.reason}"
    )
