"""pytest から DeepEval を回すための共通設定。

判定役のモデルは 1 度だけ作ってテスト間で使い回す。
テストごとに作り直しても動くが、接続の初期化を人数分繰り返す意味がない。
"""

import os

import pytest
from deepeval.models import GeminiModel

MODEL = "gemini-3.6-flash"


@pytest.fixture(scope="session")
def judge():
    """判定役の LLM。

    キーが無ければ skip する。ただし EVAL_REQUIRE=1 なら落とす。
    CI で skip は緑になるため、「評価に通った」と「評価が走らなかった」を
    テスト結果の色で区別できない。キーの設定漏れやクォータ枯渇を
    緑のまま見逃す事故は、この分岐で防げる。
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return GeminiModel(model=MODEL, api_key=key)

    if os.environ.get("EVAL_REQUIRE") == "1":
        pytest.fail("GEMINI_API_KEY が未設定。EVAL_REQUIRE=1 のため skip せず失敗させた")

    pytest.skip("GEMINI_API_KEY が未設定のため skip。CI では EVAL_REQUIRE=1 を立てる")
