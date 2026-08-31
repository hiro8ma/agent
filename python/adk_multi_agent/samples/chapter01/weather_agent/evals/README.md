# 天気エージェントの評価セット

## 構成

| ファイル | 役割 |
|---|---|
| `build_eval_set.py` | ADK の型から評価セットを組み立てる |
| `weather_agent_v1.evalset.json` | 生成物。`adk eval` へ渡す |
| `test_eval_set.py` | 評価セット自体の検査 |

## 使い方

```bash
# 評価セットを作り直す
uv run python samples/chapter01/weather_agent/evals/build_eval_set.py

# 評価セット自体を検査する（API キー不要）
uv run pytest samples/chapter01/weather_agent/evals/ -q

# エージェントを評価する（API キーが要る）
adk eval samples/chapter01/weather_agent \
  samples/chapter01/weather_agent/evals/weather_agent_v1.evalset.json
```

## 10 件の内訳

| 観点 | 件数 | eval_id |
|---|---|---|
| 正常系 | 4 | tokyo_basic / osaka_basic / sapporo_japanese / two_cities |
| ツールが error を返す | 1 | unknown_city |
| 範囲外 | 1 | out_of_scope |
| ペルソナ | 3 | hurried_user / vague_user / prompt_injection |
| 過剰な回答 | 1 | no_extrapolation |

ツール呼び出しの回数は 0 回が 3 件、1 回が 6 件、2 回が 1 件。

## 期待するツール呼び出しを書く理由

答えが合っていても手順が違う実行を区別するため。

`unknown_city` は未登録の都市でもツールを呼ぶことを期待する。
呼ばずに答えたら、ツールではなくモデルの知識から答えている。
出力は正しく見えても、データを差し替えた時点で崩れる。

`out_of_scope` と `prompt_injection` は逆に、呼ばないことを期待する。

## 評価セット自体を検査する理由

評価セットは評価される側と同じだけ間違える。
未登録の都市を正解に書けば、正しい実装が落ちる。

`test_expected_answers_match_tool_output` は、
正解の文に含まれる気温が、ツールが実際に返す値と一致するかを見る。
ツールの固定データを変えて評価セットを直し忘れると、ここで落ちる。
