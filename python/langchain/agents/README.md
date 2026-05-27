# agents/

個別エージェントの実装を置くディレクトリ。各エージェントは独立したサブディレクトリに閉じ、Core / CLI / Action 層には変更を加えずに追加できる。

## エージェントの作法

新しいエージェント `{name}` を追加する手順:

1. `agents/{name}/` ディレクトリを作る
2. 以下のファイルを置く

```
agents/{name}/
├── __init__.py
├── prompt.py       system prompt（定数）
├── runner.py       本体（CLI からも Action からも呼ばれる）
└── output.py       結果の整形・書き出し（任意）
```

3. `bin/{name}.py` に CLI エントリを置く
4. （CI 統合する場合）`bin/{name}_action.py` に Action エントリを置く

## runner.py の構造

```python
from cli.agent import build_agent
from core.providers.factory import select_provider

from .prompt import SYSTEM_PROMPT


def run(input: dict) -> dict:
    agent = build_agent(
        provider=select_provider(),
        system_prompt=SYSTEM_PROMPT,
        tools=[...],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": format_input(input)}]})
    return parse_output(result)
```

CLI / Action の両方から `run` 関数を呼び出す。エントリ側は引数パースと結果出力だけを担う。

## 計画中のエージェント

| 名前 | 用途 | 状態 |
|---|---|---|
| search_agent | PDF を Chroma に索引して検索 Tool で回答 | 実装済 |
| web_researcher | Web を Tavily で調べ HTML レポートを書き出す（書き込みは HITL 承認） | 実装済 |
| repo_reader | OSS リポを読み解いて要約を生成 | 設計中 |
| dev_digest | URL / PDF を要約してナレッジ化 | アイデア段階 |
| memory_curator | `.claude/memory/` の重複検出・整理 | アイデア段階 |

`web_researcher` は HITL の参考実装。`runner.run(topic, approve, ...)` が承認コールバックを受け取り、`build_web_researcher()` がコンパイル済みグラフを返す（CLI と Streamlit GUI で共有）。

ts/ と同じエージェント名で実装することで、エコシステム比較ができる。
