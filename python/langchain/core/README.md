# Core 層

LLM API 抽象化レイヤー。LangChain の `ChatModel` SPI を Provider（OpenAI / Anthropic / Google Gemini）越しに使えるようにする層と、エージェント全体で共有する型を置く。

## 責務

- LangChain `BaseChatModel` をラップした Provider 抽象化
- `model_factory` で env から Provider を選択
- 共通型（pydantic BaseModel）

## 依存

- LangChain（`langchain-core`、各 provider パッケージ）
- pydantic
- 上位層（CLI / Action）から呼び出されるが、自身は何にも依存しない

## ファイル構成（予定）

```
core/
├── __init__.py
├── types.py              共通型（pydantic）
├── providers/
│   ├── __init__.py
│   ├── openai.py
│   ├── anthropic.py
│   ├── google.py
│   └── factory.py
```

## 設計方針

- Provider 切替は env 変数 1 つで済むように（`LLM_PROVIDER=anthropic`）
- 型は LangChain `Message` / `Tool` を活用（独自定義は最小限）
- ストリーミングは `astream` を使う
- エラーは Provider 固有エラーをそのまま伝搬（LangChain が共通化済み）

## ts/ との対比

| 観点 | ts/ | python/ |
|---|---|---|
| Provider 抽象 | 自前 SPI（`LanguageModel` interface） | LangChain の `BaseChatModel` |
| 型 | zod | pydantic |
| Tool 定義 | `Tool<Schema>` 型 | LangChain `BaseTool` |
| Streaming | AsyncIterable | `astream` |

ts/ は「フレームワーク非依存で 4 層全部自分で組む」、python/ は「LangChain の抽象化を正面から使う」。同じ問題への異なるアプローチ。
