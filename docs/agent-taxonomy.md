# エージェント分類のメンタルモデルとリポ資産マッピング

## 受領インプット

「AI エージェントプロダクトの分類メンタルモデル」を受けて、本リポの全エージェント実装を 2 軸で整理する。コードは変更せず、各実装の実体（runner / agent / tool）を読んで現状を可視化し、手薄な象限と次に作る候補を特定する。

## 2 軸メンタルモデル

エージェントプロダクトを 2 つの軸で位置づける。

- **X 軸（実行形態）** バックグラウンド型 ↔ 消費者対応型
  - バックグラウンド型 人間の対話を介さず裏で走る。バッチ / CI / cron。完了して結果を返す
  - 消費者対応型 ユーザーと対話しながら進む。REPL / チャット / GUI。応答性とターン管理が要件
- **Y 軸（適用範囲）** タスク特化型 ↔ 汎用型
  - タスク特化型 単一ユースケースに最適化（例 PDF を索引して答える、リポを読んで要約する）
  - 汎用型 任意のタスクを Tool 群で解く。ドメイン非依存

この 2 軸で 4 象限ができる。各象限で必要な設計（プロンプト / Tool セット / メモリ / HITL）が変わる。

## 3 つの柱

各エージェントの成熟度を 3 つの柱で評価する。

- **学習** 会話履歴 / メモリ / RAG など、過去や外部知識を取り込む仕組み
- **意思決定** どの Tool をいつ呼ぶか、いつ終わるかを LLM 自身が判断する Tool Calling ループ
- **自律行動** 人間の承認なしにどこまで行動できるか。書き込み / 実行 / 多段ステップの実行力

評価記号 ✅ 実装済 / 🟡 部分的・骨格のみ / 🔴 未実装

## リポ資産の 2 軸マッピング

実体を読んで確認した内容にもとづく。playwright / 名前付き deep-research エージェントは存在しない。

| 実装 | 実体 | X 軸（実行形態） | Y 軸（適用範囲） | 象限 |
|---|---|---|---|---|
| ts repo-reader | `ts/agents/repo-reader/runner.ts`（`listFiles` + `readFile` の 2 tool を Agent に配線） | 消費者対応型（CLI 実行） | タスク特化型（リポ読解 → 要約） | 消費者 × 特化 |
| ts Core / CLI 基盤 | `ts/core/`（Provider 抽象 + metrics）/ `ts/cli/agent.ts`（Tool ループ）/ `ts/cli/tools/`（readFile / listFiles / grep / writeFile / workspace ガード） | 横断（基盤層） | 汎用型（任意の tool を載せられる） | 基盤（象限非依存） |
| python langchain search_agent | `python/langchain/agents/search_agent/runner.py`（PDF → split → Chroma → retriever → 検索 tool → `create_react_agent`） | バックグラウンド型（CLI 1-shot） | タスク特化型（PDF QA / RAG） | BG × 特化 |
| python langchain web_researcher | `python/langchain/agents/web_researcher/runner.py`（Tavily `web_search` + `write_file`、`build_hitl_agent` で write 前に `interrupt` 承認） | 両（CLI 承認 + Streamlit GUI `bin/web_researcher_app.py`） | タスク特化型（Web 調査 → HTML レポート） | 両 × 特化 |
| python langchain doc_reader | `python/langchain/agents/doc_reader/runner.py` | バックグラウンド型（CLI 1-shot） | タスク特化型（ドキュメント要約） | BG × 特化 |
| mastra assistant | `mastra/src/mastra/agents/assistant.ts`（instructions のみ、tool / memory / workflow なし） | 消費者対応型（mastra dev playground） | 汎用型（命令応答のみ、用途を絞っていない） | 消費者 × 汎用（ただし機能は最小） |
| go genai | `go/cmd/genai/`（usecase 層で自前 Tool ループ、calculator / search_knowledge） | バックグラウンド型（CLI デモ） | タスク特化型（Tool ループ比較） | BG × 特化 |
| go genkit | `go/cmd/genkit/`（Genkit Generate 内部ループ） | バックグラウンド型（CLI デモ） | タスク特化型 | BG × 特化 |
| go adk | `go/cmd/adk/`（ADK Runner + Session Service） | バックグラウンド型（CLI デモ） | タスク特化型（同一 tool セット） | BG × 特化 |

象限サマリ（機能が最小の mastra は括弧付き）

```
                    汎用型
                      │
   (mastra assistant) │   ts Core / CLI 基盤
   命令応答のみ最小    │  （基盤・象限非依存）
                      │
─ バックグラウンド ───┼─── 消費者対応 ─
                      │
  go genai/genkit/adk │   ts repo-reader
  langchain search    │   langchain web_researcher
  langchain doc_reader│   （CLI 承認 + GUI、両軸にまたがる）
 （BG × 特化）         │  （消費者 × 特化）
                      │
                   タスク特化型
```

## 3 つの柱の現状評価

| 実装 | 学習（メモリ / RAG） | 意思決定（Tool ループ） | 自律行動（書き込み / 多段 / HITL） |
|---|---|---|---|
| ts repo-reader | 🟡 会話履歴あり、RAG なし | ✅ listFiles / readFile の Tool ループ（`cli/agent.ts`） | 🔴 read 系 2 tool のみ配線、write / exec なし |
| ts Core / CLI 基盤 | 🟡 messages 管理あり | ✅ Provider 抽象 + Tool 定義 + maxSteps | 🟡 writeFile tool は存在するが repo-reader 未配線、Action 層 HITL / Sandbox 未着手（ts-phase3） |
| langchain search_agent | ✅ Chroma + embeddings + retriever（RAG 完備） | ✅ `create_react_agent` の ReAct ループ | 🔴 read（検索）のみ、書き込みなし |
| langchain web_researcher | 🟡 MemorySaver checkpointer（スレッド継続）、RAG なし | ✅ web_search + write_file の ReAct ループ | ✅ write 前 `interrupt` で HITL 承認、CLI / GUI 両方で resume |
| langchain doc_reader | 🟡 会話履歴 | ✅ ReAct ループ | 🔴 要約のみ |
| mastra assistant | 🔴 instructions のみ | 🔴 tool 未配線 | 🔴 行動なし（応答のみ） |
| go genai | 🟡 ドメイン層で履歴保持 | ✅ usecase 自前ループ（iteration / token 上限 / 同一 tool 検出） | 🔴 calculator / search_knowledge のみ、副作用なし |
| go genkit | 🟡 `ai.WithMessages` | ✅ Genkit Generate 内部ループ | 🔴 同上 |
| go adk | ✅ Session Service（state 永続の枠組み） | ✅ ADK Runner 内部ループ | 🔴 同 tool セット、副作用なし |

## 手薄な象限

- **消費者対応 × 汎用** が薄い。mastra assistant が位置的にはここだが tool / memory / workflow 未実装で機能が最小。「対話しながら任意タスクを tool で解く汎用アシスタント」が実質ゼロ
- **自律行動は langchain web_researcher の 1 本だけが ✅**。HITL 承認付き write を持つのはこの実装のみ。ts / go / mastra には書き込み・実行の自律行動がない。ts は writeFile tool を持つが repo-reader に配線していない
- **RAG（学習の柱）は langchain search_agent のみ**。ts / mastra / go に retrieval がなく、横並び比較が成立していない
- **go の 3 実装は tool セットが同一で副作用なし**。ライブラリ比較が目的のため意図的だが、自律行動の観点では 3 本とも 🔴 で並ぶ

## 次に作ると学習効果が高いエージェント

langchain が HITL（web_researcher）と RAG（search_agent）の参照実装を既に持つ。これを他エコシステムへ横展開すると 2 軸 × 複数実装の比較が完成する。

1. **ts repo-reader への writeFile 配線 + HITL 承認（自律行動を初めて ✅ に）** writeFile tool は `cli/tools/` に既存。repo-reader へ配線し、`cli/agent.ts` のループに承認ゲート（ts-phase3 の `needsApproval`）を足すと、langchain web_researcher と同じ HITL パターンを TS フレームワーク非依存で再現できる。消費者 × 特化を深化させる
2. **mastra への tool + memory 追加（消費者 × 汎用の空白を埋める）** assistant に Mastra の createTool / `@mastra/memory` を載せ、langchain web_researcher と同一ユースケース（Web 調査 → レポート）を Mastra 標準構造で実装する。「フレームワークが HITL / memory をどう抽象化するか」を langchain と直接比較できる
3. **ts / mastra への RAG（retriever tool）追加（学習の柱を全実装で揃える）** langchain search_agent の Chroma + retriever を ts / mastra にも実装し、retrieval の有無・フレームワーク差を横並びにする
