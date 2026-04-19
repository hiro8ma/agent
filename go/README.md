# Go 実装

Go で AI エージェントを 3 種類のライブラリ（`google.golang.org/genai` / Firebase Genkit / Google ADK）で実装して比較する

## アーキテクチャ

ヘキサゴナル / クリーンアーキテクチャを単一 Go モジュール内で採用

```
go/
├── cmd/
│   ├── genai/main.go           # DI: genai adapter を注入
│   ├── genkit/main.go          # DI: genkit adapter を注入
│   └── adk/main.go             # DI: adk adapter を注入
├── internal/
│   ├── agent/                  # 3 実装で共有
│   │   ├── prompt.go           # SystemPrompt 定数
│   │   ├── domain/
│   │   │   ├── model/          # Message / ToolCall / ToolSchema
│   │   │   └── externalservice/
│   │   │       ├── llm.go      # LLMProvider IF
│   │   │       └── tool.go     # ToolExecutor IF
│   │   └── usecase/
│   │       ├── ask.go          # ユースケース IF
│   │       └── ask_impl.go     # Tool Dispatch ループ参照実装（genai 用）
│   ├── tool/                   # 共有 Tool 実装
│   │   ├── registry.go
│   │   ├── calculator.go
│   │   └── search_knowledge.go
│   ├── handler/cli/            # 共有 CLI ランナー
│   ├── genai/                  # genai adapter（LLMProvider 実装）
│   ├── genkit/                 # genkit adapter（usecase.Ask 直接実装）
│   ├── adk/                    # adk adapter（usecase.Ask 直接実装）
│   └── lib/                    # 開放レイヤー
│       └── liberrors/          # エラー正規化
├── build/
│   ├── Dockerfile              # 3 実装共通（TARGET ARG で cmd を切替）
│   └── compose.yaml            # docker compose up で 3 サービス起動
├── Makefile
└── go.mod
```

### LLMProvider と usecase.Ask の使い分け

3 ライブラリの Tool ループ粒度が違うため、境界を 2 つ用意した

| 実装 | 実装する IF | Tool ループの所在 |
|---|---|---|
| genai | `domain/externalservice.LLMProvider` | usecase の参照実装で自前ループ（iteration 上限 / トークン累積上限 / 同一 Tool 検出） |
| genkit | `usecase.Ask` | Genkit 内部の Generate ループに任せる |
| adk | `usecase.Ask` | ADK Runner に任せる |

→ 各ライブラリが何を抽象化しているかが実装パターンの差として現れる

## Quick Start

```bash
cd go
cp ../.env.example ../.env      # GEMINI_API_KEY を設定
make setup                      # 依存取得

make demo-genai                 # genai 実装デモ
make demo-genkit                # genkit 実装デモ
make demo-adk                   # adk 実装デモ

make demo-all                   # 3 実装を順次実行
```

Docker Compose でも 3 実装を同じ環境で回せる

```bash
make compose-up                 # build + 3 サービス起動
```

## 比較観点

| 評価軸 | genai | genkit | adk |
|---|---|---|---|
| 依存ライブラリ | `google.golang.org/genai` のみ | Genkit + googlegenai plugin | ADK + genai |
| Tool Dispatch | usecase 層で自前ループ | Genkit Generate 内部ループ | ADK Runner 内部ループ |
| システムプロンプト | `SystemInstruction` 直指定 | `ai.WithSystem` | `llmagent.Config.Instruction` |
| 会話履歴 | ドメイン層で保持 + 各ターン変換 | `ai.WithMessages` | Session Service（InMemory） |
| 観測性 | 自前計装が必要 | Dev UI + OTel 自動計装 | OTel native + Callback |
| ループ停止条件 | usecase で完全制御 | Genkit の内部制御 | Runner の内部制御 |
| MVP 実装行数（参考） | ~60 行（ask_impl + llm.go） | ~80 行（ask.go） | ~110 行（ask.go + tools.go） |

## 共有レイヤー

- `internal/agent/domain/` — 3 実装で同じドメインモデル（Message / ToolCall / ToolSchema）を使う
- `internal/tool/` — calculator / search_knowledge の Handler は 1 箇所で定義、3 実装から参照
- `internal/agent/prompt.go` — システムプロンプトも 1 箇所
- `internal/handler/cli/` — デモランナーも 1 箇所

3 実装の差異は `internal/{genai,genkit,adk}/` の adapter コードのみに閉じ込めている

## Tech Stack

- **Language** Go 1.25
- **SDK** google.golang.org/genai v1.51
- **Frameworks** Firebase Genkit v1.4 / Google ADK v0.6
- **Model** Gemini 2.0 Flash
