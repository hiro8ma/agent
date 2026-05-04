---
title: "エージェントフレームワーク比較（2026 年 5 月）"
date: "2026-05-04"
tags: [agent, framework, langgraph, claude-sdk, openai-sdk, mcp, comparison]
---

# エージェントフレームワーク比較（2026 年 5 月）

主要フレームワークの最新状況、共通トレンド、選択指針を整理する。

## 2024 → 2026 の構造変化

| 観点 | 2024 | 2026 |
|---|---|---|
| 主流 | LangChain / AutoGen / CrewAI の三択 | クラウドベンダー固有 SDK + 各ベンダー自社 SDK |
| AutoGen | 主要 OSS | EOL（メンテナンスモード）、Microsoft Agent Framework に統合 |
| ツール統合 | 各フレームワーク固有 | MCP 一択（新規 92% / enterprise 78% 採用） |
| Tracing | ベンダー固有 | OpenTelemetry + OpenInference が事実上の標準 |
| 制御フロー | Graph DSL を書く | model-driven（LLM が制御フロー決定）が優勢に |
| 言語 | Python 一強 | TypeScript 勢急成長（Mastra / Vercel AI SDK） |

## 主要 4 フレームワーク

### LangGraph（LangChain）

- **最新**: v1.0 GA（2025-10）、no breaking changes until 2.0
- **特徴**: グラフベース + 共有 state、durable state、HITL API、time travel with interrupts
- **採用**: Uber / LinkedIn / Klarna の production 運用
- **強み**: 複雑な state machine、条件分岐 + リトライ + HITL を綺麗に表現
- **Repo**: https://github.com/langchain-ai/langgraph

### AutoGen（Microsoft）— EOL

- **状態**: v0.4 でメンテナンスモード入り
- **後継**: Microsoft Agent Framework（2025-10 public preview、Q1 2026 GA）
- **判断**: 新規プロジェクトでは選ばない、既存資産の維持のみ
- **後継リポ**: https://github.com/microsoft/agent-framework

### Claude Agent SDK（Anthropic）

- **2026 新機能**: SessionStore protocol（TS パリティ）、W3C trace context 伝播、tool 実行 timing metrics、MCP 接続並列化
- **Managed Agents（2026-04）**: Anthropic ホスト型サービス、長期 horizon 用に session/harness/sandbox の安定 IF
- **強み**: Claude モデル + Code 系の sandbox / tool 実行、HITL 設計の中心
- **Repo**: https://github.com/anthropics/claude-agent-sdk-python

### OpenAI Agents SDK

- **最新**: 2026-04-29
- **新機能**: Realtime Agents（default `gpt-realtime-1.5`）、Sandbox Agents（β）、100+ 非 OpenAI LLM サポート
- **強み**: Voice / Realtime、組み込み tracing UI、handoff モデル
- **Repo**: https://github.com/openai/openai-agents-python

## その他注目フレームワーク

| Framework | 言語 | 位置付け |
|---|---|---|
| **CrewAI** | Python | Role-based multi-agent、built-in memory、MCP native |
| **Mastra** | TypeScript | TS 第一、opinionated toolkit、built-in memory + MCP native |
| **Pydantic AI** | Python | 型安全、FastAPI 風 API、output schema 自動 retry |
| **Strands Agents（AWS）** | Python | Bedrock 統合、Graph/Swarm/Workflow + A2A + MCP |
| **Vertex AI Agent Engine（GCP）** | Python / Go | ADK + Agent Engine runtime、Gemini/Claude 200+ モデル |
| **Microsoft Agent Framework** | C# / Python | AutoGen + Semantic Kernel の後継、enterprise 向け |

## 共通トレンド

### 1. MCP がデファクト

- 2025〜Q1 2026 リリースの新規フレームワークの 92% が MCP native
- enterprise の 78% が MCP backed agent を本番運用（2026-04）
- public registry: 1.2k → 9.4k+ servers

### 2. Streaming / Realtime

- OpenAI が `gpt-realtime-1.5` を default に
- LangGraph が型安全 streaming を 1.0 で標準化
- 音声 + Realtime API が新しい入出力チャネルに

### 3. Multi-agent パターン

各 SDK で標準提供: Graph / Swarm / Workflow / Handoffs / A2A protocol。

### 4. Observability の標準化

- OpenTelemetry GenAI semantic conventions
- OpenInference（Arize）

## 選択指針（2026 年版）

| ケース | 推奨 |
|---|---|
| 複雑な state machine、durable + HITL が必須 | LangGraph 1.0 |
| TypeScript 第一、フルスタック | Mastra or Vercel AI SDK |
| 型安全 Python、構造化 output | Pydantic AI |
| Voice / Realtime、OpenAI スタック | OpenAI Agents SDK |
| Claude モデル + Code 系 | Claude Agent SDK |
| AWS Bedrock 中心 | Strands Agents |
| Microsoft / Azure | Microsoft Agent Framework |
| GCP / Gemini | Vertex AI Agent Engine + ADK |
| Role-based multi-agent prototyping | CrewAI |
| フレームワーク非依存 / 動作原理理解 | nano-code 流の自作 |

## Production 成熟度の判断軸

エージェントフレームワークを評価する時の 6 つの観点:

1. **Durable state**: プロセス再起動を跨いで状態を保持できるか
2. **HITL API**: 人間承認フックが標準で組み込まれているか
3. **Sandbox 隔離**: 危険操作を隔離環境で実行できるか
4. **OTel / OpenInference trace**: 観測性が標準で出せるか
5. **MCP 対応**: 既存 / 将来の Tool 資産を活用できるか
6. **Managed runtime**: ベンダーホスト版があるか

すべて満たせば production 即投入可能。半分以下なら自前で補完が必要。

## 参考

- LangGraph 1.0 GA: https://changelog.langchain.com/announcements/langgraph-1-0-is-now-generally-available
- Microsoft retires AutoGen: https://venturebeat.com/ai/microsoft-retires-autogen-and-debuts-agent-framework-to-unify-and-govern
- Building agents with the Claude Agent SDK: https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk
- The next evolution of the Agents SDK: https://openai.com/index/the-next-evolution-of-the-agents-sdk/
- Strands Agents (AWS): https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/
- MCP Adoption Statistics 2026: https://www.digitalapplied.com/blog/mcp-adoption-statistics-2026-model-context-protocol

## 関連

- `agent-system-design-principles.md` — 5 原則
- `coding-agent-design-patterns.md` — 設計パターン
