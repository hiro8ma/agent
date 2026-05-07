---
title: "エージェントフロー型分類と 2026 動向（Interleave / Plan-and-Execute / Simulation + Conductor-Swarm）"
date: "2026-05-07"
tags: [agent, react, plan-and-execute, mcts, multi-agent, conductor-swarm]
---

# エージェントフロー型分類と 2026 動向

エージェントワークフローの 3 型分類（Interleave / Plan-and-Execute / Simulation）に、2026 で主役級となった Conductor + Swarm パターンを加えて整理する。

## 3 型の基本定義

| 型 | 構造 | 向く用途 |
|---|---|---|
| **Plan-and-Execute** | タスク受領時にサブタスク系列を生成し逐次実行。サブタスクは JSON / コードで表現 | 業務用途が固定、Tool 集合が事前確定 |
| **Interleave（ReAct）** | 「思考 → 行動 → 観測」を逐次反復。LangChain / LlamaIndex 等で標準実装 | 高汎用性、未知タスク、対話型 |
| **Simulation（MCTS）** | モンテカルロ木探索で探索的に行動選択 | 複雑動的環境、戦略的計画、組合せ最適化 |

## 2026 全体トレンド

- **マルチエージェント問い合わせ 1,445% 増**（Gartner、Q1 2024 → Q2 2025）。シングル中心からフロー設計中心へ重心移動
- **エージェントフレームワーク採用率 約 2 倍**（2025 初頭 9% → 2026 初頭 18%）
- **Flow Engineering（制御フロー / 状態遷移 / 判断境界の設計）が AI エンジニアリングで最高レバレッジのスキル**と位置づけられた
- **MCP（Model Context Protocol）が事実上の標準**（Anthropic 公開 2024-11、18 ヶ月で OpenAI / MS / Google / Amazon 全部採用）
- Gartner 予測、2026 年末までにエンタープライズアプリの 40% がタスク特化エージェントを搭載（2025 年は 5% 未満）

## 各型の現状

### Interleave（ReAct）

- **Production-ready のデファクト**として定着。全エージェントフレームワークの基盤
- 単独利用は短期 / 単一ドメイン向けに固定化
- long-horizon タスクで脱線・コスト増の弱点が広く認識され、**他パターンとのハイブリッド化が標準**
- LangGraph `create_react_agent`、Genkit Go の Tool Dispatch ループ、OpenAI Agents SDK、Anthropic Agent SDK すべて ReAct 系の prebuilt を提供

### Plan-and-Execute

- **2026 で実装急増**。フロンティアモデルで Plan、安価モデルで Execute する構成が流行
- **コスト削減 90% 実例**（フロンティアモデルに全部任せる構成と比較）
- long-horizon タスク（複数ステップにわたる戦略処理）で ReAct を上回る
- LangGraph / Vercel AI SDK に reference 実装が同梱される段階

### Simulation（MCTS / Tree Search）

- **研究フェーズから実用フェーズへ移行中**だが production 採用は限定的
- 主要研究
  - **LATS (Language Agent Tree Search)** — 推論 / 行動 / 計画を MCTS で統一
  - **SWE-Search** — repo-level コーディングタスクで MCTS + 自己改善エージェント、SWE-bench 成績向上
  - **ToolTree** — Tool 計画を MCTS で探索、pre/post スコアによる枝刈り
  - **MCT Self-Refine** — Llama-3 8B + MCTS で GPT-4 級の数学推論を実現
- 適用領域は数学 / 戦略計画 / コード生成 / ゲーム
- 業務 Q&A のような対話型タスクには探索コスト > 期待値で重すぎる

## 4 つ目の潮流 — Conductor + Swarm

3 型分類とは別軸で、**マルチエージェントオーケストレーション** が 2026 の主役級。

| パターン | 構造 | 用途 |
|---|---|---|
| **Conductor**（指揮者型） | Master Agent が分解・委譲・統合 | 戦略立案、技術ディープリサーチ、複雑判断 |
| **Swarm**（群知能型） | 自律エージェントが並列協調 | 並列実行可能な定型タスク、スケール処理 |
| **ハイブリッド** | Conductor 層 + Swarm 層 | エンタープライズの主流構成 |

## フレームワーク勢力図（2026）

| フレームワーク | 公開 / 強み | 状況 |
|---|---|---|
| **LangGraph** | 月間検索 27,100、有向グラフ + 条件付きエッジ | 最大シェア |
| **OpenAI Agents SDK** | 2026-03 公開、明示的 handoff | OpenAI モデル限定 |
| **Google ADK** | 2026-04 公開、Gemini Enterprise Agent Platform 統合 | エンタープライズ向け |
| **Anthropic Agent SDK** | Claude 4.6 と同時公開、Constitutional AI による安全性をモデル層で評価 | オーケストレーション機能は限定的 |
| **Claude Agent SDK / Strands** | Agent loop 特化 | コーディング / CLI 用途 |

## 設計原則として持ち帰るもの

- **Flow Engineering 視点で設計する** — どのモデル呼び出しの精度を上げるかではなく、制御フロー / 状態遷移 / 判断境界をどう設計するかが第一論点
- **ReAct を入口にしてハイブリッド化を見越す** — 単純ユースケースで ReAct 単独でよいが、Plan-and-Execute / Conductor 層を後付けできる構造を維持
- **MCP を将来の前提にしておく** — Tool 抽象化は MCP 互換のシグネチャを意識（`name` / `description` / `input_schema`）
- **MCTS は罠を避ける** — 対話型の Q&A への適用は探索コストで割に合わない。コード自動修正など限定領域で再考

## 進化線（一般形）

```
Step 1  ReAct 単独（単一ドメイン Q&A、Tool 5-10 個）
  ↓
Step 2  ReAct + Plan-and-Execute（定型ワークフロー混在、Skill / Playbook 起動）
  ↓
Step 3  Conductor 層追加（Router で経路選択）+ 既存 ReAct / Skill = Swarm 層
  ↓
Step 4  限定領域で MCTS（コード自動修正、戦略立案）を実験的に導入
```

## 参考

- Sitepoint, [Agentic Design Patterns: The 2026 Guide](https://www.sitepoint.com/the-definitive-guide-to-agentic-design-patterns-in-2026/)
- ML Mastery, [7 Agentic AI Trends to Watch in 2026](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
- Datadog, [State of AI Engineering](https://www.datadoghq.com/state-of-ai-engineering/)
- AGIX Tech, [Conductor vs Swarm: Multi-Agent AI Guide (2026)](https://agixtech.com/insights/conductor-vs-swarm-multi-agent-ai-orchestration/)
- StackAI, [The 2026 Guide to Agentic Workflow Architectures](https://www.stackai.com/blog/the-2026-guide-to-agentic-workflow-architectures)
- arXiv, [Language Agent Tree Search (LATS)](https://arxiv.org/pdf/2310.04406)
- arXiv, [ToolTree: MCTS-based LLM Agent Tool Planning](https://arxiv.org/html/2603.12740v1)
- OpenReview, [SWE-Search: MCTS for Software Agents](https://openreview.net/forum?id=G7sIFXugTX)
- QubitTool, [2026 AI Agent Framework Showdown](https://qubittool.com/blog/ai-agent-framework-comparison-2026)
- Softmax, [Definitive Guide to Agentic Frameworks in 2026](https://softmaxdata.com/blog/definitive-guide-to-agentic-frameworks-in-2026-langgraph-crewai-ag2-openai-and-more/)
