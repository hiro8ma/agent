---
title: "エージェントシステム設計の 5 原則"
date: "2026-05-04"
tags: [agent, system-design, principles, scalability, modularity, resilience]
---

# エージェントシステム設計の 5 原則

エージェントを PoC から本番システムに育てるために必要な 5 つの設計原則。各原則を「失敗例」と「設計上の押さえどころ」で具体化する。

## 1. 拡張性（Scalability）

分散アーキテクチャ、クラウド基盤、並列処理、リソース最適化アルゴリズムでワークロード増加に対応する。

**失敗例**: 1 分あたり 10 件のリクエストを捌くサポートエージェントが、トラフィック 1,000 件/分への急増時にオートスケール基盤がなくて停止。

**設計上の押さえどころ**:

- worker は stateless に保ち、状態は外部（Firestore / Redis / DB 等）に持つ
- LLM API のレート制限を意識した backpressure / queue 設計
- Cloud Run / Lambda / Cloud Functions 系のオートスケールに任せる
- Prompt cache とコンテキスト圧縮でトークン消費を抑制

## 2. モジュール性（Modularity）

明確なインターフェースで結ばれた独立・交換可能なコンポーネント群として設計する。

**失敗例**: ツール呼び出しをサービス内にハードコードしたエージェントは、わずかな変更でも全面再デプロイが必要。

**設計上の押さえどころ**:

- LLM Provider を SPI（Service Provider Interface）で抽象化、モデル乗り換え可能に
- Tool 定義と Tool 実行を分離（proto annotation / 設定ファイル経由）
- Skill / Playbook を外部化して動的にロード
- 3 層構造（Core 層 LLM 抽象化 / CLI 層 ループ + Tool / Action 層 CI/CD 統合）の分離

## 3. 継続的学習（Continuous Learning）

コンテキスト内学習やフィードバックループでユーザー反応を反映し、変化するタスクに性能を保つ。

**失敗例**: フィードバックループを無視すると、誤分類やエスカレーション忘れを繰り返す。

**設計上の押さえどころ**:

- ユーザーフィードバック（👍 / 👎 / 自由記述）の収集と Golden QA 化
- Skill / Prompt のバージョン管理 + A/B テスト
- Ragas / LLM-as-judge / Golden QA 等の評価基盤で回帰検出
- 失敗ケースを次の Few-shot example に取り込む経路

## 4. レジリエンス（Resilience）

エラー、セキュリティ脅威、タイムアウト、想定外事象をしなやかに処理する。

**失敗例**: リトライやフォールバックを備えないエージェントは、API 呼び出しが一度失敗しただけで停止し利用者を困惑させる。

**設計上の押さえどころ**:

- リトライ可否のエラー分類（一時エラーはリトライ、Safety filter ブロックはリトライ不可）
- Tool 実行のタイムアウト + 同一 Tool の連続呼び出し抑制
- Stream 中エラーの partial result の扱い設計
- HITL（Human-in-the-loop）承認を破壊的操作に挟む
- max iterations でループ暴走を防ぐ

## 5. 将来性（Future-proofing）

オープンスタンダードとスケーラブルな基盤、新技術への迅速な適応文化。

**失敗例**: 特定ベンダーのプロンプトフォーマットに強く依存するとモデル乗り換えが難しくなり、実験の幅が狭まる。

**設計上の押さえどころ**:

- ベンダー固有 SDK を Provider 抽象化レイヤーに閉じ込める
- MCP / OpenAPI / OpenInference / OTel 等のオープンスタンダードを採用
- Prompt は外部化（dotprompt / Markdown 等）してモデル間ポータビリティを確保
- フレームワーク非依存の最小実装を 1 つ持っておくと、フレームワーク選定の判断材料になる

## 5 原則のチェックリスト的活用

エージェントシステムの設計レビュー時、各原則について以下を問う。

| 原則 | 問い |
|---|---|
| 拡張性 | 10 倍のトラフィックが来た時、どこで詰まる？ |
| モジュール性 | LLM Provider を変えるのに何を書き換える？ |
| 継続的学習 | ユーザーフィードバックは収集できている？評価指標で回帰を検知できる？ |
| レジリエンス | API が落ちた / タイムアウトした / Safety filter で弾かれた時の挙動は？ |
| 将来性 | 6 ヶ月後に新モデルが出たら、どれくらいの工数で乗り換えられる？ |

5 つすべてに即答できれば設計は健全。詰まる箇所が「次に手を入れるべき場所」。

## 参考

- Building Effective AI Agents (Anthropic): https://www.anthropic.com/research/building-effective-agents
- nano-code: https://github.com/laiso/nano-code
