# AI Engineering Agent (Go ADK)

Google ADK for Go で構築した、AIエンジニアリング知識に特化した自律エージェント。

## 機能

- **ReActパターン**: 思考→ツール選択→実行→観察→再思考のループ
- **AI知識検索**: mcp/ai_knowledge/ のFT済みモデルに問い合わせ
- **Web検索**: mcp/external_api/ で最新情報を取得
- **Gemini 2.0 Flash**: Google ADK経由でプランニング・生成

## セットアップ

```bash
cp .env.example .env  # GEMINI_API_KEY を設定
make setup
```

## 使い方

```bash
# CLIモード
make run

# WebUIモード（http://localhost:8080）
make web
```

## アーキテクチャ

```
ユーザー
  ↓
ADKエージェント（ReActループ）
  ├→ search_knowledge（AI知識検索）
  └→ web_search（Web検索）
  ↓
最終応答
```
