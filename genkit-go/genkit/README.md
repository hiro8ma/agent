# genkit パッケージ

Genkit Go SDK を使った AI エージェントの実装。テキスト生成、Tool Calling、構造化出力、Flow、RAG をサポート。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `app.go` | `App` 構造体・初期化・ツール/フロー定義 |
| `types.go` | 入出力スキーマの型定義 |
| `vectorstore.go` | インメモリベクトルストア（コサイン類似度検索） |
| `rag.go` | RAG（知識検索型生成）のインデックス・クエリ |
| `demo.go` | デモモード（全機能のデモンストレーション） |
| `server.go` | REST API サーバーモード |

## 対応モデルプロバイダ

- **Google AI (Gemini)** — `GEMINI_API_KEY` を設定。Embedding / RAG も利用可能
- **Ollama** — `USE_OLLAMA=true` を設定。ローカル LLM で動作（RAG は非対応）

## 機能一覧

### テキスト生成

LLM によるシンプルなテキスト生成。

### Tool Calling

LLM が外部ツールを呼び出して処理を実行。現在 `calculator`（四則演算）ツールを定義済み。

### 構造化出力

LLM の出力を Go の構造体に直接マッピング。`JokeOutput`、`SentimentOutput` など。

### Flow

再利用可能なワークフローを定義し、REST API エンドポイントとして公開可能。

- `tellJoke` — トピックを指定してジョークを生成
- `analyzeSentiment` — テキストの感情分析

### RAG（Retrieval-Augmented Generation）

Embedding でドキュメントをベクトル化し、類似検索で関連知識を取得してから LLM に回答を生成させる。Gemini API 使用時のみ利用可能。

## 使い方

```go
ctx := context.Background()
app := genkit.New(ctx)  // 環境変数に基づいて初期化
app.Run(ctx)            // MODE=demo or MODE=server
```

## REST API エンドポイント（サーバーモード）

| メソッド | パス | 説明 | リクエスト例 |
|---|---|---|---|
| POST | `/api/tellJoke` | ジョーク生成 | `{"data":{"topic":"Go言語"}}` |
| POST | `/api/analyzeSentiment` | 感情分析 | `{"data":"今日は最高！"}` |
| POST | `/api/chat` | チャット | `{"message":"こんにちは"}` |
| POST | `/api/rag` | RAG 質問応答 | `{"question":"Genkitとは？"}` |
| GET | `/health` | ヘルスチェック | — |

## RAG Search Characteristics

This agent implements **vector search** for RAG.

| Aspect | Detail |
|---|---|
| **Engine** | In-memory vector store with Google AI Embedder (text-embedding-004) |
| **Matching** | Cosine similarity between query and document embeddings |
| **Strengths** | Semantic similarity, handles natural language queries, no keyword engineering needed |
| **Weaknesses** | In-memory only (lost on restart), no persistence, may miss exact term matches |
| **Query style** | Natural language questions |
| **Top-K** | 3 (hardcoded) |

### How It Works

```
1. Documents → Embedding (text-embedding-004) → Vectors stored in memory
2. User query → Embedding → Cosine similarity with all stored vectors
3. Top-K most similar documents → Injected as context → LLM generates answer
```

### Limitations

- No full-text search fallback (pure vector search)
- No persistence (documents re-indexed on every restart)
- No chunking for large documents
- Ollama mode has no RAG support (no embedder available)
