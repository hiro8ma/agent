# tei

Hugging Face の text-embeddings-inference（TEI）を手元の Docker で動かし、日本語の埋め込みモデル Ruri v3（`cl-nagoya/ruri-v3-310m`、Apache-2.0、768 次元）で文字列をベクトルにする

- 文書は外部の API に送らない
- `Client` は `search.Embedder` を満たし、`search.Flat` / `search.IVF` / `search.Hybrid` にそのまま渡せる

## 起動

```sh
docker run --rm -p 8080:80 -v tei-ruri-data:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4 \
  --model-id cl-nagoya/ruri-v3-310m --max-batch-tokens 2048 --auto-truncate
```

- Apple Silicon では `cpu-arm64-*` のタグを使う（`cpu-1.9` は amd64 だけ）
- ONNX の重みが無いので、TEI は safetensors を candle の ModernBERT で読む
- `--max-batch-tokens` を既定（16384）のままにすると、起動時の試運転でメモリが足りずに落ちる（Docker の上限 11.6 GiB で終了コード 137）

## 接頭辞

Ruri v3 は用途ごとに入力の前に接頭辞を付ける

| 用途 | 接頭辞 |
|---|---|
| 意味の近さ | なし |
| 分類 / クラスタリング | `トピック: ` |
| 検索のクエリ | `検索クエリ: ` |
| 検索の文書 | `検索文書: ` |

`WithPrefixes(RuriQueryPrefix, RuriDocumentPrefix)` を渡すと、`Embed` / `EmbedQueries` はクエリの接頭辞を、`EmbedDocuments` は文書の接頭辞を付ける

## テスト

```sh
make test                                                                # 保存した埋め込みだけを使う
TEI_LIVE=1 go test -run Live ./internal/embedding/tei/                    # 起動済みの TEI に繋ぐ
TEI_LIVE=1 TEI_UPDATE_EMBEDDINGS=1 go test -run LiveWriteRuriFixture ./internal/embedding/tei/
```

- `TEI_URL` で TEI の URL を変える（既定 `http://localhost:8080`）
- `TEI_ES_URL` を渡すと、`testdata/ruri_kenpou.json` に Elasticsearch（kuromoji）の上位 10 件も記録する
- `testdata/ruri_kenpou.json` は日本国憲法 103 条（接頭辞あり / なし）、検索のクエリ、例文の埋め込みで、成分を小数 4 桁に丸めてある
