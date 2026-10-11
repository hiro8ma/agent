# tei

Hugging Face の text-embeddings-inference（TEI）を手元の Docker で動かし、日本語の埋め込みモデル Ruri v3（`cl-nagoya/ruri-v3-310m`、Apache-2.0、768 次元）で文字列をベクトルにする

- 文書は外部の API に送らない
- `Client` は `search.Embedder` を満たし、`search.Flat` / `search.IVF` / `search.Hybrid` にそのまま渡せる
- 交差エンコーダを読み込んだ TEI に繋いだ `Client` は `search.Reranker` を満たし、`search.CrossRerank` に渡せる

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

## トークンごとのベクトル

`EmbedTokens` / `EmbedDocumentTokens` は `/embed_all` でプーリング前のトークンごとのベクトルを返し、`search.MaxSim` / `search.RankMaxSim` に渡せる

- 応答は入力が1件でも「入力 × トークン × 次元」の3段の配列になる
- ベクトルは最後の層の LayerNorm の後の値で、正規化していない。クライアントで長さ1にする（Ruri v3 の生のベクトルの長さは30から40程度）
- パディングは含まないが、トークナイザが足す特殊トークンは残る。Ruri v3 では先頭の `<s>` と末尾の `</s>` で、`search.DropEnds` で外せる
- `--pooling` は `/embed_all` に効かない
- Ruri v3 は文単位の埋め込みとして学習したモデルで、ColBERT のような128次元への射影も遅延相互作用の学習も無い。ここでの MaxSim は比較のための近似

## 再ランカー

交差エンコーダ `hotchpotch/japanese-reranker-xsmall-v2`（ModernBERT 系、約37M パラメータ、MIT）を、埋め込みとは別の TEI で8081番に立てる

```sh
docker run --rm -p 8081:80 -v tei-rerank-data:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4 \
  --model-id hotchpotch/japanese-reranker-xsmall-v2 --max-client-batch-size 128 --auto-truncate
```

- 別のモデルを使うときは `--model-id` と `TEI_RERANK_MODEL` を揃える
- 精度の上限を見るなら `cl-nagoya/ruri-v3-reranker-310m`（Apache-2.0、315M パラメータ）を使う。このリポではまだ測っていない
- 旧世代の `hotchpotch/japanese-reranker-cross-encoder-xsmall-v1`（BERT 系）は `--max-batch-tokens 2048` を付けて立てた
- `Rerank` は `/rerank` に `raw_scores: false` で送るので、出力が1つのモデルの点数はシグモイドを通した0から1の値になる
- TEI は点数の高い順に返す。`Rerank` は `index` で入力の順に戻す
- `texts` が `--max-client-batch-size`（既定32）を超えると TEI は断る。`Rerank` は `WithBatchSize`（既定32）件ずつ分けて送る
- 接頭辞は付けない。交差エンコーダには Ruri の接頭辞の約束が無い
- 文書ごとに1回推論するので、`search.CrossRerank` の第1段で候補を絞ってから使う

日本国憲法の103条と10個のクエリで比べた。v2 の結果は `testdata/rerank_kenpou.json` にある（Apple Silicon の CPU、Docker の上限 11.6 GiB）

| 方式 | hit@1 | hit@3 | MRR@10 | 1クエリの時間（中央値） | p95 |
|---|---|---|---|---|---|
| Ruri | 0.90 | 1.00 | 0.950 | 88ms | 97ms |
| Ruri の上位20件 + v2 | 0.90 | 0.90 | 0.914 | 187ms | 204ms |
| Ruri の上位50件 + v2 | 0.90 | 0.90 | 0.900 | 361ms | 419ms |
| Ruri の上位20件 + v1 | 1.00 | 1.00 | 1.000 | 573ms | 609ms |
| Ruri の上位50件 + v1 | 1.00 | 1.00 | 1.000 | 1.29s | 1.56s |
| Ruri のトークンで MaxSim（全トークン） | 0.90 | 1.00 | 0.950 | 118ms | 124ms |
| Ruri のトークンで MaxSim（`<s>` と `</s>` を除く） | 0.80 | 1.00 | 0.900 | 115ms | 137ms |

- 時間はクエリを TEI で埋め込むところから上位10件が決まるまで。文書のベクトルは前もって求めておく
- v1 の行は既定の `--max-client-batch-size`（32）の TEI で測った。上位50件でもクライアントが32件ずつ分けて送るので断られない
- Ruri だけで hit@3 が1.00なので、hit@3 では差が出ない
- v1 と v2 はどちらも「働く権利」の1位を第二十八条（団結権）から第二十七条（勤労の権利）に入れ替える
- v2 は「好きな仕事を選べる」で103条すべてを0.071以下と採点し、正解の第二十二条（職業選択の自由）を70位に置く。言い換えのクエリを拾えず、候補を50件に広げると正解が上位10件から落ちる
- v2 は v1 より約3倍速い
- MaxSim は全トークンなら Ruri と同じ順位で、`<s>` と `</s>` を除くと「差別されない権利」の正解が2位に下がる
- 条文のトークンのベクトルは103条で5,562トークン、求めるのに約29秒かかる

## テスト

```sh
make test                                                                # 保存した埋め込みだけを使う
TEI_LIVE=1 go test -run Live ./internal/embedding/tei/                    # 起動済みの TEI に繋ぐ
TEI_LIVE=1 TEI_UPDATE_EMBEDDINGS=1 go test -run LiveWriteRuriFixture ./internal/embedding/tei/
TEI_LIVE=1 go test -run LiveRerankKenpou -v ./internal/embedding/tei/     # 埋め込みと再ランカーの2つの TEI に繋ぐ
TEI_LIVE=1 TEI_UPDATE_EMBEDDINGS=1 go test -run LiveRerankKenpou ./internal/embedding/tei/
```

- `TEI_URL` で TEI の URL を変える（既定 `http://localhost:8080`）
- `TEI_RERANK_URL` で再ランカーの TEI の URL を変える（既定 `http://localhost:8081`）。`TEI_RERANK_MODEL` は `/info` で確かめるモデル（既定 `hotchpotch/japanese-reranker-xsmall-v2`）
- `-run Live` だけで流すと再ランカーの TEI も待つ。埋め込みだけを確かめるときは `-run 'Live[^R]'` のように絞る
- `TEI_ES_URL` を渡すと、`testdata/ruri_kenpou.json` に Elasticsearch（kuromoji）の上位 10 件も記録する
- `testdata/ruri_kenpou.json` は日本国憲法 103 条（接頭辞あり / なし）、検索のクエリ、例文の埋め込みで、成分を小数 4 桁に丸めてある
