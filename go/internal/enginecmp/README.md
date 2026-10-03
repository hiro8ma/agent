# enginecmp

日本国憲法の 103 条を Elasticsearch（kuromoji）、Meilisearch、`internal/search` に入れ、同じクエリの上位を比べる

## データ

- `kenpou.json` は 1 条を 1 つの文字列にした配列
- 出典は e-Gov 法令 API v2 の `https://laws.e-gov.go.jp/api/2/law_data/321CONSTITUTION?law_full_text_format=json`（2026-10-03 取得）
- 法令は著作権法第 13 条により著作権の目的とならない
- 取り直すときは `go generate ./internal/enginecmp`

## 起動と実行

```sh
docker run --rm -p 9200:9200 -p 9300:9300 -e discovery.type=single-node -e xpack.security.enabled=false -e xpack.license.self_generated.type=basic docker.elastic.co/elasticsearch/elasticsearch:9.1.3 bash -c 'elasticsearch-plugin install --batch analysis-kuromoji && exec bin/elasticsearch'
docker run --rm -p 7700:7700 -e MEILI_NO_ANALYTICS=true getmeili/meilisearch:v1.20

go run ./cmd/enginecmp                       # 既定のクエリ
go run ./cmd/enginecmp すべて国民 会議        # クエリを指定する
ENGINECMP_LIVE=1 go test -run Live ./internal/enginecmp/
```

`make test` はコンテナもネットワークも使わない
