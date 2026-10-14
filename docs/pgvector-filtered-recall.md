# フィルタ付き検索を起動時の Recall 測定にも反映する

テナントを指定して検索する構成では、起動時の品質測定も同じテナント条件で行う。
全体で Recall が高くても、HNSW の探索後にテナント条件で候補が落ちると、実際の検索では必要な文書が返らないことがある。

`go/internal/genkitagent/adapter/infrastructure/knowledgepgvector/` では次を揃える。

- 選択率を先に測り、その値から探索設定を決めてから Recall を測定する。
- 測定クエリを作る文書のサンプリングと近傍の選択にも `FilterColumn` / `FilterValue` を適用する。
- 厳密検索と近似検索の正解集合・候補集合に同じ条件を適用する。
- 値は SQL のパラメーターで渡す。

これはベクトル近傍の再現率の測定であり、利用者にとっての関連性の評価ではない。
件数が LIMIT に届いても、正解の近傍が揃ったとは限らない。選択率と `ef_search` の関係も探索幅を決める目安で、Recall の保証ではない。
二値化による候補検索を導入する場合も同じ条件で正解集合を作り、候補に厳密検索の上位文書が入った割合を測る。再ランキングは候補漏れを回復できない。

## 回帰テスト

```sh
cd go
go test ./internal/genkitagent/adapter/infrastructure/knowledgepgvector/
# vector 拡張が作成済みの、テスト専用データベースを指定する。
PGVECTOR_TEST_REQUIRE=1 PGVECTOR_TEST_DSN="$TEST_DATABASE_URL" \
  go test -count=1 ./internal/genkitagent/adapter/infrastructure/knowledgepgvector/
```

統合テストでは対象テナントが全体の 1% で、別テナントとベクトル分布が異なるデータを使う。
サンプリング・中点作成・厳密/近似検索の各段階に別テナントが入らないことを確認する。
対象行が 0 件または 1 件でも起動時検証が破綻しないことを確認する。

入力元：pgvector の Sparse Vector・Binary Embedding と RAG スキーマ設計の提供教材。
仕様の参照：[pgvector filtering / iterative scan](https://github.com/pgvector/pgvector/blob/v0.8.7/README.md#filtering)。
