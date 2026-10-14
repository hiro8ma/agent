# pgvector の一括投入とモデル変更

既存の検索アダプターは [knowledgepgvector](../go/internal/genkitagent/adapter/infrastructure/knowledgepgvector/pgvector.go)。投入の実例は [統合試験の insertRows](../go/internal/genkitagent/adapter/infrastructure/knowledgepgvector/pgvector_integration_test.go) にある。検索 API に投入処理を追加せず、検証用データの準備として扱っている。

## COPY で一括投入する

pgx の `CopyFrom` は PostgreSQL の binary COPY を使う。`CopyFromFunc` で行を送信時に生成すれば、全件の `[][]any` とベクトルを事前に保持する必要がなくなる。試験では ID、テナント、ベクトルの生成規則を保持し、戻り値の投入件数を要求件数と照合する。COPY は生成した値の意味や検索品質を検証する機能ではない。

vector の binary エンコードには接続ごとの型登録が要る。[NewPool](../go/internal/genkitagent/adapter/infrastructure/knowledgepgvector/pgvector_pool.go) は `AfterConnect` で pgvector の型を登録する。既存プールを使う場合も、各接続で同じ登録を行う。

新しい独立テーブルを準備するときは、代表的なデータを COPY で投入し、距離演算子に合う索引を構築して `ANALYZE` を実行する。IVFFlat は代表データを基にリストを学習する。HNSW も初期投入後の構築を検討できる。索引使用、同じフィルタの厳密検索に対する recall、返却件数、レイテンシを確認してから切り替える。

稼働中のデータを更新する際は、試験の `DROP TABLE` 手順を持ち込まない。既存テーブルへの投入と索引更新を扱い、再構築が必要なら負荷とロックを評価する。`CREATE INDEX CONCURRENTLY` を選ぶ場合は、その制約と失敗時の索引状態も確認する。

## 1 テーブルで使うモデルを明確にする

現在のアダプターは `vector(n)` の列とクエリ埋め込みの次元を起動時に照合し、コサイン距離に対応する索引を確認する。**同じ次元数の別モデルは検出できない。**モデル A の文書にモデル B のクエリを投げると、SQL が成功しても距離の意味が失われる。モデル名、版、前処理を投入と検索で揃える契約が必要になる。

モデル変更を評価するには、新モデル専用テーブルと埋め込み器を用意し、`PgVectorConfig.Table` を分けて旧モデルと並行評価する。同じ文書・チャンク、問い合わせ集合、権限フィルタで検索品質と速度を比較する。各モデル内の exact に対する ANN recall は索引の近似品質であり、モデル間の検索品質は適合性ラベルなどで別に評価する。評価後に検索先を切り替え、旧データの保持期間を決める。

次元が違うモデルを型指定なしの `vector` 列に混在させる方式も PostgreSQL では可能だが、このアダプターの対応範囲外になる。モデルごとの部分式索引、`embedding::vector(n)` の固定次元キャスト、距離演算子に合う opclass が要る。クエリ側にもモデルの条件と同じキャストを入れ、索引を利用する計画を確認する。固定次元列を前提にした現在の起動検査を通すためだけに設定を変える方法では対応できない。

## フィルタとパーティション

部署・テナントで絞る検索では、全体の索引に対する後段フィルタで ANN 候補が減ることがある。部分索引やパーティションは候補の母集団を狭める選択肢だが、索引の数、更新負荷、パーティションの剪定を確認する。モデル分離とテナント分離の目的を明確にし、同じフィルタで exact と ANN を比較する。[フィルタ付き recall の検証](pgvector-filtered-recall.md) を参照。

## 公式資料

- [pgvector: Loading / Indexing / Filtering / Indexing vectors with different dimensions](https://github.com/pgvector/pgvector)
- [pgvector-go: pgx integration and COPY](https://github.com/pgvector/pgvector-go)
- [pgx: CopyFrom / CopyFromFunc](https://pkg.go.dev/github.com/jackc/pgx/v5#CopyFromFunc)
- [PostgreSQL: ANALYZE](https://www.postgresql.org/docs/current/sql-analyze.html)
- [PostgreSQL: CREATE INDEX](https://www.postgresql.org/docs/current/sql-createindex.html)
- [PostgreSQL: Table Partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html)
