package knowledge

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// 実 DB に対してガードが実際に発火するかを確かめる。
//
// ガードを書いたことと、ガードが働くことは別になる。
// 落ちるべき状況を作って、落ちることを確認する。
//
//	PGVECTOR_TEST_DSN=postgresql://... go test -run Integration ./internal/genkitagent/knowledge/
const dim = 128

func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()

	dsn := os.Getenv("PGVECTOR_TEST_DSN")
	if dsn == "" {
		if os.Getenv("PGVECTOR_TEST_REQUIRE") == "1" {
			t.Fatal("PGVECTOR_TEST_DSN が未設定。PGVECTOR_TEST_REQUIRE=1 のため skip せず失敗させた")
		}
		t.Skip("PGVECTOR_TEST_DSN が未設定のため skip。CI では PGVECTOR_TEST_REQUIRE=1 を立てる")
	}

	// NewPool を使う。素の pgxpool.New だと vector 型が登録されず、
	// 投入時に次元と無関係なエラーになる。
	pool, err := NewPool(context.Background(), dsn)
	if err != nil {
		t.Fatalf("接続に失敗: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// fakeEmbedder は決め打ちのベクトルを返す。埋め込みモデルを呼ばずに検索の挙動を試す。
type fakeEmbedder struct{ seed int }

func (f fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return vectorFor(len(text) + f.seed), nil
}

// vectorFor は単位球上の点を決め打ちで作る。同じ i なら同じベクトルになる。
//
// 分布の作り方で 2 回失敗した。
//
//	sin だけの列    点が滑らかな曲線上に並び、近傍が 1 か所に固まる。
//	               probes=1 でも再現率 1.000 になり、探索の広さが結果に効かない
//	一様乱数        128 次元ではすべての点がほぼ等距離になる（次元の呪い）。
//	               上位 5 件が僅差で入れ替わり、全リストを走査しても再現率が 0.67 止まり
//
// 実際の埋め込みはクラスタを作る。意味の近い文書が空間の一角に集まり、
// 遠い話題は離れる。クラスタ中心の周りに小さな揺らぎを足して、その形にする。
const clusters = 64

func vectorFor(i int) []float32 {
	rng := func(seed uint64) func() float64 {
		st := seed*0x9E3779B97F4A7C15 + 0x2545F4914F6CDD1D
		return func() float64 {
			st ^= st << 13
			st ^= st >> 7
			st ^= st << 17
			return float64(st>>11)/float64(1<<53)*2 - 1
		}
	}

	center := rng(uint64(i % clusters))
	jitter := rng(uint64(i) + 1e6)

	v := make([]float32, dim)
	var norm float64
	for d := 0; d < dim; d++ {
		// 中心が支配的で、揺らぎは小さい。同じクラスタの点同士が近くなる。
		x := center() + jitter()*0.15
		v[d] = float32(x)
		norm += x * x
	}
	norm = math.Sqrt(norm)
	for d := range v {
		v[d] /= float32(norm)
	}
	return v
}

// setup はテスト用のテーブルを作る。rows 件を tenants 個のテナントに配る。
func setup(t *testing.T, pool *pgxpool.Pool, table string, rows, tenants int) {
	t.Helper()
	ctx := context.Background()

	exec := func(sql string, args ...any) {
		t.Helper()
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatalf("%s: %v", strings.SplitN(sql, "\n", 2)[0], err)
		}
	}

	exec(fmt.Sprintf("DROP TABLE IF EXISTS %s CASCADE", table))
	exec(fmt.Sprintf(`CREATE TABLE %s (
		id int PRIMARY KEY, tenant_id text NOT NULL,
		title text NOT NULL, content text NOT NULL, embedding vector(%d) NOT NULL)`, table, dim))
	t.Cleanup(func() {
		pool.Exec(context.Background(), fmt.Sprintf("DROP TABLE IF EXISTS %s CASCADE", table))
	})

	if rows == 0 {
		return
	}
	insertRows(t, pool, table, 0, rows, tenants)
	exec(fmt.Sprintf("ANALYZE %s", table))
}

// insertRows は CopyFrom で一括投入する。1 行ずつの INSERT だと
// 数万行の準備に時間がかかり、テストが実行されなくなる。
func insertRows(t *testing.T, pool *pgxpool.Pool, table string, from, to, tenants int) {
	t.Helper()

	src := make([][]any, 0, to-from)
	for i := from; i < to; i++ {
		src = append(src, []any{
			// テナントはクラスタと独立に割り当てる。i%tenants にすると
			// クラスタ数との公約数でテナントが特定のクラスタに偏り、
			// 絞り込んだ先に近傍が 1 つも無い状況が生まれる。
			i, fmt.Sprintf("tenant-%02d", (i/clusters)%tenants),
			fmt.Sprintf("題名 %d", i), fmt.Sprintf("本文 %d", i), toVector(vectorFor(i)),
		})
	}
	_, err := pool.CopyFrom(context.Background(), pgx.Identifier{table},
		[]string{"id", "tenant_id", "title", "content", "embedding"},
		pgx.CopyFromRows(src))
	if err != nil {
		t.Fatalf("投入に失敗: %v", err)
	}
}

func baseCfg(table string) PgVectorConfig {
	return PgVectorConfig{Table: table, EmbeddingColumn: "embedding",
		TitleColumn: "title", ContentColumn: "content"}
}

func TestIntegrationVerifyPassesOnCorrectSetup(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_ok"
	setup(t, pool, table, 3000, 1)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	s, err := NewPgVector(ctx, pool, fakeEmbedder{}, baseCfg(table))
	if err != nil {
		t.Fatalf("正しい構成なのに Verify が落ちた: %v", err)
	}

	docs, err := s.Search(ctx, "問い合わせ", 5)
	if err != nil {
		t.Fatalf("検索に失敗: %v", err)
	}
	if len(docs) != 5 {
		t.Errorf("%d 件, 期待 5 件", len(docs))
	}
}

// opclass が合わないと Seq Scan に落ちる。結果は正しいので実行時には気づけない。
func TestIntegrationVerifyCatchesWrongOpclass(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_l2"
	setup(t, pool, table, 3000, 1)

	// L2 用の索引を作る。検索はコサインで投げるため対応しない。
	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_l2_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	_, err := NewPgVector(ctx, pool, fakeEmbedder{}, baseCfg(table))
	if err == nil {
		t.Fatal("opclass が合わないのに Verify が通ってしまった")
	}
	if !strings.Contains(err.Error(), "vector_cosine_ops") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// 索引そのものが無い場合。小規模なら動くが、規模が増えると遅くなる。
func TestIntegrationVerifyCatchesMissingIndex(t *testing.T) {
	pool := testPool(t)
	const table = "kb_noidx"
	setup(t, pool, table, 3000, 1)

	_, err := NewPgVector(context.Background(), pool, fakeEmbedder{}, baseCfg(table))
	if err == nil {
		t.Fatal("索引が無いのに Verify が通ってしまった")
	}
	if !strings.Contains(err.Error(), "索引が使えない") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// 再現率の測定が方式の違いを捉えられることを確かめる。
//
// 当初は「IVFFlat を空テーブルに作ると再現率 0.000 になる」という前提で
// テストを書いたが、測り直すと再現しなかった。作成順による差は無く、
// 0.26〜0.32 は lists=40 を probes=1 で引いたときの IVFFlat の通常の値だった。
// 空テーブルでも CREATE INDEX が成功する点は事実だが、壊れた索引はできない。
//
// そこで測るのは「再現率の測定そのものが働くか」に変えた。
// 探索を狭めれば再現率が下がることを確かめる。下がらないなら測定が機能していない。
func TestIntegrationRecallMeasurementReflectsProbes(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_recall"
	setup(t, pool, table, 40000, 1)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200)",
		table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	measure := func(probes int) float64 {
		t.Helper()
		cfg := baseCfg(table)
		cfg.IvfflatProbes = probes
		cfg.MinRecall = 0.0001 // 測るだけで落とさない
		s, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
		if err != nil {
			t.Fatalf("probes=%d で Verify に失敗: %v", probes, err)
		}
		return s.MeasuredRecall
	}

	narrow := measure(1)
	wide := measure(200)
	t.Logf("probes=1 の再現率 %.3f / probes=200 の再現率 %.3f", narrow, wide)

	// 確かめたいのは「測定が索引の設定を反映するか」になる。
	// 絶対値は合成データの同点の出方に左右されるため主張しない。
	// 実際、一様乱数では全リスト走査でも 0.67 にしかならなかった。
	if wide <= narrow*1.5 {
		t.Errorf("探索を広げても再現率が十分に上がらない (probes=1: %.3f, probes=200: %.3f)。"+
			"測定が索引の設定を反映していない", narrow, wide)
	}
}

// 再現率が閾値を下回れば起動時に落ちることを確かめる。
func TestIntegrationLowRecallIsRejected(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_lowrecall"
	setup(t, pool, table, 40000, 1)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200)",
		table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}
	cfg := baseCfg(table)
	cfg.IvfflatProbes = 1
	cfg.MinRecall = 0.99 // probes=1 では届かない水準を求める

	_, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
	if err == nil {
		t.Fatal("再現率が閾値を下回るのに Verify が通ってしまった")
	}
	if !strings.Contains(err.Error(), "再現率") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// 選択率の設定が実測より大きいと ef_search が足りず取りこぼす。
func TestIntegrationVerifyCatchesOverstatedSelectivity(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_sel"
	// 1 テナントが 5% にあたる規模にする。行数が少ないと絞り込んだ結果が小さく、
	// Seq Scan のほうが速いためプランナが索引を選ばない。
	setup(t, pool, table, 40000, 20)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	cfg := baseCfg(table)
	cfg.FilterColumn = "tenant_id"
	cfg.FilterValue = "tenant-00"
	cfg.Selectivity = 0.5 // 実測は 0.05。10 倍の過大申告

	_, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
	if err == nil {
		t.Fatal("選択率が過大なのに Verify が通ってしまった")
	}
	if !strings.Contains(err.Error(), "選択率") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// 選択率を実測に任せると、絞り込みがあっても取りこぼさない。
// これが無いと ef_search=40 のまま 5% を引いて 2 件程度しか返らない。
func TestIntegrationMeasuredSelectivityPreventsShortfall(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_auto"
	setup(t, pool, table, 40000, 20)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	cfg := baseCfg(table)
	cfg.FilterColumn = "tenant_id"
	cfg.FilterValue = "tenant-00"
	cfg.Selectivity = 0 // 実測に任せる

	s, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
	if err != nil {
		t.Fatalf("Verify に失敗: %v", err)
	}
	if s.cfg.Selectivity > 0.1 {
		t.Errorf("実測した選択率 %.4f が想定より大きい（20 テナントなので 0.05 前後）", s.cfg.Selectivity)
	}

	docs, err := s.Search(ctx, "問い合わせ", 10)
	if err != nil {
		t.Fatalf("検索に失敗: %v", err)
	}
	if len(docs) != 10 {
		t.Errorf("%d 件, 期待 10 件。ef_search=%d では足りていない",
			len(docs), s.EfSearch(10))
	}
}

// ef_search を既定に固定すると取りこぼし、それを検出できることを確かめる。
// ガードが無ければ件数の少ない結果が黙って返る。
func TestIntegrationShortfallIsDetected(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_short"
	setup(t, pool, table, 40000, 20)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	cfg := baseCfg(table)
	cfg.FilterColumn = "tenant_id"
	cfg.FilterValue = "tenant-00"
	cfg.Selectivity = 0
	cfg.EfSearchCap = 40 // 既定に固定して取りこぼす状況を作る

	s, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
	if err != nil {
		t.Fatalf("Verify に失敗: %v", err)
	}

	docs, err := s.Search(ctx, "問い合わせ", 10)
	if err == nil {
		t.Fatalf("取りこぼしたのにエラーにならなかった（%d 件返った）", len(docs))
	}
	if !strings.Contains(err.Error(), "取りこぼした") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// 拡張のバージョンが要求に満たなければ起動時に落とす。
//
// 古い版では索引の種類や関数が足りず、CREATE INDEX が構文エラーになる。
// 原因が索引の定義側に見えるため、バージョンとして先に出す。
func TestIntegrationVerifyChecksExtensionVersion(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_extver"
	setup(t, pool, table, 500, 1)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	cfg := baseCfg(table)
	cfg.MinExtensionVersion = "0.1.0"
	s, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg)
	if err != nil {
		t.Fatalf("満たしているのに落ちた: %v", err)
	}
	if s.ExtensionVersion == "" {
		t.Error("実際のバージョンを記録していない")
	}
	t.Logf("vector %s / vector(%d)", s.ExtensionVersion, s.Dimension)

	cfg.MinExtensionVersion = "99.0.0"
	if _, err := NewPgVector(ctx, pool, fakeEmbedder{}, cfg); err == nil {
		t.Fatal("満たさないバージョンを通した")
	} else if !strings.Contains(err.Error(), "ALTER EXTENSION") {
		t.Errorf("対処が示されない: %v", err)
	}
}

// 埋め込み器の次元と列の次元が食い違えば起動時に落とす。
//
// モデルを差し替えて次元が変わると、投入も検索も
// different vector dimensions で落ちる。落ちる場所がリクエスト単位になる。
func TestIntegrationVerifyCatchesDimensionMismatch(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	const table = "kb_dim"
	setup(t, pool, table, 500, 1)

	if _, err := pool.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX ON %s USING hnsw (embedding vector_cosine_ops)", table)); err != nil {
		t.Fatalf("索引作成: %v", err)
	}

	if _, err := NewPgVector(ctx, pool, fakeEmbedder{}, baseCfg(table)); err != nil {
		t.Fatalf("次元が合っているのに落ちた: %v", err)
	}

	_, err := NewPgVector(ctx, pool, shortEmbedder{}, baseCfg(table))
	if err == nil {
		t.Fatal("次元違いの埋め込み器を通した")
	}
	if !strings.Contains(err.Error(), "作り直す") {
		t.Errorf("対処が示されない: %v", err)
	}
	t.Logf("検出: %v", err)
}

// shortEmbedder はモデルを差し替えて次元が減った状況を作る。
type shortEmbedder struct{}

func (shortEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, dim/2), nil
}
