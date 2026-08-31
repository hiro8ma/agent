// Package knowledge の pgvector 実装。
//
// この実装が InMemory と違うのは、検索結果の質が静かに落ちる経路を塞いでいる点にある。
// pgvector でベクトル検索を書くと、次のいずれも「エラーにならず結果だけ劣化する」。
//
//	フィルタ付き検索       HNSW は ef_search 件の候補を集めてから WHERE を適用する。
//	                      対象が全体の一部だと候補がほぼ削られ、LIMIT 10 に対して
//	                      2 件しか返らない。実測で選択率 6.2% のとき再現率 0.20
//	ef_search の天井       LIMIT を大きく書いても ef_search 件を超えては返らない。
//	                      実測で ef_search=40 のとき LIMIT 320 に対して 43 件
//	opclass の不一致       コサイン索引に <#> を投げると Seq Scan に落ちる。
//	                      正しい結果は返るため、遅くなったことに気づけない
//	空テーブルへの IVFFlat  作成は成功するが k-means の重心が学習されず、
//	                      実測で再現率 0.000
//
// いずれも本番で顕在化するのは「検索の精度が出ない」という形になり、
// 原因の切り分けに時間がかかる。起動時に検証し、検索時に取りこぼしを検出する。
package knowledge

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"

	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
)

// Embedder はクエリ文字列をベクトルに変換する。
//
// 埋め込みモデルは検索側と投入側で一致している必要がある。
// 違うモデルで作った索引を引くと、次元が同じなら動いてしまい結果だけ無意味になる。
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// PgVector は pgvector を使う KnowledgeSearcher。
type PgVector struct {
	pool     *pgxpool.Pool
	embedder Embedder
	cfg      PgVectorConfig

	// ExtensionVersion と Dimension は起動時に確かめた実際の値。
	ExtensionVersion string
	Dimension        int

	// MeasuredRecall は起動時に測った厳密検索との一致率。
	// 索引の設定を変えたときの影響を追えるよう、起動ログに出す用途を想定する。
	MeasuredRecall float64
}

// PgVectorConfig は接続とクエリの設定。
type PgVectorConfig struct {
	// Table / EmbeddingColumn / TitleColumn / ContentColumn は対象のスキーマ。
	Table           string
	EmbeddingColumn string
	TitleColumn     string
	ContentColumn   string

	// FilterColumn と FilterValue はテナントや部署での絞り込み。空なら絞らない。
	FilterColumn string
	FilterValue  string

	// Selectivity は FilterValue が全体に占める割合の見積もり（0 < s <= 1）。
	// ef_search をここから決める。0 なら起動時に実測する。
	//
	// HNSW は ef_search 件の候補を集めてから WHERE を適用するため、
	// limit 件を返すには ef_search >= limit / Selectivity が要る。
	// 実測でこの関係が成り立つことを確かめてある。
	Selectivity float64

	// EfSearchCap は ef_search の上限。選択率が極端に低いと際限なく大きくなり、
	// 厳密検索より遅くなる。頭打ちにして、届かない分は取りこぼしとして報告する。
	EfSearchCap int

	// IvfflatProbes は IVFFlat で走査するリスト数。0 なら設定しない（既定の 1）。
	//
	// ef_search と同じくセッション単位なので、プールから借りた接続に毎回設定する。
	// ALTER DATABASE で設定しても既存の接続には届かない。
	IvfflatProbes int

	// MinExtensionVersion は要求する vector 拡張の最小バージョン。空なら確かめない。
	//
	// 索引の種類や関数はバージョンで増える。古い版では
	// CREATE INDEX が構文エラーになり、原因が索引の定義側に見える。
	MinExtensionVersion string

	// MinRecall は起動時に測る再現率の下限。0 なら 0.05 を使う。
	//
	// 近似索引の再現率は方式と設定で大きく変わる。IVFFlat を probes=1 で引けば
	// 0.3 前後になるのが普通で、これは壊れているのではなく設定どおりの動作になる。
	// そのため既定は「明らかに何も返っていない」水準まで下げてある。
	// 求める品質が決まっているなら、その値をここに置いて起動時に落とす。
	MinRecall float64
}

func (c *PgVectorConfig) applyDefaults() {
	if c.Table == "" {
		c.Table = "documents"
	}
	if c.EmbeddingColumn == "" {
		c.EmbeddingColumn = "embedding"
	}
	if c.TitleColumn == "" {
		c.TitleColumn = "title"
	}
	if c.ContentColumn == "" {
		c.ContentColumn = "content"
	}
	if c.EfSearchCap <= 0 {
		c.EfSearchCap = 1000
	}
}

var _ agent.KnowledgeSearcher = (*PgVector)(nil)

// NewPgVector は接続プールと埋め込み器から検索器を作る。
//
// 起動時に Verify を呼んで索引の整合を確かめる。ここで落としておかないと、
// 検索の精度が出ない理由をリクエスト単位で追うことになる。
func NewPgVector(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, cfg PgVectorConfig) (*PgVector, error) {
	if pool == nil {
		return nil, errors.New("pgvector: pool is required")
	}
	if embedder == nil {
		return nil, errors.New("pgvector: embedder is required")
	}
	cfg.applyDefaults()

	s := &PgVector{pool: pool, embedder: embedder, cfg: cfg}
	if err := s.Verify(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// EfSearch は limit 件を返すのに要る ef_search を返す。
//
// HNSW は ef_search 件の候補を集めてから WHERE を適用する。
// 選択率 s なら候補のうち残るのは期待値で ef_search * s 件になるため、
// limit 件を返すには ef_search >= limit / s が要る。
// 期待値ちょうどでは半分の確率で足りないので倍の余裕を持たせる。
func (s *PgVector) EfSearch(limit int) int {
	const hnswDefaultEfSearch = 40

	sel := s.cfg.Selectivity
	if sel <= 0 || sel > 1 {
		sel = 1
	}
	need := int(float64(limit)/sel) * 2
	if need < hnswDefaultEfSearch {
		need = hnswDefaultEfSearch
	}
	if need > s.cfg.EfSearchCap {
		need = s.cfg.EfSearchCap
	}
	return need
}

// Search はクエリを埋め込み、コサイン距離で近傍を引く。
//
// 取りこぼし（limit 件に届かない）を検出したらエラーにする。
// 件数が少ないだけの結果を黙って返すと、上位のエージェントは
// 「そのドキュメントは存在しない」と解釈して回答してしまう。
func (s *PgVector) Search(ctx context.Context, query string, limit int) ([]agent.KnowledgeDoc, error) {
	if strings.TrimSpace(query) == "" {
		return nil, errors.New("pgvector: query is required")
	}
	if limit <= 0 {
		return nil, fmt.Errorf("pgvector: limit must be positive, got %d", limit)
	}

	vec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("pgvector: embed query: %w", err)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("pgvector: acquire connection: %w", err)
	}
	defer conn.Release()

	// ef_search はセッション単位。プールから借りた接続に毎回設定する。
	// 使い回された接続に前回の値が残るため、既定に頼らず明示する。
	ef := s.EfSearch(limit)
	if err := s.applySessionSettings(ctx, conn.Conn(), ef); err != nil {
		return nil, err
	}

	sql, args := s.buildQuery(toVector(vec), limit)
	rows, err := conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("pgvector: query: %w", err)
	}
	defer rows.Close()

	var docs []agent.KnowledgeDoc
	for rows.Next() {
		var d agent.KnowledgeDoc
		if err := rows.Scan(&d.Title, &d.Content); err != nil {
			return nil, fmt.Errorf("pgvector: scan: %w", err)
		}
		docs = append(docs, d)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: rows: %w", err)
	}

	// 件数が足りないとき、母集団そのものが少ないのか、索引が取りこぼしたのかを分ける。
	if len(docs) < limit {
		total, err := s.matchingRows(ctx, conn)
		if err != nil {
			return nil, err
		}
		if total > len(docs) {
			return nil, fmt.Errorf(
				"pgvector: 索引が取りこぼした。%d 件要求して %d 件しか返らなかったが、"+
					"条件に一致する行は %d 件ある (ef_search=%d, 選択率の見積もり=%.4f)。"+
					"選択率の設定を下げるか EfSearchCap を上げる",
				limit, len(docs), total, ef, s.cfg.Selectivity)
		}
	}
	return docs, nil
}

// applySessionSettings は探索の広さをこの接続に設定する。
//
// セッション単位の設定なので、プールから借りた接続に毎回入れる。
// 使い回された接続に前回の値が残るため、既定に頼らず明示する。
func (s *PgVector) applySessionSettings(ctx context.Context, conn *pgx.Conn, efSearch int) error {
	if _, err := conn.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", efSearch)); err != nil {
		return fmt.Errorf("pgvector: set ef_search: %w", err)
	}
	if s.cfg.IvfflatProbes > 0 {
		if _, err := conn.Exec(ctx,
			fmt.Sprintf("SET ivfflat.probes = %d", s.cfg.IvfflatProbes)); err != nil {
			return fmt.Errorf("pgvector: set ivfflat.probes: %w", err)
		}
	}
	return nil
}

// toVector は pgvector のドライバ型に包む。テストから同じ形で呼べるようにしてある。
func toVector(v []float32) pgvector.Vector { return pgvector.NewVector(v) }

func (s *PgVector) buildQuery(vec pgvector.Vector, limit int) (string, []any) {
	var b strings.Builder
	fmt.Fprintf(&b, "SELECT %s, %s FROM %s", s.cfg.TitleColumn, s.cfg.ContentColumn, s.cfg.Table)

	args := []any{}
	if s.cfg.FilterColumn != "" {
		args = append(args, s.cfg.FilterValue)
		fmt.Fprintf(&b, " WHERE %s = $%d", s.cfg.FilterColumn, len(args))
	}

	args = append(args, vec)
	// <=> はコサイン距離。索引は vector_cosine_ops で作る必要がある。
	// 別の opclass だと Seq Scan に落ちる。Verify で確かめている。
	fmt.Fprintf(&b, " ORDER BY %s <=> $%d", s.cfg.EmbeddingColumn, len(args))

	args = append(args, limit)
	fmt.Fprintf(&b, " LIMIT $%d", len(args))
	return b.String(), args
}

func (s *PgVector) matchingRows(ctx context.Context, conn *pgxpool.Conn) (int, error) {
	sql := fmt.Sprintf("SELECT count(*) FROM %s", s.cfg.Table)
	args := []any{}
	if s.cfg.FilterColumn != "" {
		args = append(args, s.cfg.FilterValue)
		sql += fmt.Sprintf(" WHERE %s = $1", s.cfg.FilterColumn)
	}
	var n int
	if err := conn.QueryRow(ctx, sql, args...).Scan(&n); err != nil {
		return 0, fmt.Errorf("pgvector: count matching rows: %w", err)
	}
	return n, nil
}
