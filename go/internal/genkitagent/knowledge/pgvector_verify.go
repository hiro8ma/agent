package knowledge

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// Verify は索引の整合を起動時に確かめる。
//
// pgvector で結果が劣化する原因は、どれも実行時にエラーを出さない。
// 「検索の精度が出ない」という形でしか現れないため、リクエスト単位で追うと高くつく。
// 起動時に落として、原因を 1 行で示す。
func (s *PgVector) Verify(ctx context.Context) error {
	for _, check := range []func(context.Context) error{
		s.verifyVectorTypeRegistered,
		s.verifyIndexUsed,
		s.verifyIndexRecall,
		s.verifySelectivity,
	} {
		if err := check(ctx); err != nil {
			return err
		}
	}
	return nil
}

// verifyVectorTypeRegistered は接続で vector 型が登録されていることを確かめる。
//
// 登録を忘れると、ベクトルが誤った形式で送られる。症状は
// 「16000 次元を超える」のような実際の次元と無関係なエラーになり、
// 原因に行き着くまで遠回りする。NewPool を使えば起きないが、
// 既存のプールを渡された場合はここで検出する。
func (s *PgVector) verifyVectorTypeRegistered(ctx context.Context) error {
	probe := toVector([]float32{1, 0, 0})
	var got pgvector.Vector
	if err := s.pool.QueryRow(ctx, "SELECT $1::vector(3)", probe).Scan(&got); err != nil {
		return fmt.Errorf(
			"pgvector: verify: 接続で vector 型が登録されていない。"+
				"knowledge.NewPool を使うか、pgxpool.Config.AfterConnect で "+
				"pgvector-go/pgx.RegisterTypes を呼ぶ: %w", err)
	}
	out := got.Slice()
	if len(out) != 3 || out[0] != 1 {
		return fmt.Errorf("pgvector: verify: ベクトルの往復が壊れている: %v", out)
	}
	return nil
}

// verifyIndexUsed は索引がこのクエリを処理できることを確かめる。
//
// 索引の定義を読んで opclass を照合するのではなく、投げるクエリそのもので確かめる。
// opclass の対応表を自前で持つと、pgvector 側の追加に追従できない。
//
// 判定は enable_seqscan = off の下で行う。区別したいのは 2 つの状態になる。
//
//	索引が使えない      opclass が対応しない。強制しても Index Scan にならない
//	索引を使う価値がない 行数が少ない、絞り込みで対象が小さい。強制すれば使える
//
// 後者は Seq Scan が正しい選択なので落とす理由が無い。
// 強制せずに判定すると、小規模なテーブルや絞り込みの強い構成で誤検知する。
func (s *PgVector) verifyIndexUsed(ctx context.Context) error {
	zero := make([]float32, 0)
	dim, err := s.dimension(ctx)
	if err != nil {
		return err
	}
	zero = make([]float32, dim)
	zero[0] = 1 // 全要素 0 のベクトルはコサイン距離が未定義になる

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("pgvector: verify: 接続の取得: %w", err)
	}
	defer conn.Release()

	// 強制はこの接続のセッションに閉じる。プールに戻る前に元へ戻す。
	if _, err := conn.Exec(ctx, "SET enable_seqscan = off"); err != nil {
		return fmt.Errorf("pgvector: verify: enable_seqscan の設定: %w", err)
	}
	defer conn.Exec(ctx, "RESET enable_seqscan")

	sql, args := s.buildQuery(toVector(zero), 1)
	rows, err := conn.Query(ctx, "EXPLAIN "+sql, args...)
	if err != nil {
		return fmt.Errorf("pgvector: verify: EXPLAIN に失敗: %w", err)
	}
	defer rows.Close()

	var plan strings.Builder
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			return fmt.Errorf("pgvector: verify: EXPLAIN の読み取り: %w", err)
		}
		plan.WriteString(line)
		plan.WriteString("\n")
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("pgvector: verify: EXPLAIN: %w", err)
	}

	if !strings.Contains(plan.String(), "Index Scan") {
		return fmt.Errorf(
			"pgvector: verify: %s.%s に対する検索で索引が使えない"+
				"（enable_seqscan = off でも Index Scan にならない）。"+
				"コサイン距離 <=> を使うため索引は vector_cosine_ops で作る必要がある。"+
				"opclass が違うと結果は正しいまま Seq Scan に落ちる\n実行計画:\n%s",
			s.cfg.Table, s.cfg.EmbeddingColumn, plan.String())
	}
	return nil
}

// verifyIndexRecall は索引が厳密検索とどれだけ一致するかを起動時に測る。
//
// 索引が「存在する」ことと「使い物になる」ことは別になる。
// ただし近似索引の再現率は方式と設定で大きく変わる。
// IVFFlat を probes=1 で引けば再現率 0.3 前後になるのが普通で、
// これは壊れているのではなく設定どおりの動作にあたる。
// そのため既定の閾値は「明らかに何も返っていない」水準まで下げてある。
// 実際の値は MeasuredRecall に残るので、運用では起動ログに出して推移を見る。
//
// 判定に使うクエリは表に存在しないベクトルにする。
// 表にある行をそのままクエリにすると、その行は自分が属するリストに必ず含まれるため、
// 索引の質と関係なく見つかってしまう。実際にその形で書いて何も検出できなかった。
func (s *PgVector) verifyIndexRecall(ctx context.Context) error {
	const (
		probes = 3 // クエリの本数
		topK   = 5 // 1 クエリで比べる件数
	)
	threshold := s.cfg.MinRecall
	if threshold <= 0 {
		threshold = 0.05
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("pgvector: verify: 接続の取得: %w", err)
	}
	defer conn.Release()

	queries, err := s.sampleQueries(ctx, conn, probes)
	if err != nil || len(queries) == 0 {
		return err // 行が足りなければ確かめようが無い
	}

	var total float64
	for _, q := range queries {
		exact, err := s.topIDs(ctx, conn, q, topK, false)
		if err != nil {
			return err
		}
		if len(exact) == 0 {
			return nil
		}
		indexed, err := s.topIDs(ctx, conn, q, topK, true)
		if err != nil {
			return err
		}

		hit := 0
		for _, id := range indexed {
			for _, e := range exact {
				if id == e {
					hit++
					break
				}
			}
		}
		total += float64(hit) / float64(len(exact))
	}

	recall := total / float64(len(queries))
	s.MeasuredRecall = recall
	if recall < threshold {
		return fmt.Errorf(
			"pgvector: verify: %s の索引が厳密検索とほとんど一致しない（再現率 %.2f、閾値 %.2f）。"+
				"索引が壊れているか、探索の設定が狭すぎる。"+
				"IVFFlat なら probes、HNSW なら ef_search を上げるか、REINDEX で作り直す",
			s.cfg.Table, recall, threshold)
	}
	return nil
}

// sampleQueries は表に存在しないクエリベクトルを作る。
//
// 2 行の平均を取る。実際の分布の内側にありながら、どの行とも一致しない点になる。
// 乱数で作ると分布から外れ、索引の善し悪しと関係なく再現率が下がる。
func (s *PgVector) sampleQueries(ctx context.Context, conn *pgxpool.Conn, n int) ([]pgvector.Vector, error) {
	rows, err := conn.Query(ctx, fmt.Sprintf(
		"SELECT %s FROM %s LIMIT $1", s.cfg.EmbeddingColumn, s.cfg.Table), n*2)
	if err != nil {
		return nil, fmt.Errorf("pgvector: verify: 標本の取得: %w", err)
	}
	defer rows.Close()

	var vecs [][]float32
	for rows.Next() {
		var v pgvector.Vector
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("pgvector: verify: 標本の読み取り: %w", err)
		}
		vecs = append(vecs, v.Slice())
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: verify: 標本: %w", err)
	}
	if len(vecs) < 2 {
		return nil, nil
	}

	var out []pgvector.Vector
	for i := 0; i+1 < len(vecs); i += 2 {
		mid := make([]float32, len(vecs[i]))
		for d := range mid {
			mid[d] = (vecs[i][d] + vecs[i+1][d]) / 2
		}
		out = append(out, toVector(mid))
	}
	return out, nil
}

// topIDs は上位 topK の ctid を返す。useIndex=false なら索引を切って厳密に引く。
//
// 主キーの名前に依存しないよう ctid を使う。行の同一性が分かればよい。
func (s *PgVector) topIDs(ctx context.Context, conn *pgxpool.Conn, q pgvector.Vector,
	topK int, useIndex bool) ([]string, error) {

	setting := "SET enable_indexscan = off"
	if useIndex {
		setting = "SET enable_seqscan = off"
		// 本番と同じ探索設定で測る。設定が違えば、起動時に測った再現率は
		// 実際のリクエストで得られる値と関係の無い数字になる。
		if err := s.applySessionSettings(ctx, conn.Conn(), s.EfSearch(topK)); err != nil {
			return nil, err
		}
	}
	if _, err := conn.Exec(ctx, setting); err != nil {
		return nil, fmt.Errorf("pgvector: verify: プランの強制: %w", err)
	}
	defer conn.Exec(ctx, "RESET enable_indexscan")
	defer conn.Exec(ctx, "RESET enable_seqscan")

	rows, err := conn.Query(ctx, fmt.Sprintf(
		"SELECT ctid::text FROM %s ORDER BY %s <=> $1 LIMIT $2",
		s.cfg.Table, s.cfg.EmbeddingColumn), q, topK)
	if err != nil {
		return nil, fmt.Errorf("pgvector: verify: 再現率の測定: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("pgvector: verify: ctid の読み取り: %w", err)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// verifySelectivity は設定された選択率が実測と合っているかを確かめる。
//
// 選択率から ef_search を決めるため、この値がずれると取りこぼす。
// 設定が無ければ実測値を採用する。
func (s *PgVector) verifySelectivity(ctx context.Context) error {
	if s.cfg.FilterColumn == "" {
		s.cfg.Selectivity = 1
		return nil
	}

	var total, matching int64
	if err := s.pool.QueryRow(ctx,
		fmt.Sprintf("SELECT count(*) FROM %s", s.cfg.Table)).Scan(&total); err != nil {
		return fmt.Errorf("pgvector: verify: 全行数の取得: %w", err)
	}
	if total == 0 {
		return nil
	}
	if err := s.pool.QueryRow(ctx,
		fmt.Sprintf("SELECT count(*) FROM %s WHERE %s = $1", s.cfg.Table, s.cfg.FilterColumn),
		s.cfg.FilterValue).Scan(&matching); err != nil {
		return fmt.Errorf("pgvector: verify: 一致行数の取得: %w", err)
	}

	measured := float64(matching) / float64(total)
	if s.cfg.Selectivity <= 0 {
		s.cfg.Selectivity = measured
		return nil
	}

	// 設定値が実測より大きいと ef_search が足りず取りこぼす。
	// 小さいぶんには安全側なので、過大な場合だけ落とす。
	if s.cfg.Selectivity > measured*1.5 {
		return fmt.Errorf(
			"pgvector: verify: 選択率の設定 %.4f が実測 %.4f より大きい"+
				"（%s = %s は %d / %d 行）。"+
				"ef_search が足りず検索が取りこぼす。設定を下げるか 0 にして実測値を使う",
			s.cfg.Selectivity, measured, s.cfg.FilterColumn, s.cfg.FilterValue, matching, total)
	}
	return nil
}

func (s *PgVector) dimension(ctx context.Context) (int, error) {
	var mod int
	err := s.pool.QueryRow(ctx, `
		SELECT a.atttypmod
		FROM pg_attribute a
		JOIN pg_class c ON c.oid = a.attrelid
		WHERE c.relname = $1 AND a.attname = $2`,
		s.cfg.Table, s.cfg.EmbeddingColumn).Scan(&mod)
	if err != nil {
		return 0, fmt.Errorf("pgvector: verify: %s.%s の次元取得に失敗: %w",
			s.cfg.Table, s.cfg.EmbeddingColumn, err)
	}
	if mod <= 0 {
		return 0, fmt.Errorf(
			"pgvector: verify: %s.%s に次元が宣言されていない。"+
				"vector(n) で宣言しないと索引が張れず、次元違いの投入も検出できない",
			s.cfg.Table, s.cfg.EmbeddingColumn)
	}
	return mod, nil
}
