package knowledge

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// NewPool は pgvector の型を登録した接続プールを作る。
//
// pgx は vector 型を知らないため、接続ごとに登録が要る。登録を忘れると
// ベクトルが誤った形式で送られ、投入時に「16000 次元を超える」といった
// 実際の次元と無関係なエラーになる。原因が分かりにくいので、
// 呼び出し側に任せずここで済ませる。
//
// 既存のプールを使う場合は、pgxpool.Config.AfterConnect で同じ登録を行う。
func NewPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("pgvector: DSN の解釈: %w", err)
	}
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("pgvector: 接続プールの作成: %w", err)
	}
	return pool, nil
}
