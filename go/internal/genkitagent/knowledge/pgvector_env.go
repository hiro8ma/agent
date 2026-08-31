package knowledge

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// verifyExtension は vector 拡張の有無とバージョンを確かめる。
//
// 拡張が無くても接続は成立し、最初のクエリまで落ちない。
// 起動時に見ないと、原因を追う場所が検索の実装側になる。
func (s *PgVector) verifyExtension(ctx context.Context) error {
	var installed, available *string
	err := s.pool.QueryRow(ctx, `
		SELECT
			(SELECT extversion FROM pg_extension WHERE extname = 'vector'),
			(SELECT default_version FROM pg_available_extensions WHERE name = 'vector')`,
	).Scan(&installed, &available)
	if err != nil {
		return fmt.Errorf("pgvector: verify: 拡張の確認に失敗: %w", err)
	}

	if installed == nil {
		if available == nil {
			return fmt.Errorf(
				"pgvector: verify: vector 拡張がこのサーバーに存在しない。" +
					"イメージが pgvector 入りか、マネージドサービスが対応しているかを確かめる")
		}
		var superuser bool
		_ = s.pool.QueryRow(ctx,
			`SELECT rolsuper FROM pg_roles WHERE rolname = current_user`).Scan(&superuser)
		hint := "CREATE EXTENSION vector を実行する"
		if !superuser {
			hint = "接続ユーザーに拡張作成の権限が無い。マスターユーザーで CREATE EXTENSION vector を実行する"
		}
		return fmt.Errorf(
			"pgvector: verify: vector 拡張が未作成（利用可能な版 %s）。%s", *available, hint)
	}

	s.ExtensionVersion = *installed
	if s.cfg.MinExtensionVersion == "" {
		return nil
	}
	ok, err := versionAtLeast(*installed, s.cfg.MinExtensionVersion)
	if err != nil {
		return fmt.Errorf("pgvector: verify: バージョンの比較に失敗: %w", err)
	}
	if !ok {
		return fmt.Errorf(
			"pgvector: verify: vector %s は要求する %s に満たない。"+
				"ALTER EXTENSION vector UPDATE を実行する（利用可能な版 %s）",
			*installed, s.cfg.MinExtensionVersion, deref(available))
	}
	return nil
}

// verifyEmbedderDimension は埋め込み器の出力次元と列の次元を照合する。
//
// 埋め込みモデルを差し替えて次元が変わると、投入も検索も
// different vector dimensions で落ちる。落ちる場所がリクエスト単位になる。
func (s *PgVector) verifyEmbedderDimension(ctx context.Context) error {
	column, err := s.dimension(ctx)
	if err != nil {
		return err
	}
	vec, err := s.embedder.Embed(ctx, "次元の確認")
	if err != nil {
		return fmt.Errorf("pgvector: verify: 埋め込みの取得に失敗: %w", err)
	}
	if len(vec) != column {
		return fmt.Errorf(
			"pgvector: verify: 埋め込み器は %d 次元を返すが %s.%s は vector(%d)。"+
				"モデルを差し替えたなら列と索引を作り直す",
			len(vec), s.cfg.Table, s.cfg.EmbeddingColumn, column)
	}
	s.Dimension = column
	return nil
}

// versionAtLeast は got が want 以上かを返す。
func versionAtLeast(got, want string) (bool, error) {
	g, err := parseVersion(got)
	if err != nil {
		return false, err
	}
	w, err := parseVersion(want)
	if err != nil {
		return false, err
	}
	for i := range g {
		switch {
		case g[i] > w[i]:
			return true, nil
		case g[i] < w[i]:
			return false, nil
		}
	}
	return true, nil
}

// parseVersion は 0.8.6 を [0 8 6] にする。欠けた桁は 0 で埋める。
func parseVersion(v string) ([3]int, error) {
	var out [3]int
	parts := strings.SplitN(strings.TrimSpace(v), ".", 4)
	for i := 0; i < len(parts) && i < 3; i++ {
		n, err := strconv.Atoi(strings.TrimFunc(parts[i], func(r rune) bool {
			return r < '0' || r > '9'
		}))
		if err != nil {
			return out, fmt.Errorf("バージョン %q を解釈できない", v)
		}
		out[i] = n
	}
	return out, nil
}

func deref(s *string) string {
	if s == nil {
		return "不明"
	}
	return *s
}
