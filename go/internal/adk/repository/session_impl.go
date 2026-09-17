package repository

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/glebarez/sqlite"
	"google.golang.org/adk/v2/session"
	dbsession "google.golang.org/adk/v2/session/database"
	vertexsession "google.golang.org/adk/v2/session/vertexai"

	"github.com/hiro8ma/agent/go/internal/adk/domain"
)

// 環境変数の名前。Session の保管先はこの 2 つで決まる。
// vertex を選んだ場合は、Memory Bank と同じ 3 つの環境変数を使う。
const (
	EnvSessionBackend = "ADK_SESSION_BACKEND"
	EnvSessionDSN     = "ADK_SESSION_DSN"
)

// DefaultSessionDSN は database を選んだときの既定の保管先。
//
// adk の CLI が session.db を置く場所に合わせてある。
const DefaultSessionDSN = "file:./.adk/sessions.db"

// NewInMemorySessions はプロセス内に持つ Session サービスを返す。
func NewInMemorySessions() session.Service {
	return session.InMemoryService()
}

// NewDatabaseSessions は sqlite のファイルに持つ Session サービスを返す。
//
// ドライバは純 Go の実装を使う。CGO を要求すると検証の敷居が上がる。
func NewDatabaseSessions(dsn string) (session.Service, error) {
	if dsn == "" {
		dsn = DefaultSessionDSN
	}
	if err := ensureParentDir(dsn); err != nil {
		return nil, err
	}
	svc, err := dbsession.NewSessionService(sqlite.Open(dsn))
	if err != nil {
		return nil, fmt.Errorf("session の database サービス作成: %w", err)
	}
	// 表は接続時に作られない。呼ばないと最初の Create が no such table で落ちる。
	if err := dbsession.AutoMigrate(svc); err != nil {
		return nil, fmt.Errorf("session の表の作成: %w", err)
	}
	return svc, nil
}

// NewVertexSessions は Agent Engine に持つ Session サービスを返す。
//
// 接続先が欠けている場合は、接続を試みる前に落とす。
func NewVertexSessions(ctx context.Context, at domain.BankLocation) (session.Service, error) {
	if !at.Complete() {
		return nil, fmt.Errorf("vertex session の接続先が不足している: %v", at.Missing())
	}
	svc, err := vertexsession.NewSessionService(ctx, vertexsession.VertexAIServiceConfig{
		ProjectID:       at.ProjectID,
		Location:        at.Location,
		ReasoningEngine: at.ReasoningEngine,
	})
	if err != nil {
		return nil, fmt.Errorf("vertex session サービスの作成: %w", err)
	}
	return svc, nil
}

// NewSessionsFromEnv は環境変数で選んだ Session サービスを返す。
//
// 第 2 戻り値は選ばれた種類。起動時に何で動いているかを表示するために返す。
func NewSessionsFromEnv(ctx context.Context) (session.Service, domain.SessionBackend, error) {
	backend, err := domain.ParseSessionBackend(os.Getenv(EnvSessionBackend))
	if err != nil {
		return nil, "", err
	}

	switch backend {
	case domain.SessionDatabase:
		svc, err := NewDatabaseSessions(os.Getenv(EnvSessionDSN))
		if err != nil {
			return nil, "", err
		}
		return svc, backend, nil
	case domain.SessionVertex:
		svc, err := NewVertexSessions(ctx, BankLocationFromEnv())
		if err != nil {
			return nil, "", err
		}
		return svc, backend, nil
	default:
		return NewInMemorySessions(), domain.SessionInMemory, nil
	}
}

// ensureParentDir は sqlite のファイルを置くディレクトリを用意する。
//
// gorm はディレクトリを作らないので、初回に unable to open database file で落ちる。
func ensureParentDir(dsn string) error {
	path, _ := strings.CutPrefix(dsn, "file:")
	if path == "" || path == ":memory:" {
		return nil
	}
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("session の保管先ディレクトリ作成: %w", err)
	}
	return nil
}
