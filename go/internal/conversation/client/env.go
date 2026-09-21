package client

import (
	"os"

	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
)

const (
	// EnvURL が設定されていれば、別プロセスの ConversationService を呼ぶ。
	EnvURL = "CONVERSATION_URL"
	// EnvDSN は EnvURL が無いときにプロセス内で使う SQLite の保管先。
	EnvDSN     = "CONVERSATION_DSN"
	DefaultDSN = "file:./.conversation/conversation.db"
)

// FromEnv は環境変数から保管先を選ぶ。どちらを選んでも所有者の確認は同じ usecase で行う。
func FromEnv() (store agentcore.SessionCreator, where string, err error) {
	if url := os.Getenv(EnvURL); url != "" {
		return NewDefaultSessionStore(url), url, nil
	}
	dsn := os.Getenv(EnvDSN)
	if dsn == "" {
		dsn = DefaultDSN
	}
	repo, err := repository.NewSQLite(dsn)
	if err != nil {
		return nil, "", err
	}
	return NewLocal(usecase.New(repo)), dsn, nil
}
