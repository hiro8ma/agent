package conversationclient

import (
	"os"

	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	sessionrepo "github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
)

const (
	EnvURL     = "CONVERSATION_URL"
	EnvDSN     = "CONVERSATION_DSN"
	DefaultDSN = "file:./.conversation/conversation.db"
)

// FromEnv は CONVERSATION_URL があれば別プロセスの ConversationService を、無ければプロセス内の SQLite を使う。
func FromEnv() (store sessionrepo.SessionCreator, where string, err error) {
	if url := os.Getenv(EnvURL); url != "" {
		return NewDefaultRemote(url), url, nil
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
