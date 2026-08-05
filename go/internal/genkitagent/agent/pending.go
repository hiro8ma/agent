package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"

	"github.com/hiro8ma/agent/go/internal/agentcore"
)

// ErrPendingNotFound は agentcore の sentinel を共有する。transport がこのエラーで NotFound を返す。
var ErrPendingNotFound = agentcore.ErrPendingNotFound

// PendingStore は承認待ちツール呼び出しの永続化。
// Take は取得と同時に削除し、同じ承認の二重実行を防ぐ。
type PendingStore interface {
	Save(ctx context.Context, p PendingToolCall) error
	Take(ctx context.Context, id string) (*PendingToolCall, error)
}

func newToolCallID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err) // crypto/rand の失敗は環境異常
	}
	return hex.EncodeToString(b)
}
