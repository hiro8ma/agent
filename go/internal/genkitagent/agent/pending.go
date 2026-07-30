package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
)

// ErrPendingNotFound は承認待ちツール呼び出しが存在しない（または実行済み）ことを示す。
var ErrPendingNotFound = errors.New("pending tool call not found")

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
