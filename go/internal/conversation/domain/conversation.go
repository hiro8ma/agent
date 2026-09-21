// Package domain は会話の履歴の型と、保管先に求める契約を持つ。
package domain

import (
	"context"
	"encoding/base64"
	"strconv"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Role はメッセージの話者。
type Role string

const (
	RoleUser  Role = "user"
	RoleModel Role = "model"
)

// Rating はアシスタントのメッセージへの評価。
type Rating string

const (
	RatingGood Rating = "good"
	RatingBad  Rating = "bad"
)

// Session は会話のまとまり。作った利用者だけが読み書きできる。
type Session struct {
	ID        string
	Owner     identity.UserID
	AgentID   string
	CreatedAt time.Time
}

// Authorize は呼び出し元がこのセッションの所有者かを確かめる。
// 他人のセッションは、ID を総当たりされても存在が分からないよう、無いときと同じ ErrSessionNotFound を返す。
func (s Session) Authorize(caller identity.UserID) error {
	if s.Owner != caller {
		return ErrSessionNotFound
	}
	return nil
}

// Message は会話の 1 件。Seq はセッション内の通し番号で、ページングの位置に使う。
type Message struct {
	ID        string
	SessionID string
	Seq       int64
	Role      Role
	Text      string
	CreatedAt time.Time
}

// Feedback はメッセージへの評価。同じ利用者が同じメッセージに付け直すと上書きする。
type Feedback struct {
	SessionID string
	MessageID string
	Owner     identity.UserID
	Rating    Rating
	Comment   string
}

// ErrSessionNotFound はセッションが無いことを示す。
var ErrSessionNotFound = liberrors.Newf(liberrors.CodeNotFound, "セッションが無い")

// ErrMessageNotFound はメッセージが無いことを示す。
var ErrMessageNotFound = liberrors.Newf(liberrors.CodeNotFound, "メッセージが無い")

// Repository は会話の保管先。所有者の確認は呼び出し側（usecase）が行う。
type Repository interface {
	CreateSession(ctx context.Context, s Session) error
	// GetSession は無ければ ErrSessionNotFound を返す。
	GetSession(ctx context.Context, id string) (Session, error)
	// EnsureSession は無ければ s で作り、あれば既存のものを返す。同時に呼ばれても所有者は 1 人に決まる。
	EnsureSession(ctx context.Context, s Session) (Session, error)
	// AppendMessages は Seq を採番して追記し、採番後のメッセージを返す。
	AppendMessages(ctx context.Context, sessionID string, msgs []Message) ([]Message, error)
	// ListMessages は afterSeq より後を Seq の昇順で最大 limit 件返す。
	ListMessages(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]Message, error)
	// GetMessage は無ければ ErrMessageNotFound を返す。
	GetMessage(ctx context.Context, sessionID, messageID string) (Message, error)
	SaveFeedback(ctx context.Context, f Feedback) error
}

const pageTokenPrefix = "v1:"

// EncodePageToken は次のページの位置を不透明な文字列にする。
// offset ではなく最後に返した Seq を持つので、件数が増えても位置がずれず、読み飛ばしも起きない。
func EncodePageToken(lastSeq int64) string {
	return base64.RawURLEncoding.EncodeToString([]byte(pageTokenPrefix + strconv.FormatInt(lastSeq, 10)))
}

// DecodePageToken は EncodePageToken の逆。空文字は先頭を表す。
func DecodePageToken(token string) (int64, error) {
	if token == "" {
		return 0, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		return 0, liberrors.Newf(liberrors.CodeInvalidArgument, "page_token が不正")
	}
	rest, ok := strings.CutPrefix(string(raw), pageTokenPrefix)
	if !ok {
		return 0, liberrors.Newf(liberrors.CodeInvalidArgument, "page_token が不正")
	}
	seq, err := strconv.ParseInt(rest, 10, 64)
	// 同じ位置の別表現（007 や +7）を受け付けると、トークンを鍵にしたキャッシュがぶれる。
	if err != nil || seq < 0 || EncodePageToken(seq) != token {
		return 0, liberrors.Newf(liberrors.CodeInvalidArgument, "page_token が不正")
	}
	return seq, nil
}

// PageSize は要求された件数を上限と既定値に収める。
func PageSize(requested int32) int {
	const (
		defaultSize = 50
		maxSize     = 200
	)
	switch {
	case requested <= 0:
		return defaultSize
	case requested > maxSize:
		return maxSize
	default:
		return int(requested)
	}
}

// Validate はメッセージの中身を検証する。
func (m Message) Validate() error {
	if m.Role != RoleUser && m.Role != RoleModel {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "role は user か model")
	}
	if m.Text == "" {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "text が空")
	}
	const maxRunes = 32_000
	if len([]rune(m.Text)) > maxRunes {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "text は %d 文字まで", maxRunes)
	}
	return nil
}
