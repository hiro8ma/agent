// Package identity は呼び出し元の利用者を context で運ぶ。
//
// 利用者 ID は API の境界（インターセプタ）でだけ context に入れる。
// ハンドラやリポジトリはリクエスト本文の値ではなく、ここから取り出した値で所有者を確かめる。
package identity

import (
	"context"
	"errors"
	"regexp"
)

// ErrUnauthenticated は context に利用者が無いことを示す。
var ErrUnauthenticated = errors.New("identity: 利用者が特定できない")

// UserID は検証済みの利用者 ID。
type UserID string

var userIDPattern = regexp.MustCompile(`^[A-Za-z0-9._@-]{1,128}$`)

// ParseUserID は利用者 ID の形を検証する。
func ParseUserID(s string) (UserID, error) {
	if !userIDPattern.MatchString(s) {
		return "", errors.New("identity: 利用者 ID は英数字と ._@- で 128 文字まで")
	}
	return UserID(s), nil
}

type ctxKey struct{}

// With は利用者を context に入れる。
func With(ctx context.Context, id UserID) context.Context {
	return context.WithValue(ctx, ctxKey{}, id)
}

// From は context から利用者を取り出す。無ければ ErrUnauthenticated を返す。
func From(ctx context.Context) (UserID, error) {
	id, ok := ctx.Value(ctxKey{}).(UserID)
	if !ok || id == "" {
		return "", ErrUnauthenticated
	}
	return id, nil
}
