// Package libconnect は Connect RPC のサービス間で共有するインターセプタとエラー変換。
package libconnect

import (
	"context"
	"net/http"

	"connectrpc.com/connect"

	"github.com/hiro8ma/agent/go/internal/lib/identity"
)

// UserHeader は利用者 ID を運ぶヘッダ。
//
// ローカル検証用の口で、値を信じる。外に公開するときは、ID トークンを検証して
// 利用者を取り出す Authenticator に差し替える。
const UserHeader = "X-User-Id"

// Authenticator はリクエストヘッダから利用者を特定する。
type Authenticator func(ctx context.Context, h http.Header) (identity.UserID, error)

// HeaderAuthenticator は UserHeader の値をそのまま利用者 ID として受け取る。
func HeaderAuthenticator(_ context.Context, h http.Header) (identity.UserID, error) {
	v := h.Get(UserHeader)
	if v == "" {
		return "", identity.ErrUnauthenticated
	}
	return identity.ParseUserID(v)
}

// ServerIdentity は利用者を特定して context に入れる。特定できなければ Unauthenticated で止める。
// public に挙げた手続き（一覧取得など）は利用者なしで通す。
func ServerIdentity(auth Authenticator, public ...string) connect.Interceptor {
	open := make(map[string]bool, len(public))
	for _, p := range public {
		open[p] = true
	}
	authenticate := func(ctx context.Context, procedure string, h http.Header) (context.Context, error) {
		id, err := auth(ctx, h)
		if err == nil {
			return identity.With(ctx, id), nil
		}
		if open[procedure] {
			return ctx, nil
		}
		return nil, connect.NewError(connect.CodeUnauthenticated, err)
	}
	return &serverIdentity{authenticate: authenticate}
}

type serverIdentity struct {
	authenticate func(context.Context, string, http.Header) (context.Context, error)
}

func (s *serverIdentity) WrapUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
		if req.Spec().IsClient {
			return next(ctx, req)
		}
		ctx, err := s.authenticate(ctx, req.Spec().Procedure, req.Header())
		if err != nil {
			return nil, err
		}
		return next(ctx, req)
	}
}

func (s *serverIdentity) WrapStreamingClient(next connect.StreamingClientFunc) connect.StreamingClientFunc {
	return next
}

func (s *serverIdentity) WrapStreamingHandler(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return func(ctx context.Context, conn connect.StreamingHandlerConn) error {
		ctx, err := s.authenticate(ctx, conn.Spec().Procedure, conn.RequestHeader())
		if err != nil {
			return err
		}
		return next(ctx, conn)
	}
}

// ForwardIdentity は context の利用者を下流のサービスへのヘッダに載せる。
// エージェントのサービスが会話や検索のサービスを呼ぶとき、呼び出し元の利用者のまま呼ぶ。
func ForwardIdentity() connect.Interceptor {
	return &forwardIdentity{}
}

type forwardIdentity struct{}

func (forwardIdentity) WrapUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
		if req.Spec().IsClient {
			if id, err := identity.From(ctx); err == nil {
				req.Header().Set(UserHeader, string(id))
			}
		}
		return next(ctx, req)
	}
}

func (forwardIdentity) WrapStreamingClient(next connect.StreamingClientFunc) connect.StreamingClientFunc {
	return func(ctx context.Context, spec connect.Spec) connect.StreamingClientConn {
		conn := next(ctx, spec)
		if id, err := identity.From(ctx); err == nil {
			conn.RequestHeader().Set(UserHeader, string(id))
		}
		return conn
	}
}

func (forwardIdentity) WrapStreamingHandler(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return next
}
