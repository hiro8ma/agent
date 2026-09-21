package libconnect

import (
	"errors"

	"connectrpc.com/connect"

	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

var codes = map[liberrors.Code]connect.Code{
	liberrors.CodeInvalidArgument: connect.CodeInvalidArgument,
	liberrors.CodeUnauthenticated: connect.CodeUnauthenticated,
	liberrors.CodePermissionDeny:  connect.CodePermissionDenied,
	liberrors.CodeNotFound:        connect.CodeNotFound,
	liberrors.CodeUnavailable:     connect.CodeUnavailable,
	liberrors.CodeInternal:        connect.CodeInternal,
	liberrors.CodeFailedPrecond:   connect.CodeFailedPrecondition,
}

// Error は内部のエラーを Connect のエラーに変換する。ハンドラはここを通して返す。
// Code の無いエラーは Internal にし、内部の文言を外に出さない。
func Error(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := errors.AsType[*connect.Error](err); ok {
		return err
	}
	if errors.Is(err, identity.ErrUnauthenticated) {
		return connect.NewError(connect.CodeUnauthenticated, err)
	}
	if e, ok := errors.AsType[*liberrors.Error](err); ok {
		if code, ok := codes[e.Code]; ok {
			return connect.NewError(code, errors.New(e.Msg))
		}
	}
	return connect.NewError(connect.CodeInternal, errors.New("内部エラー"))
}

// FromConnect は下流のサービスから返った Connect のエラーを内部のエラーに戻す。
// 呼び出し側は liberrors の Code で分岐でき、Connect に依存しない。
func FromConnect(err error, msg string) error {
	ce, ok := errors.AsType[*connect.Error](err)
	if !ok {
		return liberrors.Wrap(liberrors.CodeUnavailable, err, "%s", msg)
	}
	for lc, cc := range codes {
		if cc == ce.Code() {
			return liberrors.Wrap(lc, err, "%s", msg)
		}
	}
	return liberrors.Wrap(liberrors.CodeInternal, err, "%s", msg)
}
