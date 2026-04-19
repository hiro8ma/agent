// Package liberrors は開放レイヤーのエラーユーティリティ。
// API 境界で返すエラーを Code 付きで包み、呼び出し側で分岐しやすくする。
package liberrors

import (
	"errors"
	"fmt"
)

// Code はエラー区分。
type Code string

const (
	CodeUnknown         Code = "UNKNOWN"
	CodeInvalidArgument Code = "INVALID_ARGUMENT"
	CodeUnauthenticated Code = "UNAUTHENTICATED"
	CodePermissionDeny  Code = "PERMISSION_DENIED"
	CodeNotFound        Code = "NOT_FOUND"
	CodeInternal        Code = "INTERNAL"
	CodeUnavailable     Code = "UNAVAILABLE"
)

// Error は Code 付きエラー。
type Error struct {
	Code Code
	Msg  string
	Err  error
}

// Error は fmt.Stringer 互換。
func (e *Error) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Msg, e.Err)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Msg)
}

// Unwrap は errors.Is / errors.As で内包エラーを取り出せるようにする。
func (e *Error) Unwrap() error { return e.Err }

// Newf は Code 付きエラーを生成する。
func Newf(code Code, format string, args ...any) *Error {
	return &Error{Code: code, Msg: fmt.Sprintf(format, args...)}
}

// Wrap は既存エラーを Code 付きで包む。
func Wrap(code Code, err error, format string, args ...any) *Error {
	return &Error{Code: code, Msg: fmt.Sprintf(format, args...), Err: err}
}

// Convert は任意 error を Code 付きエラーに正規化する（既に *Error ならそのまま返す）。
func Convert(err error) *Error {
	if err == nil {
		return nil
	}
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	return &Error{Code: CodeUnknown, Msg: err.Error(), Err: err}
}
