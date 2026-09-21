// Package action は変更系の操作を承認の後ろに置く。
//
// エージェントのツールは Gate.Authorize で自動実行か承認待ちか拒否かを受け取り、
// 承認待ちなら依頼の ID を返すだけで実行しない。承認の後、依頼者が ExecuteConfirmedToolCall を呼ぶと、
// Executor が Gate.Take で承認された引数を受け取り、その引数で実行する。モデルに呼び直させない。
package action

import (
	"context"
	"errors"
	"os"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Gate は承認の窓口。プロセス内の Local と、ActionService を呼ぶ client.Remote がある。
type Gate interface {
	// Authorize は呼び出し元の利用者の依頼として判定する。
	Authorize(ctx context.Context, tool string, args map[string]any) (approval.Decision, error)
	// Take は呼び出し元の利用者の、承認済みの依頼を使用済みにして返す。
	Take(ctx context.Context, id string) (approval.Request, error)
}

// Local は approval.Service をプロセス内で使う。
type Local struct {
	Service *approval.Service
}

func (l Local) Authorize(ctx context.Context, tool string, args map[string]any) (approval.Decision, error) {
	who, err := identity.From(ctx)
	if err != nil {
		return approval.Decision{}, err
	}
	return l.Service.Authorize(string(who), tool, args)
}

func (l Local) Take(ctx context.Context, id string) (approval.Request, error) {
	who, err := identity.From(ctx)
	if err != nil {
		return approval.Request{}, err
	}
	r, err := l.Service.Take(string(who), id)
	return r, Coded(err)
}

// Coded は approval のエラーに Code を付ける。API の境界で Connect の Code に写すために使う。
func Coded(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, approval.ErrNotFound):
		return liberrors.Wrap(liberrors.CodeNotFound, err, "%s", err.Error())
	case errors.Is(err, approval.ErrNotYours), errors.Is(err, approval.ErrSelfApproval),
		errors.Is(err, approval.ErrNotApprover), errors.Is(err, approval.ErrNotRequester):
		return liberrors.Wrap(liberrors.CodePermissionDeny, err, "%s", err.Error())
	case errors.Is(err, approval.ErrNotPending), errors.Is(err, approval.ErrNotApproved),
		errors.Is(err, approval.ErrExpired), errors.Is(err, approval.ErrForbiddenRisk):
		return liberrors.Wrap(liberrors.CodeFailedPrecond, err, "%s", err.Error())
	}
	return err
}

// ToolUpdatePaymentMethod は注文の支払い方法を変えるツールの名前。ADK と Genkit で揃える。
const ToolUpdatePaymentMethod = "update_order_payment_method"

// DefaultRules は変更系のツールのリスク。支払い方法の変更はお金の流れを変えるので、承認者の承認を要る。
func DefaultRules() map[string]approval.Rule {
	return map[string]approval.Rule{
		ToolUpdatePaymentMethod: func(map[string]any) approval.Risk { return approval.High },
	}
}

// NewLocalFromEnv はプロセス内の承認窓口を作る。承認者は ACTION_APPROVERS（カンマ区切り）で決める。
func NewLocalFromEnv() Local {
	var approvers []string
	for a := range strings.SplitSeq(os.Getenv("ACTION_APPROVERS"), ",") {
		if a = strings.TrimSpace(a); a != "" {
			approvers = append(approvers, a)
		}
	}
	return Local{Service: approval.NewService(DefaultRules(), approvers, time.Hour)}
}
