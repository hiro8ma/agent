// Package actiongate は承認の窓口（internal/action の Gate）を、Genkit 版のエージェントの型で呼べるようにする。
package actiongate

import (
	"context"
	"errors"

	"github.com/hiro8ma/agent/go/internal/action"
	actionclient "github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

type Gate struct {
	gate action.Gate
}

var _ externalservice.ActionGate = (*Gate)(nil)

func New(gate action.Gate) *Gate {
	return &Gate{gate: gate}
}

// FromEnv は ACTION_URL があれば別プロセスの ActionService を、無ければプロセス内の承認窓口を使う。
func FromEnv() (gate *Gate, where string) {
	g, where := actionclient.FromEnv()
	return New(g), where
}

func (g *Gate) Authorize(ctx context.Context, tool string, args map[string]any) (*model.ActionDecision, error) {
	d, err := g.gate.Authorize(ctx, tool, args)
	if err != nil {
		return nil, err
	}
	out := &model.ActionDecision{Risk: d.Risk.String()}
	switch d.Outcome {
	case approval.Allow:
		out.Outcome = model.ActionAllow
	case approval.NeedsApproval:
		out.Outcome = model.ActionNeedsApproval
	case approval.Deny:
		out.Outcome = model.ActionDeny
	}
	if d.Request != nil {
		out.RequestID = d.Request.ID
	}
	return out, nil
}

func (g *Gate) Take(ctx context.Context, id string) (*model.ApprovedAction, error) {
	r, err := g.gate.Take(ctx, id)
	if err != nil {
		if liberrors.Convert(err).Code == liberrors.CodeNotFound {
			return nil, errors.Join(model.ErrPendingNotFound, err)
		}
		return nil, err
	}
	return &model.ApprovedAction{ID: r.ID, Tool: r.Tool, Args: r.Args}, nil
}
