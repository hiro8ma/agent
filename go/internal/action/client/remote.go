// Package client はエージェントのサービスから ActionService を呼ぶ。action.Gate を満たす。
package client

import (
	"context"
	"net/http"
	"os"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/types/known/structpb"

	actionv1 "github.com/hiro8ma/agent/go/gen/action/v1"
	"github.com/hiro8ma/agent/go/gen/action/v1/actionv1connect"
	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// EnvURL が設定されていれば、別プロセスの ActionService を呼ぶ。
const EnvURL = "ACTION_URL"

// Remote は ActionService を呼び出し元の利用者のまま呼ぶ。
type Remote struct {
	client actionv1connect.ActionServiceClient
}

var _ action.Gate = (*Remote)(nil)

func NewRemote(httpClient connect.HTTPClient, baseURL string) *Remote {
	return &Remote{client: actionv1connect.NewActionServiceClient(httpClient, baseURL,
		connect.WithInterceptors(libconnect.Telemetry(), libconnect.ForwardIdentity()))}
}

func (r *Remote) Authorize(ctx context.Context, tool string, args map[string]any) (approval.Decision, error) {
	s, err := structpb.NewStruct(args)
	if err != nil {
		return approval.Decision{}, err
	}
	res, err := r.client.Authorize(ctx, connect.NewRequest(&actionv1.AuthorizeRequest{Tool: tool, Args: s}))
	if err != nil {
		return approval.Decision{}, libconnect.FromConnect(err, "action: 判定")
	}
	d := approval.Decision{Risk: riskFromProto(res.Msg.GetRisk())}
	switch res.Msg.GetOutcome() {
	case actionv1.Outcome_OUTCOME_ALLOW:
		d.Outcome = approval.Allow
	case actionv1.Outcome_OUTCOME_PENDING:
		d.Outcome = approval.NeedsApproval
		req := fromProto(res.Msg.GetRequest())
		d.Request = &req
	default:
		d.Outcome = approval.Deny
	}
	return d, nil
}

func (r *Remote) Take(ctx context.Context, id string) (approval.Request, error) {
	res, err := r.client.Take(ctx, connect.NewRequest(&actionv1.TakeRequest{RequestId: id}))
	if err != nil {
		return approval.Request{}, libconnect.FromConnect(err, "action: 取り出し")
	}
	return fromProto(res.Msg.GetRequest()), nil
}

func fromProto(p *actionv1.ApprovalRequest) approval.Request {
	return approval.Request{
		ID: p.GetId(), Tool: p.GetTool(), Args: p.GetArgs().AsMap(), Requester: p.GetRequester(),
		Risk: riskFromProto(p.GetRisk()), Approver: p.GetApprover(),
		CreatedAt: p.GetCreateTime().AsTime(), ExpiresAt: p.GetExpireTime().AsTime(),
	}
}

func riskFromProto(r actionv1.Risk) approval.Risk {
	switch r {
	case actionv1.Risk_RISK_MEDIUM:
		return approval.Medium
	case actionv1.Risk_RISK_HIGH:
		return approval.High
	case actionv1.Risk_RISK_FORBIDDEN, actionv1.Risk_RISK_UNSPECIFIED:
		return approval.Forbidden
	case actionv1.Risk_RISK_LOW:
	}
	return approval.Low
}

// FromEnv は ACTION_URL があれば別プロセスの ActionService を、無ければプロセス内の承認窓口を返す。
func FromEnv() (gate action.Gate, where string) {
	if url := os.Getenv(EnvURL); url != "" {
		return NewRemote(http.DefaultClient, url), url
	}
	return action.NewLocalFromEnv(), "in-process"
}
