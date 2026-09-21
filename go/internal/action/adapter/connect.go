// Package adapter は ActionService を Connect RPC で公開する。
package adapter

import (
	"context"
	"net/http"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	actionv1 "github.com/hiro8ma/agent/go/gen/action/v1"
	"github.com/hiro8ma/agent/go/gen/action/v1/actionv1connect"
	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// Handler は approval.Service を Connect の型に写すだけで、判断は持たない。
type Handler struct {
	svc *approval.Service
}

var _ actionv1connect.ActionServiceHandler = (*Handler)(nil)

// NewHandler は利用者の特定まで含めたハンドラを返す。
func NewHandler(svc *approval.Service, auth libconnect.Authenticator) (string, http.Handler) {
	return actionv1connect.NewActionServiceHandler(&Handler{svc: svc},
		connect.WithInterceptors(libconnect.ServerIdentity(auth)))
}

func caller(ctx context.Context) (string, error) {
	id, err := identity.From(ctx)
	return string(id), err
}

func (h *Handler) Authorize(ctx context.Context, req *connect.Request[actionv1.AuthorizeRequest]) (*connect.Response[actionv1.AuthorizeResponse], error) {
	who, err := caller(ctx)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	d, err := h.svc.Authorize(who, req.Msg.GetTool(), req.Msg.GetArgs().AsMap())
	if err != nil {
		return nil, libconnect.Error(action.Coded(err))
	}
	res := &actionv1.AuthorizeResponse{Outcome: outcomeToProto(d.Outcome), Risk: riskToProto(d.Risk)}
	if d.Request != nil {
		res.Request = requestToProto(*d.Request)
	}
	return connect.NewResponse(res), nil
}

func (h *Handler) Decide(ctx context.Context, req *connect.Request[actionv1.DecideRequest]) (*connect.Response[actionv1.DecideResponse], error) {
	who, err := caller(ctx)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	if err := h.svc.Decide(who, req.Msg.GetRequestId(), req.Msg.GetApprove()); err != nil {
		return nil, libconnect.Error(action.Coded(err))
	}
	r, err := h.svc.Get(req.Msg.GetRequestId())
	if err != nil {
		return nil, libconnect.Error(action.Coded(err))
	}
	return connect.NewResponse(&actionv1.DecideResponse{Request: requestToProto(r)}), nil
}

func (h *Handler) Take(ctx context.Context, req *connect.Request[actionv1.TakeRequest]) (*connect.Response[actionv1.TakeResponse], error) {
	who, err := caller(ctx)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	r, err := h.svc.Take(who, req.Msg.GetRequestId())
	if err != nil {
		return nil, libconnect.Error(action.Coded(err))
	}
	return connect.NewResponse(&actionv1.TakeResponse{Request: requestToProto(r)}), nil
}

func (h *Handler) ListRequests(ctx context.Context, req *connect.Request[actionv1.ListRequestsRequest]) (*connect.Response[actionv1.ListRequestsResponse], error) {
	who, err := caller(ctx)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	f := approval.Mine
	if req.Msg.GetView() == actionv1.View_VIEW_TO_APPROVE {
		f = approval.ToApprove
	}
	res := &actionv1.ListRequestsResponse{}
	for _, r := range h.svc.List(who, f) {
		res.Requests = append(res.Requests, requestToProto(r))
	}
	return connect.NewResponse(res), nil
}

func (h *Handler) ListAudit(ctx context.Context, req *connect.Request[actionv1.ListAuditRequest]) (*connect.Response[actionv1.ListAuditResponse], error) {
	who, err := caller(ctx)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	entries, err := h.svc.Audit(who, req.Msg.GetRequestId())
	if err != nil {
		return nil, libconnect.Error(action.Coded(err))
	}
	res := &actionv1.ListAuditResponse{}
	for _, e := range entries {
		res.Entries = append(res.Entries, &actionv1.AuditEntry{
			RequestId: e.RequestID, Action: e.Action, Actor: e.Actor, Time: timestamppb.New(e.At), Detail: e.Detail,
		})
	}
	return connect.NewResponse(res), nil
}

func requestToProto(r approval.Request) *actionv1.ApprovalRequest {
	args, _ := structpb.NewStruct(r.Args)
	return &actionv1.ApprovalRequest{
		Id: r.ID, Tool: r.Tool, Args: args, Requester: r.Requester, Risk: riskToProto(r.Risk),
		Status: statusToProto(r.Status), Approver: r.Approver,
		CreateTime: timestamppb.New(r.CreatedAt), ExpireTime: timestamppb.New(r.ExpiresAt),
	}
}

func riskToProto(r approval.Risk) actionv1.Risk {
	return map[approval.Risk]actionv1.Risk{
		approval.Low: actionv1.Risk_RISK_LOW, approval.Medium: actionv1.Risk_RISK_MEDIUM,
		approval.High: actionv1.Risk_RISK_HIGH, approval.Forbidden: actionv1.Risk_RISK_FORBIDDEN,
	}[r]
}

func outcomeToProto(o approval.Outcome) actionv1.Outcome {
	return map[approval.Outcome]actionv1.Outcome{
		approval.Allow: actionv1.Outcome_OUTCOME_ALLOW, approval.NeedsApproval: actionv1.Outcome_OUTCOME_PENDING,
		approval.Deny: actionv1.Outcome_OUTCOME_FORBIDDEN,
	}[o]
}

func statusToProto(s approval.Status) actionv1.Status {
	return map[approval.Status]actionv1.Status{
		approval.Pending: actionv1.Status_STATUS_PENDING, approval.Approved: actionv1.Status_STATUS_APPROVED,
		approval.Rejected: actionv1.Status_STATUS_REJECTED, approval.Used: actionv1.Status_STATUS_USED,
	}[s]
}
