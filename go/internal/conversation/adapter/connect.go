// Package adapter は ConversationService を Connect RPC で公開する。
package adapter

import (
	"context"
	"net/http"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/types/known/timestamppb"

	conversationv1 "github.com/hiro8ma/agent/go/gen/conversation/v1"
	"github.com/hiro8ma/agent/go/gen/conversation/v1/conversationv1connect"
	"github.com/hiro8ma/agent/go/internal/conversation/domain"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Handler は usecase を Connect の型に写すだけで、判断は持たない。
type Handler struct {
	svc *usecase.Service
}

var _ conversationv1connect.ConversationServiceHandler = (*Handler)(nil)

// NewHandler は利用者の特定まで含めたハンドラを返す。
func NewHandler(svc *usecase.Service, auth libconnect.Authenticator) (string, http.Handler) {
	return conversationv1connect.NewConversationServiceHandler(&Handler{svc: svc},
		connect.WithInterceptors(libconnect.Telemetry(), libconnect.ServerIdentity(auth)))
}

func (h *Handler) CreateSession(ctx context.Context, req *connect.Request[conversationv1.CreateSessionRequest]) (*connect.Response[conversationv1.CreateSessionResponse], error) {
	s, err := h.svc.CreateSession(ctx, req.Msg.GetAgentId())
	if err != nil {
		return nil, libconnect.Error(err)
	}
	return connect.NewResponse(&conversationv1.CreateSessionResponse{Session: &conversationv1.Session{
		Id: s.ID, AgentId: s.AgentID, CreateTime: timestamppb.New(s.CreatedAt),
	}}), nil
}

func (h *Handler) AppendTurn(ctx context.Context, req *connect.Request[conversationv1.AppendTurnRequest]) (*connect.Response[conversationv1.AppendTurnResponse], error) {
	msgs := make([]domain.Message, len(req.Msg.GetMessages()))
	for i, m := range req.Msg.GetMessages() {
		role, err := roleFromProto(m.GetRole())
		if err != nil {
			return nil, libconnect.Error(err)
		}
		msgs[i] = domain.Message{Role: role, Text: m.GetText()}
	}
	saved, err := h.svc.AppendTurn(ctx, req.Msg.GetSessionId(), req.Msg.GetAgentId(), msgs)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	return connect.NewResponse(&conversationv1.AppendTurnResponse{Messages: messagesToProto(saved)}), nil
}

func (h *Handler) ListMessages(ctx context.Context, req *connect.Request[conversationv1.ListMessagesRequest]) (*connect.Response[conversationv1.ListMessagesResponse], error) {
	msgs, next, err := h.svc.ListMessages(ctx, req.Msg.GetSessionId(), req.Msg.GetPageSize(), req.Msg.GetPageToken())
	if err != nil {
		return nil, libconnect.Error(err)
	}
	return connect.NewResponse(&conversationv1.ListMessagesResponse{Messages: messagesToProto(msgs), NextPageToken: next}), nil
}

func (h *Handler) SubmitFeedback(ctx context.Context, req *connect.Request[conversationv1.SubmitFeedbackRequest]) (*connect.Response[conversationv1.SubmitFeedbackResponse], error) {
	var rating domain.Rating
	switch req.Msg.GetRating() {
	case conversationv1.Rating_RATING_GOOD:
		rating = domain.RatingGood
	case conversationv1.Rating_RATING_BAD:
		rating = domain.RatingBad
	case conversationv1.Rating_RATING_UNSPECIFIED:
	}
	err := h.svc.SubmitFeedback(ctx, req.Msg.GetSessionId(), req.Msg.GetMessageId(), rating, req.Msg.GetComment())
	if err != nil {
		return nil, libconnect.Error(err)
	}
	return connect.NewResponse(&conversationv1.SubmitFeedbackResponse{}), nil
}

func roleFromProto(r conversationv1.Role) (domain.Role, error) {
	switch r {
	case conversationv1.Role_ROLE_USER:
		return domain.RoleUser, nil
	case conversationv1.Role_ROLE_MODEL:
		return domain.RoleModel, nil
	case conversationv1.Role_ROLE_UNSPECIFIED:
	}
	return "", liberrors.Newf(liberrors.CodeInvalidArgument, "role が未指定")
}

func roleToProto(r domain.Role) conversationv1.Role {
	if r == domain.RoleModel {
		return conversationv1.Role_ROLE_MODEL
	}
	return conversationv1.Role_ROLE_USER
}

func messagesToProto(msgs []domain.Message) []*conversationv1.Message {
	out := make([]*conversationv1.Message, len(msgs))
	for i, m := range msgs {
		out[i] = &conversationv1.Message{
			Id: m.ID, Role: roleToProto(m.Role), Text: m.Text, CreateTime: timestamppb.New(m.CreatedAt),
		}
	}
	return out
}
