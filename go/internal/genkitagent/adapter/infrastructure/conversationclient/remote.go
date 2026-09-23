// Package conversationclient は会話の履歴を ConversationService に保存する。
// 呼び出し元の利用者のまま呼ぶので、他人のセッションは ConversationService が拒否する。
package conversationclient

import (
	"context"
	"net/http"

	"connectrpc.com/connect"

	conversationv1 "github.com/hiro8ma/agent/go/gen/conversation/v1"
	"github.com/hiro8ma/agent/go/gen/conversation/v1/conversationv1connect"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

type Remote struct {
	client conversationv1connect.ConversationServiceClient
}

var _ repository.SessionCreator = (*Remote)(nil)

func NewRemote(httpClient connect.HTTPClient, baseURL string) *Remote {
	return &Remote{
		client: conversationv1connect.NewConversationServiceClient(httpClient, baseURL,
			connect.WithInterceptors(libconnect.Telemetry(), libconnect.ForwardIdentity())),
	}
}

func NewDefaultRemote(baseURL string) *Remote {
	return NewRemote(http.DefaultClient, baseURL)
}

func (s *Remote) CreateSession(ctx context.Context, agentID string) (string, error) {
	res, err := s.client.CreateSession(ctx, connect.NewRequest(&conversationv1.CreateSessionRequest{AgentId: agentID}))
	if err != nil {
		return "", libconnect.FromConnect(err, "conversation: セッションの作成")
	}
	return res.Msg.GetSession().GetId(), nil
}

// ListMessages は全ページを読む。まだ 1 件も追記されていないセッションは空の履歴を返す。
func (s *Remote) ListMessages(ctx context.Context, sessionID string) ([]model.Message, error) {
	var out []model.Message
	token := ""
	for {
		res, err := s.client.ListMessages(ctx, connect.NewRequest(&conversationv1.ListMessagesRequest{
			SessionId: sessionID, PageToken: token,
		}))
		if connect.CodeOf(err) == connect.CodeNotFound && token == "" {
			return nil, nil
		}
		if err != nil {
			return nil, libconnect.FromConnect(err, "conversation: 履歴の取得")
		}
		for _, m := range res.Msg.GetMessages() {
			out = append(out, model.Message{Role: roleFromProto(m.GetRole()), Text: m.GetText()})
		}
		token = res.Msg.GetNextPageToken()
		if token == "" {
			return out, nil
		}
	}
}

func (s *Remote) CreateMessages(ctx context.Context, sessionID string, messages []model.Message) error {
	msgs := make([]*conversationv1.Message, len(messages))
	for i, m := range messages {
		msgs[i] = &conversationv1.Message{Role: roleToProto(m.Role), Text: m.Text}
	}
	_, err := s.client.AppendTurn(ctx, connect.NewRequest(&conversationv1.AppendTurnRequest{
		SessionId: sessionID, Messages: msgs,
	}))
	if err != nil {
		return libconnect.FromConnect(err, "conversation: 履歴の追記")
	}
	return nil
}

func roleFromProto(r conversationv1.Role) string {
	if r == conversationv1.Role_ROLE_MODEL {
		return model.RoleModel
	}
	return model.RoleUser
}

func roleToProto(role string) conversationv1.Role {
	if role == model.RoleModel {
		return conversationv1.Role_ROLE_MODEL
	}
	return conversationv1.Role_ROLE_USER
}
