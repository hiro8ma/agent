// Package client はエージェントのサービスから ConversationService を呼ぶ。
// agentcore.SessionStore を満たすので、プロセス内の保管先と差し替えられる。
package client

import (
	"context"
	"net/http"

	"connectrpc.com/connect"

	conversationv1 "github.com/hiro8ma/agent/go/gen/conversation/v1"
	"github.com/hiro8ma/agent/go/gen/conversation/v1/conversationv1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// SessionStore は ConversationService を呼ぶ agentcore.SessionStore。
// 呼び出し元の利用者のまま呼ぶので、他人のセッションは下流で拒否される。
type SessionStore struct {
	client conversationv1connect.ConversationServiceClient
}

var _ agentcore.SessionCreator = (*SessionStore)(nil)

func NewSessionStore(httpClient connect.HTTPClient, baseURL string) *SessionStore {
	return &SessionStore{
		client: conversationv1connect.NewConversationServiceClient(httpClient, baseURL,
			connect.WithInterceptors(libconnect.Telemetry(), libconnect.ForwardIdentity())),
	}
}

// NewDefaultSessionStore は http.DefaultClient で繋ぐ。
func NewDefaultSessionStore(baseURL string) *SessionStore {
	return NewSessionStore(http.DefaultClient, baseURL)
}

func (s *SessionStore) Create(ctx context.Context, agentID string) (string, error) {
	res, err := s.client.CreateSession(ctx, connect.NewRequest(&conversationv1.CreateSessionRequest{AgentId: agentID}))
	if err != nil {
		return "", libconnect.FromConnect(err, "conversation: セッションの作成")
	}
	return res.Msg.GetSession().GetId(), nil
}

// Load は全ページを読む。まだ 1 件も追記されていないセッションは空の履歴を返す。
func (s *SessionStore) Load(ctx context.Context, sessionID string) ([]agentcore.Message, error) {
	var out []agentcore.Message
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
			role := "user"
			if m.GetRole() == conversationv1.Role_ROLE_MODEL {
				role = "model"
			}
			out = append(out, agentcore.Message{Role: role, Text: m.GetText()})
		}
		token = res.Msg.GetNextPageToken()
		if token == "" {
			return out, nil
		}
	}
}

func (s *SessionStore) Append(ctx context.Context, sessionID string, messages ...agentcore.Message) error {
	msgs := make([]*conversationv1.Message, len(messages))
	for i, m := range messages {
		role := conversationv1.Role_ROLE_USER
		if m.Role == "model" {
			role = conversationv1.Role_ROLE_MODEL
		}
		msgs[i] = &conversationv1.Message{Role: role, Text: m.Text}
	}
	_, err := s.client.AppendTurn(ctx, connect.NewRequest(&conversationv1.AppendTurnRequest{
		SessionId: sessionID, Messages: msgs,
	}))
	if err != nil {
		return libconnect.FromConnect(err, "conversation: 履歴の追記")
	}
	return nil
}
