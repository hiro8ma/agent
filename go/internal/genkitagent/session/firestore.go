package session

import (
	"context"
	"errors"
	"fmt"
	"time"

	"cloud.google.com/go/firestore"
	"google.golang.org/api/iterator"

	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
)

const (
	sessionCollection = "agent_sessions"
	messageCollection = "messages"
)

// Firestore はメッセージをサブコレクションで 1 件ずつ保持する。
// 1 ドキュメント 1MB 上限を回避し、長い会話でも破綻しない。
// FIRESTORE_EMULATOR_HOST が設定されていればエミュレータに接続する。
type Firestore struct {
	client *firestore.Client
}

var _ Store = (*Firestore)(nil)

func NewFirestore(client *firestore.Client) *Firestore {
	return &Firestore{client: client}
}

type messageDoc struct {
	Role      string    `firestore:"role"`
	Text      string    `firestore:"text"`
	CreatedAt time.Time `firestore:"createdAt"`
}

func (s *Firestore) messages(sessionID string) *firestore.CollectionRef {
	return s.client.Collection(sessionCollection).Doc(sessionID).Collection(messageCollection)
}

func (s *Firestore) Load(ctx context.Context, sessionID string) ([]agent.Message, error) {
	iter := s.messages(sessionID).OrderBy("createdAt", firestore.Asc).Documents(ctx)
	defer iter.Stop()

	var history []agent.Message
	for {
		snap, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			return history, nil
		}
		if err != nil {
			return nil, fmt.Errorf("load session %s: %w", sessionID, err)
		}
		var doc messageDoc
		if err := snap.DataTo(&doc); err != nil {
			return nil, fmt.Errorf("decode message %s/%s: %w", sessionID, snap.Ref.ID, err)
		}
		history = append(history, agent.Message{Role: doc.Role, Text: doc.Text})
	}
}

func (s *Firestore) Append(ctx context.Context, sessionID string, messages ...agent.Message) error {
	// createdAt を単調増加させ、同時刻書き込みでも順序を保つ
	base := time.Now()
	for i, m := range messages {
		doc := messageDoc{Role: m.Role, Text: m.Text, CreatedAt: base.Add(time.Duration(i) * time.Microsecond)}
		if _, _, err := s.messages(sessionID).Add(ctx, doc); err != nil {
			return fmt.Errorf("append session %s: %w", sessionID, err)
		}
	}
	return nil
}
