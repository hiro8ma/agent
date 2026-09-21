// Package usecase は会話の操作。呼び出し元が所有者かをここで確かめる。
package usecase

import (
	"context"
	"time"

	"github.com/google/uuid"

	"github.com/hiro8ma/agent/go/internal/conversation/domain"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Service は会話の操作。
type Service struct {
	repo domain.Repository
}

func New(repo domain.Repository) *Service {
	return &Service{repo: repo}
}

func (s *Service) CreateSession(ctx context.Context, agentID string) (domain.Session, error) {
	caller, err := identity.From(ctx)
	if err != nil {
		return domain.Session{}, err
	}
	sess := domain.Session{ID: uuid.NewString(), Owner: caller, AgentID: agentID, CreatedAt: time.Now()}
	if err := s.repo.CreateSession(ctx, sess); err != nil {
		return domain.Session{}, liberrors.Wrap(liberrors.CodeInternal, err, "セッションの作成")
	}
	return sess, nil
}

// AppendTurn は 1 往復分を追記する。セッションが無ければ呼び出し元の所有で作る。
func (s *Service) AppendTurn(ctx context.Context, sessionID, agentID string, msgs []domain.Message) ([]domain.Message, error) {
	caller, err := identity.From(ctx)
	if err != nil {
		return nil, err
	}
	if sessionID == "" {
		return nil, liberrors.Newf(liberrors.CodeInvalidArgument, "session_id が空")
	}
	if len(msgs) == 0 {
		return nil, liberrors.Newf(liberrors.CodeInvalidArgument, "messages が空")
	}
	now := time.Now()
	for i := range msgs {
		if err := msgs[i].Validate(); err != nil {
			return nil, err
		}
		msgs[i].ID = uuid.NewString()
		msgs[i].CreatedAt = now
	}
	sess, err := s.repo.EnsureSession(ctx, domain.Session{ID: sessionID, Owner: caller, AgentID: agentID, CreatedAt: now})
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "セッションの取得")
	}
	if err := sess.Authorize(caller); err != nil {
		return nil, err
	}
	saved, err := s.repo.AppendMessages(ctx, sessionID, msgs)
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "メッセージの追記")
	}
	return saved, nil
}

// ListMessages は古い順に 1 ページ分を返す。次のページが無ければ nextToken は空。
func (s *Service) ListMessages(ctx context.Context, sessionID string, pageSize int32, pageToken string) (msgs []domain.Message, nextToken string, err error) {
	if _, err := s.ownedSession(ctx, sessionID); err != nil {
		return nil, "", err
	}
	after, err := domain.DecodePageToken(pageToken)
	if err != nil {
		return nil, "", err
	}
	limit := domain.PageSize(pageSize)
	// 1 件多く読み、次のページがあるかを追加の問い合わせなしで判定する。
	msgs, err = s.repo.ListMessages(ctx, sessionID, after, limit+1)
	if err != nil {
		return nil, "", liberrors.Wrap(liberrors.CodeInternal, err, "メッセージの取得")
	}
	if len(msgs) > limit {
		msgs = msgs[:limit]
		nextToken = domain.EncodePageToken(msgs[limit-1].Seq)
	}
	return msgs, nextToken, nil
}

func (s *Service) SubmitFeedback(ctx context.Context, sessionID, messageID string, rating domain.Rating, comment string) error {
	sess, err := s.ownedSession(ctx, sessionID)
	if err != nil {
		return err
	}
	if rating != domain.RatingGood && rating != domain.RatingBad {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "rating は good か bad")
	}
	const maxCommentRunes = 2000
	if len([]rune(comment)) > maxCommentRunes {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "comment は %d 文字まで", maxCommentRunes)
	}
	msg, err := s.repo.GetMessage(ctx, sessionID, messageID)
	if err != nil {
		return err
	}
	if msg.Role != domain.RoleModel {
		return liberrors.Newf(liberrors.CodeInvalidArgument, "評価できるのはアシスタントのメッセージだけ")
	}
	err = s.repo.SaveFeedback(ctx, domain.Feedback{
		SessionID: sessionID, MessageID: messageID, Owner: sess.Owner, Rating: rating, Comment: comment,
	})
	if err != nil {
		return liberrors.Wrap(liberrors.CodeInternal, err, "評価の保存")
	}
	return nil
}

func (s *Service) ownedSession(ctx context.Context, sessionID string) (domain.Session, error) {
	caller, err := identity.From(ctx)
	if err != nil {
		return domain.Session{}, err
	}
	sess, err := s.repo.GetSession(ctx, sessionID)
	if err != nil {
		return domain.Session{}, err
	}
	if err := sess.Authorize(caller); err != nil {
		return domain.Session{}, err
	}
	return sess, nil
}
