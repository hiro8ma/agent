// Package repository は会話の保管先の実装。
package repository

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/logger"

	"github.com/hiro8ma/agent/go/internal/conversation/domain"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
)

type sessionRow struct {
	ID        string `gorm:"primaryKey"`
	Owner     string `gorm:"index;not null"`
	AgentID   string
	CreatedAt time.Time
}

func (sessionRow) TableName() string { return "sessions" }

type messageRow struct {
	ID        string `gorm:"primaryKey"`
	SessionID string `gorm:"uniqueIndex:idx_session_seq;not null"`
	Seq       int64  `gorm:"uniqueIndex:idx_session_seq;not null"`
	Role      string `gorm:"not null"`
	Text      string `gorm:"not null"`
	CreatedAt time.Time
}

func (messageRow) TableName() string { return "messages" }

type feedbackRow struct {
	SessionID string `gorm:"not null"`
	MessageID string `gorm:"primaryKey"`
	Owner     string `gorm:"primaryKey"`
	Rating    string `gorm:"not null"`
	Comment   string
	UpdatedAt time.Time
}

func (feedbackRow) TableName() string { return "feedbacks" }

// SQLite は会話を SQLite に持つ。ローカル検証と単体テストで使う。
type SQLite struct {
	db *gorm.DB
}

var _ domain.Repository = (*SQLite)(nil)

// NewSQLite は dsn（例 file:./.conversation/conversation.db）に接続し、表を作る。
func NewSQLite(dsn string) (*SQLite, error) {
	if err := ensureParentDir(dsn); err != nil {
		return nil, err
	}
	// トランザクションの開始時に書き込みロックを取り、採番の読み取りと追記の間に他の書き込みを入れない。
	db, err := gorm.Open(sqlite.Open(dsn+sep(dsn)+"_pragma=busy_timeout(5000)&_txlock=immediate"),
		&gorm.Config{Logger: logger.Discard})
	if err != nil {
		return nil, fmt.Errorf("conversation: sqlite の接続: %w", err)
	}
	if err := db.AutoMigrate(&sessionRow{}, &messageRow{}, &feedbackRow{}); err != nil {
		return nil, fmt.Errorf("conversation: 表の作成: %w", err)
	}
	return &SQLite{db: db}, nil
}

func sep(dsn string) string {
	if strings.Contains(dsn, "?") {
		return "&"
	}
	return "?"
}

func ensureParentDir(dsn string) error {
	path, ok := strings.CutPrefix(dsn, "file:")
	if !ok {
		return nil
	}
	path, _, _ = strings.Cut(path, "?")
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o750); err != nil { //nolint:gosec // DSN は運用者が環境変数で与える値で、利用者の入力ではない
		return fmt.Errorf("conversation: 保管先ディレクトリの作成: %w", err)
	}
	return nil
}

func (r *SQLite) CreateSession(ctx context.Context, s domain.Session) error {
	return r.db.WithContext(ctx).Create(toSessionRow(s)).Error
}

func (r *SQLite) GetSession(ctx context.Context, id string) (domain.Session, error) {
	var row sessionRow
	err := r.db.WithContext(ctx).Where("id = ?", id).Take(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return domain.Session{}, domain.ErrSessionNotFound
	}
	if err != nil {
		return domain.Session{}, err
	}
	return fromSessionRow(row), nil
}

func (r *SQLite) EnsureSession(ctx context.Context, s domain.Session) (domain.Session, error) {
	// 同時に作ろうとしても、主キーの衝突で先に入った 1 件だけが残る。所有者はその 1 件で決まる。
	err := r.db.WithContext(ctx).Clauses(clause.OnConflict{DoNothing: true}).Create(toSessionRow(s)).Error
	if err != nil {
		return domain.Session{}, err
	}
	return r.GetSession(ctx, s.ID)
}

func (r *SQLite) AppendMessages(ctx context.Context, sessionID string, msgs []domain.Message) ([]domain.Message, error) {
	out := make([]domain.Message, len(msgs))
	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var last int64
		if err := tx.Model(&messageRow{}).Where("session_id = ?", sessionID).
			Select("COALESCE(MAX(seq), 0)").Scan(&last).Error; err != nil {
			return err
		}
		rows := make([]messageRow, len(msgs))
		for i, m := range msgs {
			m.SessionID = sessionID
			m.Seq = last + int64(i) + 1
			out[i] = m
			rows[i] = toMessageRow(m)
		}
		return tx.Create(&rows).Error
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (r *SQLite) ListMessages(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]domain.Message, error) {
	var rows []messageRow
	err := r.db.WithContext(ctx).Where("session_id = ? AND seq > ?", sessionID, afterSeq).
		Order("seq").Limit(limit).Find(&rows).Error
	if err != nil {
		return nil, err
	}
	out := make([]domain.Message, len(rows))
	for i, row := range rows {
		out[i] = fromMessageRow(row)
	}
	return out, nil
}

func (r *SQLite) GetMessage(ctx context.Context, sessionID, messageID string) (domain.Message, error) {
	var row messageRow
	err := r.db.WithContext(ctx).Where("session_id = ? AND id = ?", sessionID, messageID).Take(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return domain.Message{}, domain.ErrMessageNotFound
	}
	if err != nil {
		return domain.Message{}, err
	}
	return fromMessageRow(row), nil
}

func (r *SQLite) SaveFeedback(ctx context.Context, f domain.Feedback) error {
	row := feedbackRow{
		SessionID: f.SessionID, MessageID: f.MessageID, Owner: string(f.Owner),
		Rating: string(f.Rating), Comment: f.Comment,
	}
	return r.db.WithContext(ctx).Clauses(clause.OnConflict{UpdateAll: true}).Create(&row).Error
}

func toSessionRow(s domain.Session) *sessionRow {
	return &sessionRow{ID: s.ID, Owner: string(s.Owner), AgentID: s.AgentID, CreatedAt: s.CreatedAt}
}

func fromSessionRow(row sessionRow) domain.Session {
	return domain.Session{ID: row.ID, Owner: identity.UserID(row.Owner), AgentID: row.AgentID, CreatedAt: row.CreatedAt}
}

func toMessageRow(m domain.Message) messageRow {
	return messageRow{ID: m.ID, SessionID: m.SessionID, Seq: m.Seq, Role: string(m.Role), Text: m.Text, CreatedAt: m.CreatedAt}
}

func fromMessageRow(row messageRow) domain.Message {
	return domain.Message{
		ID: row.ID, SessionID: row.SessionID, Seq: row.Seq,
		Role: domain.Role(row.Role), Text: row.Text, CreatedAt: row.CreatedAt,
	}
}
