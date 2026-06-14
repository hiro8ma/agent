package workflow

import (
	"encoding/json"
	"os"
	"sync"

	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Event は完了したステップ 1 件のチェックポイント。WHY: 再実行時に
// ジャーナルにあるステップはスキップし、途中から再開する（決定論リプレイの簡易版）。
type Event struct {
	StepName string         `json:"step_name"`
	Attempts int            `json:"attempts"`
	Result   ActivityResult `json:"result"`
}

// Journal は完了ステップのイベント履歴を読み書きする。
// インメモリ実装とファイル永続化実装を差し替えられるよう interface 分離する。
type Journal interface {
	// Load は記録済みイベントを順序通りに返す。
	Load() ([]Event, error)
	// Append は完了ステップを 1 件追記し、即時に永続化する。
	Append(e Event) error
}

// MemoryJournal はプロセス内のインメモリ実装。テストや単発実行向け。
type MemoryJournal struct {
	mu     sync.Mutex
	events []Event
}

// NewMemoryJournal はインメモリ Journal を生成する。
func NewMemoryJournal() *MemoryJournal { return &MemoryJournal{} }

// Load は Journal を満たす。
func (j *MemoryJournal) Load() ([]Event, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]Event, len(j.events))
	copy(out, j.events)
	return out, nil
}

// Append は Journal を満たす。
func (j *MemoryJournal) Append(e Event) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.events = append(j.events, e)
	return nil
}

// FileJournal は JSON ファイルへ追記する永続化実装。
// WHY: プロセスがクラッシュしても完了済みステップがディスクに残り、
// 再起動後の別プロセスがそこから再開できる。
type FileJournal struct {
	mu   sync.Mutex
	path string
}

// NewFileJournal は指定パスの JSON ファイルを使う Journal を生成する。
func NewFileJournal(path string) *FileJournal { return &FileJournal{path: path} }

// Load は Journal を満たす。ファイル未作成なら空履歴を返す。
func (j *FileJournal) Load() ([]Event, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	data, err := os.ReadFile(j.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "journal read")
	}
	var events []Event
	if len(data) == 0 {
		return nil, nil
	}
	if err := json.Unmarshal(data, &events); err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "journal unmarshal")
	}
	return events, nil
}

// Append は Journal を満たす。全イベントを読み直して 1 件足し、ファイル全体を書き戻す。
func (j *FileJournal) Append(e Event) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	var events []Event
	data, err := os.ReadFile(j.path)
	switch {
	case err == nil && len(data) > 0:
		if err := json.Unmarshal(data, &events); err != nil {
			return liberrors.Wrap(liberrors.CodeInternal, err, "journal unmarshal")
		}
	case err != nil && !os.IsNotExist(err):
		return liberrors.Wrap(liberrors.CodeInternal, err, "journal read")
	}

	events = append(events, e)
	out, err := json.MarshalIndent(events, "", "  ")
	if err != nil {
		return liberrors.Wrap(liberrors.CodeInternal, err, "journal marshal")
	}
	if err := os.WriteFile(j.path, out, 0o600); err != nil {
		return liberrors.Wrap(liberrors.CodeInternal, err, "journal write")
	}
	return nil
}
