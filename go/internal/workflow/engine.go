package workflow

import (
	"context"
	"time"

	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Step は Workflow の 1 フェーズ。Activity・入力・リトライ設定を束ねる。
type Step struct {
	Activity Activity
	Input    ActivityInput
	Retry    RetryPolicy
}

// Workflow は順序付きステップのリスト。供給チェーン例では
// inventory -> transportation -> supplier の 3 フェーズを並べる。
type Workflow struct {
	Name  string
	Steps []Step
}

// Logger はエンジンの進捗通知。デモ表示やテスト検証に使う。nil 可。
type Logger func(format string, args ...any)

// Engine はジャーナルを使って Workflow を耐久実行する。
type Engine struct {
	journal Journal
	log     Logger
	// sleep はバックオフ待機。テストで時間を進めずに差し替えられるよう注入する。
	sleep func(time.Duration)
}

// NewEngine はエンジンを生成する。log が nil なら進捗通知は無視する。
func NewEngine(journal Journal, log Logger) *Engine {
	if log == nil {
		log = func(string, ...any) {}
	}
	return &Engine{journal: journal, log: log, sleep: time.Sleep}
}

// Result は Workflow 完了時の出力。各ステップ結果を順序通りに保持する。
type Result struct {
	Completed []Event
}

// Execute は Workflow を頭から流す。WHY: ジャーナルに完了済みのステップがあれば
// そのステップは Activity を呼ばずにスキップし、未完了のステップから再開する。
// これによりクラッシュ後の再実行でも完了済みフェーズを二重実行しない。
func (e *Engine) Execute(ctx context.Context, wf Workflow) (Result, error) {
	done, err := e.journal.Load()
	if err != nil {
		return Result{}, liberrors.Wrap(liberrors.CodeInternal, err, "load journal")
	}
	completedByName := make(map[string]Event, len(done))
	for _, ev := range done {
		completedByName[ev.StepName] = ev
	}

	result := Result{Completed: make([]Event, 0, len(wf.Steps))}

	for i, step := range wf.Steps {
		name := step.Activity.Name()

		if ev, ok := completedByName[name]; ok {
			e.log("[%s] step %d/%d %q: skip (journal hit)", wf.Name, i+1, len(wf.Steps), name)
			result.Completed = append(result.Completed, ev)
			continue
		}

		ev, err := e.runStep(ctx, wf.Name, i, len(wf.Steps), step)
		if err != nil {
			return result, err
		}
		if err := e.journal.Append(ev); err != nil {
			return result, liberrors.Wrap(liberrors.CodeInternal, err, "append journal")
		}
		result.Completed = append(result.Completed, ev)
	}

	e.log("[%s] workflow completed (%d steps)", wf.Name, len(result.Completed))
	return result, nil
}

// runStep は 1 ステップを RetryPolicy に従って試行する。WHY: 失敗は該当ステップ内で
// 再試行し、最大試行数を超えたら error を返してワークフロー全体を停止する。
func (e *Engine) runStep(ctx context.Context, wf string, idx, total int, step Step) (Event, error) {
	name := step.Activity.Name()
	policy := step.Retry.normalized()

	var lastErr error
	for attempt := 1; attempt <= policy.MaximumAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return Event{}, liberrors.Wrap(liberrors.CodeUnavailable, err, "context canceled at %q", name)
		}

		res, err := step.Activity.Execute(ctx, step.Input)
		if err == nil {
			e.log("[%s] step %d/%d %q: ok (attempt %d)", wf, idx+1, total, name, attempt)
			return Event{StepName: name, Attempts: attempt, Result: res}, nil
		}

		lastErr = err
		e.log("[%s] step %d/%d %q: fail attempt %d/%d: %v", wf, idx+1, total, name, attempt, policy.MaximumAttempts, err)

		if attempt < policy.MaximumAttempts {
			e.sleep(policy.backoff(attempt))
		}
	}

	return Event{}, liberrors.Wrap(liberrors.CodeUnavailable, lastErr,
		"step %q exhausted %d attempts", name, policy.MaximumAttempts)
}
