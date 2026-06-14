package workflow

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func newTestEngine(j Journal) *Engine {
	e := NewEngine(j, nil)
	e.sleep = func(time.Duration) {} // バックオフ待機を無効化して高速にテストする
	return e
}

func TestExecute_HappyPath(t *testing.T) {
	j := NewMemoryJournal()
	e := newTestEngine(j)

	res, err := e.Execute(context.Background(), SupplyChainWorkflow(DefaultRetryPolicy()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Completed) != 3 {
		t.Fatalf("want 3 completed steps, got %d", len(res.Completed))
	}
	want := []string{ActivityInventory, ActivityTransportation, ActivitySupplier}
	for i, ev := range res.Completed {
		if ev.StepName != want[i] {
			t.Errorf("step %d: want %q, got %q", i, want[i], ev.StepName)
		}
		if ev.Attempts != 1 {
			t.Errorf("step %q: want 1 attempt, got %d", ev.StepName, ev.Attempts)
		}
	}
}

func TestExecute_StepRetrySucceeds(t *testing.T) {
	j := NewMemoryJournal()
	e := newTestEngine(j)

	wf := SupplyChainWorkflow(DefaultRetryPolicy())
	// transportation を 2 回失敗→3 回目成功にする。該当ステップだけ再試行されること。
	wf.Steps[1].Activity = &FlakyActivity{Inner: transportationActivity{}, FailBeforeSucces: 2}

	res, err := e.Execute(context.Background(), wf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := res.Completed[1].Attempts; got != 3 {
		t.Fatalf("transportation: want 3 attempts, got %d", got)
	}
	// inventory は 1 回で成功している（全体やり直しが起きていない）。
	if got := res.Completed[0].Attempts; got != 1 {
		t.Fatalf("inventory should run once, got %d attempts", got)
	}
}

func TestExecute_RetryExhausted(t *testing.T) {
	j := NewMemoryJournal()
	e := newTestEngine(j)

	wf := SupplyChainWorkflow(RetryPolicy{MaximumAttempts: 2})
	wf.Steps[0].Activity = &FlakyActivity{Inner: inventoryActivity{}, FailBeforeSucces: 99}

	_, err := e.Execute(context.Background(), wf)
	if err == nil {
		t.Fatal("want error when attempts exhausted, got nil")
	}
	// 失敗ステップはジャーナルに残らない。
	events, _ := j.Load()
	if len(events) != 0 {
		t.Fatalf("want empty journal on failure, got %d events", len(events))
	}
}

func TestExecute_CrashThenResume(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.json")

	// 1st プロセス: supplier で「クラッシュ」して止まる。
	{
		j := NewFileJournal(path)
		e := newTestEngine(j)
		wf := SupplyChainWorkflow(RetryPolicy{MaximumAttempts: 1})
		wf.Steps[2].Activity = CrashActivity{Inner: supplierActivity{}}

		_, err := e.Execute(context.Background(), wf)
		if err == nil {
			t.Fatal("want crash error on first run")
		}
	}

	// ジャーナルには inventory + transportation の 2 件だけ残るはず。
	events, err := NewFileJournal(path).Load()
	if err != nil {
		t.Fatalf("load journal: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("want 2 checkpointed steps after crash, got %d", len(events))
	}

	// 2nd プロセス: 別エンジンで再実行。完了済み 2 件はスキップ、supplier だけ実行。
	var executed []string
	track := func(name string, inner Activity) Activity {
		return ActivityFunc{ActivityName: name, Fn: func(ctx context.Context, in ActivityInput) (ActivityResult, error) {
			executed = append(executed, name)
			return inner.Execute(ctx, in)
		}}
	}

	j := NewFileJournal(path)
	e := newTestEngine(j)
	wf := Workflow{Name: "supply-chain", Steps: []Step{
		{Activity: track(ActivityInventory, inventoryActivity{}), Retry: RetryPolicy{MaximumAttempts: 1}},
		{Activity: track(ActivityTransportation, transportationActivity{}), Retry: RetryPolicy{MaximumAttempts: 1}},
		{Activity: track(ActivitySupplier, supplierActivity{}), Retry: RetryPolicy{MaximumAttempts: 1}},
	}}

	res, err := e.Execute(context.Background(), wf)
	if err != nil {
		t.Fatalf("resume failed: %v", err)
	}
	if len(res.Completed) != 3 {
		t.Fatalf("want 3 completed after resume, got %d", len(res.Completed))
	}
	if len(executed) != 1 || executed[0] != ActivitySupplier {
		t.Fatalf("resume should run only supplier, ran: %v", executed)
	}
}
