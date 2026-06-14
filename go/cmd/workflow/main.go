// Package main はミニ耐久ワークフローエンジンのデモエントリ。
// 完全オフラインで (1) 正常実行 (2) 途中失敗→リトライ (3) クラッシュ→再開 を表示する。
//
// production では本物の Temporal Go SDK（go.temporal.io/sdk）に差し替える。
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hiro8ma/agent/go/internal/workflow"
)

func main() {
	ctx := context.Background()
	logln := func(format string, args ...any) { fmt.Printf(format+"\n", args...) }

	demoHappyPath(ctx, logln)
	fmt.Println()
	demoRetry(ctx, logln)
	fmt.Println()
	demoCrashResume(ctx, logln)
}

// demoHappyPath は全ステップが 1 回で完了するケース。
func demoHappyPath(ctx context.Context, log workflow.Logger) {
	fmt.Println("=== (1) happy path ===")
	e := workflow.NewEngine(workflow.NewMemoryJournal(), log)
	if _, err := e.Execute(ctx, workflow.SupplyChainWorkflow(workflow.DefaultRetryPolicy())); err != nil {
		fmt.Println("error:", err)
	}
}

// demoRetry は transportation を 2 回失敗→3 回目成功にし、該当ステップだけ再試行されることを示す。
func demoRetry(ctx context.Context, log workflow.Logger) {
	fmt.Println("=== (2) step retry ===")
	wf := workflow.SupplyChainWorkflow(workflow.DefaultRetryPolicy())
	wf.Steps[1].Activity = &workflow.FlakyActivity{
		Inner:            workflow.NewTransportationActivity(),
		FailBeforeSucces: 2,
	}
	e := workflow.NewEngine(workflow.NewMemoryJournal(), log)
	if _, err := e.Execute(ctx, wf); err != nil {
		fmt.Println("error:", err)
	}
}

// demoCrashResume は supplier 到達時にクラッシュ→別プロセス相当の再実行で
// ジャーナルから完了済みをスキップし残りを完了させる。
func demoCrashResume(ctx context.Context, log workflow.Logger) {
	fmt.Println("=== (3) crash then resume ===")
	path := filepath.Join(os.TempDir(), "miniworkflow-demo-journal.json")
	_ = os.Remove(path) // デモ再現のため前回ジャーナルを消す

	fmt.Println("-- first run (crashes at supplier) --")
	wf := workflow.SupplyChainWorkflow(workflow.RetryPolicy{MaximumAttempts: 1})
	wf.Steps[2].Activity = workflow.CrashActivity{Inner: workflow.NewSupplierActivity()}
	e1 := workflow.NewEngine(workflow.NewFileJournal(path), log)
	if _, err := e1.Execute(ctx, wf); err != nil {
		fmt.Println("crashed:", err)
	}

	fmt.Println("-- second run (resumes from journal) --")
	e2 := workflow.NewEngine(workflow.NewFileJournal(path), log)
	res, err := e2.Execute(ctx, workflow.SupplyChainWorkflow(workflow.RetryPolicy{MaximumAttempts: 1}))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("resume done: %d steps completed\n", len(res.Completed))
	_ = os.Remove(path)
}
