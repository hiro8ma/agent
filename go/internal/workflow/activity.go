// Package workflow は Temporal の核（ステップ単位リトライ + 進捗チェックポイント +
// クラッシュからの再開）を外部サーバなしの純 Go で再現するミニ耐久ワークフローエンジン。
//
// 二層構造の前提:
//   - フェーズ遷移と耐久性 = このエンジン（ジャーナルで進捗を残し、途中から再開する）
//   - フェーズ内の LLM 往復 = 既存 internal/genkit（Activity 実装として後から注入する）
//
// production では本物の Temporal Go SDK（go.temporal.io/sdk）に差し替える。
// 今回は依存追加せず、標準ライブラリのみで核を再現する。
package workflow

import "context"

// ActivityInput は 1 Activity への入力。供給チェーン例の operation / messages を渡す。
type ActivityInput struct {
	Operation string         `json:"operation"`
	Messages  []string       `json:"messages"`
	Params    map[string]any `json:"params,omitempty"`
}

// ActivityResult は 1 Activity の実行結果。ジャーナルに記録され、再実行時のスキップ判定に使う。
type ActivityResult struct {
	Output map[string]any `json:"output"`
}

// Activity は冪等に再試行可能な処理単位。
// interface 分離により fake（決定論）と genkit の Ask（LLM 往復）を差し替えられる。
type Activity interface {
	// Name は Activity を一意に識別する。ジャーナルのイベントキーになる。
	Name() string
	// Execute は 1 回分の試行。error を返すとエンジンが RetryPolicy に従って再試行する。
	Execute(ctx context.Context, in ActivityInput) (ActivityResult, error)
}

// ActivityFunc は関数を Activity に適合させるアダプタ。
type ActivityFunc struct {
	ActivityName string
	Fn           func(ctx context.Context, in ActivityInput) (ActivityResult, error)
}

// Name は Activity を満たす。
func (a ActivityFunc) Name() string { return a.ActivityName }

// Execute は Activity を満たす。
func (a ActivityFunc) Execute(ctx context.Context, in ActivityInput) (ActivityResult, error) {
	return a.Fn(ctx, in)
}
