// Package dynamicflow は分岐とループをコードで書く Workflow。
//
// Graph-based Workflow は順序を edges で固定する。
// こちらは 1 つのノードの中で RunNode を呼び、
// ループと打ち切りを Go の for と if で書く。
//
//	下書き → 評価 → 承認なら終了 / 未承認なら修正して再評価（最大 3 周）
//
// 回数の上限をコードで持つのが要点になる。
// 「承認されるまで繰り返す」を LLM に任せると、止まる保証が無い。
package dynamicflow

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// MaxRounds は評価を繰り返す上限。
const MaxRounds = 3

// Approved は評価者が承認を表す語。
const Approved = "APPROVED"

// Result はループの結果。
//
// 何周したかと、承認で終わったのか上限で終わったのかを残す。
// 下書きだけ返すと、通ったのか諦めたのかが区別できない。
type Result struct {
	Draft    string `json:"draft"`
	Rounds   int    `json:"rounds"`
	Approved bool   `json:"approved"`
}

// New は下書きと評価のノードから、ループするノードを組む。
func New(draft, review workflow.Node) workflow.Node {
	rerun := true
	return workflow.NewDynamicNode[string, Result]("review_loop",
		func(ctx agent.Context, request string, emit func(*session.Event) error) (Result, error) {
			text, err := workflow.RunNode[string](ctx, draft, request)
			if err != nil {
				return Result{}, fmt.Errorf("first draft: %w", err)
			}

			for round := 1; round <= MaxRounds; round++ {
				verdict, err := workflow.RunNode[string](ctx, review, text)
				if err != nil {
					return Result{}, fmt.Errorf("review round %d: %w", round, err)
				}
				if strings.Contains(verdict, Approved) {
					return Result{Draft: text, Rounds: round, Approved: true}, nil
				}
				text, err = workflow.RunNode[string](ctx, draft,
					fmt.Sprintf("修正指摘: %s\n元の下書き: %s", verdict, text))
				if err != nil {
					return Result{}, fmt.Errorf("revise round %d: %w", round, err)
				}
			}
			return Result{Draft: text, Rounds: MaxRounds, Approved: false}, nil
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)
}

// NewWithApproval は初回の評価のあとに人の承認を挟むループを組む。
//
// 教材は rerun_on_resume=True にすると「完了済みの子ノードを無駄に
// 再実行せず」再開できると書いている。RerunOnResume=&true の意味は
// 「中断したノードを最初から再実行する」なので、ループ全体が
// やり直されるなら子も再実行されるはず。どちらが起きるかを
// 子ノードの呼び出し回数で判定する。
func NewWithApproval(draft, review workflow.Node) workflow.Node {
	rerun := true
	return workflow.NewDynamicNode[string, Result]("review_loop_hitl",
		func(ctx agent.Context, request string, emit func(*session.Event) error) (Result, error) {
			text, err := workflow.RunNode[string](ctx, draft, request)
			if err != nil {
				return Result{}, fmt.Errorf("first draft: %w", err)
			}
			verdict, err := workflow.RunNode[string](ctx, review, text)
			if err != nil {
				return Result{}, fmt.Errorf("first review: %w", err)
			}

			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: "publish-" + ctx.InvocationID(),
				Message:     fmt.Sprintf("この評価で公開してよいですか。%s", verdict),
			})
			if err != nil {
				return Result{}, err
			}
			if !accepted(reply) {
				return Result{Draft: text, Rounds: 1, Approved: false}, nil
			}
			return Result{Draft: text, Rounds: 1, Approved: true}, nil
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)
}

func accepted(reply any) bool {
	switch v := reply.(type) {
	case bool:
		return v
	case string:
		return v == "はい" || v == "yes" || v == "ok"
	case map[string]any:
		if inner, ok := v["payload"]; ok {
			return accepted(inner)
		}
	}
	return false
}

// NewHandoff は重い処理を、中断するノードより前で完了させる。
//
// 再入では中断したノードが最初から走るため、その中の RunNode も
// もう一度走る。中断の手前に置いた処理は、別のノードとして
// 完了させておけば再実行されない。
//
//	expensive（下書きと評価）→ ask（聞くだけ）→ finish（返答を受け取る）
//
// ask は返答を読まない。RerunOnResume を既定のままにすると
// handoff になり、返答は後続ノードの入力として渡る。
//
// 代償として、finish は返答しか受け取らない。上流の値も要るなら
// state へ置く。再入は上流の値を持ち回せるが、走り直す。
func NewHandoff(draft, review workflow.Node) []workflow.Edge {
	// RunNode は dynamic node の中でしか呼べない。
	// 関数ノードにすると "RunNode called outside a dynamic node" で落ちる。
	expensive := workflow.NewDynamicNode[string, string]("expensive",
		func(ctx agent.Context, request string, _ func(*session.Event) error) (string, error) {
			text, err := workflow.RunNode[string](ctx, draft, request)
			if err != nil {
				return "", fmt.Errorf("draft: %w", err)
			}
			verdict, err := workflow.RunNode[string](ctx, review, text)
			if err != nil {
				return "", fmt.Errorf("review: %w", err)
			}
			return text + "\n評価: " + verdict, nil
		},
		workflow.NodeConfig{},
	)

	ask := workflow.NewEmittingFunctionNode[string, string]("ask",
		func(ctx agent.Context, in string, emit func(*session.Event) error) (string, error) {
			// 返答を読まない。読むと再入が要る。
			if err := emit(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "publish-" + ctx.InvocationID(),
				Message:     "公開してよいですか。" + in,
			})); err != nil {
				return "", err
			}
			return "", workflow.ErrNodeInterrupted
		},
		workflow.NodeConfig{},
	)

	finish := workflow.NewFunctionNode[any, Result]("finish",
		func(_ agent.Context, reply any) (Result, error) {
			return Result{Rounds: 1, Approved: accepted(reply)}, nil
		},
		workflow.NodeConfig{},
	)

	return workflow.Chain(workflow.Start, expensive, ask, finish)
}
