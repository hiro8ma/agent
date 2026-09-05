package hitl

import (
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// Request は承認を求める対象。
type Request struct {
	Action string         `json:"action"`
	Args   map[string]any `json:"args"`
}

// Result は判断の結果。
//
// Status は allowed / denied / approved / rejected のいずれか。
// Allow と Deny は人を待たずに決まるので allowed / denied、
// 人に聞いた結果は approved / rejected になる。
// 4 つに分けるのは「聞かずに通した」と「聞いて通した」を
// 事後に区別するため。
type Result struct {
	Status string `json:"status"`
	Reason string `json:"reason,omitempty"`
	Action string `json:"action,omitempty"`
}

// ApprovalNode は Policy の判断を ADK の中断と再開へ接続する。
//
// Gate は自前で止めるところまで持っていたが、こちらは中断と再開を
// Workflow に委ねる。セッションをまたいだ承認待ちが扱えるようになる。
//
// RerunOnResume を &true にするのは ResumeOrRequestInput が再実行を
// 前提にしているため。既定の nil は handoff で、人の返答は後続ノードの
// 入力になり、このノードは二度と実行されない。
func ApprovalNode(name string, policy Policy) *workflow.FunctionNode {
	rerun := true
	return workflow.NewEmittingFunctionNode[Request, Result](name,
		func(ctx agent.Context, in Request, emit func(*session.Event) error) (Result, error) {
			decision, reason := policy(in.Args)
			switch decision {
			case Allow:
				return Result{Status: "allowed", Action: in.Action}, nil
			case Deny:
				return Result{Status: "denied", Reason: reason, Action: in.Action}, nil
			}

			// InterruptID に InvocationID を混ぜる。固定文字列にすると
			// 同じセッションの 2 回目以降で再確認が出なくなる。
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: name + "-" + ctx.InvocationID(),
				Message:     fmt.Sprintf("%s を実行してよいですか。%s", in.Action, reason),
			})
			if err != nil {
				return Result{}, err
			}
			if approved(reply) {
				return Result{Status: "approved", Reason: reason, Action: in.Action}, nil
			}
			return Result{Status: "rejected", Reason: "利用者が承認しなかった", Action: in.Action}, nil
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)
}

// approved は人の返答を可否へ落とす。
//
// 判定できない返答は拒否側に倒す。承認は明示されたときだけ成立させる。
func approved(reply any) bool {
	switch v := reply.(type) {
	case bool:
		return v
	case string:
		switch v {
		case "yes", "y", "ok", "approve", "approved", "true", "はい", "承認":
			return true
		}
	case map[string]any:
		if b, ok := v["approved"].(bool); ok {
			return b
		}
		// console launcher は平文を {"payload": "..."} で包んで返す。
		if inner, ok := v["payload"]; ok {
			return approved(inner)
		}
	}
	return false
}
