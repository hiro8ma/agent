// Package supportcontext はマルチエージェントでの Instruction の配置を扱う。
package supportcontext

import (
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"

	"github.com/hiro8ma/agent/go/internal/adkbook/blackboard"
)

const GlobalPolicy = "顧客の認証情報や決済情報を要求しません。" +
	"確証のない内容を事実として断定しません。" +
	"担当外の操作は実行せず、人間の担当者へ引き継ぎます。"

var StateSchema = blackboard.Schema{
	"policy_version": blackboard.App,
	"display_name":   blackboard.User,
	"issue_kind":     blackboard.Session,
	"order_result":   blackboard.Invocation,
}

func SupportInstruction(ctx agent.ReadonlyContext) (string, error) {
	displayName := "お客様"
	issueKind := "未分類"
	if ctx != nil && ctx.ReadonlyState() != nil {
		if value, err := ctx.ReadonlyState().Get(StateSchema.MustKey("display_name")); err == nil {
			if name, ok := value.(string); ok && name != "" {
				displayName = name
			}
		}
		if value, err := ctx.ReadonlyState().Get(StateSchema.MustKey("issue_kind")); err == nil {
			if kind, ok := value.(string); ok && kind != "" {
				issueKind = kind
			}
		}
	}
	return "あなたは技術サポート担当です。" +
		fmt.Sprintf("対応中の利用者は %s です。", displayName) +
		fmt.Sprintf("問い合わせ分類は %s です。", issueKind) +
		"利用者が提示した事実とツール結果だけを使って回答してください。", nil
}

func orderConfig(m model.LLM) llmagent.Config {
	return llmagent.Config{
		Name:        "order_agent",
		Model:       m,
		Description: "注文状況と返品手続きを扱う",
		Instruction: "注文番号を確認し、注文ツールの結果だけを使って案内してください。" +
			"注文内容を推測しないでください。",
	}
}

func supportConfig(m model.LLM) llmagent.Config {
	return llmagent.Config{
		Name:                "support_agent",
		Model:               m,
		Description:         "技術的な問い合わせを扱う",
		InstructionProvider: SupportInstruction,
	}
}

func rootConfig(m model.LLM, children []agent.Agent) llmagent.Config {
	return llmagent.Config{
		Name:              "support_router",
		Model:             m,
		Description:       "問い合わせを注文管理または技術サポートへ振り分ける",
		Instruction:       "問い合わせ内容に対応する担当へ振り分けてください。",
		GlobalInstruction: GlobalPolicy,
		SubAgents:         children,
	}
}

func New(m model.LLM) (agent.Agent, error) {
	orderAgent, err := llmagent.New(orderConfig(m))
	if err != nil {
		return nil, fmt.Errorf("注文エージェントの作成: %w", err)
	}
	supportAgent, err := llmagent.New(supportConfig(m))
	if err != nil {
		return nil, fmt.Errorf("技術サポートエージェントの作成: %w", err)
	}
	rootAgent, err := llmagent.New(rootConfig(m, []agent.Agent{orderAgent, supportAgent}))
	if err != nil {
		return nil, fmt.Errorf("ルートエージェントの作成: %w", err)
	}
	return rootAgent, nil
}
