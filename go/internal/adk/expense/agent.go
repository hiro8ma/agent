package expense

import (
	"regexp"
	"slices"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/guardrail"
)

// ModelName は経費エージェントのモデル。
const ModelName = "gemini-3.8-flash"

// InjectionPatterns は共通のパターンに、経費エージェントで確かめる形を足したもの。
var InjectionPatterns = append(slices.Clone(guardrail.InjectionPatterns),
	regexp.MustCompile(`(前|以前|これまで)の指示を(すべて)?(忘れ|無視)`),
	regexp.MustCompile(`(管理者|開発者|デバッグ)モード`),
	// 大文字小文字を無視すると「Dan さん」のような人名で止まる。
	regexp.MustCompile(`\bDAN\b|(?i:do\s+anything\s+now)`),
	regexp.MustCompile(`制限(を|の)?(解除|無視|なし)`),
)

const instruction = `あなたは経費精算の担当です。経費の申請、照会、承認を扱います。
- 申請は submit_expense を使う。category は費目名（交通費、会議費など）、description は業務目的や補足だけにする
- 照会は query_expenses を使う
- 承認は approve_expense を使う
- 50 万円以上の申請は承認待ちになる。承認待ちになったら、その旨と経費の ID を伝える
- 経費と関係の無い依頼には答えない`

// NewAgent は経費エージェントを組み立てる。
func NewAgent(m model.LLM, d Deps, log *guardrail.Log) (agent.Agent, error) {
	tools, err := d.Tools()
	if err != nil {
		return nil, err
	}
	zero := float32(0)
	return llmagent.New(llmagent.Config{
		Name:        "expense_agent",
		Model:       m,
		Instruction: instruction,
		Tools:       tools,
		// 評価のために揺れを減らす。Gemini では 0 にしても同じ出力になる保証は無い。
		GenerateContentConfig: &genai.GenerateContentConfig{Temperature: &zero},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			guardrail.DetectInjection(log, InjectionPatterns, "その依頼には応じられません。経費の申請、照会、承認についてお尋ねください。"),
		},
		AfterModelCallbacks: []llmagent.AfterModelCallback{guardrail.MaskPII(log)},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
			guardrail.FallbackOnModelError(log, "いま処理できません。少し時間をおいて試してください。"),
		},
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{guardrail.StructureToolError(log)},
	})
}
