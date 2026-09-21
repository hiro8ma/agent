// Package adkeval は ADK の評価セット（*.evalset.json）を Go で読み、REST で立てたエージェントを採点する。
//
// Go の ADK には adk eval が無く、REST の評価の口も未実装のまま。
// Python の adk api_server と Go の ADK は同じ /run の口を持つので、
// ここから両方を同じ評価セットと同じ採点で比べる。
package adkeval

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// EvalSet は ADK の評価セット。Python の EvalSet が書き出す snake_case の JSON を読む。
type EvalSet struct {
	EvalSetID string     `json:"eval_set_id"`
	EvalCases []EvalCase `json:"eval_cases"`
}

// EvalCase は 1 件のケース。Conversation の順に利用者の発話を流すか、
// ConversationScenario に従って LLM が利用者役を演じる。
type EvalCase struct {
	EvalID               string       `json:"eval_id"`
	Conversation         []Invocation `json:"conversation"`
	ConversationScenario *Scenario    `json:"conversation_scenario,omitempty"`
}

// Scenario は利用者役に渡す台本。ADK の ConversationScenario と同じ形。
type Scenario struct {
	StartingPrompt   string   `json:"starting_prompt"`
	ConversationPlan string   `json:"conversation_plan"`
	UserPersona      *Persona `json:"user_persona,omitempty"`
}

// Persona は利用者役の振る舞い。ADK の UserPersona と同じ形。
//
// ViolationRubrics は利用者役がペルソナどおりに振る舞えたかを見る基準で、エージェントを採点する基準ではない。
// エージェントに起きてはいけないこと（アンチゴール）は Options.Forbidden で見る。
type Persona struct {
	ID          string     `json:"id"`
	Description string     `json:"description"`
	Behaviors   []Behavior `json:"behaviors"`
}

// UnmarshalJSON は、ペルソナが ID の文字列（"NOVICE" など）で書かれていたら止める。
// Python の ADK は読み込み時に組み込みのペルソナへ展開するが、Go は組み込みのペルソナの中身を持たない。
func (p *Persona) UnmarshalJSON(b []byte) error {
	var id string
	if json.Unmarshal(b, &id) == nil {
		return fmt.Errorf("adkeval: ペルソナ %q が ID のまま。Python の EvalSet で読み書きしてオブジェクトに展開する", id)
	}
	type plain Persona
	return json.Unmarshal(b, (*plain)(p))
}

// Behavior はペルソナの 1 つの振る舞い。
type Behavior struct {
	Name                 string   `json:"name"`
	Description          string   `json:"description"`
	BehaviorInstructions []string `json:"behavior_instructions"`
	ViolationRubrics     []string `json:"violation_rubrics"`
}

// Invocation は 1 ターン。評価セットでは期待値、推論の結果では実際の値を持つ。
type Invocation struct {
	UserContent      Content          `json:"user_content"`
	FinalResponse    *Content         `json:"final_response,omitempty"`
	IntermediateData IntermediateData `json:"intermediate_data"`
}

// IntermediateData はターンの途中のツール呼び出し。
type IntermediateData struct {
	ToolUses []ToolUse `json:"tool_uses"`
}

// ToolUse は 1 回のツール呼び出し。
type ToolUse struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// Content は genai の Content のうち、採点に使う部分。
type Content struct {
	Role  string `json:"role"`
	Parts []Part `json:"parts"`
}

// Part は genai の Part のうち、採点に使う部分。REST のイベントは camelCase で返る。
type Part struct {
	Text         string   `json:"text,omitempty"`
	FunctionCall *ToolUse `json:"functionCall,omitempty"`
}

// Text は text の部分をつなげて返す。
func (c *Content) Text() string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

// Load は評価セットを読み、ケースと発話が空でないことを確かめる。
func Load(path string) (*EvalSet, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // 評価セットのパスは実行する人が指定する
	if err != nil {
		return nil, fmt.Errorf("adkeval: 評価セットの読み込み: %w", err)
	}
	var set EvalSet
	if err := json.Unmarshal(raw, &set); err != nil {
		return nil, fmt.Errorf("adkeval: 評価セットの解析: %w", err)
	}
	if len(set.EvalCases) == 0 {
		return nil, fmt.Errorf("adkeval: %s にケースが無い", path)
	}
	for _, c := range set.EvalCases {
		if c.ConversationScenario != nil {
			if c.ConversationScenario.StartingPrompt == "" {
				return nil, fmt.Errorf("adkeval: %s の starting_prompt が空", c.EvalID)
			}
			continue
		}
		if len(c.Conversation) == 0 {
			return nil, fmt.Errorf("adkeval: %s の conversation も conversation_scenario も無い", c.EvalID)
		}
		for i, turn := range c.Conversation {
			if turn.UserContent.Text() == "" {
				return nil, fmt.Errorf("adkeval: %s の %d ターン目に利用者の発話が無い", c.EvalID, i+1)
			}
		}
	}
	return &set, nil
}
