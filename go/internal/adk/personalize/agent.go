// Package personalize は利用者ごとに答え方を変える技術アシスタントを組み立てる。
//
// 利用者の輪郭（技術レベル / 使う言語 / 回答の長さ / 関心領域）を user: の State に溜め、
// 毎回の system instruction に差し込む。user: は同じ利用者の別 Session へ引き継がれ、
// 別の利用者には渡らない。
//
// 輪郭の書き込みはツールからだけ行う。コールバックの State では保存されない経路がある。
// 書ける項目は domain.ProfileField で許可制にし、権限の鍵を書き換えられないようにしている。
package personalize

import (
	"errors"
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"github.com/hiro8ma/agent/go/internal/adk/domain"
)

const baseInstruction = `あなたは開発の相談に乗る技術アシスタントです。
利用者の技術レベル、主に使う言語、好む回答の長さ、関心のある領域が会話から分かったら、remember_profile で記録してください。
記録できる項目は expertise / language / answer_style / focus だけです。
expertise は beginner / intermediate / advanced、answer_style は concise / detailed のどれかです。
利用者が明言していないことを推測して記録しないでください。`

// ReadProfile は State から利用者の輪郭を読む。
func ReadProfile(st session.ReadonlyState) domain.Profile {
	get := func(f domain.ProfileField) string {
		if st == nil {
			return ""
		}
		v, err := st.Get(f.StateKey())
		if err != nil {
			return ""
		}
		s, _ := v.(string)
		return s
	}
	return domain.Profile{
		Expertise:   get(domain.FieldExpertise),
		Language:    get(domain.FieldLanguage),
		AnswerStyle: get(domain.FieldAnswerStyle),
		Focus:       get(domain.FieldFocus),
	}
}

// BuildInstruction は基本の指示に、利用者の輪郭を描き出した文を足す。
func BuildInstruction(ctx agent.ReadonlyContext) (string, error) {
	var st session.ReadonlyState
	if ctx != nil {
		st = ctx.ReadonlyState()
	}
	profile := ReadProfile(st).Instruction()
	if profile == "" {
		return baseInstruction, nil
	}
	return baseInstruction + "\n\n" + profile, nil
}

// RememberInput はツールの入力。
type RememberInput struct {
	// Field は記録する項目。expertise / language / answer_style / focus のどれか。
	Field string `json:"field"`
	// Value は記録する値。
	Value string `json:"value"`
}

// RememberOutput はツールの出力。
//
// 拒否もエラーではなく結果として返す。モデルが理由を読んで言い直せるようにする。
type RememberOutput struct {
	Saved bool   `json:"saved"`
	Field string `json:"field"`
	Value string `json:"value,omitempty"`
	Error string `json:"error,omitempty"`
}

// Remember は利用者の輪郭を 1 項目記録する。
func Remember(ctx agent.Context, in RememberInput) (RememberOutput, error) {
	field, err := domain.ParseProfileField(strings.TrimSpace(in.Field))
	if err != nil {
		return RememberOutput{Field: in.Field, Error: err.Error()}, nil
	}
	value, err := field.Normalize(in.Value)
	if err != nil {
		return RememberOutput{Field: in.Field, Error: err.Error()}, nil
	}
	if ctx == nil || ctx.State() == nil {
		return RememberOutput{}, errors.New("state にアクセスできない")
	}
	if err := ctx.State().Set(field.StateKey(), value); err != nil {
		return RememberOutput{}, fmt.Errorf("輪郭の保存: %w", err)
	}
	return RememberOutput{Saved: true, Field: string(field), Value: value}, nil
}

// NewAgent は利用者ごとに答え方を変える技術アシスタントを組み立てる。
func NewAgent(m model.LLM) (agent.Agent, error) {
	rememberTool, err := functiontool.New(functiontool.Config{
		Name:        "remember_profile",
		Description: "利用者の輪郭を 1 項目記録する。記録できる項目は expertise / language / answer_style / focus だけ。",
	}, Remember)
	if err != nil {
		return nil, fmt.Errorf("remember_profile の作成: %w", err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:                "tech_assistant",
		Model:               m,
		Description:         "利用者ごとに答え方を変える技術アシスタント",
		InstructionProvider: BuildInstruction,
		Tools:               []tool.Tool{rememberTool},
	})
	if err != nil {
		return nil, fmt.Errorf("エージェントの作成: %w", err)
	}
	return a, nil
}
