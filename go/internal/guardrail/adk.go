// ADK 側の適用。コールバック 4 点に規則を載せる。
//
// どの段の検査かがシグネチャで決まる代わりに、
// before で見た値を after で使うには外の状態が要る。
package guardrail

import (
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// BlockInput は禁止語を含む入力を LLM へ渡す前に止める。
//
// LLMResponse を返すとモデルの呼び出しを飛ばし、その内容が応答になる。
func BlockInput(log *Log, banned []string) llmagent.BeforeModelCallback {
	return func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		text := requestText(req)
		for _, w := range banned {
			if w != "" && strings.Contains(text, w) {
				log.add(Verdict{Stage: "before_model", Rule: "禁止語", Blocked: true,
					Detail: fmt.Sprintf("入力に %q が含まれる", w)})
				return refuse(fmt.Sprintf("この内容には回答できません（%s）", w)), nil
			}
		}
		log.add(Verdict{Stage: "before_model"})
		return nil, nil
	}
}

// RedactOutput は出力に現れた機微な語を伏せる。
// 止めずに書き換え、外へ出してはいけない部分だけを消す。
func RedactOutput(log *Log, secrets []string) llmagent.AfterModelCallback {
	return func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err != nil || resp == nil || resp.Content == nil {
			return nil, nil
		}
		hit := ""
		for i, part := range resp.Content.Parts {
			if part == nil || part.Text == "" {
				continue
			}
			t := part.Text
			for _, s := range secrets {
				if s != "" && strings.Contains(t, s) {
					t = strings.ReplaceAll(t, s, "***")
					hit = s
				}
			}
			resp.Content.Parts[i].Text = t
		}
		if hit != "" {
			log.add(Verdict{Stage: "after_model", Rule: "伏せ字", Blocked: true,
				Detail: fmt.Sprintf("出力の %q を伏せた", hit)})
			return resp, nil
		}
		log.add(Verdict{Stage: "after_model"})
		return nil, nil
	}
}

// RequireArgs はツールの必須引数が埋まっているかを実行前に確かめる。
//
// 引数が落ちたまま実行すると、ツール側が空文字を既定値として扱い
// それらしい結果を返す。実行前に止めれば落ちたことが記録に残る。
func RequireArgs(log *Log, toolName string, required ...string) llmagent.BeforeToolCallback {
	return func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
		if t.Name() != toolName {
			log.add(Verdict{Stage: "before_tool"})
			return nil, nil
		}
		for _, k := range required {
			v, ok := args[k]
			if !ok || v == nil || v == "" {
				log.add(Verdict{Stage: "before_tool", Rule: "必須引数", Blocked: true,
					Detail: fmt.Sprintf("%s に %s が無い", toolName, k)})
				// map を返すとツールを呼ばず、その値が結果になる。
				return map[string]any{
					"error": fmt.Sprintf("%s は必須です。指定してから呼び直してください", k),
				}, nil
			}
		}
		log.add(Verdict{Stage: "before_tool"})
		return nil, nil
	}
}

// RejectEmptyResult はツールが空を返したことを失敗として扱う。
//
// 検索の 0 件はエラーにならず、そのまま次の判定へ流れる。
func RejectEmptyResult(log *Log, keys ...string) llmagent.AfterToolCallback {
	return func(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
		if err != nil {
			log.add(Verdict{Stage: "after_tool", Rule: "実行失敗", Blocked: true,
				Detail: fmt.Sprintf("%s: %v", t.Name(), err)})
			return nil, nil
		}
		if empty(result, keys) {
			log.add(Verdict{Stage: "after_tool", Rule: "空の結果", Blocked: true,
				Detail: fmt.Sprintf("%s が空を返した", t.Name())})
			return map[string]any{
				"error": "結果が空でした。条件を変えて呼び直すか、分からないと答えてください",
			}, nil
		}
		log.add(Verdict{Stage: "after_tool"})
		return nil, nil
	}
}

// refuse はモデルを呼ばずに返す応答を作る。
func refuse(msg string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: msg}},
		},
	}
}

func requestText(req *model.LLMRequest) string {
	if req == nil {
		return ""
	}
	var b strings.Builder
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil {
				b.WriteString(p.Text)
			}
		}
	}
	return b.String()
}
