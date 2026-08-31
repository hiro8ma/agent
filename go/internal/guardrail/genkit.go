// Genkit 側の適用。同じ規則を 2 つのラッパーへ載せる。
//
// ADK が before/after × model/tool の 4 点に分けるのに対し、
// Genkit は WrapModel と WrapTool の 2 つで包む。
// 前と後は 1 つの関数の中で next の前後に置く。
//
//	ADK     4 つの関数。どの段かがシグネチャで決まる
//	Genkit  2 つの関数。next の呼び出しで前後が分かれる
//
// 包む形は、前で見た値を後でそのまま使える。
// ADK では before で見た値を after へ渡すのに外の状態が要る。
// 分ける形は、書き分けが強制される代わりに受け渡しに手間が乗る。
//
// どちらも「止める」は同じ考え方になる。
// ADK は非 nil を返して次を飛ばし、Genkit は next を呼ばずに返す。
package guardrail

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// ModelGuard は LLM 呼び出しの前後を検査するフックを返す。
//
// banned は入力に含まれてはいけない語、
// secrets は出力に現れたら伏せる語になる。
func ModelGuard(log *Log, banned, secrets []string) func(context.Context, *ai.ModelParams, ai.ModelNext) (*ai.ModelResponse, error) {
	return func(ctx context.Context, p *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
		// next の前が ADK の before_model にあたる。
		text := promptText(p.Request)
		for _, w := range banned {
			if w != "" && strings.Contains(text, w) {
				log.add(Verdict{Stage: "before_model", Rule: "禁止語", Blocked: true,
					Detail: fmt.Sprintf("入力に %q が含まれる", w)})
				// next を呼ばずに返すとモデルを呼ばない。
				return &ai.ModelResponse{
					Message: ai.NewModelTextMessage(
						fmt.Sprintf("この内容には回答できません（%s）", w)),
				}, nil
			}
		}
		log.add(Verdict{Stage: "before_model"})

		resp, err := next(ctx, p)
		if err != nil {
			return nil, err
		}

		// next の後が after_model にあたる。
		// 前で見た text がそのまま使える。ADK では外の状態が要る。
		hit := ""
		if resp != nil && resp.Message != nil {
			for _, part := range resp.Message.Content {
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
				part.Text = t
			}
		}
		if hit != "" {
			log.add(Verdict{Stage: "after_model", Rule: "伏せ字", Blocked: true,
				Detail: fmt.Sprintf("出力の %q を伏せた", hit)})
		} else {
			log.add(Verdict{Stage: "after_model"})
		}
		return resp, nil
	}
}

// ToolGuard はツール実行の前後を検査するフックを返す。
//
// required はツール名ごとの必須引数、
// emptyKeys は空だったら失敗として扱う出力の鍵になる。
func ToolGuard(log *Log, required map[string][]string, emptyKeys ...string) func(context.Context, *ai.ToolParams, ai.ToolNext) (*ai.MultipartToolResponse, error) {
	return func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
		name := ""
		if p.Request != nil {
			name = p.Request.Name
		}

		// before_tool にあたる。
		if keys, ok := required[name]; ok {
			args, _ := p.Request.Input.(map[string]any)
			for _, k := range keys {
				v, present := args[k]
				if !present || v == nil || v == "" {
					log.add(Verdict{Stage: "before_tool", Rule: "必須引数", Blocked: true,
						Detail: fmt.Sprintf("%s に %s が無い", name, k)})
					return &ai.MultipartToolResponse{Output: map[string]any{
						"error": fmt.Sprintf("%s は必須です。指定してから呼び直してください", k),
					}}, nil
				}
			}
		}
		log.add(Verdict{Stage: "before_tool"})

		res, err := next(ctx, p)
		if err != nil {
			log.add(Verdict{Stage: "after_tool", Rule: "実行失敗", Blocked: true,
				Detail: fmt.Sprintf("%s: %v", name, err)})
			return nil, err
		}

		// after_tool にあたる。空を成功として通さない。
		var out map[string]any
		if res != nil {
			out, _ = res.Output.(map[string]any)
		}
		if empty(out, emptyKeys) {
			log.add(Verdict{Stage: "after_tool", Rule: "空の結果", Blocked: true,
				Detail: fmt.Sprintf("%s が空を返した", name)})
			return &ai.MultipartToolResponse{Output: map[string]any{
				"error": "結果が空でした。条件を変えて呼び直すか、分からないと答えてください",
			}}, nil
		}
		log.add(Verdict{Stage: "after_tool"})
		return res, nil
	}
}

// Use は 2 つのフックを Genkit のミドルウェアへまとめる。
//
// ai.WithUse(guardrail.Use(log, ...)) の形で Generate に渡す。
func Use(log *Log, banned, secrets []string, required map[string][]string, emptyKeys ...string) ai.MiddlewareFunc {
	return func(context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapModel: ModelGuard(log, banned, secrets),
			WrapTool:  ToolGuard(log, required, emptyKeys...),
		}, nil
	}
}

func promptText(req *ai.ModelRequest) string {
	if req == nil {
		return ""
	}
	var b strings.Builder
	for _, m := range req.Messages {
		if m == nil {
			continue
		}
		for _, p := range m.Content {
			if p != nil {
				b.WriteString(p.Text)
			}
		}
	}
	return b.String()
}
