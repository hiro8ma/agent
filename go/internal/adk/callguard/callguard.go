// Package callguard は 1 回の Invocation の中で、モデルの呼び出しの総数と、同じツールを同じ引数で呼ぶ回数に上限を置く。
//
// ADK Go v2.2.0 の通常の実行には、モデルの呼び出しの上限が無い（MaxLLMCalls は Live と REST のサーバーだけ）。
// 同じツールを呼び続けるモデルは、context が切れるまで止まらない。
package callguard

import (
	"encoding/json"
	"fmt"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

// Limits は上限。0 は上限なし。
type Limits struct {
	MaxLLMCalls      int
	MaxSameToolCalls int
}

type state struct {
	llmCalls int
	same     map[string]int
	stopped  string
}

// New は上限を超えたら打ち切るプラグインを返す。
//
// 同じ呼び出しが上限を超えたら、そのツールを実行せずに理由を返し、次のモデルの呼び出しの代わりに打ち切りの応答を返す。
func New(l Limits) (*plugin.Plugin, error) {
	var mu sync.Mutex
	states := map[string]*state{}
	get := func(id string) *state {
		s, ok := states[id]
		if !ok {
			s = &state{same: map[string]int{}}
			states[id] = s
		}
		return s
	}
	stop := func(reason string) *model.LLMResponse {
		return &model.LLMResponse{Content: genai.NewContentFromText("処理を打ち切りました: "+reason, genai.RoleModel)}
	}
	return plugin.New(plugin.Config{
		Name: "callguard",
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			mu.Lock()
			defer mu.Unlock()
			s := get(ctx.InvocationID())
			if s.stopped != "" {
				return stop(s.stopped), nil
			}
			s.llmCalls++
			if l.MaxLLMCalls > 0 && s.llmCalls > l.MaxLLMCalls {
				s.stopped = fmt.Sprintf("モデルの呼び出しが %d 回を超えた", l.MaxLLMCalls)
				return stop(s.stopped), nil
			}
			return nil, nil
		},
		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			if l.MaxSameToolCalls <= 0 {
				return nil, nil
			}
			raw, err := json.Marshal(args)
			if err != nil {
				return nil, err
			}
			key := t.Name() + ":" + string(raw)
			mu.Lock()
			defer mu.Unlock()
			s := get(ctx.InvocationID())
			s.same[key]++
			if s.same[key] <= l.MaxSameToolCalls {
				return nil, nil
			}
			s.stopped = fmt.Sprintf("%s を同じ引数で %d 回呼んだ", t.Name(), l.MaxSameToolCalls)
			return map[string]any{"status": "error", "error": s.stopped}, nil
		},
		AfterRunCallback: func(ctx agent.InvocationContext) {
			mu.Lock()
			defer mu.Unlock()
			delete(states, ctx.InvocationID())
		},
	})
}
