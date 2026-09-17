// Package memorybank は Session を越えて覚えるエージェントを組み立てる。
//
// 記憶の保管先は internal/adk/domain のポート越しに扱う。
// 実装の選択と接続は internal/adk/repository が持つ。
//
// Go の保存経路は Python と違う。
// Python は after_agent_callback から ctx.add_session_to_memory() を呼べるが、
// Go のコールバック用ラッパーは Memory() に未対応で nil を返す。
//
//	func (c *callbackContextWrapper) Memory() Memory {
//		log.Print("Memory() is not supported for callback context")
//
// 保存できるのは InvocationContext を受け取る Plugin の AfterRunCallback になる。
// ADK 同梱の examples/agentengine も同じ形をとっている。
//
// 検索は事情が違う。tool_context_wrapper.SearchMemory は素通しで委譲するので、
// ツールからも preloadmemorytool からも呼べる。
package memorybank

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/loadmemorytool"
	"google.golang.org/adk/v2/tool/preloadmemorytool"

	"github.com/hiro8ma/agent/go/internal/adk/repository"
)

const instruction = `あなたは利用者の好みを覚える案内役です。
関連する記憶は自動で渡されます。
それで足りない場合だけ load_memory ツールで追加の記憶を探してください。
記憶に無いことを推測して答えないでください。`

// NewSavePlugin は対話の終わりに Session を記憶へ取り込む Plugin を作る。
//
// BeforeRunCallback で差分の基準時刻を State へ置き、AfterRunCallback で保存する。
// 保存の失敗で対話そのものを止めない。AfterRunCallback は戻り値を持たない契約になっている。
func NewSavePlugin() (*plugin.Plugin, error) {
	p, err := plugin.New(plugin.Config{
		Name: "memory-writer",
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			return nil, ic.Session().State().Set(repository.LastUpdateKey, ic.Session().LastUpdateTime())
		},
		AfterRunCallback: func(ic agent.InvocationContext) {
			m := ic.Memory()
			if m == nil {
				return
			}
			_ = m.AddSessionToMemory(ic, ic.Session())
		},
	})
	if err != nil {
		return nil, fmt.Errorf("plugin の作成: %w", err)
	}
	return p, nil
}

// NewAgent は記憶する案内役を組み立てる。
//
// ツールを 2 つ渡す。
// preload は毎リクエストで直前の発話を検索して instruction へ足す。
// load はモデルが必要と判断したときだけ呼ぶ。
// 前者だけでは検索語が直前の発話に固定され、後者だけでは呼び忘れる。
func NewAgent(m model.LLM) (agent.Agent, error) {
	a, err := llmagent.New(llmagent.Config{
		Name:        "memory_agent",
		Model:       m,
		Description: "過去の対話を覚えて次の対話で使う案内役",
		Instruction: instruction,
		Tools: []tool.Tool{
			preloadmemorytool.New(),
			loadmemorytool.New(),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("エージェントの作成: %w", err)
	}
	return a, nil
}
