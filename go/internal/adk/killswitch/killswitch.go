// Package killswitch は、全体、エージェント、ツールの単位で実行を止めるプラグイン。
//
// 状態はプロセスの変数ではなく共有の保存先から読む。インスタンスが複数あっても、1 か所の操作で全て止まる。
// 保存先が読めないときは止める側に倒す。
package killswitch

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

// State は止めている範囲。
type State struct {
	Global bool     `json:"global"`
	Agents []string `json:"agents"`
	Tools  []string `json:"tools"`
}

// Store は State の共有の保存先。
type Store interface {
	Load(ctx context.Context) (State, error)
}

// Memory はプロセスの中の Store。テストと、1 インスタンスの構成で使う。
type Memory struct {
	mu    sync.Mutex
	state State
}

func (m *Memory) Load(context.Context) (State, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state, nil
}

// Set は止める範囲を置き換える。
func (m *Memory) Set(s State) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = s
}

// File は JSON のファイルを読む Store。共有のボリュームや ConfigMap に置く。
type File string

func (f File) Load(context.Context) (State, error) {
	var s State
	raw, err := os.ReadFile(string(f))
	if err != nil {
		return s, err
	}
	return s, json.Unmarshal(raw, &s)
}

// Switch は Store の State を ttl の間だけ手元に持つ。
type Switch struct {
	store Store
	ttl   time.Duration
	now   func() time.Time

	mu       sync.Mutex
	cached   State
	err      error
	loadedAt time.Time
}

// New は Switch を作る。ttl は止めてから全インスタンスに効くまでの最大の遅れになる。
func New(store Store, ttl time.Duration) *Switch {
	return &Switch{store: store, ttl: ttl, now: time.Now}
}

// WithClock は時刻を差し替える。テスト用。
func (s *Switch) WithClock(now func() time.Time) *Switch {
	s.now = now
	return s
}

func (s *Switch) load(ctx context.Context) (State, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.loadedAt.IsZero() && s.now().Sub(s.loadedAt) < s.ttl {
		return s.cached, s.err
	}
	s.cached, s.err = s.store.Load(ctx)
	s.loadedAt = s.now()
	return s.cached, s.err
}

// Reason は止める理由を返す。止めないなら空。保存先が読めなければ止める。
func (s *Switch) Reason(ctx context.Context, agentName, toolName string) string {
	st, err := s.load(ctx)
	switch {
	case err != nil:
		return "停止の状態を確かめられないため止めている"
	case st.Global:
		return "全体を止めている"
	case slices.Contains(st.Agents, agentName):
		return "このエージェントを止めている"
	case toolName != "" && slices.Contains(st.Tools, toolName):
		return "このツールを止めている"
	}
	return ""
}

// Plugin はモデルの呼び出しとツールの実行の前に確かめるプラグインを返す。
func (s *Switch) Plugin() (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "killswitch",
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			if r := s.Reason(ctx, ctx.AgentName(), ""); r != "" {
				return &model.LLMResponse{Content: genai.NewContentFromText("現在このエージェントは利用できません（"+r+"）", genai.RoleModel)}, nil
			}
			return nil, nil
		},
		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, _ map[string]any) (map[string]any, error) {
			if r := s.Reason(ctx, ctx.AgentName(), t.Name()); r != "" {
				return map[string]any{"status": "error", "error": r}, nil
			}
			return nil, nil
		},
	})
}
