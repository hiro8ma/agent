// Package toolgov はエージェントごとのツールの権限と、ツールの呼び出しの監査を、ランナーのプラグインとして持つ。
//
// 監査をエージェントのコールバックに置くと、前に並べた別のコールバックが呼び出しを止めたとき、監査まで届かない。
// プラグインのコールバックはエージェントのコールバックより先に走るので、止められた呼び出しも記録できる。
package toolgov

import (
	"errors"
	"slices"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
)

// Policy はエージェントの名前ごとに、使ってよいツールの名前を持つ。載っていないツールは使えない。
// 拒否の一覧は持たない。許可と拒否の両方を書くと、どちらにも無いツールの扱いが曖昧になる。
type Policy map[string][]string

// Allows はエージェントがツールを使ってよいかを返す。
func (p Policy) Allows(agentName, toolName string) bool {
	return slices.Contains(p[agentName], toolName)
}

// Entry は監査ログの 1 行。
type Entry struct {
	Time    time.Time      `json:"time"`
	Agent   string         `json:"agent"`
	User    string         `json:"user"`
	Session string         `json:"session"`
	Tool    string         `json:"tool"`
	Args    map[string]any `json:"args"`
	// Event は call（呼び出し）、denied（権限で拒否）、ok（成功）、error（失敗）のどれか。
	Event string `json:"event"`
	Error string `json:"error,omitempty"`
}

// Sink は監査ログの書き出し先。実運用では Cloud Logging などに流す。
type Sink interface {
	Write(Entry)
}

// Memory は監査ログをメモリに持つ Sink。
type Memory struct {
	mu      sync.Mutex
	entries []Entry
}

func (m *Memory) Write(e Entry) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, e)
}

// Entries は書かれた順に返す。
func (m *Memory) Entries() []Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.entries)
}

// sensitiveKeys は値を伏せる引数の名前に含まれる語。
var sensitiveKeys = []string{"password", "token", "secret", "api_key", "apikey", "credential"}

// Redact は秘密らしい名前の引数の値を伏せた写しを返す。
func Redact(args map[string]any) map[string]any {
	out := make(map[string]any, len(args))
	for k, v := range args {
		lower := strings.ToLower(k)
		if slices.ContainsFunc(sensitiveKeys, func(s string) bool { return strings.Contains(lower, s) }) {
			v = "[REDACTED]"
		}
		out[k] = v
	}
	return out
}

// New は権限の確認と監査をするプラグインを返す。
func New(policy Policy, sink Sink) (*plugin.Plugin, error) {
	entry := func(ctx agent.Context, t tool.Tool, args map[string]any, event string) Entry {
		return Entry{
			Time: time.Now(), Agent: ctx.AgentName(), User: ctx.UserID(), Session: ctx.SessionID(),
			Tool: t.Name(), Args: Redact(args), Event: event,
		}
	}
	return plugin.New(plugin.Config{
		Name: "toolgov",
		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			if !policy.Allows(ctx.AgentName(), t.Name()) {
				sink.Write(entry(ctx, t, args, "denied"))
				return map[string]any{"status": "denied", "message": "このエージェントはこのツールを使えない"}, nil
			}
			sink.Write(entry(ctx, t, args, "call"))
			return nil, nil
		},
		// エージェントのコールバックが呼び出しを止めると、止めた側が返した結果がここに来る。
		// ツールが動いたかは区別できないので、結果に error があれば失敗として残す。
		AfterToolCallback: func(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
			if !policy.Allows(ctx.AgentName(), t.Name()) {
				return nil, nil
			}
			e := entry(ctx, t, args, "ok")
			if msg, ok := result["error"].(string); ok && err == nil {
				err = errors.New(msg)
			}
			if err != nil {
				e.Event, e.Error = "error", err.Error()
			}
			sink.Write(e)
			return nil, nil
		},
	})
}
