package toolgov_test

import (
	"context"
	"errors"
	"iter"
	"slices"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/toolgov"
)

type call struct {
	name string
	args map[string]any
}

func (c call) Name() string { return "scripted" }

func (c call) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "終わりました"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: c.name, Args: c.args}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

type queryIn struct {
	SQL      string `json:"sql"`
	Password string `json:"password,omitempty"`
}

type env struct {
	sink     *toolgov.Memory
	executed *atomic.Int32
}

func run(t *testing.T, agentName string, c call, guards ...llmagent.BeforeToolCallback) env {
	t.Helper()
	var executed atomic.Int32
	query, err := functiontool.New(functiontool.Config{Name: "run_query", Description: "SQL を実行する"},
		func(_ agent.Context, in queryIn) (map[string]any, error) {
			executed.Add(1)
			if in.SQL == "broken" {
				return nil, errors.New("syntax error")
			}
			return map[string]any{"rows": 1}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	del, err := functiontool.New(functiontool.Config{Name: "kubectl_delete", Description: "Pod を消す"},
		func(_ agent.Context, _ struct {
			Name string `json:"name"`
		},
		) (map[string]any, error) {
			executed.Add(1)
			return map[string]any{"deleted": true}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name: agentName, Model: c, Tools: []tool.Tool{query, del}, BeforeToolCallbacks: guards,
	})
	if err != nil {
		t.Fatal(err)
	}
	sink := &toolgov.Memory{}
	gov, err := toolgov.New(toolgov.Policy{
		"data_analyst": {"run_query"},
		"sre_operator": {"run_query", "kubectl_delete"},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "infra", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{gov}},
	})
	if err != nil {
		t.Fatal(err)
	}
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "お願い"}}}
	for _, err := range r.Run(t.Context(), "alice", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	return env{sink: sink, executed: &executed}
}

func events(s *toolgov.Memory) []string {
	var out []string
	for _, e := range s.Entries() {
		out = append(out, e.Tool+":"+e.Event)
	}
	return out
}

func TestPolicyAndAudit(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		agent    string
		call     call
		want     []string
		executed int32
	}{
		"分析担当は SQL を実行できる":  {agent: "data_analyst", call: call{"run_query", map[string]any{"sql": "SELECT 1"}}, want: []string{"run_query:call", "run_query:ok"}, executed: 1},
		"分析担当は Pod を消せない":   {agent: "data_analyst", call: call{"kubectl_delete", map[string]any{"name": "web-1"}}, want: []string{"kubectl_delete:denied"}},
		"運用担当は Pod を消せる":    {agent: "sre_operator", call: call{"kubectl_delete", map[string]any{"name": "web-1"}}, want: []string{"kubectl_delete:call", "kubectl_delete:ok"}, executed: 1},
		"表に無いエージェントは何も使えない": {agent: "intern", call: call{"run_query", map[string]any{"sql": "SELECT 1"}}, want: []string{"run_query:denied"}},
		"ツールの失敗も結果として残す":    {agent: "data_analyst", call: call{"run_query", map[string]any{"sql": "broken"}}, want: []string{"run_query:call", "run_query:error"}, executed: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			e := run(t, tc.agent, tc.call)
			if got := events(e.sink); !slices.Equal(got, tc.want) {
				t.Errorf("監査ログ = %v, want %v", got, tc.want)
			}
			if e.executed.Load() != tc.executed {
				t.Errorf("実行された回数 = %d, want %d", e.executed.Load(), tc.executed)
			}
		})
	}
}

func TestAuditSeesCallsBlockedByAgentCallbacks(t *testing.T) {
	t.Parallel()
	guard := func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
		return map[string]any{"error": "止めた"}, nil
	}
	e := run(t, "data_analyst", call{"run_query", map[string]any{"sql": "SELECT 1"}}, guard)
	if got, want := events(e.sink), []string{"run_query:call", "run_query:error"}; !slices.Equal(got, want) {
		t.Errorf("監査ログ = %v, want %v", got, want)
	}
	if e.executed.Load() != 0 {
		t.Error("止めたはずのツールが実行された")
	}
}

func TestSecretsAreRedacted(t *testing.T) {
	t.Parallel()
	e := run(t, "data_analyst", call{"run_query", map[string]any{"sql": "SELECT 1", "password": "hunter2"}})
	for _, entry := range e.sink.Entries() {
		if entry.Args["password"] != "[REDACTED]" || entry.Args["sql"] != "SELECT 1" {
			t.Errorf("引数 = %v", entry.Args)
		}
	}
}
