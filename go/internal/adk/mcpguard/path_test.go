package mcpguard_test

import (
	"context"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/mcpguard"
)

func TestWithin(t *testing.T) {
	t.Parallel()
	roots := []string{"/srv/allowed"}
	testCases := map[string]struct {
		path string
		want bool
	}{
		"許可したディレクトリの絶対パスは通す":    {path: "/srv/allowed/report.txt", want: true},
		"相対パスは許可したディレクトリから解決する": {path: "sub/report.txt", want: true},
		"名前に .. を含むだけなら通す":      {path: "notes..txt", want: true},
		"上へ出る相対パスは止める":          {path: "../../etc/passwd", want: false},
		"中を経由して外へ出るパスも止める":      {path: "/srv/allowed/sub/../../secret", want: false},
		"名前の前半が同じ別のディレクトリは止める":  {path: "/srv/allowed-evil/x", want: false},
		"別の絶対パスは止める":            {path: "/etc/passwd", want: false},
		"空は止める":                 {path: "", want: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := mcpguard.Within(roots, tc.path); got != tc.want {
				t.Errorf("Within(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

type pathIn struct {
	Path        string   `json:"path,omitempty"`
	Paths       []string `json:"paths,omitempty"`
	Source      string   `json:"source,omitempty"`
	Destination string   `json:"destination,omitempty"`
}

// filesServer は公式のファイルシステムの MCP サーバーと同じ名前と引数の 3 つのツールを持つ。
func filesServer(calls *atomic.Int32) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "files", Version: "v1"}, nil)
	handler := func(context.Context, *mcp.CallToolRequest, pathIn) (*mcp.CallToolResult, map[string]any, error) {
		calls.Add(1)
		return nil, map[string]any{"ok": true}, nil
	}
	mcp.AddTool(s, &mcp.Tool{Name: "read_file", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}, handler)
	mcp.AddTool(s, &mcp.Tool{Name: "read_multiple_files", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}}, handler)
	mcp.AddTool(s, &mcp.Tool{Name: "move_file"}, handler)
	return s
}

// scriptedCall は 1 回目に決めた引数でツールを呼び、応答を受けたら答える。
type scriptedCall struct {
	name string
	args map[string]any
}

func (c scriptedCall) Name() string { return "scripted" }

func (c scriptedCall) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "終わりました"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: c.name, Args: c.args}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func run(t *testing.T, a agent.Agent) error {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "files", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return err
	}
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "ファイルを扱って"}}}
	for _, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

func TestPathGuardChecksEveryPathArgument(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		tool     string
		args     map[string]any
		executed bool
	}{
		"許可したディレクトリの中は実行する":             {tool: "read_file", args: map[string]any{"path": "/srv/allowed/a.txt"}, executed: true},
		"path で外へ出るのは止める":               {tool: "read_file", args: map[string]any{"path": "../../etc/passwd"}},
		"move_file の source で外へ出るのは止める": {tool: "move_file", args: map[string]any{"source": "../../etc/passwd", "destination": "/srv/allowed/x"}},
		"paths の 1 つでも外なら止める":           {tool: "read_multiple_files", args: map[string]any{"paths": []any{"/srv/allowed/a", "/etc/shadow"}}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var calls atomic.Int32
			ts, err := mcptoolset.New(mcptoolset.Config{Transport: connect(t, filesServer(&calls))})
			if err != nil {
				t.Fatal(err)
			}
			m := scriptedCall{name: tc.tool, args: tc.args}
			a, err := llmagent.New(llmagent.Config{
				Name: "files_agent", Model: m, Toolsets: []tool.Toolset{ts},
				BeforeToolCallbacks: []llmagent.BeforeToolCallback{mcpguard.PathGuard([]string{"/srv/allowed"})},
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := run(t, a); err != nil {
				t.Fatal(err)
			}
			if got := calls.Load() == 1; got != tc.executed {
				t.Errorf("実行された = %v, want %v", got, tc.executed)
			}
		})
	}
}
