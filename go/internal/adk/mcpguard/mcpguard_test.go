package mcpguard_test

import (
	"context"
	"iter"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
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

type noteIn struct {
	ID string `json:"id,omitempty"`
}

// notesServer は読み取り、追加、削除の 3 つのツールを持つ MCP サーバー。削除された回数を数える。
func notesServer(deleted *atomic.Int32) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "notes", Version: "v1"}, nil)
	destructive := true
	mcp.AddTool(s, &mcp.Tool{Name: "list_notes", Description: "メモの一覧", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		func(context.Context, *mcp.CallToolRequest, noteIn) (*mcp.CallToolResult, map[string]any, error) {
			return nil, map[string]any{"notes": []string{"n1", "n2"}}, nil
		})
	// 注釈を付けない。MCP の仕様の既定では破壊的とみなす。
	mcp.AddTool(s, &mcp.Tool{Name: "add_note", Description: "メモを足す"},
		func(context.Context, *mcp.CallToolRequest, noteIn) (*mcp.CallToolResult, map[string]any, error) {
			return nil, map[string]any{"added": true}, nil
		})
	mcp.AddTool(s, &mcp.Tool{Name: "delete_note", Description: "メモを消す", Annotations: &mcp.ToolAnnotations{DestructiveHint: &destructive}},
		func(context.Context, *mcp.CallToolRequest, noteIn) (*mcp.CallToolResult, map[string]any, error) {
			deleted.Add(1)
			return nil, map[string]any{"deleted": true}, nil
		})
	return s
}

// connect はサーバーをメモリ内のトランスポートで立て、クライアント側のトランスポートを返す。
func connect(t *testing.T, s *mcp.Server) mcp.Transport {
	t.Helper()
	serverT, clientT := mcp.NewInMemoryTransports()
	ss, err := s.Connect(t.Context(), serverT, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ss.Close() })
	return clientT
}

// recorder はクライアントが送った要求と通知のメソッド名を記録する。
type recorder struct {
	mcp.Transport
	mu      sync.Mutex
	methods []string
}

type recordingConn struct {
	mcp.Connection
	r *recorder
}

func (r *recorder) Connect(ctx context.Context) (mcp.Connection, error) {
	c, err := r.Transport.Connect(ctx)
	return &recordingConn{Connection: c, r: r}, err
}

func (c *recordingConn) Write(ctx context.Context, msg jsonrpc.Message) error {
	if req, ok := msg.(*jsonrpc.Request); ok {
		c.r.mu.Lock()
		c.r.methods = append(c.r.methods, req.Method)
		c.r.mu.Unlock()
	}
	return c.Connection.Write(ctx, msg)
}

func TestLifecycleHasNoShutdownMessage(t *testing.T) {
	t.Parallel()
	var deleted atomic.Int32
	rec := &recorder{Transport: connect(t, notesServer(&deleted))}
	client := mcp.NewClient(&mcp.Implementation{Name: "probe", Version: "v1"}, nil)
	cs, err := client.Connect(t.Context(), rec, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cs.ListTools(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := cs.CallTool(t.Context(), &mcp.CallToolParams{Name: "list_notes", Arguments: map[string]any{}}); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	// プロトコル版 2026-07-28（SEP-2575）では initialize の往復が server/discover に置き換わる。
	// 古い版のサーバーには initialize と notifications/initialized で接続する。どちらの版も shutdown は送らない。
	want := []string{"server/discover", "tools/list", "tools/call"}
	if !slices.Equal(rec.methods, want) {
		t.Errorf("送ったメッセージ = %v, want %v", rec.methods, want)
	}
}

func TestInspectSortsToolsByAnnotations(t *testing.T) {
	t.Parallel()
	var deleted atomic.Int32
	client := mcp.NewClient(&mcp.Implementation{Name: "probe", Version: "v1"}, nil)
	cs, err := client.Connect(t.Context(), connect(t, notesServer(&deleted)), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()

	c, err := mcpguard.Inspect(t.Context(), cs)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(c.ReadOnly, []string{"list_notes"}) {
		t.Errorf("読み取り専用 = %v", c.ReadOnly)
	}
	if !c.Destructive["delete_note"] || !c.Destructive["add_note"] || c.Destructive["list_notes"] {
		t.Errorf("破壊的 = %v（注釈の無い add_note も破壊的とみなす）", c.Destructive)
	}
}

// callOnce は 1 回目に決めたツールを呼び、応答を受けたら答える。
type callOnce string

func (c callOnce) Name() string { return "scripted" }

func (c callOnce) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "終わりました"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: string(c), Args: map[string]any{"id": "n1"}}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

type outcome struct {
	calls     []string
	responses []map[string]any
	err       error
}

func runAgent(t *testing.T, m model.LLM, ts tool.Toolset) outcome {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "notes_agent", Model: m, Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "notes", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	var out outcome
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "n1 を消して"}}}
	for ev, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			out.err = err
			break
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				out.calls = append(out.calls, p.FunctionCall.Name)
			}
			if p.FunctionResponse != nil {
				out.responses = append(out.responses, p.FunctionResponse.Response)
			}
		}
	}
	return out
}

func toolset(t *testing.T, s *mcp.Server, cfg mcptoolset.Config) tool.Toolset {
	t.Helper()
	cfg.Transport = connect(t, s)
	ts, err := mcptoolset.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

func TestFilteredToolCannotBeCalledButServerStillAllowsIt(t *testing.T) {
	t.Parallel()
	var deleted atomic.Int32
	s := notesServer(&deleted)
	readOnly := tool.FilterToolset(toolset(t, s, mcptoolset.Config{}), tool.StringPredicate([]string{"list_notes"}))

	out := runAgent(t, callOnce("delete_note"), readOnly)
	// Go の ADK は見つからないツールの呼び出しを、エラーの応答としてモデルに返して対話を続ける。
	// Python の ADK は同じ場面で ValueError を投げ、対話ごと止まる。
	if out.err != nil || len(out.responses) != 1 {
		t.Fatalf("err = %v, responses = %v", out.err, out.responses)
	}
	if msg, _ := out.responses[0]["error"].(string); !strings.Contains(msg, "delete_note' not found") {
		t.Errorf("応答 = %v", out.responses[0])
	}
	if deleted.Load() != 0 {
		t.Fatal("絞り込みで外したツールが実行された")
	}

	// 同じサーバーに別のクライアントでつなげば、外したツールも呼べる。
	client := mcp.NewClient(&mcp.Implementation{Name: "other", Version: "v1"}, nil)
	cs, err := client.Connect(t.Context(), connect(t, s), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	if _, err := cs.CallTool(t.Context(), &mcp.CallToolParams{Name: "delete_note", Arguments: map[string]any{"id": "n1"}}); err != nil {
		t.Fatal(err)
	}
	if deleted.Load() != 1 {
		t.Error("サーバーの権限は絞られていないはず")
	}
}

func TestDestructiveToolAsksForConfirmation(t *testing.T) {
	t.Parallel()
	var deleted atomic.Int32
	s := notesServer(&deleted)

	client := mcp.NewClient(&mcp.Implementation{Name: "probe", Version: "v1"}, nil)
	cs, err := client.Connect(t.Context(), connect(t, s), nil)
	if err != nil {
		t.Fatal(err)
	}
	c, err := mcpguard.Inspect(t.Context(), cs)
	_ = cs.Close()
	if err != nil {
		t.Fatal(err)
	}

	testCases := map[string]struct {
		tool        string
		wantDeleted int32
		wantConfirm bool
	}{
		"破壊的なツールは確認を求めて実行しない":  {tool: "delete_note", wantConfirm: true},
		"読み取り専用のツールは確認なしで実行する": {tool: "list_notes"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			ts := toolset(t, s, mcptoolset.Config{RequireConfirmationProvider: c.ConfirmDestructive})
			out := runAgent(t, callOnce(tc.tool), ts)
			if out.err != nil {
				t.Fatalf("Run() error = %v", out.err)
			}
			if got := slices.Contains(out.calls, "adk_request_confirmation"); got != tc.wantConfirm {
				t.Errorf("確認の要求 = %v, want %v（calls %v）", got, tc.wantConfirm, out.calls)
			}
			if deleted.Load() != tc.wantDeleted {
				t.Errorf("削除された回数 = %d", deleted.Load())
			}
		})
	}
}
