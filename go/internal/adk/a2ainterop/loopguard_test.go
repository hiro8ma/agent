package a2ainterop_test

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
)

// mesh は名前ごとの A2A のサーバーと、次に呼ぶ相手の表を持つ。
type mesh struct {
	mu   sync.Mutex
	urls map[string]string
	next map[string]string
}

func (m *mesh) url(name string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.urls[name]
}

// relay は、次の相手が決まっていれば呼び、返ってきた経路の文に自分の名前を足して返す。
// LLM を使わずに、呼び出しの経路だけを確かめる。
type relay struct {
	self string
	m    *mesh
}

func (r relay) Execute(ctx context.Context, _ *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		text := r.self
		if next := r.m.next[r.self]; next != "" {
			text += " -> " + call(ctx, r.m.url(next))
		}
		yield(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(text)), nil)
	}
}

func (relay) Cancel(context.Context, *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(func(a2a.Event, error) bool) {}
}

func call(ctx context.Context, url string) string {
	client, err := a2aclient.NewFromEndpoints(ctx,
		[]*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC)},
		a2aclient.WithCallInterceptors(a2ainterop.TracePropagator{}))
	if err != nil {
		return "接続できない: " + err.Error()
	}
	defer func() { _ = client.Destroy() }()
	res, err := client.SendMessage(ctx, &a2a.SendMessageRequest{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("お願い"))})
	if err != nil {
		return "拒否"
	}
	if msg, ok := res.(*a2a.Message); ok && len(msg.Parts) > 0 {
		return msg.Parts[0].Text()
	}
	return "?"
}

func newMesh(t *testing.T, next map[string]string, maxDepth int) *mesh {
	t.Helper()
	m := &mesh{urls: map[string]string{}, next: next}
	for _, name := range []string{"A", "B", "C"} {
		handler := a2asrv.NewHandler(relay{self: name, m: m},
			a2asrv.WithCallInterceptors(a2ainterop.LoopGuard{Self: name, MaxDepth: maxDepth}))
		mux := http.NewServeMux()
		mux.Handle("/", a2asrv.NewJSONRPCHandler(handler))
		srv := httptest.NewServer(mux)
		t.Cleanup(srv.Close)
		m.urls[name] = srv.URL + "/"
	}
	return m
}

func TestLoopGuard(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		next     map[string]string
		maxDepth int
		want     string
	}{
		"一方向の呼び出しは通る":          {next: map[string]string{"A": "B", "B": "C"}, want: "A -> B -> C"},
		"A から B を経て A に戻ると止める": {next: map[string]string{"A": "B", "B": "A"}, want: "A -> B -> 拒否"},
		"自分を呼ぶと止める":            {next: map[string]string{"A": "A"}, want: "A -> 拒否"},
		"深さの上限を超えると止める":        {next: map[string]string{"A": "B", "B": "C"}, maxDepth: 2, want: "A -> B -> 拒否"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			m := newMesh(t, tc.next, tc.maxDepth)
			got := call(t.Context(), m.url("A"))
			if !strings.HasPrefix(got, tc.want) {
				t.Errorf("経路 = %q, want %q", got, tc.want)
			}
		})
	}
}
