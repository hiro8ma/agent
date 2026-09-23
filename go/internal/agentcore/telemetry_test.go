package agentcore_test

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"connectrpc.com/connect"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/adkagent"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/agentcore/actionexec"
	"github.com/hiro8ma/agent/go/internal/agentcore/backend"
	convadapter "github.com/hiro8ma/agent/go/internal/conversation/adapter"
	convclient "github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	knadapter "github.com/hiro8ma/agent/go/internal/knowledge/adapter"
	knclient "github.com/hiro8ma/agent/go/internal/knowledge/client"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
)

const (
	secretQuestion = "秘密の質問-7f3a"
	secretQuery    = "機密の検索語-9c1d"
	secretUser     = "user-secret-42"
)

var (
	recordersOnce sync.Once
	redacted      *tracetest.InMemoryExporter
	raw           *tracetest.InMemoryExporter
)

// recorders は global の provider を 1 回だけ差し替える。ADK は global の provider を使う。
func recorders() (*tracetest.InMemoryExporter, *tracetest.InMemoryExporter) {
	recordersOnce.Do(func() {
		redacted = tracetest.NewInMemoryExporter()
		raw = tracetest.NewInMemoryExporter()
		otel.SetTracerProvider(sdktrace.NewTracerProvider(
			sdktrace.WithSyncer(libotel.Redact(redacted)),
			sdktrace.WithSyncer(raw),
		))
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	})
	return redacted, raw
}

// searchOnce は 1 回目に search_knowledge を呼び、ツールの結果を受けたら答える台本のモデル。
type searchOnce struct{}

func (searchOnce) Name() string { return "scripted" }

func (searchOnce) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "見つかりました"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: "search_knowledge", Args: map[string]any{"query": secretQuery}}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func serve(t *testing.T) func(string, http.Handler) *httptest.Server {
	t.Helper()
	return func(path string, h http.Handler) *httptest.Server {
		mux := http.NewServeMux()
		mux.Handle(path, h)
		srv := httptest.NewServer(mux)
		t.Cleanup(srv.Close)
		return srv
	}
}

// system は エージェント → conversation / エージェント → ADK → knowledge の 3 サービスを立てる。
func system(t *testing.T) agentv1connect.AgentServiceClient {
	t.Helper()
	kn := serve(t)(knadapter.NewHandler(backend.NewInMemoryKnowledge(), libconnect.HeaderAuthenticator))

	repo, err := repository.NewSQLite("file:" + filepath.Join(t.TempDir(), "conversation.db"))
	if err != nil {
		t.Fatal(err)
	}
	conv := serve(t)(convadapter.NewHandler(usecase.New(repo), libconnect.HeaderAuthenticator))

	tools, err := adkagent.ResearchTools(knclient.NewSearcher(kn.Client(), kn.URL))
	if err != nil {
		t.Fatal(err)
	}
	research, err := adkagent.New(adkagent.Definition{ID: "research", Instruction: "x", Model: searchOnce{}, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	core := agentcore.NewHandler(agentcore.NewRegistry(research), convclient.NewSessionStore(conv.Client(), conv.URL), actionexec.Executor{}, slog.New(slog.DiscardHandler))
	edge := serve(t)(agentcore.NewConnectHandler(core, libconnect.HeaderAuthenticator))
	return agentv1connect.NewAgentServiceClient(edge.Client(), edge.URL, connect.WithInterceptors(libconnect.ForwardIdentity()))
}

// chatTraced は traceparent を付けて Chat を呼び、呼び出し元の trace ID を返す。
func chatTraced(t *testing.T, client agentv1connect.AgentServiceClient, agentID string) trace.TraceID {
	t.Helper()
	callerCtx := trace.ContextWithSpanContext(identity.With(t.Context(), secretUser), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{0x0c, 0xa1, 0x1e, 0xe0}, SpanID: trace.SpanID{0x01}, TraceFlags: trace.FlagsSampled, Remote: true,
	}))
	req := connect.NewRequest(&agentv1.ChatRequest{AgentId: agentID, Message: secretQuestion})
	otel.GetTextMapPropagator().Inject(callerCtx, propagation.HeaderCarrier(req.Header()))
	stream, err := client.Chat(callerCtx, req)
	if err != nil {
		t.Fatal(err)
	}
	for stream.Receive() {
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	return trace.SpanContextFromContext(callerCtx).TraceID()
}

func byName(spans tracetest.SpanStubs, prefix string) []tracetest.SpanStub {
	var out []tracetest.SpanStub
	for _, s := range spans {
		if strings.HasPrefix(s.Name, prefix) {
			out = append(out, s)
		}
	}
	return out
}

func inTrace(spans tracetest.SpanStubs, id trace.TraceID) tracetest.SpanStubs {
	var out tracetest.SpanStubs
	for _, s := range spans {
		if s.SpanContext.TraceID() == id {
			out = append(out, s)
		}
	}
	return out
}

func TestTraceSpansServicesAndHidesContent(t *testing.T) {
	redactedExp, rawExp := recorders()
	callerTrace := chatTraced(t, system(t), "research")

	all := redactedExp.GetSpans()
	edge := byName(all, "agent.v1.AgentService/Chat")
	var chatSpan tracetest.SpanStub
	for _, s := range edge {
		if s.SpanKind == trace.SpanKindServer {
			chatSpan = s
		}
	}
	if !chatSpan.SpanContext.IsValid() {
		t.Fatal("エージェントのサーバーの span が無い")
	}
	spans := inTrace(all, chatSpan.SpanContext.TraceID())

	t.Run("外からの trace は信用せず新しい trace を始め、リンクだけ残す", func(t *testing.T) {
		if chatSpan.SpanContext.TraceID() == callerTrace {
			t.Errorf("エッジのサーバーが呼び出し元の trace にそのまま入った")
		}
		if len(chatSpan.Links) != 1 || chatSpan.Links[0].SpanContext.TraceID() != callerTrace {
			t.Errorf("links = %v", chatSpan.Links)
		}
	})

	parentOf := map[string]string{}
	names := map[string]string{}
	for _, s := range spans {
		names[s.SpanContext.SpanID().String()] = s.Name
		parentOf[s.SpanContext.SpanID().String()] = s.Parent.SpanID().String()
	}
	ancestors := func(s tracetest.SpanStub) []string {
		var chain []string
		for id := s.Parent.SpanID().String(); names[id] != ""; id = parentOf[id] {
			chain = append(chain, names[id])
		}
		return chain
	}

	testCases := map[string]struct {
		span      string
		kind      trace.SpanKind
		ancestors []string
	}{
		"ADK のエージェントはエッジのサーバーの子": {
			span: "invoke_agent research", kind: trace.SpanKindInternal,
			ancestors: []string{"agent.v1.AgentService/Chat"},
		},
		"ツールの実行は ADK のエージェントの子": {
			span: "execute_tool search_knowledge", kind: trace.SpanKindInternal,
			ancestors: []string{"invoke_agent research", "agent.v1.AgentService/Chat"},
		},
		"knowledge のサーバーはツールの中のクライアント呼び出しの子": {
			span: "knowledge.v1.KnowledgeService/Search", kind: trace.SpanKindServer,
			ancestors: []string{"knowledge.v1.KnowledgeService/Search", "execute_tool search_knowledge", "invoke_agent research", "agent.v1.AgentService/Chat"},
		},
		"conversation のサーバーも同じ trace に入る": {
			span: "conversation.v1.ConversationService/AppendTurn", kind: trace.SpanKindServer,
			ancestors: []string{"conversation.v1.ConversationService/AppendTurn", "agent.v1.AgentService/Chat"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			var found bool
			for _, s := range byName(spans, tc.span) {
				if s.SpanKind != tc.kind {
					continue
				}
				found = true
				got := ancestors(s)
				if fmt.Sprint(got[:min(len(got), len(tc.ancestors))]) != fmt.Sprint(tc.ancestors) {
					t.Errorf("祖先 = %v, want 先頭が %v", got, tc.ancestors)
				}
			}
			if !found {
				t.Errorf("%s の span が trace に無い", tc.span)
			}
		})
	}

	t.Run("送る span に本文と利用者 ID が残らない", func(t *testing.T) {
		for _, s := range spans {
			for _, kv := range s.Attributes {
				v := kv.Value.String()
				for _, secret := range []string{secretQuestion, secretQuery, secretUser} {
					if strings.Contains(v, secret) {
						t.Errorf("%s の %s に %q が残った", s.Name, kv.Key, secret)
					}
				}
			}
		}
	})

	t.Run("落とさなければ ADK はツールの引数を載せる", func(t *testing.T) {
		var leaked bool
		for _, s := range inTrace(rawExp.GetSpans(), chatSpan.SpanContext.TraceID()) {
			for _, kv := range s.Attributes {
				if strings.Contains(kv.Value.String(), secretQuery) {
					leaked = true
				}
			}
		}
		if !leaked {
			t.Error("対照の exporter にもツールの引数が無い。落とす処理を確かめられていない")
		}
	})
}
