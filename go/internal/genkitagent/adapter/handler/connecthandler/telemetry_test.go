package connecthandler_test

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"connectrpc.com/connect"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	convadapter "github.com/hiro8ma/agent/go/internal/conversation/adapter"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	convusecase "github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/handler/connecthandler"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/conversationclient"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/inmemory"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/genkitagent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
)

const (
	secretQuestion = "秘密の質問-7f3a"
	secretUser     = "user-secret-42"
)

var (
	recordersOnce sync.Once
	redacted      *tracetest.InMemoryExporter
	raw           *tracetest.InMemoryExporter
)

// recorders は global の provider を 1 回だけ差し替える。Genkit は global の provider を使う。
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

// tracedSystem は エージェント → conversation の 2 サービスを立て、Genkit のエージェントを載せる。
func tracedSystem(t *testing.T) agentv1connect.AgentServiceClient {
	t.Helper()
	repo, err := repository.NewSQLite("file:" + filepath.Join(t.TempDir(), "conversation.db"))
	if err != nil {
		t.Fatal(err)
	}
	conv := serve(t)(convadapter.NewHandler(convusecase.New(repo), libconnect.HeaderAuthenticator))

	g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/echo"))
	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("echo: " + req.Messages[len(req.Messages)-1].Text()), FinishReason: ai.FinishReasonStop}, nil
		})
	echo := service.NewGenkitAgent(g, service.Definition{ID: "echo", SystemPrompt: "x"})
	uc := usecase.NewAgentService(service.NewRegistry(echo), conversationclient.NewRemote(conv.Client(), conv.URL),
		noPendingGate{}, inmemory.NewOrders(), slog.New(slog.DiscardHandler))
	edge := serve(t)(connecthandler.Route(connecthandler.New(uc), libconnect.HeaderAuthenticator))
	return agentv1connect.NewAgentServiceClient(edge.Client(), edge.URL, connect.WithInterceptors(libconnect.ForwardIdentity()))
}

func chatTraced(t *testing.T, client agentv1connect.AgentServiceClient, agentID string) {
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

func TestGenkitSpansJoinTheTraceWithoutContent(t *testing.T) {
	redactedExp, rawExp := recorders()
	chatTraced(t, tracedSystem(t), "echo")

	var chatSpan tracetest.SpanStub
	for _, s := range byName(redactedExp.GetSpans(), "agent.v1.AgentService/Chat") {
		if s.SpanKind == trace.SpanKindServer {
			chatSpan = s
		}
	}
	id := chatSpan.SpanContext.TraceID()
	isGenkit := func(s tracetest.SpanStub) bool {
		for _, kv := range s.Attributes {
			if strings.HasPrefix(string(kv.Key), "genkit:") {
				return true
			}
		}
		return false
	}

	var genkitSpans int
	for _, s := range inTrace(redactedExp.GetSpans(), id) {
		if !isGenkit(s) {
			continue
		}
		genkitSpans++
		for _, kv := range s.Attributes {
			if kv.Key == "genkit:input" || kv.Key == "genkit:output" {
				t.Errorf("%s に %s が残った", s.Name, kv.Key)
			}
		}
	}
	if genkitSpans == 0 {
		t.Fatal("Genkit の span がエッジのサーバーの trace に無い")
	}

	var rawInput bool
	for _, s := range inTrace(rawExp.GetSpans(), id) {
		for _, kv := range s.Attributes {
			if kv.Key == "genkit:input" && strings.Contains(kv.Value.String(), secretQuestion) {
				rawInput = true
			}
		}
	}
	if !rawInput {
		t.Error("対照の exporter に Genkit の入力が無い。落とす処理を確かめられていない")
	}
}
