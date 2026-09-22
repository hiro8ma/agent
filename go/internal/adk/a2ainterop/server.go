package a2ainterop

import (
	"net/http"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

// V0Path は v0.3 のクライアントのための互換の口。
const V0Path = "/v0"

// NewHandler は ADK のエージェントを A2A v1.0 のサーバーとして公開し、v0.3 の互換の口も足す。
//
// baseURL は Agent Card に書く、クライアントから届く名前（例 http://localhost:8002）。
// 待ち受けのアドレス（0.0.0.0 など）を書くと、ほかのマシンのクライアントは接続できない。
// Agent Card には v1.0 の口と v0.3 の口の両方を並べる。0.3 の口が無いと互換の Agent Card が 500 になる。
func NewHandler(a agent.Agent, card a2a.AgentCard, baseURL string, plugins ...*plugin.Plugin) http.Handler {
	exec := adka2a.NewExecutor(adka2a.ExecutorConfig{RunnerConfig: runner.Config{
		AppName: a.Name(), Agent: a, SessionService: session.InMemoryService(),
		PluginConfig: runner.PluginConfig{Plugins: plugins},
	}})
	return NewExecutorHandler(exec, card, baseURL)
}

// NewExecutorHandler は任意の実行器（ADK 以外のエージェントを包んだものなど）を、NewHandler と同じ口と Agent Card で公開する。
func NewExecutorHandler(exec a2asrv.AgentExecutor, card a2a.AgentCard, baseURL string) http.Handler {
	handler := a2asrv.NewHandler(exec)

	base := strings.TrimSuffix(baseURL, "/")
	v0 := a2a.NewAgentInterface(base+V0Path, a2a.TransportProtocolJSONRPC)
	v0.ProtocolVersion = a2av0.Version
	card.SupportedInterfaces = []*a2a.AgentInterface{
		a2a.NewAgentInterface(base+"/", a2a.TransportProtocolJSONRPC),
		v0,
	}

	mux := http.NewServeMux()
	mux.Handle("/", a2asrv.NewJSONRPCHandler(handler))
	mux.Handle(V0Path, a2av0.NewJSONRPCHandler(handler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewAgentCardHandler(a2av0.NewStaticAgentCardProducer(&card)))
	return mux
}
