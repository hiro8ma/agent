package a2aserve_test

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/a2aserve"
	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
	"github.com/hiro8ma/agent/go/internal/devidp"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/handler/a2ahandler"
	genkitmodel "github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	genkitservice "github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/lib/libauth"
	"github.com/hiro8ma/agent/go/internal/toolscope"
)

const (
	issuer   = "https://idp.test"
	audience = "a2a://agents"
)

// seen はツールの中から見えた呼び出し元を記録する。
type seen struct {
	mu    sync.Mutex
	calls []string
}

func (s *seen) record(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, ok := libauth.FromContext(ctx)
	if !ok {
		s.calls = append(s.calls, "(no principal)")
		return
	}
	s.calls = append(s.calls, p.Subject)
}

func (s *seen) list() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.calls...)
}

type world struct {
	ca       *devidp.CA
	idpURL   string
	verifier *libauth.Verifier
	certs    map[string]*devidp.Issued
	cfg      a2aserve.Config
}

func newWorld(t *testing.T) *world {
	t.Helper()
	ca, err := devidp.NewCA("test-ca")
	if err != nil {
		t.Fatal(err)
	}
	idp, err := devidp.New(issuer, audience, map[string][]string{
		"caller-full": {"a2a.invoke", "orders.read", "knowledge.read"},
		"caller-min":  {"a2a.invoke"},
	})
	if err != nil {
		t.Fatal(err)
	}
	idpCert, err := ca.IssueServer("idp")
	if err != nil {
		t.Fatal(err)
	}
	idpSrv := httptest.NewUnstartedServer(idp.Handler())
	idpSrv.TLS = &tls.Config{Certificates: []tls.Certificate{idpCert.TLS}, ClientCAs: ca.Pool(), ClientAuth: tls.VerifyClientCertIfGiven}
	idpSrv.StartTLS()
	t.Cleanup(idpSrv.Close)
	v, err := libauth.NewVerifier(t.Context(), libauth.Config{
		JWKSURL: idpSrv.URL + devidp.JWKSPath, Issuer: issuer, Audience: audience, RequireCertBinding: true,
		HTTPClient: &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: ca.Pool()}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	certs := map[string]*devidp.Issued{}
	for _, name := range []string{"caller-full", "caller-min"} {
		if certs[name], err = ca.IssueClient(name); err != nil {
			t.Fatal(err)
		}
	}
	return &world{
		ca: ca, idpURL: idpSrv.URL, verifier: v, certs: certs,
		cfg: a2aserve.Config{TokenURL: idpSrv.URL + devidp.TokenPath, Required: []string{"a2a.invoke"}},
	}
}

// serve は build が返す A2A のハンドラを、クライアント証明書を必須にした TLS と Verifier で守って立てる。
func (w *world) serve(t *testing.T, build func(baseURL string) http.Handler) string {
	t.Helper()
	cert, err := w.ca.IssueServer("agent")
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(nil)
	srv.TLS = &tls.Config{Certificates: []tls.Certificate{cert.TLS}, ClientCAs: w.ca.Pool(), ClientAuth: tls.RequireAndVerifyClientCert, MinVersion: tls.VersionTLS13}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	srv.Config.Handler = a2aserve.Protect(build(srv.URL), w.verifier, w.cfg)
	return srv.URL
}

func (w *world) tls(name string) *tls.Config {
	return &tls.Config{Certificates: []tls.Certificate{w.certs[name].TLS}, RootCAs: w.ca.Pool(), MinVersion: tls.VersionTLS13}
}

func (w *world) client(t *testing.T, name string) *http.Client {
	t.Helper()
	return libauth.ClientCredentials(t.Context(), libauth.ClientConfig{TokenURL: w.cfg.TokenURL, ClientID: name, TLS: w.tls(name)})
}

func ask(t *testing.T, hc *http.Client, baseURL, text string) (a2a.TaskState, string) {
	t.Helper()
	c, err := a2aclient.NewFromEndpoints(t.Context(),
		[]*a2a.AgentInterface{a2a.NewAgentInterface(baseURL+"/", a2a.TransportProtocolJSONRPC)},
		a2aclient.WithJSONRPCTransport(hc))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Destroy() })
	res, err := c.SendMessage(t.Context(), &a2a.SendMessageRequest{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart(text))})
	if err != nil {
		t.Fatal(err)
	}
	task, ok := res.(*a2a.Task)
	if !ok {
		t.Fatalf("result = %T", res)
	}
	var b strings.Builder
	for _, art := range task.Artifacts {
		for _, p := range art.Parts {
			if s, ok := p.Content.(a2a.Text); ok {
				b.WriteString(string(s))
			}
		}
	}
	return task.Status.State, b.String()
}

// scriptedADK は注文を 1 回照会し、ツールの結果をそのまま答えにする。
type scriptedADK struct{}

func (scriptedADK) Name() string { return "scripted" }

func (scriptedADK) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1].Parts[0]
	part := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "get_order", Args: map[string]any{"orderId": "A-1"}}}
	if last.FunctionResponse != nil {
		raw, _ := json.Marshal(last.FunctionResponse.Response)
		part = &genai.Part{Text: "結果: " + string(raw)}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func adkAgent(t *testing.T, s *seen) agent.Agent {
	t.Helper()
	getOrder, err := functiontool.New(functiontool.Config{Name: "get_order", Description: "注文を照会する"},
		func(ctx agent.Context, _ struct {
			OrderID string `json:"orderId"`
		},
		) (map[string]any, error) {
			s.record(ctx)
			return map[string]any{"orderId": "A-1", "status": "shipped"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "operations", Model: scriptedADK{}, Tools: []tool.Tool{getOrder}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func (w *world) serveADK(t *testing.T, s *seen) string {
	t.Helper()
	scope, err := toolscope.ADKPlugin(a2aserve.Policy)
	if err != nil {
		t.Fatal(err)
	}
	a := adkAgent(t, s)
	return w.serve(t, func(base string) http.Handler {
		return a2ainterop.NewHandler(a, a2aserve.Card(w.cfg, "operations", "注文の照会"), base, scope)
	})
}

func genkitAgent(t *testing.T, s *seen) *genkitservice.GenkitAgent {
	t.Helper()
	g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/scripted"))
	genkit.DefineModel(g, "test/scripted", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true}},
		func(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			last := req.Messages[len(req.Messages)-1]
			if last.Role == ai.RoleTool {
				raw, _ := json.Marshal(last.Content[0].ToolResponse.Output)
				return &ai.ModelResponse{Message: ai.NewModelTextMessage("結果: " + string(raw)), FinishReason: ai.FinishReasonStop}, nil
			}
			return &ai.ModelResponse{
				Message:      ai.NewModelMessage(ai.NewToolRequestPart(&ai.ToolRequest{Name: "get_order", Input: map[string]any{"orderId": "A-1"}})),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})
	getOrder := genkit.DefineTool(g, "get_order", "注文を照会する",
		func(tc *ai.ToolContext, _ struct {
			OrderID string `json:"orderId"`
		},
		) (map[string]any, error) {
			s.record(tc.Context)
			return map[string]any{"orderId": "A-1", "status": "shipped"}, nil
		})
	return genkitservice.NewGenkitAgent(g, genkitservice.Definition{
		ID: "operations", SystemPrompt: "x", Tools: []ai.ToolRef{getOrder},
		Use: []ai.Middleware{toolscope.GenkitMiddleware(a2aserve.Policy)},
	})
}

func (w *world) serveGenkit(t *testing.T, s *seen) string {
	t.Helper()
	a := genkitAgent(t, s)
	return w.serve(t, func(base string) http.Handler {
		return a2ainterop.NewExecutorHandler(&a2ahandler.Executor{Agent: a}, a2aserve.Card(w.cfg, "operations", "注文の照会"), base)
	})
}

func TestBothFrameworksOverTwoFactorA2A(t *testing.T) {
	t.Parallel()
	w := newWorld(t)
	testCases := map[string]struct {
		framework  string
		caller     string
		wantAnswer string
		wantSeen   []string
	}{
		"ADK は orders.read を持つ呼び出し元にツールを使わせる":    {framework: "adk", caller: "caller-full", wantAnswer: `"status":"shipped"`, wantSeen: []string{"caller-full"}},
		"ADK は orders.read の無い呼び出し元のツールを拒否する":    {framework: "adk", caller: "caller-min", wantAnswer: "insufficient scope"},
		"Genkit は orders.read を持つ呼び出し元にツールを使わせる": {framework: "genkit", caller: "caller-full", wantAnswer: `"status":"shipped"`, wantSeen: []string{"caller-full"}},
		"Genkit は orders.read の無い呼び出し元のツールを拒否する": {framework: "genkit", caller: "caller-min", wantAnswer: "insufficient scope"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			s := &seen{}
			base := w.serveADK(t, s)
			if tc.framework == "genkit" {
				base = w.serveGenkit(t, s)
			}
			state, answer := ask(t, w.client(t, tc.caller), base, "注文 A-1 を見せて")
			if state != a2a.TaskStateCompleted || !strings.Contains(answer, tc.wantAnswer) {
				t.Fatalf("state = %s, answer = %q, want %q", state, answer, tc.wantAnswer)
			}
			if got := s.list(); fmt.Sprint(got) != fmt.Sprint(tc.wantSeen) {
				t.Errorf("ツールから見えた呼び出し元 = %v, want %v", got, tc.wantSeen)
			}
		})
	}
}

func TestAgentCardIsPublicAndRequiresBothFactors(t *testing.T) {
	t.Parallel()
	w := newWorld(t)
	base := w.serveADK(t, &seen{})
	hc := &http.Client{Transport: &http.Transport{TLSClientConfig: w.tls("caller-min")}}
	cardReq, err := http.NewRequestWithContext(t.Context(), http.MethodGet, base+a2asrv.WellKnownAgentCardPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := hc.Do(cardReq)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	body, _ := io.ReadAll(res.Body)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("Agent Card = %d", res.StatusCode)
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, body); err != nil {
		t.Fatal(err)
	}
	raw := compact.Bytes()
	for _, want := range []string{`"security":[{"mtls":[],"oauth2":["a2a.invoke"]}]`, `"mutualTLS"`, `"clientCredentials"`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("Agent Card に %s が無い: %s", want, raw)
		}
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, base+"/", strings.NewReader("{}"))
	res2, err := hc.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res2.Body.Close()
	if res2.StatusCode != http.StatusUnauthorized {
		t.Errorf("証明書だけの呼び出し = %d, want 401", res2.StatusCode)
	}
}

func TestADKAgentCallsGenkitAgentWithClientCredentials(t *testing.T) {
	t.Parallel()
	w := newWorld(t)
	s := &seen{}
	base := w.serveGenkit(t, s)
	card := a2aserve.Card(w.cfg, "operations", "注文の照会")
	card.SupportedInterfaces = []*a2a.AgentInterface{a2a.NewAgentInterface(base+"/", a2a.TransportProtocolJSONRPC)}
	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name: "orders_remote", Description: "注文の照会（A2A）", AgentCard: &card,
		ClientProvider: remoteagent.NewA2AClientProvider(a2aclient.NewFactory(a2aclient.WithJSONRPCTransport(w.client(t, "caller-full")))),
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "front", Agent: remote, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	var answer strings.Builder
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("注文 A-1 を見せて", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				answer.WriteString(p.Text)
			}
		}
	}
	if !strings.Contains(answer.String(), `"status":"shipped"`) {
		t.Fatalf("ADK から Genkit を呼んだ答え = %q", answer.String())
	}
	if got := s.list(); fmt.Sprint(got) != "[caller-full]" {
		t.Errorf("Genkit のツールから見えた呼び出し元 = %v", got)
	}
}

func TestStateCannotGrantScope(t *testing.T) {
	t.Parallel()
	s := &seen{}
	scope, err := toolscope.ADKPlugin(a2aserve.Policy)
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "ops", UserID: "u", SessionID: "s",
		State: map[string]any{"user_role": "admin", "scopes": "orders.read orders.write"},
	}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "ops", Agent: adkAgent(t, s), SessionService: svc, PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{scope}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx := libauth.WithPrincipal(t.Context(), libauth.Principal{Subject: "caller-min", Scopes: []string{"a2a.invoke"}})
	var answer strings.Builder
	for ev, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("注文 A-1", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				answer.WriteString(p.Text)
			}
		}
	}
	if len(s.list()) != 0 || !strings.Contains(answer.String(), "insufficient scope") {
		t.Errorf("State の役割で権限が上がった: seen=%v answer=%q", s.list(), answer.String())
	}
}

func TestGenkitMiddlewarePassesOutsideA2A(t *testing.T) {
	t.Parallel()
	s := &seen{}
	a := genkitAgent(t, s)
	var final *genkitmodel.ChatOutput
	for _, out := range a.Chat(t.Context(), &genkitmodel.ChatInput{SessionID: "s", UserMessage: "注文 A-1"}) {
		if out != nil {
			final = out
		}
	}
	if final == nil || !strings.Contains(final.Answer, `"status":"shipped"`) {
		t.Fatalf("A2A 以外の口（Connect）で、ツールが止められた: %+v", final)
	}
}

func TestConfigFromEnv(t *testing.T) {
	full := map[string]string{
		"A2A_ADDR": "127.0.0.1:19940", "A2A_TLS_CERT": "c", "A2A_TLS_KEY": "k", "A2A_CLIENT_CA": "ca",
		"A2A_JWKS_URL": "https://idp/jwks", "A2A_ISSUER": "https://idp", "A2A_AUDIENCE": "a2a://agents", "A2A_TOKEN_URL": "https://idp/token",
	}
	testCases := map[string]struct {
		env         map[string]string
		wantEnabled bool
		wantErr     bool
		wantUnbound bool
	}{
		"A2A_ADDR が無ければ口を出さない":    {env: map[string]string{}, wantEnabled: false},
		"そろっていれば 2 要素で出す":         {env: full, wantEnabled: true},
		"証明書の設定が欠ければ起動を止める":       {env: without(full, "A2A_CLIENT_CA"), wantEnabled: true, wantErr: true},
		"IdP の設定が欠ければ起動を止める":      {env: without(full, "A2A_AUDIENCE"), wantEnabled: true, wantErr: true},
		"開発用の設定でだけ結び付けの無いトークンを通す": {env: with(full, "A2A_DEV_ALLOW_UNBOUND_TOKENS", "true"), wantEnabled: true, wantUnbound: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			for _, k := range []string{"A2A_ADDR", "A2A_TLS_CERT", "A2A_TLS_KEY", "A2A_CLIENT_CA", "A2A_JWKS_URL", "A2A_ISSUER", "A2A_AUDIENCE", "A2A_TOKEN_URL", "A2A_DEV_ALLOW_UNBOUND_TOKENS"} {
				t.Setenv(k, tc.env[k])
			}
			cfg, enabled, err := a2aserve.ConfigFromEnv()
			if enabled != tc.wantEnabled || (err != nil) != tc.wantErr || cfg.DevAllowUnboundTokens != tc.wantUnbound {
				t.Errorf("enabled=%v err=%v unbound=%v", enabled, err, cfg.DevAllowUnboundTokens)
			}
		})
	}
}

func without(m map[string]string, key string) map[string]string {
	out := map[string]string{}
	for k, v := range m {
		if k != key {
			out[k] = v
		}
	}
	return out
}

func with(m map[string]string, key, value string) map[string]string {
	out := without(m, key)
	out[key] = value
	return out
}
