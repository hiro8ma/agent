package a2ainterop_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"

	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
)

const testToken = "test-token"

func securedCard() a2a.AgentCard {
	card := testCard()
	a2ainterop.DeclareOAuth2(&card, "oauth2", "https://auth.example.com/oauth/token",
		map[string]string{"expense:read": "経費の読み取り", "expense:write": "経費の書き込み"}, "expense:read")
	return card
}

func newSecuredServer(t *testing.T, enforce bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(nil)
	srv.Start()
	t.Cleanup(srv.Close)
	h := a2ainterop.NewHandler(echoAgent(t), securedCard(), srv.URL)
	if enforce {
		h = a2ainterop.RequireBearer(h, a2ainterop.StaticToken(testToken, "expense:read"), "expense:read")
	}
	srv.Config.Handler = h
	return srv
}

func postAuth(t *testing.T, url, body, auth string) (int, string) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	return res.StatusCode, string(raw)
}

func TestDeclaredSchemeIsNotEnforced(t *testing.T) {
	t.Parallel()
	srv := newSecuredServer(t, false)
	// Agent Card で OAuth 2 を宣言しても、検証を置かなければトークンの無い呼び出しが通る。
	status, body := postAuth(t, srv.URL+"/", v1Send, "")
	if status != http.StatusOK || !strings.Contains(body, "受け付けました") {
		t.Fatalf("status = %d body = %s", status, body)
	}
}

func TestRequireBearerCoversBothEndpoints(t *testing.T) {
	t.Parallel()
	srv := newSecuredServer(t, true)
	endpoints := map[string]struct{ path, body string }{
		"v1.0 の口":    {path: "/", body: v1Send},
		"v0.3 の互換の口": {path: a2ainterop.V0Path, body: v0Send},
	}
	testCases := map[string]struct {
		auth string
		want int
	}{
		"トークンが無ければ 401":    {auth: "", want: http.StatusUnauthorized},
		"Bearer でなければ 401": {auth: "Basic " + testToken, want: http.StatusUnauthorized},
		"違うトークンは 401":      {auth: "Bearer other", want: http.StatusUnauthorized},
		"正しいトークンは通る":       {auth: "Bearer " + testToken, want: http.StatusOK},
	}
	for en, ep := range endpoints {
		for tn, tc := range testCases {
			t.Run(en+"/"+tn, func(t *testing.T) {
				t.Parallel()
				status, body := postAuth(t, srv.URL+ep.path, ep.body, tc.auth)
				if status != tc.want {
					t.Fatalf("status = %d, want %d: %s", status, tc.want, body)
				}
				if tc.want == http.StatusOK && !strings.Contains(body, "受け付けました") {
					t.Errorf("body = %s", body)
				}
			})
		}
	}
}

func TestRequireBearerRejectsMissingScope(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(a2ainterop.RequireBearer(
		http.NotFoundHandler(), a2ainterop.StaticToken(testToken, "expense:write"), "expense:read"))
	t.Cleanup(srv.Close)
	status, _ := postAuth(t, srv.URL+"/", v1Send, "Bearer "+testToken)
	if status != http.StatusForbidden {
		t.Errorf("status = %d, want 403", status)
	}
}

func TestAgentCardStaysPublicAndDeclaresRequirement(t *testing.T) {
	t.Parallel()
	srv := newSecuredServer(t, true)
	card := get(t, srv.URL+a2asrv.WellKnownAgentCardPath)

	// 互換の Agent Card は、v0.3 の security と v1.0 の securityRequirements の両方に要件を書く。
	raw, _ := json.Marshal(card)
	for _, key := range []string{`"security":[{"oauth2":["expense:read"]}]`, `"securityRequirements"`, `"clientCredentials"`, `"tokenUrl"`} {
		if !strings.Contains(string(raw), key) {
			t.Errorf("Agent Card に %s が無い: %s", key, raw)
		}
	}
}

func TestWrappingOnlyV1LeavesCompatEndpointOpen(t *testing.T) {
	t.Parallel()
	exec := adka2a.NewExecutor(adka2a.ExecutorConfig{RunnerConfig: runner.Config{
		AppName: "expense", Agent: echoAgent(t), SessionService: session.InMemoryService(),
	}})
	handler := a2asrv.NewHandler(exec)
	verify := a2ainterop.StaticToken(testToken, "expense:read")
	mux := http.NewServeMux()
	mux.Handle("/", a2ainterop.RequireBearer(a2asrv.NewJSONRPCHandler(handler), verify, "expense:read"))
	mux.Handle(a2ainterop.V0Path, a2av0.NewJSONRPCHandler(handler))
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	if status, _ := postAuth(t, srv.URL+"/", v1Send, ""); status != http.StatusUnauthorized {
		t.Fatalf("v1.0 の口 status = %d, want 401", status)
	}
	// v1.0 の口だけを守ると、互換の口からトークンなしで呼べてしまう。
	status, body := postAuth(t, srv.URL+a2ainterop.V0Path, v0Send, "")
	if status != http.StatusOK || !strings.Contains(body, "受け付けました") {
		t.Fatalf("互換の口 status = %d body = %s", status, body)
	}
}
