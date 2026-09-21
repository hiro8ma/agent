package a2ainterop_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/a2asrv/push"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"

	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
)

// recorder は Webhook に届いた要求のヘッダーと、受け入れた出来事を記録する。
type recorder struct {
	mu      sync.Mutex
	headers []http.Header
	states  []a2a.TaskState
}

func (r *recorder) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.mu.Lock()
		r.headers = append(r.headers, req.Header.Clone())
		r.mu.Unlock()
		next.ServeHTTP(w, req)
	})
}

func (r *recorder) onEvent(_ context.Context, e a2a.Event) {
	var state a2a.TaskState
	switch v := e.(type) {
	case *a2a.Task:
		state = v.Status.State
	case *a2a.TaskStatusUpdateEvent:
		state = v.Status.State
	default:
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.states = append(r.states, state)
}

func (r *recorder) snapshot() ([]http.Header, []a2a.TaskState) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]http.Header(nil), r.headers...), append([]a2a.TaskState(nil), r.states...)
}

// pushServer は Push Notification の送り手を組み込んだ A2A v1.0 のサーバー。
func pushServer(t *testing.T, sender push.Sender) *httptest.Server {
	t.Helper()
	exec := adka2a.NewExecutor(adka2a.ExecutorConfig{RunnerConfig: runner.Config{
		AppName: "expense", Agent: echoAgent(t), SessionService: session.InMemoryService(),
	}})
	h := a2asrv.NewHandler(exec, a2asrv.WithPushNotifications(push.NewInMemoryStore(), sender))
	srv := httptest.NewServer(a2asrv.NewJSONRPCHandler(h))
	t.Cleanup(srv.Close)
	return srv
}

func sendWithPush(t *testing.T, srvURL string, cfg *a2a.PushConfig) {
	t.Helper()
	client, err := a2aclient.NewFromEndpoints(t.Context(), []*a2a.AgentInterface{
		a2a.NewAgentInterface(srvURL, a2a.TransportProtocolJSONRPC),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Destroy() })
	msg := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("先月の経費"))
	if _, err := client.SendMessage(t.Context(), &a2a.SendMessageRequest{
		Message: msg, Config: &a2a.SendMessageConfig{PushConfig: cfg},
	}); err != nil {
		t.Fatal(err)
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("時間内に条件が満たされなかった")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestGoSenderSendsTokenAndCredentials(t *testing.T) {
	t.Parallel()
	notes := a2ainterop.NewNotifications()
	rec := &recorder{}
	webhook := httptest.NewServer(rec.wrap(notes.Handler(rec.onEvent)))
	t.Cleanup(webhook.Close)

	srv := pushServer(t, push.NewHTTPPushSender(&push.HTTPSenderConfig{AllowPrivateNetworks: true}))
	token := notes.Issue()
	sendWithPush(t, srv.URL, &a2a.PushConfig{
		URL: webhook.URL, Token: token, Auth: &a2a.PushAuthInfo{Scheme: "Bearer", Credentials: "webhook-secret"},
	})

	waitFor(t, func() bool {
		_, states := rec.snapshot()
		return len(states) > 0 && states[len(states)-1] == a2a.TaskStateCompleted
	})
	headers, _ := rec.snapshot()
	// a2a-go の送り手は、トークンを v1.0 の名前のヘッダーで送り、認証情報も Authorization で送る。
	for _, h := range headers {
		if h.Get("A2A-Notification-Token") != token || h.Get("X-A2A-Notification-Token") != "" {
			t.Errorf("トークンのヘッダー = %v", h)
		}
		if h.Get("Authorization") != "Bearer webhook-secret" {
			t.Errorf("Authorization = %q", h.Get("Authorization"))
		}
	}
}

func TestGoSenderBlocksLoopbackByDefault(t *testing.T) {
	t.Parallel()
	notes := a2ainterop.NewNotifications()
	rec := &recorder{}
	webhook := httptest.NewServer(rec.wrap(notes.Handler(rec.onEvent)))
	t.Cleanup(webhook.Close)

	// 既定の送り手は SSRF の対策で、ループバックや社内のアドレスへ送らない。呼び出し自体は成功する。
	srv := pushServer(t, push.NewHTTPPushSender(nil))
	sendWithPush(t, srv.URL, &a2a.PushConfig{URL: webhook.URL, Token: notes.Issue()})

	// 許可した送り手で同じ構成を送り、届くまでの時間を上回るだけ待ってから確かめる。
	allowed := pushServer(t, push.NewHTTPPushSender(&push.HTTPSenderConfig{AllowPrivateNetworks: true}))
	control := &recorder{}
	controlHook := httptest.NewServer(control.wrap(notes.Handler(control.onEvent)))
	t.Cleanup(controlHook.Close)
	sendWithPush(t, allowed.URL, &a2a.PushConfig{URL: controlHook.URL, Token: notes.Issue()})
	waitFor(t, func() bool { h, _ := control.snapshot(); return len(h) > 0 })

	if headers, _ := rec.snapshot(); len(headers) != 0 {
		t.Errorf("既定の送り手がループバックへ %d 件送った", len(headers))
	}
}

func TestNotificationHandlerChecksToken(t *testing.T) {
	t.Parallel()
	notes := a2ainterop.NewNotifications()
	rec := &recorder{}
	srv := httptest.NewServer(notes.Handler(rec.onEvent))
	t.Cleanup(srv.Close)
	token := notes.Issue()
	revoked := notes.Issue()
	notes.Revoke(revoked)

	body := func(task string) string {
		raw, err := json.Marshal(a2a.StreamResponse{Event: &a2a.TaskStatusUpdateEvent{
			TaskID: a2a.TaskID(task), ContextID: "c", Status: a2a.TaskStatus{State: a2a.TaskStateWorking},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	post := func(header, token, body string) int {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, srv.URL, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if header != "" {
			req.Header.Set(header, token)
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		return res.StatusCode
	}

	// 順序に意味がある。最初の通知でトークンがタスク t1 に結び付く。
	steps := []struct {
		name, header, token, body string
		want                      int
	}{
		{name: "トークンが無ければ 401", header: "", body: body("t1"), want: http.StatusUnauthorized},
		{name: "発行していないトークンは 401", header: "A2A-Notification-Token", token: "guess", body: body("t1"), want: http.StatusUnauthorized},
		{name: "無効にしたトークンは 401", header: "A2A-Notification-Token", token: revoked, body: body("t1"), want: http.StatusUnauthorized},
		{name: "本文が v1.0 の形でなければ 400", header: "A2A-Notification-Token", token: token, body: `{"id":"t1","status":{"state":"working"}}`, want: http.StatusBadRequest},
		{name: "最初の通知でタスクに結び付く", header: "A2A-Notification-Token", token: token, body: body("t1"), want: http.StatusNoContent},
		{name: "v0.3 のヘッダー名でも読む", header: "X-A2A-Notification-Token", token: token, body: body("t1"), want: http.StatusNoContent},
		{name: "ほかのタスクの通知には使えない", header: "A2A-Notification-Token", token: token, body: body("t2"), want: http.StatusForbidden},
	}
	for _, s := range steps {
		if got := post(s.header, s.token, s.body); got != s.want {
			t.Errorf("%s: status = %d, want %d", s.name, got, s.want)
		}
	}
	if _, states := rec.snapshot(); len(states) != 2 {
		t.Errorf("受け入れた通知 = %d 件, want 2", len(states))
	}
}
