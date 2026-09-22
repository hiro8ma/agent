package llmretry_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/llmretry"
)

func apiErr(code int, status string, details ...map[string]any) error {
	return genai.APIError{Code: code, Status: status, Message: "x", Details: details}
}

func retryInfo(delay string) map[string]any {
	return map[string]any{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
}

func quotaFailure(quotaID string) map[string]any {
	return map[string]any{
		"@type":      "type.googleapis.com/google.rpc.QuotaFailure",
		"violations": []any{map[string]any{"quotaId": quotaID}},
	}
}

func policy() llmretry.Policy {
	p := llmretry.DefaultPolicy()
	p.Jitter = func(d time.Duration) time.Duration { return d }
	return p
}

func TestDecide(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		err         error
		attempt     int
		maxAttempts int
		want        llmretry.Decision
	}{
		"400 は直しても同じなので再試行しない":         {err: apiErr(400, "INVALID_ARGUMENT"), attempt: 1, want: llmretry.Decision{}},
		"403 は再試行しない":                  {err: apiErr(403, "PERMISSION_DENIED"), attempt: 1, want: llmretry.Decision{}},
		"404 は再試行しない":                  {err: apiErr(404, "NOT_FOUND"), attempt: 1, want: llmretry.Decision{}},
		"429 は RetryInfo の待ち時間に従う":     {err: apiErr(429, "RESOURCE_EXHAUSTED", retryInfo("7.5s")), attempt: 1, want: llmretry.Decision{Retry: true, Wait: 7500 * time.Millisecond}},
		"429 で RetryInfo が無ければ指数バックオフ": {err: apiErr(429, "RESOURCE_EXHAUSTED"), attempt: 3, want: llmretry.Decision{Retry: true, Wait: 4 * time.Second}},
		"429 の日次の上限は待っても回復しないので再試行しない": {err: apiErr(429, "RESOURCE_EXHAUSTED", quotaFailure("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), retryInfo("10s")), attempt: 1, want: llmretry.Decision{}},
		"429 の分単位の上限は再試行する":            {err: apiErr(429, "RESOURCE_EXHAUSTED", quotaFailure("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), retryInfo("20s")), attempt: 1, want: llmretry.Decision{Retry: true, Wait: 20 * time.Second}},
		"429 の指示が上限を超えたら諦める":           {err: apiErr(429, "RESOURCE_EXHAUSTED", retryInfo("45s")), attempt: 1, want: llmretry.Decision{}},
		"503 は試行回数の上限まで再試行する":          {err: apiErr(503, "UNAVAILABLE"), attempt: 3, want: llmretry.Decision{Retry: true, Wait: 4 * time.Second}},
		"503 も上限に達したら諦める":              {err: apiErr(503, "UNAVAILABLE"), attempt: 4, want: llmretry.Decision{Wait: 8 * time.Second}},
		"504 は再試行する":                   {err: apiErr(504, "DEADLINE_EXCEEDED"), attempt: 1, want: llmretry.Decision{Retry: true, Wait: time.Second}},
		"500 は 1 回だけ再試行する":             {err: apiErr(500, "INTERNAL"), attempt: 1, want: llmretry.Decision{Retry: true, Wait: time.Second}},
		"500 の 2 回目は諦める":               {err: apiErr(500, "INTERNAL"), attempt: 2, want: llmretry.Decision{Wait: 2 * time.Second}},
		"ADK が包んだエラーも中身で判断する":          {err: fmt.Errorf("failed to call model: %w", apiErr(503, "UNAVAILABLE")), attempt: 1, want: llmretry.Decision{Retry: true, Wait: time.Second}},
		"呼び出し元の取り消しは再試行しない":            {err: fmt.Errorf("x: %w", context.Canceled), attempt: 1, want: llmretry.Decision{}},
		"種類の分からないエラーは再試行しない":           {err: errors.New("boom"), attempt: 1, want: llmretry.Decision{}},
		"待ち時間は上限で頭打ちになる":               {err: apiErr(503, "UNAVAILABLE"), attempt: 10, maxAttempts: 20, want: llmretry.Decision{Retry: true, Wait: 30 * time.Second}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			p := policy()
			if tc.maxAttempts > 0 {
				p.MaxAttempts = tc.maxAttempts
			}
			if got := llmretry.Decide(tc.err, tc.attempt, p); got != tc.want {
				t.Errorf("Decide = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// script は呼ばれるたびに決めた結果を返す台本のモデル。
type script struct {
	mu    sync.Mutex
	calls int
	steps [][]step
}

type step struct {
	text string
	err  error
}

func (s *script) Name() string { return "scripted" }

func (s *script) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	s.mu.Lock()
	steps := s.steps[min(s.calls, len(s.steps)-1)]
	s.calls++
	s.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		for _, st := range steps {
			if st.err != nil {
				yield(nil, st.err)
				return
			}
			resp := &model.LLMResponse{Content: genai.NewContentFromText(st.text, genai.RoleModel)}
			if !yield(resp, nil) {
				return
			}
		}
	}
}

func TestWrap(t *testing.T) {
	t.Parallel()
	unavailable := apiErr(503, "UNAVAILABLE")
	testCases := map[string]struct {
		steps     [][]step
		wantTexts []string
		wantErr   bool
		wantCalls int
		wantWaits []time.Duration
	}{
		"503 が 2 回続いても 3 回目で返る": {
			steps:     [][]step{{{err: unavailable}}, {{err: unavailable}}, {{text: "7月の経費は12件です"}}},
			wantTexts: []string{"7月の経費は12件です"}, wantCalls: 3, wantWaits: []time.Duration{time.Second, 2 * time.Second},
		},
		"400 は 1 回で諦める": {
			steps:   [][]step{{{err: apiErr(400, "INVALID_ARGUMENT")}}},
			wantErr: true, wantCalls: 1,
		},
		"一部を返した後の失敗は再試行しない": {
			steps:     [][]step{{{text: "7月の"}, {err: unavailable}}, {{text: "7月の経費は12件です"}}},
			wantTexts: []string{"7月の"}, wantErr: true, wantCalls: 1,
		},
		"試行回数の上限で諦める": {
			steps:   [][]step{{{err: unavailable}}},
			wantErr: true, wantCalls: 4, wantWaits: []time.Duration{time.Second, 2 * time.Second, 4 * time.Second},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			inner := &script{steps: tc.steps}
			var waits []time.Duration
			p := policy()
			p.Sleep = func(_ context.Context, d time.Duration) error { waits = append(waits, d); return nil }
			var texts []string
			var gotErr error
			for resp, err := range llmretry.Wrap(inner, p).GenerateContent(t.Context(), &model.LLMRequest{}, true) {
				if err != nil {
					gotErr = err
					continue
				}
				texts = append(texts, resp.Content.Parts[0].Text)
			}
			if (gotErr != nil) != tc.wantErr {
				t.Errorf("err = %v, wantErr %v", gotErr, tc.wantErr)
			}
			if !slices.Equal(texts, tc.wantTexts) {
				t.Errorf("texts = %v, want %v", texts, tc.wantTexts)
			}
			if inner.calls != tc.wantCalls {
				t.Errorf("calls = %d, want %d", inner.calls, tc.wantCalls)
			}
			if !slices.Equal(waits, tc.wantWaits) {
				t.Errorf("waits = %v, want %v", waits, tc.wantWaits)
			}
		})
	}
}

func TestWrapStopsWaitingWhenCanceled(t *testing.T) {
	t.Parallel()
	inner := &script{steps: [][]step{{{err: apiErr(503, "UNAVAILABLE")}}}}
	p := llmretry.DefaultPolicy()
	p.BaseDelay = time.Hour
	p.MaxDelay = time.Hour
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	start := time.Now()
	for _, err := range llmretry.Wrap(inner, p).GenerateContent(ctx, &model.LLMRequest{}, false) {
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled を含む", err)
		}
	}
	if time.Since(start) > time.Second || inner.calls != 1 {
		t.Errorf("取り消し後も待ったか呼んだ: %v, calls=%d", time.Since(start), inner.calls)
	}
}

func TestWrapInsideAgent(t *testing.T) {
	t.Parallel()
	inner := &script{steps: [][]step{{{err: apiErr(429, "RESOURCE_EXHAUSTED", retryInfo("0.01s"))}}, {{text: "7月の経費は12件です"}}}}
	a, err := llmagent.New(llmagent.Config{Name: "expense", Model: llmretry.Wrap(inner, llmretry.DefaultPolicy())})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "office", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for ev, err := range r.Run(t.Context(), "alice", "s1", genai.NewContentFromText("7月の経費は？", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Content != nil {
			texts = append(texts, ev.Content.Parts[0].Text)
		}
	}
	if !slices.Equal(texts, []string{"7月の経費は12件です"}) || inner.calls != 2 {
		t.Errorf("texts = %v, calls = %d", texts, inner.calls)
	}
}
