package main

import (
	"context"
	"iter"
	"math/rand/v2"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/action"
)

// scripted は本物のモデルを呼ばずに、利用者の文の語でツールを選ぶ台本のモデル。運用の画面を確かめる負荷のためだけに使う。
//
//	検索 → search_knowledge（knowledge-server へ）
//	注文 → get_order（ord-404 を含めばツールが失敗する）
//	エリア → resolve_area_names（権限の表で拒否される）
//	支払い → update_order_payment_method（action-server で承認待ちになる）
//	混雑 → 1 回目は 503、再試行で成功する
//	障害 → 常に 503
type scripted struct {
	name     string
	thinking bool

	mu      sync.Mutex
	rng     *rand.Rand
	flaky   map[*model.LLMRequest]bool
	latency func() time.Duration
}

func newScripted(name string, thinking bool, seed uint64) *scripted {
	s := &scripted{name: name, thinking: thinking, rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)), flaky: map[*model.LLMRequest]bool{}}
	s.latency = s.randomLatency
	return s
}

func (s *scripted) Name() string { return s.name }

func (s *scripted) randomLatency() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.rng.IntN(100) < 4 {
		return time.Duration(1500+s.rng.IntN(2500)) * time.Millisecond
	}
	return time.Duration(60+s.rng.IntN(540)) * time.Millisecond
}

func (s *scripted) intn(n int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rng.IntN(n)
}

func (s *scripted) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		select {
		case <-time.After(s.latency()):
		case <-ctx.Done():
			yield(nil, ctx.Err())
			return
		}
		last := req.Contents[len(req.Contents)-1]
		var b strings.Builder
		var answered bool
		for _, p := range last.Parts {
			b.WriteString(p.Text)
			answered = answered || p.FunctionResponse != nil
		}
		text := b.String()
		if strings.Contains(text, "障害") {
			yield(nil, genai.APIError{Code: 503, Status: "UNAVAILABLE", Message: "scripted outage"})
			return
		}
		if strings.Contains(text, "混雑") && s.firstAttempt(req) {
			yield(nil, genai.APIError{Code: 503, Status: "UNAVAILABLE", Message: "scripted overload"})
			return
		}
		resp := &model.LLMResponse{UsageMetadata: s.usage(req)}
		switch call := s.pick(text); {
		case answered || call == nil:
			resp.Content = genai.NewContentFromText("確認しました。", genai.RoleModel)
		default:
			resp.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: call}}}
		}
		yield(resp, nil)
	}
}

func (s *scripted) firstAttempt(req *model.LLMRequest) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.flaky[req] {
		delete(s.flaky, req)
		return false
	}
	s.flaky[req] = true
	return true
}

func (s *scripted) pick(text string) *genai.FunctionCall {
	switch {
	case strings.Contains(text, "検索"):
		return &genai.FunctionCall{Name: "search_knowledge", Args: map[string]any{"query": "運用"}}
	case strings.Contains(text, "注文"):
		id := "ord-001"
		if strings.Contains(text, "ord-404") {
			id = "ord-404"
		}
		return &genai.FunctionCall{Name: "get_order", Args: map[string]any{"orderId": id}}
	case strings.Contains(text, "エリア"):
		return &genai.FunctionCall{Name: "resolve_area_names", Args: map[string]any{"areaIds": []any{"area-13104"}}}
	case strings.Contains(text, "支払い"):
		return &genai.FunctionCall{Name: action.ToolUpdatePaymentMethod, Args: map[string]any{"orderId": "ord-001", "paymentMethod": "クレジットカード"}}
	}
	return nil
}

// usage は履歴の長さに比例した入力と、ばらつく出力を返す。2 回目以降の呼び出しは入力の一部をキャッシュから読んだことにする。
func (s *scripted) usage(req *model.LLMRequest) *genai.GenerateContentResponseUsageMetadata {
	prompt := 800 + 350*len(req.Contents) + s.intn(400)
	cached := 0
	if len(req.Contents) > 1 {
		cached = prompt * (30 + s.intn(40)) / 100
	}
	thoughts := 0
	if s.thinking {
		thoughts = 100 + s.intn(600)
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(prompt),
		CachedContentTokenCount: int32(cached),
		CandidatesTokenCount:    int32(40 + s.intn(260)),
		ThoughtsTokenCount:      int32(thoughts),
	}
}
