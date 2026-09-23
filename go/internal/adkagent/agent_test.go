package adkagent

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
)

// changeOnce は 1 回目に支払い方法の変更を呼び、ツールの応答を受けたら答える。
type changeOnce struct{}

func (changeOnce) Name() string { return "scripted" }

func (changeOnce) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "承認待ちです"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{
			Name: action.ToolUpdatePaymentMethod,
			Args: map[string]any{"orderId": "ord-001", "paymentMethod": "クレジットカード"},
		}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func TestWriteToolReturnsPendingAsTheCaller(t *testing.T) {
	t.Parallel()
	gate := action.Local{Service: approval.NewService(action.DefaultRules(), []string{"manager"}, time.Hour)}
	orders := backend.NewInMemoryOrders()
	tools, err := OperationsTools(orders, backend.NewInMemoryGeo(), gate)
	if err != nil {
		t.Fatal(err)
	}
	a, err := New(Definition{ID: "operations", Model: changeOnce{}, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}

	ctx := identity.With(t.Context(), "alice")
	var out *agentcore.ChatOutput
	for _, o := range a.Chat(ctx, &agentcore.ChatInput{SessionID: "s1", UserMessage: "支払い方法をカードに"}) {
		if o != nil {
			out = o
		}
	}
	if out == nil || out.ErrorMessage != "" || len(out.PendingToolCalls) != 1 {
		t.Fatalf("ChatOutput = %+v", out)
	}
	p := out.PendingToolCalls[0]
	r, err := gate.Service.Get(p.ID)
	if err != nil {
		t.Fatalf("承認の依頼が無い: %v", err)
	}
	if r.Requester != "alice" || p.Input["paymentMethod"] != "クレジットカード" {
		t.Errorf("依頼者 = %q, 引数 = %v", r.Requester, p.Input)
	}
	if o, _ := orders.GetOrder(ctx, "ord-001"); o.PaymentMethod != "銀行振込" {
		t.Errorf("承認の前に支払い方法が変わった: %s", o.PaymentMethod)
	}
}
