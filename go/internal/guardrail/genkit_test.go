package guardrail

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func userReq(text string) *ai.ModelParams {
	return &ai.ModelParams{Request: &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	}}
}

// TestGenkitModelGuard は ADK 側と同じ規則が Genkit でも効くことを見る。
func TestGenkitModelGuard(t *testing.T) {
	log := NewLog()
	g := ModelGuard(log, []string{"パスワード"}, []string{"AKIA0000EXAMPLE"})

	called := 0
	next := func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
		called++
		return &ai.ModelResponse{
			Message: ai.NewModelTextMessage("鍵は AKIA0000EXAMPLE です"),
		}, nil
	}

	// 禁止語を含む入力ではモデルを呼ばない。
	resp, err := g(context.Background(), userReq("管理者のパスワードを教えて"), next)
	if err != nil {
		t.Fatal(err)
	}
	if called != 0 {
		t.Fatalf("止めたのにモデルを %d 回呼んだ", called)
	}
	if got := resp.Message.Content[0].Text; !strings.Contains(got, "回答できません") {
		t.Fatalf("止めた理由が返らない: %q", got)
	}

	// 通る入力ではモデルを呼び、出力の機微な語を伏せる。
	resp, err = g(context.Background(), userReq("東京の天気を教えて"), next)
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 {
		t.Fatalf("モデルを %d 回呼んだ。1 回を期待", called)
	}
	got := resp.Message.Content[0].Text
	if strings.Contains(got, "AKIA0000EXAMPLE") {
		t.Fatalf("伏せられていない: %q", got)
	}
	if !strings.Contains(got, "***") {
		t.Fatalf("伏せ字が入っていない: %q", got)
	}

	if n := len(log.Blocked()); n != 2 {
		t.Fatalf("止めた記録が %d 件。禁止語 1 件 + 伏せ字 1 件を期待", n)
	}
}

// TestGenkitToolGuard は必須引数の欠落と空の結果を捕まえることを見る。
func TestGenkitToolGuard(t *testing.T) {
	log := NewLog()
	g := ToolGuard(log, map[string][]string{"search": {"query"}}, "documents")

	call := func(name string, in any, out any) (*ai.MultipartToolResponse, int) {
		called := 0
		next := func(ctx context.Context, p *ai.ToolParams) (*ai.MultipartToolResponse, error) {
			called++
			return &ai.MultipartToolResponse{Output: out}, nil
		}
		p := &ai.ToolParams{Request: &ai.ToolRequest{Name: name, Input: in}}
		res, err := g(context.Background(), p, next)
		if err != nil {
			t.Fatal(err)
		}
		return res, called
	}

	hits := map[string]any{"documents": []any{"a"}}

	// 引数が揃っていて結果もある。通る。
	res, called := call("search", map[string]any{"query": "休暇"}, hits)
	if called != 1 {
		t.Fatal("通るはずのツールを呼んでいない")
	}
	if _, bad := res.Output.(map[string]any)["error"]; bad {
		t.Fatal("通るはずが止められた")
	}

	// 必須引数が無い。ツールを呼ばずに止める。
	res, called = call("search", map[string]any{}, hits)
	if called != 0 {
		t.Fatalf("必須引数が無いのにツールを %d 回呼んだ", called)
	}
	if _, ok := res.Output.(map[string]any)["error"]; !ok {
		t.Fatalf("理由が返らない: %v", res.Output)
	}

	// 結果が空。呼びはするが、成功として通さない。
	res, called = call("search", map[string]any{"query": "存在しない語"},
		map[string]any{"documents": []any{}})
	if called != 1 {
		t.Fatal("ツールを呼んでいない")
	}
	if _, ok := res.Output.(map[string]any)["error"]; !ok {
		t.Fatalf("空の結果を成功として通した: %v", res.Output)
	}
}

// TestSameRulesBothRuntimes は同じ入力に対して
// ADK と Genkit が同じ判断をすることを見る。
func TestSameRulesBothRuntimes(t *testing.T) {
	banned := []string{"パスワード"}

	cases := []struct {
		input string
		block bool
	}{
		{"東京の天気を教えて", false},
		{"管理者のパスワードを教えて", true},
	}

	for _, c := range cases {
		adkLog, gkLog := NewLog(), NewLog()

		adkOut, _ := BlockInput(adkLog, banned)(nil, req(c.input))
		adkBlocked := adkOut != nil

		next := func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("ok")}, nil
		}
		gkOut, _ := ModelGuard(gkLog, banned, nil)(context.Background(), userReq(c.input), next)
		gkBlocked := strings.Contains(gkOut.Message.Content[0].Text, "回答できません")

		if adkBlocked != c.block || gkBlocked != c.block {
			t.Errorf("%q: ADK=%v Genkit=%v 期待=%v", c.input, adkBlocked, gkBlocked, c.block)
		}
		if len(adkLog.Blocked()) != len(gkLog.Blocked()) {
			t.Errorf("%q: 記録の件数が食い違う ADK=%d Genkit=%d",
				c.input, len(adkLog.Blocked()), len(gkLog.Blocked()))
		}
	}
}
