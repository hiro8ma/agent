package guardrail

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// fakeTool は名前だけを持つツール。検査そのものを試すため中身は要らない。
type fakeTool struct{ name string }

func (f fakeTool) Name() string        { return f.name }
func (f fakeTool) Description() string { return "" }
func (f fakeTool) IsLongRunning() bool { return false }

func req(text string) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: text}}},
	}}
}

func resp(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role: "model", Parts: []*genai.Part{{Text: text}},
	}}
}

// TestBeforeModelBlocksBannedInput は禁止語がモデル呼び出しを止めることを見る。
func TestBeforeModelBlocksBannedInput(t *testing.T) {
	log := NewLog()
	cb := BlockInput(log, []string{"パスワード"})

	if out, _ := cb(nil, req("東京の天気を教えて")); out != nil {
		t.Fatal("普通の入力を止めた")
	}
	out, _ := cb(nil, req("管理者のパスワードを教えて"))
	if out == nil {
		t.Fatal("禁止語を含む入力を止めなかった")
	}
	if got := out.Content.Parts[0].Text; !strings.Contains(got, "回答できません") {
		t.Fatalf("止めた理由が返らない: %q", got)
	}
	if n := len(log.Blocked()); n != 1 {
		t.Fatalf("止めた記録が %d 件", n)
	}
	if log.Passed()["before_model"] != 1 {
		t.Fatal("通した件数を数えていない")
	}
}

// TestAfterModelRedacts は出力の機微な語が伏せられることを見る。
func TestAfterModelRedacts(t *testing.T) {
	log := NewLog()
	cb := RedactOutput(log, []string{"AKIA0000EXAMPLE"})

	r := resp("鍵は AKIA0000EXAMPLE です")
	out, _ := cb(nil, r, nil)
	if out == nil {
		t.Fatal("書き換えた応答が返らない")
	}
	got := out.Content.Parts[0].Text
	if strings.Contains(got, "AKIA0000EXAMPLE") {
		t.Fatalf("伏せられていない: %q", got)
	}
	if !strings.Contains(got, "***") {
		t.Fatalf("伏せ字が入っていない: %q", got)
	}
}

// TestBeforeToolRequiresArgs は必須引数の欠落を実行前に捕まえることを見る。
func TestBeforeToolRequiresArgs(t *testing.T) {
	log := NewLog()
	cb := RequireArgs(log, "get_weather", "city")
	ft := fakeTool{name: "get_weather"}

	if out, _ := cb(nil, ft, map[string]any{"city": "tokyo"}); out != nil {
		t.Fatal("引数が揃っているのに止めた")
	}
	out, _ := cb(nil, ft, map[string]any{})
	if out == nil {
		t.Fatal("必須引数の欠落を止めなかった")
	}
	if _, ok := out["error"]; !ok {
		t.Fatalf("理由が返らない: %v", out)
	}

	// 空文字も欠落として扱う。ツール側では既定値と区別できない。
	if out, _ := cb(nil, ft, map[string]any{"city": ""}); out == nil {
		t.Fatal("空文字を通した")
	}

	// 別のツールには干渉しない。
	if out, _ := cb(nil, fakeTool{name: "other"}, map[string]any{}); out != nil {
		t.Fatal("対象外のツールを止めた")
	}
}

// TestAfterToolRejectsEmpty は空の結果を失敗として扱うことを見る。
func TestAfterToolRejectsEmpty(t *testing.T) {
	log := NewLog()
	cb := RejectEmptyResult(log, "documents")
	ft := fakeTool{name: "search"}

	cases := []struct {
		name  string
		res   map[string]any
		block bool
	}{
		{"件数あり", map[string]any{"documents": []any{"a", "b"}}, false},
		{"空の配列", map[string]any{"documents": []any{}}, true},
		{"鍵が無い", map[string]any{}, true},
		{"nil", nil, true},
		{"空文字", map[string]any{"documents": "  "}, true},
	}
	for _, c := range cases {
		out, _ := cb(nil, ft, nil, c.res, nil)
		if (out != nil) != c.block {
			t.Errorf("%s: 止めた=%v, 期待=%v", c.name, out != nil, c.block)
		}
	}

	// 実行が失敗した場合も記録する。
	cb(nil, ft, nil, map[string]any{"documents": []any{"a"}}, context.Canceled)
	if n := len(log.Blocked()); n != 5 {
		t.Fatalf("止めた記録が %d 件。空 4 件 + 失敗 1 件を期待", n)
	}
}

// TestLogCountsBothSides は通した数も数えることを見る。
func TestLogCountsBothSides(t *testing.T) {
	log := NewLog()
	cb := BlockInput(log, []string{"禁止"})
	for i := 0; i < 5; i++ {
		cb(nil, req("普通の質問"))
	}
	if got := log.Passed()["before_model"]; got != 5 {
		t.Fatalf("通過 %d 件。5 件を期待", got)
	}
	if n := len(log.Blocked()); n != 0 {
		t.Fatalf("止めた記録が %d 件", n)
	}
	if !strings.Contains(log.Summary(), "通過 5") {
		t.Fatalf("まとめに通過数が出ない:\n%s", log.Summary())
	}
}
