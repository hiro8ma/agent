package rag_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"

	"github.com/hiro8ma/agent/go/internal/search"
	"github.com/hiro8ma/agent/go/internal/search/rag"
	"github.com/hiro8ma/agent/go/internal/search/searchtest"
)

const liveModel = "googleai/gemini-3.5-flash-lite"

// TestLiveAskTutorial は教材の例の文書に実際のモデルで質問し、回答と根拠を記録する。検索は保存した埋め込みを使うので、呼ぶのは生成だけ。
// 資料に答えが無ければ根拠が 0 件になるのが正しい振る舞いなので、根拠の件数では落とさない。
//
//	go test -run LiveAskTutorial -v ./internal/search/rag/
func TestLiveAskTutorial(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY が未設定のため skip")
	}
	fx, err := searchtest.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	vec, err := search.NewFlat(searchtest.TutorialDocs, fx.DocVectors(), fx.Embedder())
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	g := genkit.Init(t.Context(), genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: key}), genkit.WithDefaultModel(liveModel))
	r := rag.RAG{G: g, Retriever: search.Hybrid{Retrievers: []search.Retriever{search.New(searchtest.TutorialDocs), vec}}}

	for _, q := range searchtest.RAGQuestions {
		start := time.Now()
		a, err := r.Ask(t.Context(), q)
		if err != nil && strings.Contains(err.Error(), "429") {
			t.Logf("%s: 429 のため 1 分後に 1 回だけやり直す", q)
			time.Sleep(time.Minute)
			start = time.Now()
			a, err = r.Ask(t.Context(), q)
		}
		if err != nil {
			t.Fatalf("Ask(%q): %v", q, err)
		}
		t.Logf("質問 %s", q)
		t.Logf("渡した文書 %v", ids(a.Retrieved))
		t.Logf("回答 %s", a.Text)
		t.Logf("根拠 %v 取り除いた ID %v", ids(a.Sources), a.Rejected)
		if a.Usage != nil {
			t.Logf("所要 %s 入力トークン %d 出力トークン %d", time.Since(start).Round(time.Millisecond), a.Usage.InputTokens, a.Usage.OutputTokens)
		}
	}
}
