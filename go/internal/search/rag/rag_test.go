package rag_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/search"
	"github.com/hiro8ma/agent/go/internal/search/rag"
)

type fixedRetriever []search.Doc

func (f fixedRetriever) Search(_ context.Context, _ string, limit int) ([]search.Doc, error) {
	return f[:min(limit, len(f))], nil
}

// fakeModel は決まった根拠の ID を返し、受け取ったプロンプトと呼ばれた回数を残す。
type fakeModel struct {
	ids    []string
	calls  atomic.Int32
	prompt atomic.Value
}

func (f *fakeModel) generate(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	f.calls.Add(1)
	var prompt strings.Builder
	for _, m := range req.Messages {
		if m.Role == ai.RoleUser {
			prompt.WriteString(m.Text())
		}
	}
	f.prompt.Store(prompt.String())
	b, err := json.Marshal(rag.Output{Answer: "ANSWER", SourceIDs: f.ids})
	if err != nil {
		return nil, err
	}
	return &ai.ModelResponse{Message: ai.NewModelTextMessage(string(b)), FinishReason: ai.FinishReasonStop}, nil
}

func newRAG(t *testing.T, docs []search.Doc, ids []string) (rag.RAG, *fakeModel) {
	t.Helper()
	f := &fakeModel{ids: ids}
	g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/fake"))
	genkit.DefineModel(g, "test/fake", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Constrained: ai.ConstrainedSupportAll},
	}, f.generate)
	return rag.RAG{G: g, Retriever: fixedRetriever(docs)}, f
}

func ids(docs []search.Doc) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

var curry = []search.Doc{
	{ID: "d1", Content: "カレーライスはおいしい"},
	{ID: "d3", Content: "カレーはナンに合う"},
	{ID: "d7", Content: "キーマカレーの専門店"},
	{ID: "d4", Content: "猫はかわいい"},
}

func TestAskKeepsOnlyRetrievedSourceIDs(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		modelIDs     []string
		wantSources  []string
		wantRejected []string
	}{
		"渡した文書の ID はそのまま根拠にする": {
			modelIDs:    []string{"d3", "d1"},
			wantSources: []string{"d3", "d1"},
		},
		"渡していない ID は取り除く": {
			modelIDs:     []string{"d1", "d9"},
			wantSources:  []string{"d1"},
			wantRejected: []string{"d9"},
		},
		"索引にはあるが上位に入らず渡していない ID も取り除く": {
			modelIDs:     []string{"d4"},
			wantRejected: []string{"d4"},
		},
		"角括弧つきや重複した ID は 1 件にまとめる": {
			modelIDs:    []string{"[d7]", " d7 ", "d7"},
			wantSources: []string{"d7"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r, f := newRAG(t, curry, tc.modelIDs)
			a, err := r.Ask(t.Context(), "日本のカレーの特徴は？")
			if err != nil {
				t.Fatalf("Ask: %v", err)
			}
			if a.Text != "ANSWER" {
				t.Errorf("Text = %q, want ANSWER", a.Text)
			}
			if got := ids(a.Sources); !slices.Equal(got, tc.wantSources) {
				t.Errorf("Sources = %v, want %v", got, tc.wantSources)
			}
			if !slices.Equal(a.Rejected, tc.wantRejected) {
				t.Errorf("Rejected = %v, want %v", a.Rejected, tc.wantRejected)
			}
			if got := ids(a.Retrieved); !slices.Equal(got, []string{"d1", "d3", "d7"}) {
				t.Errorf("Retrieved = %v, want the top %d", got, rag.DefaultLimit)
			}
			prompt, _ := f.prompt.Load().(string)
			for _, want := range []string{"[d1]", "カレーライスはおいしい", "[d7]", "日本のカレーの特徴は？"} {
				if !strings.Contains(prompt, want) {
					t.Errorf("prompt lacks %q", want)
				}
			}
			if strings.Contains(prompt, "猫はかわいい") {
				t.Error("prompt contains a document outside the top results")
			}
		})
	}
}

func TestAskWithoutDocumentsSkipsModel(t *testing.T) {
	t.Parallel()
	r, f := newRAG(t, nil, []string{"d1"})
	a, err := r.Ask(t.Context(), "日本のカレーの特徴は？")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if a.Text != rag.NoContextAnswer || len(a.Sources) != 0 {
		t.Fatalf("answer = %+v, want NoContextAnswer without sources", a)
	}
	if n := f.calls.Load(); n != 0 {
		t.Fatalf("model called %d times, want 0", n)
	}
}

func TestPromptNamesDocsWithoutID(t *testing.T) {
	t.Parallel()
	p := rag.Prompt("q", []search.Doc{{Title: "T", Content: "a"}, {ID: "x", Content: "b"}})
	for _, want := range []string{"## [doc1] T\na", "## [x]\nb"} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt lacks %q:\n%s", want, p)
		}
	}
}
