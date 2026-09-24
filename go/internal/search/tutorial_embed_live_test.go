package search_test

import (
	"encoding/json"
	"os"
	"testing"

	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/search/searchtest"
)

const (
	embeddingModel      = "gemini-embedding-001"
	embeddingDimensions = 768
)

// TestLiveWriteTutorialEmbeddings は教材の例の埋め込みを作り直して testdata に保存する。ふだんのテストは保存したものを読むだけで API を呼ばない。
//
//	SEARCH_UPDATE_EMBEDDINGS=1 go test -run LiveWriteTutorialEmbeddings ./internal/search/
func TestLiveWriteTutorialEmbeddings(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if os.Getenv("SEARCH_UPDATE_EMBEDDINGS") != "1" || key == "" {
		t.Skip("SEARCH_UPDATE_EMBEDDINGS=1 と GEMINI_API_KEY が要る")
	}
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{APIKey: key, Backend: genai.BackendGeminiAPI})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	embed := func(texts []string, task string) [][]float32 {
		contents := make([]*genai.Content, len(texts))
		for i, s := range texts {
			contents[i] = genai.NewContentFromText(s, genai.RoleUser)
		}
		dim := int32(embeddingDimensions)
		res, err := client.Models.EmbedContent(t.Context(), embeddingModel, contents, &genai.EmbedContentConfig{TaskType: task, OutputDimensionality: &dim})
		if err != nil {
			t.Fatalf("EmbedContent(%s): %v", task, err)
		}
		if len(res.Embeddings) != len(texts) {
			t.Fatalf("embeddings = %d, want %d", len(res.Embeddings), len(texts))
		}
		out := make([][]float32, len(texts))
		for i, e := range res.Embeddings {
			out[i] = e.Values
		}
		return out
	}

	docTexts := make([]string, len(searchtest.TutorialDocs))
	for i, d := range searchtest.TutorialDocs {
		docTexts[i] = d.Content
	}
	queryTexts := append(append([]string{}, searchtest.TutorialQueries...), searchtest.RAGQuestions...)
	fx := searchtest.Fixture{Model: embeddingModel, Dimensions: embeddingDimensions}
	for i, v := range embed(docTexts, "RETRIEVAL_DOCUMENT") {
		fx.Docs = append(fx.Docs, searchtest.Vector{ID: searchtest.TutorialDocs[i].ID, Text: docTexts[i], Values: v})
	}
	for i, v := range embed(queryTexts, "RETRIEVAL_QUERY") {
		fx.Queries = append(fx.Queries, searchtest.Vector{Text: queryTexts[i], Values: v})
	}
	b, err := json.Marshal(fx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(searchtest.Path(), append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}
