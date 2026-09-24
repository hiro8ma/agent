// Package searchtest は実際の埋め込みモデルで 1 回だけ作ったベクトルを読み、外部の API を呼ばずに検索を試すためのもの。
package searchtest

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/hiro8ma/agent/go/internal/search"
)

// TutorialDocs は教材の例の文書。7 件目は「カレー」を含む語が BM25 とベクトルで扱いの分かれる例として足した。
var TutorialDocs = []search.Doc{
	{ID: "d1", Content: "カレーライスはおいしい"},
	{ID: "d2", Content: "インド料理のお店"},
	{ID: "d3", Content: "カレーはナンに合う"},
	{ID: "d4", Content: "猫はかわいい"},
	{ID: "d5", Content: "アクビをする猫"},
	{ID: "d6", Content: "転置インデックスは便利"},
	{ID: "d7", Content: "キーマカレーの専門店"},
}

var (
	TutorialQueries = []string{"インド料理のお店", "カレー", "ねこ", "転置インデックス"}
	RAGQuestions    = []string{"日本のカレーの特徴は？", "猫はどんな様子？"}
)

type Vector struct {
	ID     string    `json:"id,omitempty"`
	Text   string    `json:"text"`
	Values []float32 `json:"values"`
}

// Fixture の Docs は TutorialDocs と同じ順に並ぶ。Queries には検索の例と RAG の質問の両方を入れる。
type Fixture struct {
	Model      string   `json:"model"`
	Dimensions int      `json:"dimensions"`
	Docs       []Vector `json:"docs"`
	Queries    []Vector `json:"queries"`
}

func Path() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "testdata", "tutorial_embeddings.json")
}

func Load() (Fixture, error) {
	b, err := os.ReadFile(Path())
	if err != nil {
		return Fixture{}, err
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		return Fixture{}, err
	}
	if len(f.Docs) != len(TutorialDocs) {
		return Fixture{}, fmt.Errorf("searchtest: fixture has %d docs, want %d", len(f.Docs), len(TutorialDocs))
	}
	for i, d := range f.Docs {
		if d.ID != TutorialDocs[i].ID || d.Text != TutorialDocs[i].Content {
			return Fixture{}, fmt.Errorf("searchtest: fixture doc %d is %s %q, want %s %q", i, d.ID, d.Text, TutorialDocs[i].ID, TutorialDocs[i].Content)
		}
	}
	return f, nil
}

func (f Fixture) DocVectors() [][]float32 {
	vecs := make([][]float32, len(f.Docs))
	for i, d := range f.Docs {
		vecs[i] = d.Values
	}
	return vecs
}

// Embedder は保存したクエリのベクトルを返す。保存していない文字列はエラーにする。
func (f Fixture) Embedder() search.Embedder {
	m := make(map[string][]float32, len(f.Queries))
	for _, q := range f.Queries {
		m[q.Text] = q.Values
	}
	return mapEmbedder(m)
}

type mapEmbedder map[string][]float32

func (m mapEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v, ok := m[text]
	if !ok {
		return nil, fmt.Errorf("searchtest: no stored embedding for %q", text)
	}
	return v, nil
}
