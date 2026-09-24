package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
	"github.com/hiro8ma/agent/go/internal/search/searchtest"
)

// TestTutorialRetrievers は教材の例で BM25、ベクトル検索、RRF の上位 3 件を比べる。ベクトルは gemini-embedding-001（768 次元）で作って保存したもの。
func TestTutorialRetrievers(t *testing.T) {
	t.Parallel()
	fx, err := searchtest.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	vec, err := search.NewFlat(searchtest.TutorialDocs, fx.DocVectors(), fx.Embedder())
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	bm25 := search.New(searchtest.TutorialDocs)
	retrievers := map[string]search.Retriever{
		"bm25": bm25,
		"vec":  vec,
		"rrf":  search.Hybrid{Retrievers: []search.Retriever{bm25, vec}},
	}
	testCases := map[string]struct {
		query     string
		retriever string
		want      []string
	}{
		"BM25 はインデックスの「イン」の bigram で転置インデックスの文書も拾う": {query: "インド料理のお店", retriever: "bm25", want: []string{"d2", "d6"}},
		"ベクトルはインド料理のお店の次にカレーの文書を並べる":                {query: "インド料理のお店", retriever: "vec", want: []string{"d2", "d7", "d3"}},
		"RRF は BM25 の拾った転置インデックスの文書を 2 位に上げる":       {query: "インド料理のお店", retriever: "rrf", want: []string{"d2", "d6", "d7"}},
		"BM25 はカレーを含む 3 件を短い文書から並べる":                {query: "カレー", retriever: "bm25", want: []string{"d3", "d7", "d1"}},
		"ベクトルもカレーの 3 件を返すが順が違う":                     {query: "カレー", retriever: "vec", want: []string{"d1", "d3", "d7"}},
		"RRF は両方で上位のカレーの 3 件を返す":                    {query: "カレー", retriever: "rrf", want: []string{"d3", "d1", "d7"}},
		"BM25 はひらがなのねこで漢字の猫を引けない":                   {query: "ねこ", retriever: "bm25", want: []string{}},
		"ベクトルはひらがなのねこで猫の 2 件を上位に並べる":                {query: "ねこ", retriever: "vec", want: []string{"d4", "d5", "d1"}},
		"RRF は BM25 が 0 件でもベクトルの順位で返す":              {query: "ねこ", retriever: "rrf", want: []string{"d4", "d5", "d1"}},
		"BM25 は転置インデックスの文書とインの bigram の文書を返す":       {query: "転置インデックス", retriever: "bm25", want: []string{"d6", "d2"}},
		"ベクトルは転置インデックスの文書を大差で 1 位にする":               {query: "転置インデックス", retriever: "vec", want: []string{"d6", "d2", "d4"}},
		"RRF は両方で 1 位の文書を 1 位にする":                   {query: "転置インデックス", retriever: "rrf", want: []string{"d6", "d2", "d4"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			docs, err := retrievers[tc.retriever].Search(t.Context(), tc.query, 3)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if got := docIDs(docs); !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}
