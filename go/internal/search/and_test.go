package search_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestRankOperator(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{Title: "go", Content: "go backend service"},
		{Title: "go-cloud", Content: "go cloud run"},
		{Title: "cloud", Content: "cloud run infra"},
		{Title: "go-title", Content: "cloud infra"},
		{Title: "run-go", Content: "run the go service on cloud"},
		{Title: "golang", Content: "golang on cloud"},
	}
	testCases := map[string]struct {
		query      search.Query
		wantTitles []string
	}{
		"既定は OR でどれか 1 語を含む文書が候補になる": {
			query:      search.Query{Text: "go cloud"},
			wantTitles: []string{"cloud", "go", "go-cloud", "go-title", "golang", "run-go"},
		},
		"AND ならすべての語を含む文書だけが候補になる": {
			query:      search.Query{Text: "go cloud", Operator: search.OperatorAnd},
			wantTitles: []string{"go-cloud", "go-title", "run-go"},
		},
		"AND は 3 語でも短い postings から絞る": {
			query:      search.Query{Text: "go cloud run", Operator: search.OperatorAnd},
			wantTitles: []string{"go-cloud", "run-go"},
		},
		"AND は項目を絞ると、その項目にすべての語がある文書だけを残す": {
			query:      search.Query{Text: "go cloud", Operator: search.OperatorAnd, Fields: []search.Field{search.FieldContent}},
			wantTitles: []string{"go-cloud", "run-go"},
		},
		"AND で索引に無い語を含めば候補は無い": {
			query:      search.Query{Text: "go kubernetes", Operator: search.OperatorAnd},
			wantTitles: []string{},
		},
		"AND は同義語のどちらかの書き方ですべての語がそろえば候補にする": {
			query:      search.Query{Text: "golang cloud", Operator: search.OperatorAnd, Synonyms: map[string]string{"golang": "go"}},
			wantTitles: []string{"go-cloud", "go-title", "golang", "run-go"},
		},
		"AND とフレーズを合わせると語が隣り合う文書だけが残る": {
			query:      search.Query{Text: "cloud run", Operator: search.OperatorAnd, Phrase: true},
			wantTitles: []string{"cloud", "go-cloud"},
		},
		"フレーズは AND を指定しなくても同じ文書を返す": {
			query:      search.Query{Text: "cloud run", Phrase: true},
			wantTitles: []string{"cloud", "go-cloud"},
		},
		"AND だけなら語が離れた文書も残る": {
			query:      search.Query{Text: "cloud run", Operator: search.OperatorAnd},
			wantTitles: []string{"cloud", "go-cloud", "run-go"},
		},
	}
	for _, ratio := range []int{0, 1} {
		ix := search.New(ds, search.WithGallopRatio(ratio))
		for tn, tc := range testCases {
			t.Run(fmt.Sprintf("gallopRatio=%d/%s", ratio, tn), func(t *testing.T) {
				t.Parallel()
				res := ix.RankQuery(tc.query, -1)
				got := titles(res.Hits)
				slices.Sort(got)
				if !slices.Equal(got, tc.wantTitles) {
					t.Fatalf("titles = %v, want %v", got, tc.wantTitles)
				}
				if res.Scored != len(tc.wantTitles) {
					t.Fatalf("Scored = %d, want %d", res.Scored, len(tc.wantTitles))
				}
			})
		}
	}
}

func TestRankOperatorAndKeepsBM25Score(t *testing.T) {
	t.Parallel()
	ix := search.New(randomDocs(2_000))
	or := ix.RankQuery(search.Query{Text: benchQuery}, -1)
	and := ix.RankQuery(search.Query{Text: benchQuery, Operator: search.OperatorAnd}, -1)
	if len(and.Hits) == 0 {
		t.Fatal("AND has no hits")
	}
	orScore := make(map[string]float64, len(or.Hits))
	for _, h := range or.Hits {
		orScore[h.Doc.Title] = h.Score
	}
	for _, h := range and.Hits {
		if s, ok := orScore[h.Doc.Title]; !ok || s != h.Score {
			t.Fatalf("%s: AND score %v, OR score %v (ok=%v)", h.Doc.Title, h.Score, s, ok)
		}
	}
}

// TestRankOperatorAndMatchesNaive は長さの違う postings の組で、2 つのポインタと galloping の両方がすべての語を含む文書だけを返すことを、文書を 1 件ずつ調べた結果と比べる。
func TestRankOperatorAndMatchesNaive(t *testing.T) {
	t.Parallel()
	ds := randomDocs(3_000)
	r := rand.New(rand.NewPCG(11, 13))
	queries := []string{benchQuery, "w00001 w00002", "w00001 w00500 w03000"}
	for range 20 {
		queries = append(queries, fmt.Sprintf("w%05d w%05d", 1+r.IntN(50), 1+r.IntN(3_000)))
	}
	testCases := map[string]struct {
		ratio int
	}{
		"2 つのポインタだけ":   {ratio: 0},
		"常に galloping": {ratio: 1},
		"既定の閾値で切り替える":  {ratio: search.DefaultGallopRatio},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix := search.New(ds, search.WithGallopRatio(tc.ratio))
			for _, q := range queries {
				want := []string{}
				for _, d := range ds {
					words := strings.Fields(d.Content)
					if !slices.ContainsFunc(strings.Fields(q), func(w string) bool { return !slices.Contains(words, w) }) {
						want = append(want, d.Title)
					}
				}
				got := titles(ix.RankQuery(search.Query{Text: q, Operator: search.OperatorAnd, Fields: []search.Field{search.FieldContent}}, -1).Hits)
				slices.Sort(got)
				slices.Sort(want)
				if !slices.Equal(got, want) {
					t.Fatalf("%q: got %d docs, want %d", q, len(got), len(want))
				}
			}
		})
	}
}
