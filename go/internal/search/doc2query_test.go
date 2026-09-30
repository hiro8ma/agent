package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// handWrittenQueries は doc2query のモデルの代わりの決まった生成器。file4 の 2 つ目は文書に無い話題を作った生成の誤り。
func handWrittenQueries(d search.Doc) []string {
	switch d.ID {
	case "del3":
		return []string{"how to delete a document"}
	case "file4":
		return []string{"rename file", "refund policy for my subscription"}
	}
	return nil
}

func doc2queryDocs() []search.Doc {
	return append(evalDocs(),
		search.Doc{ID: "rf1", Title: "Refund policy", Content: "Request a refund within 30 days of purchase."},
		search.Doc{ID: "rf2", Title: "Refund status", Content: "Check the status of your refund on the billing page."},
	)
}

func TestDoc2QueryFilter(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ds := doc2queryDocs()
	testCases := map[string]struct {
		d2q          search.Doc2Query
		query        string
		want         []string
		wantFiltered int
	}{
		"生成の誤りを足すと refund で無関係な文書が出る": {
			d2q: search.Doc2Query{Generate: handWrittenQueries}, query: "refund", want: []string{"file4", "rf1", "rf2"},
		},
		"元の文書で 0 点のクエリを捨てると refund で無関係な文書が出ない": {
			d2q: search.Doc2Query{Generate: handWrittenQueries, MinScore: 1}, query: "refund", want: []string{"rf1", "rf2"}, wantFiltered: 1,
		},
		"元の文書と語が重なる生成クエリは残り delete で erase の文書が見つかる": {
			d2q: search.Doc2Query{Generate: handWrittenQueries, MinScore: 1}, query: "delete", want: []string{"del1", "del3", "del6"}, wantFiltered: 1,
		},
		"生成クエリを足さなければ delete で erase の文書は見つからない": {
			query: "delete", want: []string{"del1", "del6"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix := search.New(ds, search.WithAnalyzer(a), search.WithDoc2Query(tc.d2q))
			var got []string
			for _, h := range ix.RankQuery(search.Query{Text: tc.query}, 10).Hits {
				got = append(got, h.Doc.ID)
			}
			slices.Sort(got)
			if !slices.Equal(got, tc.want) || ix.Stats().Doc2QueryFiltered != tc.wantFiltered {
				t.Fatalf("ids = %v, filtered %d, want %v, %d", got, ix.Stats().Doc2QueryFiltered, tc.want, tc.wantFiltered)
			}
		})
	}
}

func TestDoc2QueryWeight(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ds := []search.Doc{
		{ID: "orig", Content: "delete a document"},
		{ID: "gen", Content: "erase a document"},
		{ID: "other", Content: "open a folder"},
	}
	gen := func(d search.Doc) []string {
		if d.ID == "gen" {
			return []string{"delete"}
		}
		return nil
	}
	scores := func(w float64) map[string]float64 {
		ix := search.New(ds, search.WithAnalyzer(a), search.WithDoc2Query(search.Doc2Query{Generate: gen, Weight: w}))
		out := make(map[string]float64)
		for _, h := range ix.RankQuery(search.Query{Text: "delete"}, 10).Hits {
			out[h.Doc.ID] = h.Score
		}
		return out
	}
	full, low := scores(1), scores(0.3)
	if full["gen"] != full["orig"] {
		t.Fatalf("weight 1: gen %v, orig %v, want equal", full["gen"], full["orig"])
	}
	if low["gen"] <= 0 || low["gen"] >= low["orig"] {
		t.Fatalf("weight 0.3: gen %v, orig %v, want 0 < gen < orig", low["gen"], low["orig"])
	}
}

// TestDoc2QueryField は生成したクエリを本文の後ろに足す既定と、別の項目にして BM25F で重み 0.5 を掛ける版を比べる。
// 候補の集合は変わらず、順位だけが変わる。生成の誤りで付いた refund policy では、別の項目の版だけが適合する rf2 を file4 より上に置く。
func TestDoc2QueryField(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ds := doc2queryDocs()
	build := func(field bool, minScore float64) *search.Index {
		return search.New(ds, search.WithAnalyzer(a), search.WithDoc2Query(search.Doc2Query{
			Generate: handWrittenQueries, Weight: 0.5, MinScore: minScore, Field: field,
		}))
	}
	ids := func(ix *search.Index, q string, limit int) []string {
		var out []string
		for _, h := range ix.RankQuery(search.Query{Text: q}, limit).Hits {
			out = append(out, h.Doc.ID)
		}
		return out
	}
	queries := map[string][]string{
		"refund":          {"rf1", "rf2"},
		"refund policy":   {"rf1", "rf2"},
		"delete":          {"del1", "del2", "del3", "del4", "del5", "del6"},
		"delete document": {"del1", "del2", "del3", "del4", "del5", "del6"},
		"rename file":     {"file4"},
	}
	for _, minScore := range []float64{0, 1} {
		same, field := build(false, minScore), build(true, minScore)
		for q, rel := range queries {
			s, f := ids(same, q, -1), ids(field, q, -1)
			if ss, fs := slices.Sorted(slices.Values(s)), slices.Sorted(slices.Values(f)); !slices.Equal(ss, fs) {
				t.Fatalf("MinScore %v %q: same field %v, separate field %v", minScore, q, s, f)
			}
			t.Logf("MinScore %v %-16q nDCG@10 same %.3f separate %.3f | same %v | separate %v", minScore, q,
				measure(same.RankQuery(search.Query{Text: q}, evalK).Hits, rel).ndcg,
				measure(field.RankQuery(search.Query{Text: q}, evalK).Hits, rel).ndcg, s, f)
		}
		if minScore > 0 {
			continue
		}
		if got := ids(same, "refund policy", 3); !slices.Equal(got, []string{"rf1", "file4", "rf2"}) {
			t.Fatalf("same field: %v", got)
		}
		if got := ids(field, "refund policy", 3); !slices.Equal(got, []string{"rf1", "rf2", "file4"}) {
			t.Fatalf("separate field: %v", got)
		}
	}
}
