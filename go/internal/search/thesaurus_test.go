package search_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestThesaurusExpandsGroupsAndNarrowerOnly(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	th := search.NewThesaurus(
		search.WithSynonymGroup("delete", "remove", "erase"),
		search.WithSynonymGroup("phone", "mobile"),
		search.WithNarrower("phone", "iphone", "pixel"),
		search.WithSynonymGroup("ウェブ", "ウエブ", "ウェッブ"),
	)
	testCases := map[string]struct {
		word      string
		indexTime bool
		want      []string
	}{
		"グループの語はほかのすべての語を足す": {
			word: "remove", want: []string{"delet", "eras"},
		},
		"広い語には狭い語を足す": {
			word: "phone", want: []string{"mobil", "iphon", "pixel"},
		},
		"広い語の同義語にも狭い語を足す": {
			word: "mobile", want: []string{"phone", "iphon", "pixel"},
		},
		"狭い語には広い語を足さない": {
			word: "iphone", want: nil,
		},
		"索引のときは狭い語を含む文書に広い語と同義語を足す": {
			word: "iphone", indexTime: true, want: []string{"phone", "mobil"},
		},
		"索引のときは広い語を含む文書に狭い語を足さない": {
			word: "phone", indexTime: true, want: []string{"mobil"},
		},
		"表記ゆれのグループは正規化の後の索引語で引く": {
			word: "ウエブ", want: []string{"ウェ ェブ", "ウェ ェッ ッブ"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, dropped := search.ExpandWord(th, a, tc.word, tc.indexTime)
			if !slices.Equal(got, tc.want) || dropped != 0 {
				t.Fatalf("ExpandWord(%q) = %q, dropped %d, want %q, dropped 0", tc.word, got, dropped, tc.want)
			}
		})
	}
}

func TestThesaurusLimitReportsDropped(t *testing.T) {
	t.Parallel()
	group := []string{"base"}
	for i := range 60 {
		group = append(group, fmt.Sprintf("syn%02d", i))
	}
	ds := []search.Doc{{ID: "d", Content: "base syn00 syn59"}}
	ix := search.New(ds)
	testCases := map[string]struct {
		opts        []search.ThesaurusOption
		wantAdded   int
		wantDropped int
	}{
		"既定の上限は 50 で残りの 10 を捨てる": {
			wantAdded: 50, wantDropped: 10,
		},
		"上限を 5 にすると 55 を捨てる": {
			opts: []search.ThesaurusOption{search.WithExpansionLimit(5)}, wantAdded: 5, wantDropped: 55,
		},
		"負の上限ならすべて足す": {
			opts: []search.ThesaurusOption{search.WithExpansionLimit(-1)}, wantAdded: 60, wantDropped: 0,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			th := search.NewThesaurus(append([]search.ThesaurusOption{search.WithSynonymGroup(group...)}, tc.opts...)...)
			res := ix.RankQuery(search.Query{Text: "base", Thesaurus: th}, 10)
			if res.Expanded != tc.wantAdded || res.ExpansionDropped != tc.wantDropped {
				t.Fatalf("Expanded = %d, ExpansionDropped = %d, want %d, %d", res.Expanded, res.ExpansionDropped, tc.wantAdded, tc.wantDropped)
			}
		})
	}
}

func TestThesaurusScoresBestSpelling(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ix := search.New(evalDocs(), search.WithAnalyzer(a))
	th := evalThesaurus()
	score := func(q search.Query, id string) float64 {
		for _, h := range ix.RankQuery(q, -1).Hits {
			if h.Doc.ID == id {
				return h.Score
			}
		}
		return 0
	}
	testCases := map[string]struct {
		query search.Query
		doc   string
		want  search.Query
	}{
		"同義語だけを含む文書はその同義語で検索したときと同じ点数": {
			query: search.Query{Text: "delete", Thesaurus: th}, doc: "del5", want: search.Query{Text: "erase"},
		},
		"辞書に無い語だけのクエリは辞書なしと同じ点数": {
			query: search.Query{Text: "upload file", Thesaurus: th}, doc: "file1", want: search.Query{Text: "upload file"},
		},
		"Extra の重みが 1 なら同じ語をクエリに書いたのと同じ点数": {
			query: search.Query{Text: "delete", Extra: []search.WeightedTerm{{Term: "file", Weight: 1}}}, doc: "del1", want: search.Query{Text: "delete file"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, want := score(tc.query, tc.doc), score(tc.want, tc.doc)
			if got == 0 || got != want {
				t.Fatalf("score = %v, want %v", got, want)
			}
		})
	}
}

func TestThesaurusDirectionRule(t *testing.T) {
	t.Parallel()
	s := newEvalSetup()
	qs := evalQueries()
	forced := search.NewThesaurus(search.WithNarrower("iphone", "phone"))
	run := func(q evalQuery, th *search.Thesaurus) evalMetrics {
		return measure(s.plain.RankQuery(search.Query{Text: q.text, Thesaurus: th}, evalK).Hits, q.relevant)
	}
	broad, broadExpanded := run(qs["phone battery"], nil), run(qs["phone battery"], s.th)
	if broadExpanded.recall <= broad.recall || broadExpanded.precision < broad.precision {
		t.Fatalf("phone -> iphone: recall %v -> %v, precision %v -> %v, want recall up and precision not down",
			broad.recall, broadExpanded.recall, broad.precision, broadExpanded.precision)
	}
	narrow, narrowForced := run(qs["iphone battery"], nil), run(qs["iphone battery"], forced)
	if narrowForced.precision >= narrow.precision {
		t.Fatalf("iphone -> phone: precision %v -> %v, want down", narrow.precision, narrowForced.precision)
	}
	t.Logf("phone battery: R %.3f -> %.3f, P %.2f -> %.2f, nDCG %.3f -> %.3f", broad.recall, broadExpanded.recall, broad.precision, broadExpanded.precision, broad.ndcg, broadExpanded.ndcg)
	t.Logf("iphone battery (forced iphone -> phone): R %.3f -> %.3f, P %.2f -> %.2f, nDCG %.3f -> %.3f", narrow.recall, narrowForced.recall, narrow.precision, narrowForced.precision, narrow.ndcg, narrowForced.ndcg)
}

func TestDocExpansionMatchesQueryExpansion(t *testing.T) {
	t.Parallel()
	s := newEvalSetup()
	for name, q := range evalQueries() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			byQuery := s.plain.RankQuery(search.Query{Text: q.text, Thesaurus: s.th}, evalK).Hits
			byDoc := s.docExpanded.RankQuery(search.Query{Text: q.text}, evalK).Hits
			mq, md := measure(byQuery, q.relevant), measure(byDoc, q.relevant)
			if mq.recall != md.recall {
				t.Fatalf("recall@10 query expansion %v, document expansion %v", mq.recall, md.recall)
			}
			t.Logf("overlap@10 = %.2f, nDCG query %.3f doc %.3f", overlap(byQuery, byDoc), mq.ndcg, md.ndcg)
		})
	}
}

// overlap は上位 10 件のうち両方に入る文書の割合。
func overlap(a, b []search.Hit) float64 {
	ids := make(map[string]bool)
	for _, h := range a {
		ids[h.Doc.ID] = true
	}
	n := 0
	for _, h := range b {
		if ids[h.Doc.ID] {
			n++
		}
	}
	return float64(n) / float64(max(len(a), len(b), 1))
}

func TestDictionaryChangeNeedsRebuildOnlyForDocExpansion(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ds := []search.Doc{
		{ID: "wipe", Title: "Wipe a disk", Content: "Wipe the disk before you sell the computer."},
		{ID: "erase", Title: "Erase a file", Content: "Erase a file you no longer need."},
	}
	before := search.NewThesaurus(search.WithSynonymGroup("delete", "erase"))
	after := search.NewThesaurus(search.WithSynonymGroup("delete", "erase", "wipe"))
	stale := search.New(ds, search.WithAnalyzer(a), search.WithDocExpansion(before))
	rebuilt := search.New(ds, search.WithAnalyzer(a), search.WithDocExpansion(after))
	plain := search.New(ds, search.WithAnalyzer(a))
	testCases := map[string]struct {
		ix    *search.Index
		query search.Query
		want  []string
	}{
		"文書拡張は辞書を変えても作り直すまで新しい語で見つからない": {
			ix: stale, query: search.Query{Text: "delete"}, want: []string{"erase"},
		},
		"文書拡張は作り直すと新しい語で見つかる": {
			ix: rebuilt, query: search.Query{Text: "delete"}, want: []string{"erase", "wipe"},
		},
		"クエリ拡張は同じ索引のまま新しい辞書で見つかる": {
			ix: plain, query: search.Query{Text: "delete", Thesaurus: after}, want: []string{"erase", "wipe"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var got []string
			for _, h := range tc.ix.RankQuery(tc.query, 10).Hits {
				got = append(got, h.Doc.ID)
			}
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDocExpansionPostings(t *testing.T) {
	t.Parallel()
	a := evalAnalyzer()
	ds := []search.Doc{
		{ID: "delete", Content: "delete the file"},
		{ID: "remove", Content: "remove the file"},
		{ID: "other", Content: "open the file and the folder"},
	}
	th := search.NewThesaurus(search.WithSynonymGroup("delete", "remove"))
	plain := search.New(ds, search.WithAnalyzer(a))
	full := search.New(ds, search.WithAnalyzer(a), search.WithDocExpansion(th))
	half := search.New(ds, search.WithAnalyzer(a), search.WithDocExpansion(search.NewThesaurus(search.WithSynonymGroup("delete", "remove"), search.WithExpansionWeight(0.5))))
	t.Run("足した語は元の語と同じ位置に重ねる", func(t *testing.T) {
		t.Parallel()
		got := search.PostingsOf(full, "delet")
		if len(got) != 2 || got[1].Doc != 1 || !slices.Equal(got[1].Positions, search.PostingsOf(full, "remov")[1].Positions) {
			t.Fatalf("postings of delet = %+v", got)
		}
		if n := full.RankQuery(search.Query{Text: "delete the file", Phrase: true}, 10).Hits; len(n) != 2 {
			t.Fatalf("phrase hits = %d, want 2", len(n))
		}
	})
	t.Run("足した語は文書の長さに数えない", func(t *testing.T) {
		t.Parallel()
		for d := range ds {
			if got, want := search.DocLength(full, d), search.DocLength(plain, d); got != want {
				t.Fatalf("doc %d length = %d, want %d", d, got, want)
			}
		}
		if got := full.Stats().AddedPositions; got != 2 {
			t.Fatalf("AddedPositions = %d, want 2", got)
		}
	})
	t.Run("重みが 1 なら元の語と同じ点数で、下げると足した語の点数が下がる", func(t *testing.T) {
		t.Parallel()
		scores := func(ix *search.Index) map[string]float64 {
			out := make(map[string]float64)
			for _, h := range ix.RankQuery(search.Query{Text: "delete"}, 10).Hits {
				out[h.Doc.ID] = h.Score
			}
			return out
		}
		f, h := scores(full), scores(half)
		if f["remove"] != f["delete"] {
			t.Fatalf("weight 1: remove %v, delete %v, want equal", f["remove"], f["delete"])
		}
		if h["remove"] >= h["delete"] {
			t.Fatalf("weight 0.5: remove %v, delete %v, want remove lower", h["remove"], h["delete"])
		}
	})
}
