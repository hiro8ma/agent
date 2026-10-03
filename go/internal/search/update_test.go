package search_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func updateQueries() []search.Query {
	var qs []search.Query
	for _, text := range []string{"w00050 w02000", "w00001 w00002 w00003", "w00010", "w0005 w0200", "w00007 w00008"} {
		qs = append(qs,
			search.Query{Text: text},
			search.Query{Text: text, Operator: search.OperatorAnd},
			search.Query{Text: text, Fields: []search.Field{search.FieldContent}},
			search.Query{Text: text, Typo: true, Prefix: true},
			search.Query{Text: text, Phrase: true},
		)
	}
	return qs
}

func sameHits(t *testing.T, label string, got, want []search.Hit) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d 件, want %d 件", label, len(got), len(want))
	}
	for i := range want {
		if got[i].Doc.Title != want[i].Doc.Title || math.Abs(got[i].Score-want[i].Score) > 1e-9 {
			t.Fatalf("%s: %d 位 = %s %.12f, want %s %.12f", label, i+1, got[i].Doc.Title, got[i].Score, want[i].Doc.Title, want[i].Score)
		}
	}
}

// TestUpdatableMatchesRebuild は、主の索引と補助の索引を合わせた検索が、すべての文書で作り直した索引と同じ順位と点（差 1e-9 以内）を返すことを確かめる。
func TestUpdatableMatchesRebuild(t *testing.T) {
	t.Parallel()
	docs := randomDocs(3_000)
	testCases := map[string]struct {
		opts []search.Option
	}{
		"BM25":        {},
		"BM25F":       {opts: []search.Option{search.WithFieldWeights(2, 1)}},
		"b と k1 を変える": {opts: []search.Option{search.WithB(0.3), search.WithK1(2)}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			full := search.New(docs, tc.opts...)
			u := search.NewUpdatable(docs[:2_500], tc.opts...)
			for i := 2_500; i < len(docs); i += 100 {
				u.Add(docs[i : i+100]...)
			}
			if m, a := u.Sizes(); m != 2_500 || a != 500 {
				t.Fatalf("Sizes = %d, %d, want 2500, 500", m, a)
			}
			for _, q := range updateQueries() {
				for _, limit := range []int{10, -1} {
					label := fmt.Sprintf("%+v limit=%d", q, limit)
					sameHits(t, label, u.RankQuery(q, limit).Hits, full.RankQuery(q, limit).Hits)
				}
			}
		})
	}
}

// TestUpdatableMergeMatchesRebuild は、Merge した後の索引が、作り直した索引と BM25 / TF-IDF / コサインのどれでも同じ順位と点を返すことを確かめる。
func TestUpdatableMergeMatchesRebuild(t *testing.T) {
	t.Parallel()
	corpora := map[string][]search.Doc{
		"latin":    randomDocs(2_000),
		"japanese": randomJapaneseDocs(1_000),
	}
	for cn, docs := range corpora {
		t.Run(cn, func(t *testing.T) {
			t.Parallel()
			full := search.New(docs)
			u := search.NewUpdatable(docs[:len(docs)/2])
			u.Add(docs[len(docs)/2:]...)
			u.Merge()
			if m, a := u.Sizes(); m != len(docs) || a != 0 {
				t.Fatalf("Sizes = %d, %d, want %d, 0", m, a, len(docs))
			}
			texts := []string{"w00050 w02000", "w00001 w00003", kanjiWord(50) + "には" + kanjiWord(2000), kanjiWord(3)}
			for _, text := range texts {
				for _, r := range []search.Ranking{search.RankingBM25, search.RankingTFIDF, search.RankingTFIDFCosine, search.RankingBucket} {
					q := search.Query{Text: text, Ranking: r, Typo: true}
					sameHits(t, fmt.Sprintf("%q ranking=%d", text, r), u.RankQuery(q, 20).Hits, full.RankQuery(q, 20).Hits)
				}
			}
		})
	}
}

// BenchmarkUpdatable は主の索引 10 万文書に対し、補助の索引へ 1 文書を足す時間（補助の文書数ごと）、すべて作り直す時間、Merge の時間、2 つの索引を検索する時間を比べる。
//
//	go test -run '^$' -bench Updatable -benchtime 20x ./internal/search/
func BenchmarkUpdatable(b *testing.B) {
	pool := randomDocs(101_001)
	docs, extra := pool[:101_000], pool[101_000]
	main := docs[:100_000]
	u := search.NewUpdatable(main)
	for _, aux := range []int{0, 100, 1_000} {
		search.TruncateAdded(u, 0)
		u.Add(docs[100_000 : 100_000+aux]...)
		b.Run(fmt.Sprintf("add/aux=%d", aux), func(b *testing.B) {
			for b.Loop() {
				search.TruncateAdded(u, aux)
				u.Add(extra)
			}
		})
	}
	b.Run("rebuild/N=100001", func(b *testing.B) {
		for b.Loop() {
			search.New(docs[:100_001])
		}
	})
	mainIx := search.New(main)
	b.Run("merge/aux=1000", func(b *testing.B) {
		for b.Loop() {
			b.StopTimer()
			m := search.UpdatableOf(mainIx)
			m.Add(docs[100_000:]...)
			b.StartTimer()
			m.Merge()
		}
	})
	b.Run("rebuild/N=101000", func(b *testing.B) {
		for b.Loop() {
			search.New(docs)
		}
	})
	full := search.New(docs)
	search.TruncateAdded(u, 0)
	u.Add(docs[100_000:]...)
	for _, q := range []string{benchQuery, "w00001 w00002 w00003"} {
		b.Run(fmt.Sprintf("query/%s/single", q), func(b *testing.B) {
			for b.Loop() {
				full.Rank(q, 10)
			}
		})
		b.Run(fmt.Sprintf("query/%s/main+aux", q), func(b *testing.B) {
			for b.Loop() {
				u.Rank(q, 10)
			}
		})
	}
}
