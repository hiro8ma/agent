package search_test

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// mixedQueries は頻度の高い語（w00001〜w00030）とまれな語（w00030〜w03029）を混ぜた 1〜6 語のクエリを n 個作る。
func mixedQueries(seed uint64, n int) []string {
	r := rand.New(rand.NewPCG(seed, 11))
	qs := make([]string, n)
	for i := range qs {
		ws := make([]string, 1+r.IntN(6))
		for j := range ws {
			if r.IntN(2) == 0 {
				ws[j] = fmt.Sprintf("w%05d", 1+r.IntN(30))
			} else {
				ws[j] = fmt.Sprintf("w%05d", 30+r.IntN(3000))
			}
		}
		qs[i] = strings.Join(ws, " ")
	}
	return qs
}

var prunings = []struct {
	name    string
	pruning search.Pruning
	block   int
}{
	{name: "wand", pruning: search.PruningWAND},
	{name: "bmw-32", pruning: search.PruningBlockMaxWAND, block: 32},
	{name: "bmw-64", pruning: search.PruningBlockMaxWAND, block: 64},
	{name: "bmw-128", pruning: search.PruningBlockMaxWAND, block: 128},
}

func diffHits(a, b []search.Hit) string {
	if len(a) != len(b) {
		return fmt.Sprintf("件数 %d, want %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Doc.Title != b[i].Doc.Title || a[i].Score != b[i].Score {
			return fmt.Sprintf("%d 位 %s %v, want %s %v", i+1, a[i].Doc.Title, a[i].Score, b[i].Doc.Title, b[i].Score)
		}
	}
	return ""
}

type ranker func(q search.Query, limit int) search.Result

// TestPruningMatchesExhaustive は WAND と Block-Max WAND の上位 K 件が、全候補を採点してヒープで選んだ結果と文書・点数・同点の順まで一致することを確かめる。
func TestPruningMatchesExhaustive(t *testing.T) {
	t.Parallel()
	for _, c := range compareCorpora {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			docs := c.docs(4_000)
			ix := search.New(docs)
			bm25f := search.New(docs, search.WithFieldWeights(2, 1))
			dfs := search.NewSharded(docs, 3, nil)
			dfs.Type = search.DFSQueryThenFetch
			upd := search.NewUpdatable(docs[:3_500])
			upd.Add(docs[3_500:]...)
			testCases := map[string]struct {
				rank ranker
				base search.Query
			}{
				"BM25":                    {rank: ix.RankQuery},
				"BM25 で本文だけ":              {rank: ix.RankQuery, base: search.Query{Fields: []search.Field{search.FieldContent}}},
				"BM25F":                   {rank: bm25f.RankQuery},
				"k1=2, b=1 の BM25":        {rank: search.WithBM25Of(ix, 2, 1).RankQuery},
				"TF-IDF":                  {rank: ix.RankQuery, base: search.Query{Ranking: search.RankingTFIDF}},
				"生の回数と IDFPlain の TF-IDF": {rank: ix.RankQuery, base: search.Query{Ranking: search.RankingTFIDF, TFIDF: search.TFIDF{TF: search.TFRaw, IDF: search.IDFPlain}}},
				"シャードの DFS":               {rank: func(q search.Query, k int) search.Result { return dfs.RankQuery(q, k).Result }},
				"補助の索引を持つ Updatable":      {rank: upd.RankQuery},
			}
			for tn, tc := range testCases {
				t.Run(tn, func(t *testing.T) {
					t.Parallel()
					for _, text := range mixedQueries(uint64(len(tn)), 60) {
						for _, k := range []int{1, 10, 100} {
							q := tc.base
							q.Text = text
							want := tc.rank(q, k)
							for _, p := range prunings {
								q.Pruning, q.BlockSize = p.pruning, p.block
								got := tc.rank(q, k)
								if d := diffHits(got.Hits, want.Hits); d != "" {
									t.Fatalf("%s %q k=%d: %s", p.name, text, k, d)
								}
							}
						}
					}
				})
			}
		})
	}
}

// TestPruningScoresFewerDocs は 6 語のクエリの上位 10 件で、WAND が全候補より少なく、Block-Max WAND が WAND より少ない文書だけを採点することを確かめる。
func TestPruningScoresFewerDocs(t *testing.T) {
	t.Parallel()
	ix := search.New(randomDocs(10_000))
	var exhaustive, wand, bmw int
	for _, text := range mixedQueries(5, 40) {
		q := search.Query{Text: text}
		q.Pruning = search.PruningNone
		exhaustive += ix.RankQuery(q, 10).Scored
		q.Pruning = search.PruningWAND
		wand += ix.RankQuery(q, 10).Scored
		q.Pruning = search.PruningBlockMaxWAND
		bmw += ix.RankQuery(q, 10).Scored
	}
	if wand >= exhaustive || bmw >= wand {
		t.Fatalf("採点した文書 = 全候補 %d / WAND %d / BMW %d", exhaustive, wand, bmw)
	}
}

func TestPruningFallsBackToExhaustive(t *testing.T) {
	t.Parallel()
	ix := search.New(randomDocs(2_000))
	text := "w00001 w00002 w00500"
	testCases := map[string]struct {
		q     search.Query
		limit int
	}{
		"AND":        {q: search.Query{Operator: search.OperatorAnd}, limit: 10},
		"コサイン":       {q: search.Query{Ranking: search.RankingTFIDFCosine}, limit: 10},
		"同義語で書き方が複数": {q: search.Query{Synonyms: map[string]string{"w00001": "w00003"}}, limit: 10},
		"limit が負":   {q: search.Query{}, limit: -1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := tc.q
			q.Text = text
			want := ix.RankQuery(q, tc.limit)
			q.Pruning = search.PruningBlockMaxWAND
			got := ix.RankQuery(q, tc.limit)
			if d := diffHits(got.Hits, want.Hits); d != "" || got.Scored != want.Scored {
				t.Fatalf("%s, scored %d want %d", d, got.Scored, want.Scored)
			}
		})
	}
}

// fixedLengthQueries は mixedQueries と同じ語の選び方で、words 語のクエリを n 個作る。
func fixedLengthQueries(seed uint64, n, words int) []string {
	r := rand.New(rand.NewPCG(seed, uint64(words)))
	qs := make([]string, n)
	for i := range qs {
		ws := make([]string, words)
		for j := range ws {
			if r.IntN(2) == 0 {
				ws[j] = fmt.Sprintf("w%05d", 1+r.IntN(30))
			} else {
				ws[j] = fmt.Sprintf("w%05d", 30+r.IntN(3000))
			}
		}
		qs[i] = strings.Join(ws, " ")
	}
	return qs
}

// BenchmarkPruning は全候補の採点、WAND、区間の大きさごとの Block-Max WAND で、50 個のクエリの上位 K 件を求める時間と、1 クエリあたりに採点した文書の数を比べる。
// 区間ごとの上限を float64 で持つ大きさと、メモリ上の postings の構造体の大きさに対する比も報告する。
//
//	go test -run '^$' -bench Pruning ./internal/search/
func BenchmarkPruning(b *testing.B) {
	methods := append([]struct {
		name    string
		pruning search.Pruning
		block   int
	}{{name: "exhaustive"}}, prunings...)
	for _, c := range compareCorpora {
		for _, n := range []int{10_000, 100_000} {
			ix := search.New(c.docs(n))
			structs, _ := search.PostingsBytes(ix)
			for _, block := range []int{32, 64, 128} {
				size := search.BlockMaxBytes(ix, block)
				b.Logf("%s N=%d 区間 %d 件: 上限 %d バイト / postings の構造体 %d バイト (%.2f%%)", c.name, n, block, size, structs, 100*float64(size)/float64(structs))
			}
			for _, words := range []int{2, 4, 6} {
				qs := fixedLengthQueries(uint64(n), 50, words)
				for _, k := range []int{10, 100} {
					for _, m := range methods {
						b.Run(fmt.Sprintf("%s/N=%d/terms=%d/K=%d/%s", c.name, n, words, k, m.name), func(b *testing.B) {
							scored := 0
							for _, text := range qs {
								scored += ix.RankQuery(search.Query{Text: text, Pruning: m.pruning, BlockSize: m.block}, k).Scored
							}
							for b.Loop() {
								for _, text := range qs {
									ix.RankQuery(search.Query{Text: text, Pruning: m.pruning, BlockSize: m.block}, k)
								}
							}
							b.ReportMetric(float64(scored)/float64(len(qs)), "scored/query")
							b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*len(qs)), "µs/query")
						})
					}
				}
			}
		}
	}
}
