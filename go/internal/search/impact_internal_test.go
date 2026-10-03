package search

import (
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

// zipfDocs は語の出現頻度が Zipf 分布に従う文書を固定の種で作る。
func zipfDocs(n int) []Doc {
	r := rand.New(rand.NewPCG(42, uint64(n)))
	zipf := rand.NewZipf(r, 1.1, 1, 19_999)
	ds := make([]Doc, n)
	for i := range ds {
		words := make([]string, 20+r.IntN(61))
		for j := range words {
			words[j] = fmt.Sprintf("w%05d", zipf.Uint64())
		}
		ds[i] = Doc{Title: fmt.Sprintf("doc%d", i), Content: strings.Join(words, " ")}
	}
	return ds
}

// impactList は 1 語のクエリの BM25 の点の高い順（同点は文書番号の順）に並べた出現記録。索引を作るときに語ごとに作っておく想定。
type impactList struct {
	ids    []int
	scores []float64
}

func newImpactList(ix *Index, term string) impactList {
	res, ids := ix.rankWith(Query{Text: term}, -1, nil)
	l := impactList{ids: ids, scores: make([]float64, len(ids))}
	for i, h := range res.Hits {
		l.scores[i] = h.Score
	}
	return l
}

// TestImpactOrderedTopK は 1 語の上位 K 件が、文書番号の順の postings を全部採点してヒープで選ぶ方法と、点の順に並べた postings の先頭 K 件で一致することを確かめ、読む件数と大きさを出す。
// 点の順に並べると文書番号が昇順でなくなるので、差分で縮められず、AND の 2 つのポインタも使えない。
//
//	go test -run TestImpactOrderedTopK -v ./internal/search/
func TestImpactOrderedTopK(t *testing.T) {
	t.Parallel()
	ix := New(zipfDocs(20_000))
	for _, term := range []string{"w00001", "w00010", "w00100", "w01000"} {
		l := newImpactList(ix, term)
		ps := ix.postingsOf(ix.vocab[term])
		for _, k := range []int{10, 100} {
			res, ids := ix.rankWith(Query{Text: term}, k, nil)
			n := min(k, len(l.ids))
			for i := range n {
				if ids[i] != l.ids[i] || res.Hits[i].Score != l.scores[i] {
					t.Fatalf("%s k=%d %d 位: 文書 %d %.6f, want %d %.6f", term, k, i+1, ids[i], res.Hits[i].Score, l.ids[i], l.scores[i])
				}
			}
		}
		var byDoc, byImpact []byte
		byDoc = encodePostings(byDoc, ps)
		for _, id := range l.ids {
			byImpact = binary.AppendUvarint(byImpact, uint64(id))
			byImpact = append(byImpact, 0) // 点を 1 バイトに量子化して持つ想定
		}
		t.Logf("%s postings %5d 件: 文書番号の順 %6d バイト（差分）/ 点の順 %6d バイト。上位 10 件で読む件数 %d / %d", term, len(ps), len(byDoc), len(byImpact), len(ps), min(10, len(ps)))
	}
}

// BenchmarkImpactOrderedTopK は 1 語の上位 10 件を、文書番号の順の postings を全部採点してヒープで選ぶ方法と、点の順の postings の先頭 10 件を読む方法で比べる。
//
//	go test -run '^$' -bench ImpactOrderedTopK ./internal/search/
func BenchmarkImpactOrderedTopK(b *testing.B) {
	ix := New(zipfDocs(100_000))
	for _, term := range []string{"w00001", "w00100", "w01000"} {
		l := newImpactList(ix, term)
		b.Run(term+"/doc-order+heap", func(b *testing.B) {
			for b.Loop() {
				ix.rankWith(Query{Text: term}, 10, nil)
			}
			b.ReportMetric(float64(len(l.ids)), "postings")
		})
		b.Run(term+"/impact-order", func(b *testing.B) {
			var sink int
			for b.Loop() {
				top := l.ids[:min(10, len(l.ids))]
				hits := make([]Hit, len(top))
				for i, id := range top {
					hits[i] = Hit{Doc: ix.docs[id], Score: l.scores[i]}
				}
				sink += len(hits)
			}
			_ = sink
		})
	}
}
