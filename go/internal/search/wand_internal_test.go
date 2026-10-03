package search

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

// TestWANDTutorial は教材の例を再現する。重み おいしい 1.6 / カレー 2.3 / ルー 3.0 / カツ 3.2 と最大の出現回数 8 / 5 / 1 / 2 から、上限は 12.8 / 11.5 / 3.0 / 6.4 になる。
// K=2 で 12.8 と 11.5 の文書を採点するとしきい値が 11.5 になり、ルーとカツだけを含む文書は上限の和が 9.4 なのでピボットにならず採点しない。
// 文書は 0: おいしい×8 / 1: カレー×5 / 2: ルー カツ カツ / 3: ルー / 4: カツ カツ / 5: おいしい カレー。採点するのは 0 / 1 / 5 の 3 件。
func TestWANDTutorial(t *testing.T) {
	t.Parallel()
	contents := []string{
		strings.Repeat("おいしい ", 8), strings.Repeat("カレー ", 5), "ルー カツ カツ", "ルー", "カツ カツ", "おいしい カレー",
	}
	docs := make([]Doc, len(contents))
	for i, c := range contents {
		docs[i] = Doc{ID: fmt.Sprint(i), Content: c}
	}
	ix := New(docs, WithAnalyzer(NewAnalyzer(WithWhitespaceTokenizer())))
	weights := map[string]float64{"おいしい": 1.6, "カレー": 2.3, "ルー": 3.0, "カツ": 3.2}
	r := &wandRanker{ix: ix, kind: boundTFIDF, tf: TFRaw}
	for term, want := range map[string]float64{"おいしい": 12.8, "カレー": 11.5, "ルー": 3.0, "カツ": 6.4} {
		m := slices.Max(ix.maxima(ix.vocab[term], selectFields(nil), 0, boundTFIDF, ix.avgLen))
		if got := r.exactBound(weights[term], m); got != want {
			t.Errorf("U(%s) = %v, want %v", term, got, want)
		}
	}
	testCases := map[string]struct {
		pruning Pruning
		scored  int
	}{
		"全候補を採点する":               {pruning: PruningNone, scored: 6},
		"WAND":                   {pruning: PruningWAND, scored: 3},
		"区間 2 件の Block-Max WAND": {pruning: PruningBlockMaxWAND, scored: 3},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := Query{Text: "おいしい カレー ルー カツ", Ranking: RankingTFIDF, TFIDF: TFIDF{TF: TFRaw}, Pruning: tc.pruning, BlockSize: 2}
			res, _ := ix.rankWith(q, 2, &corpus{docs: len(docs), avgLen: ix.avgLen, idf: weights})
			if len(res.Hits) != 2 || res.Hits[0].Doc.ID != "0" || res.Hits[0].Score != 12.8 || res.Hits[1].Doc.ID != "1" || res.Hits[1].Score != 11.5 {
				t.Fatalf("hits = %+v, want 文書 0 の 12.8 と文書 1 の 11.5", res.Hits)
			}
			if res.Scored != tc.scored {
				t.Errorf("採点した文書 = %d, want %d", res.Scored, tc.scored)
			}
		})
	}
}

// TestUpperBoundIsExact は語ごとの上限と区間ごとの上限が、その語だけのクエリで採点した点数の最大とちょうど等しいことを確かめる。上限が低すぎれば上位を取りこぼし、高すぎれば省ける採点が減る。
func TestUpperBoundIsExact(t *testing.T) {
	t.Parallel()
	docs := zipfDocs(3_000)
	plain := New(docs)
	bm25f := New(docs, WithFieldWeights(2, 1))
	testCases := map[string]struct {
		ix   *Index
		q    Query
		kind boundKind
	}{
		"BM25":        {ix: plain, kind: boundBM25},
		"BM25 で本文だけ":  {ix: plain, q: Query{Fields: []Field{FieldContent}}, kind: boundBM25},
		"BM25F":       {ix: bm25f, kind: boundBM25},
		"TF-IDF":      {ix: plain, q: Query{Ranking: RankingTFIDF}, kind: boundTFIDF},
		"平方根の TF-IDF": {ix: plain, q: Query{Ranking: RankingTFIDF, TFIDF: TFIDF{TF: TFSqrt}}, kind: boundTFIDF},
	}
	const block = 64
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix, sel := tc.ix, selectFields(tc.q.Fields)
			r := &wandRanker{ix: ix, kind: tc.kind, tf: tc.q.TFIDF.TF}
			for _, term := range []string{"w00001", "w00007", "w00050", "w00400", "w03000"} {
				q := tc.q
				q.Text = term
				res, ids := ix.rankWith(q, -1, nil)
				score := make(map[int]float64, len(ids))
				for i, id := range ids {
					score[id] = res.Hits[i].Score
				}
				t0 := ix.vocab[term]
				w := ix.idf(len(ix.docs), ix.docFreq([]int32{t0}, sel)[0])
				if tc.kind == boundTFIDF {
					w = tc.q.TFIDF.IDF.weight(len(ix.docs), ix.docFreq([]int32{t0}, sel)[0])
				}
				if got, want := r.exactBound(w, slices.Max(ix.maxima(t0, sel, 0, tc.kind, ix.avgLen))), res.Hits[0].Score; got != want {
					t.Fatalf("%s: 上限 %v, 最大の点数 %v", term, got, want)
				}
				ps := ix.postingsOf(t0)
				for b, m := range ix.maxima(t0, sel, block, tc.kind, ix.avgLen) {
					want := 0.0
					for _, p := range ps[b*block : min((b+1)*block, len(ps))] {
						want = max(want, score[int(p.doc)])
					}
					if got := r.exactBound(w, m); got != want {
						t.Fatalf("%s 区間 %d: 上限 %v, 最大の点数 %v", term, b, got, want)
					}
				}
			}
		})
	}
}
