package search_test

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func wsIndex(ds []search.Doc, opts ...search.Option) *search.Index {
	return search.New(ds, append([]search.Option{search.WithAnalyzer(search.NewAnalyzer(search.WithWhitespaceTokenizer()))}, opts...)...)
}

// occurrenceDocs は文書 ID ごとの語の出現回数から文書を作る。語を含まない文書は pad だけにする。
func occurrenceDocs(n int, occ map[string]map[int]int) []search.Doc {
	ds := make([]search.Doc, n)
	for i := range ds {
		words := []string{"pad"}
		for term, docs := range occ {
			for range docs[i+1] {
				words = append(words, term)
			}
		}
		slices.Sort(words)
		ds[i] = search.Doc{ID: fmt.Sprintf("文書%d", i+1), Content: strings.Join(words, " ")}
	}
	return ds
}

func rankedIDs(hits []search.Hit) []string {
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.Doc.ID
	}
	return ids
}

func TestTFIDFTextbook(t *testing.T) {
	t.Parallel()
	occ := map[string]map[int]int{
		"カツ":  {3: 2, 7: 1, 8: 2, 12: 3},
		"カレー": {1: 1, 2: 2, 3: 1, 5: 1},
	}
	ix := wsIndex(occurrenceDocs(12, occ))
	q := search.Query{Text: "カツ カレー", Ranking: search.RankingTFIDF, TFIDF: search.TFIDF{TF: search.TFRaw}}
	res := search.RankTFIDFWithIDF(ix, q, map[string]float64{"カツ": 3.0, "カレー": 2.3}, -1)
	got := make(map[string]float64)
	for _, h := range res.Hits {
		got[h.Doc.ID] = h.Score
	}
	testCases := map[string]struct {
		doc  string
		want float64
	}{
		"文書1 はカレー 1 回で 2.3":         {doc: "文書1", want: 2.3},
		"文書2 はカレー 2 回で 4.6":         {doc: "文書2", want: 4.6},
		"文書3 はカツ 2 回とカレー 1 回で 8.3":  {doc: "文書3", want: 8.3},
		"文書12 はカツ 3 回で 9.0（教材の表の外）": {doc: "文書12", want: 9.0},
		"文書8 はカツ 2 回で 6.0（教材の表の外）":  {doc: "文書8", want: 6.0},
		"どちらの語も含まない文書 4 は候補に入らず 0":  {doc: "文書4", want: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if math.Abs(got[tc.doc]-tc.want) > 1e-9 {
				t.Fatalf("score = %v, want %v", got[tc.doc], tc.want)
			}
		})
	}
	t.Run("カレーを含む文書の上位 2 件は文書3 と文書2 で、文書1 は外れる", func(t *testing.T) {
		t.Parallel()
		var curry []string
		for _, h := range res.Hits {
			if occ["カレー"][idNumber(t, h.Doc.ID)] > 0 {
				curry = append(curry, h.Doc.ID)
			}
		}
		if want := []string{"文書3", "文書2"}; !slices.Equal(curry[:2], want) {
			t.Fatalf("top 2 = %v, want %v", curry[:2], want)
		}
	})
	t.Run("OR の全候補では文書12 が文書3 より上になる", func(t *testing.T) {
		t.Parallel()
		if got, want := rankedIDs(res.Hits[:2]), []string{"文書12", "文書3"}; !slices.Equal(got, want) {
			t.Fatalf("top 2 = %v, want %v", got, want)
		}
	})
}

func idNumber(t *testing.T, id string) int {
	t.Helper()
	var n int
	if _, err := fmt.Sscanf(id, "文書%d", &n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestIDFIsInformation(t *testing.T) {
	t.Parallel()
	const n = 16
	occ := map[string]map[int]int{"all": {}, "half": {}, "rare": {1: 1, 2: 1}, "single": {3: 1}}
	for i := 1; i <= n; i++ {
		occ["all"][i] = 1
		if i%2 == 0 {
			occ["half"][i] = 1
		}
	}
	ix := wsIndex(occurrenceDocs(n, occ))
	testCases := map[string]struct {
		term string
		idf  search.IDFWeight
		want float64
	}{
		"全文書に出る語は情報量 0":          {term: "all", idf: search.IDFPlain, want: 0},
		"N/2 の文書に出る語は 1":         {term: "half", idf: search.IDFPlain, want: 1},
		"N/8 の文書に出る語は 3":         {term: "rare", idf: search.IDFPlain, want: 3},
		"1 文書だけに出る語は log2 N = 4": {term: "single", idf: search.IDFPlain, want: 4},
		"平滑化すると全文書に出る語も 1":       {term: "all", idf: search.IDFSmooth, want: 1},
		"平滑化すると N/8 の語は log2 9":  {term: "rare", idf: search.IDFSmooth, want: math.Log2(9)},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := search.Query{Text: tc.term, Ranking: search.RankingTFIDF, TFIDF: search.TFIDF{TF: search.TFRaw, IDF: tc.idf}}
			hits := ix.RankQuery(q, -1).Hits
			if len(hits) != len(occ[tc.term]) {
				t.Fatalf("hits = %d, want %d", len(hits), len(occ[tc.term]))
			}
			if got := hits[0].Score; math.Abs(got-tc.want) > 1e-12 {
				t.Fatalf("score = %v, want %v", got, tc.want)
			}
			if tc.idf == search.IDFPlain {
				p := float64(len(occ[tc.term])) / n
				if info := math.Log2(1 / p); math.Abs(hits[0].Score-info) > 1e-12 {
					t.Fatalf("score = %v, information log2(1/P) = %v", hits[0].Score, info)
				}
			}
		})
	}
}

func TestTFIDFDiffersFromBM25(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		docs      []search.Doc
		query     string
		tfidf     search.TFIDF
		wantBM25  []string
		wantTFIDF []string
	}{
		"TF の飽和: 生の回数では同じ語を繰り返す文書が上になり、BM25 では両方の語を含む文書が上になる": {
			docs:      saturationDocs(),
			query:     "cat dog",
			tfidf:     search.TFIDF{TF: search.TFRaw},
			wantBM25:  []string{"both", "spam"},
			wantTFIDF: []string{"spam", "both"},
		},
		"TF の飽和: 1 + log10(f) なら BM25 と同じく両方の語を含む文書が上になる": {
			docs:      saturationDocs(),
			query:     "cat dog",
			tfidf:     search.TFIDF{TF: search.TFLog},
			wantBM25:  []string{"both", "spam"},
			wantTFIDF: []string{"both", "spam"},
		},
		"TF の飽和: sqrt(f) は 8 回で 2.83 倍になり、2 語の和を超えて繰り返しの文書が上に残る": {
			docs:      saturationDocs(),
			query:     "cat dog",
			tfidf:     search.TFIDF{TF: search.TFSqrt},
			wantBM25:  []string{"both", "spam"},
			wantTFIDF: []string{"spam", "both"},
		},
		"文書長: BM25 は短い文書を上にし、長さを見ない TF-IDF は回数の多い長い文書を上にする": {
			docs: []search.Doc{
				{ID: "long", Content: "cat cat " + strings.Repeat("pad ", 60)},
				{ID: "short", Content: "cat pad"},
				{ID: "f1", Content: "pad pad pad"},
				{ID: "f2", Content: "pad pad pad"},
			},
			query:     "cat",
			tfidf:     search.TFIDF{TF: search.TFLog},
			wantBM25:  []string{"short", "long"},
			wantTFIDF: []string{"long", "short"},
		},
		"IDF の平滑化: 全文書に出る語は log2(N/DF) で 0 になり、平滑化すると回数が効く": {
			docs: []search.Doc{
				{ID: "common", Content: "common common common"},
				{ID: "rare", Content: "common rare"},
				{ID: "r2", Content: "common rare"},
				{ID: "c", Content: "common"},
			},
			query:     "common rare",
			tfidf:     search.TFIDF{TF: search.TFRaw, IDF: search.IDFSmooth},
			wantBM25:  []string{"rare", "r2"},
			wantTFIDF: []string{"common", "rare"},
		},
		"IDF の平滑化なし: 全文書に出る語は点にならない": {
			docs: []search.Doc{
				{ID: "common", Content: "common common common"},
				{ID: "rare", Content: "common rare"},
				{ID: "r2", Content: "common rare"},
				{ID: "c", Content: "common"},
			},
			query:     "common rare",
			tfidf:     search.TFIDF{TF: search.TFRaw, IDF: search.IDFPlain},
			wantBM25:  []string{"rare", "r2"},
			wantTFIDF: []string{"rare", "r2"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix := wsIndex(tc.docs)
			bm25 := rankedIDs(ix.RankQuery(search.Query{Text: tc.query}, 2).Hits)
			tfidf := rankedIDs(ix.RankQuery(search.Query{Text: tc.query, Ranking: search.RankingTFIDF, TFIDF: tc.tfidf}, 2).Hits)
			if !slices.Equal(bm25, tc.wantBM25) || !slices.Equal(tfidf, tc.wantTFIDF) {
				t.Fatalf("bm25 = %v (want %v), tfidf = %v (want %v)", bm25, tc.wantBM25, tfidf, tc.wantTFIDF)
			}
		})
	}
}

// saturationDocs は同じ長さの 2 文書で、cat を 8 回繰り返す文書と cat と dog を 1 回ずつ含む文書を比べる。
func saturationDocs() []search.Doc {
	ds := []search.Doc{
		{ID: "spam", Content: strings.Repeat("cat ", 8)},
		{ID: "both", Content: "cat dog pad pad pad pad pad pad"},
	}
	for i := range 6 {
		ds = append(ds, search.Doc{ID: fmt.Sprintf("f%d", i), Content: "pad pad pad pad pad pad pad pad"})
	}
	ds = append(ds, search.Doc{ID: "d", Content: "dog pad pad pad pad pad pad pad"})
	return ds
}

// dictionaryDocs は語彙 vocab の語をすべて 100 回ずつ含む辞書の文書と、先頭の 4 語を 1 回ずつ含む短い文書を作る。
// 残りの語を 4 語ずつ含む文書を足し、どの語も 2 文書に出るようにして IDF をそろえる。
func dictionaryDocs(vocab int) ([]search.Doc, string) {
	word := func(i int) string { return fmt.Sprintf("v%05d", i) }
	var dict strings.Builder
	for i := range vocab {
		for range 100 {
			dict.WriteString(word(i))
			dict.WriteByte(' ')
		}
	}
	short := strings.Join([]string{word(0), word(1), word(2), word(3)}, " ")
	ds := []search.Doc{{ID: "dict", Content: dict.String()}, {ID: "short", Content: short}}
	for i := 4; i < vocab; i += 4 {
		ds = append(ds, search.Doc{ID: fmt.Sprintf("f%d", i), Content: strings.Join([]string{word(i), word(i + 1), word(i + 2), word(i + 3)}, " ")})
	}
	return ds, short
}

func TestTFIDFCosineDictionary(t *testing.T) {
	t.Parallel()
	ds, query := dictionaryDocs(10_000)
	ix := wsIndex(ds)
	score := func(r search.Ranking, normalize bool) map[string]float64 {
		out := make(map[string]float64)
		for _, h := range ix.RankQuery(search.Query{Text: query, Ranking: r, TFIDF: search.TFIDF{NormalizeQuery: normalize}}, -1).Hits {
			out[h.Doc.ID] = h.Score
		}
		return out
	}
	cosine, plain := score(search.RankingTFIDFCosine, true), score(search.RankingTFIDF, false)
	testCases := map[string]struct {
		got, want float64
	}{
		"コサインでは辞書の文書が 0.02":               {got: cosine["dict"], want: 0.02},
		"コサインでは同じ内容の短い文書が 1":              {got: cosine["short"], want: 1},
		"単純な TF-IDF では辞書の文書が短い文書の 3 倍になる": {got: plain["dict"] / plain["short"], want: 3},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if math.Abs(tc.got-tc.want) > 1e-9 {
				t.Fatalf("got %v, want %v", tc.got, tc.want)
			}
		})
	}
}

func TestTFIDFCosineQueryNormDoesNotChangeOrder(t *testing.T) {
	t.Parallel()
	ix := search.New(randomDocs(2_000))
	testCases := map[string]struct {
		query search.Query
	}{
		"2 語":        {query: search.Query{Text: benchQuery}},
		"3 語":        {query: search.Query{Text: "w00001 w00300 w01234"}},
		"AND":        {query: search.Query{Text: "w00001 w00002", Operator: search.OperatorAnd}},
		"同義語で広げたクエリ": {query: search.Query{Text: "w00050", Synonyms: map[string]string{"w00050": "w00051"}}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := tc.query
			q.Ranking = search.RankingTFIDFCosine
			raw := ix.RankQuery(q, -1).Hits
			q.TFIDF.NormalizeQuery = true
			norm := ix.RankQuery(q, -1).Hits
			if len(raw) == 0 || !slices.Equal(hitTitles(raw), hitTitles(norm)) {
				t.Fatalf("order differs: %d hits vs %d hits", len(raw), len(norm))
			}
			ratio := raw[0].Score / norm[0].Score
			for i := range raw {
				if math.Abs(raw[i].Score/norm[i].Score-ratio) > 1e-9 {
					t.Fatalf("hit %d ratio = %v, want %v", i, raw[i].Score/norm[i].Score, ratio)
				}
			}
		})
	}
}

func hitTitles(hits []search.Hit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Doc.Title
	}
	return out
}

// titledDocs は randomDocs の本文の先頭 3 語を隣の文書のタイトルにし、タイトルと本文に同じ語が出るようにする。
func titledDocs(n int) []search.Doc {
	ds := withIDs(randomDocs(n))
	for i := range ds {
		ds[i].Title = strings.Join(strings.Fields(ds[(i+1)%n].Content)[:3], " ")
	}
	return ds
}

// withoutField は項目 f を空にした文書を返す。
func withoutField(ds []search.Doc, f search.Field) []search.Doc {
	out := slices.Clone(ds)
	for i := range out {
		if f == search.FieldTitle {
			out[i].Title = ""
		} else {
			out[i].Content = ""
		}
	}
	return out
}

func TestTFIDFCosineFieldsMatchEmptiedField(t *testing.T) {
	t.Parallel()
	small := []search.Doc{
		{ID: "長いタイトル", Title: strings.Repeat("t1 t2 t3 t4 t5 t6 t7 t8 ", 5), Content: "cat"},
		{ID: "短いタイトル", Title: "t9", Content: "cat"},
		{ID: "本文だけ", Content: "cat"},
		{ID: "o1", Title: "cat", Content: "dog"},
		{ID: "o2", Title: "cat", Content: "dog"},
		{ID: "o3", Title: "cat", Content: "dog"},
	}
	random := titledDocs(2_000)
	testCases := map[string]struct {
		docs  []search.Doc
		query string
		field search.Field
	}{
		"本文だけの検索でタイトルの長さが点に効かない":     {docs: small, query: "cat", field: search.FieldContent},
		"タイトルに多く出る語を本文だけで検索しても 1 以下": {docs: small, query: "cat dog", field: search.FieldContent},
		"本文だけの 3 語":   {docs: random, query: "w00001 w00300 w01234", field: search.FieldContent},
		"タイトルだけの 2 語": {docs: random, query: "w00001 w00050", field: search.FieldTitle},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			other := search.FieldTitle
			if tc.field == search.FieldTitle {
				other = search.FieldContent
			}
			tfidf := search.TFIDF{NormalizeQuery: true}
			got := search.New(tc.docs).RankQuery(search.Query{Text: tc.query, Fields: []search.Field{tc.field}, Ranking: search.RankingTFIDFCosine, TFIDF: tfidf}, -1).Hits
			want := search.New(withoutField(tc.docs, other)).RankQuery(search.Query{Text: tc.query, Ranking: search.RankingTFIDFCosine, TFIDF: tfidf}, -1).Hits
			if len(got) == 0 || len(got) != len(want) {
				t.Fatalf("len = %d, want %d", len(got), len(want))
			}
			for i := range want {
				g, w := got[i], want[i]
				if g.Doc.ID != w.Doc.ID || math.Abs(g.Score-w.Score) > 1e-9 {
					t.Fatalf("hit %d = {%s %v}, want {%s %v}", i, g.Doc.ID, g.Score, w.Doc.ID, w.Score)
				}
				if g.Score > 1+1e-9 {
					t.Fatalf("hit %d %s score = %v, want <= 1", i, g.Doc.ID, g.Score)
				}
			}
		})
	}
}
