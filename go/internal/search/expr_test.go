package search_test

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestParseExpr(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		expr string
		want string
	}{
		"括弧の中の OR を先にまとめる":          {expr: "(カツ OR カレー) AND おいしい", want: "((カツ OR カレー) AND おいしい)"},
		"括弧が無ければ AND が OR より先に結び付く": {expr: "カツ OR カレー AND おいしい", want: "(カツ OR (カレー AND おいしい))"},
		"同じ演算子の並びは 1 つにまとめる":        {expr: "カツ AND カレー AND おいしい", want: "(カツ AND カレー AND おいしい)"},
		"全角の括弧も受け付ける":               {expr: "（カツ OR カレー）AND おいしい", want: "((カツ OR カレー) AND おいしい)"},
		"入れ子の括弧":                    {expr: "((カツ OR カレー) AND (ライス OR ナン)) OR スパイス", want: "(((カツ OR カレー) AND (ライス OR ナン)) OR スパイス)"},
		"語が 1 つなら語そのもの":             {expr: " カレー ", want: "カレー"},
		"小文字の and は語として読む":          {expr: "and OR or", want: "(and OR or)"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			e, err := search.ParseExpr(tc.expr)
			if err != nil {
				t.Fatalf("ParseExpr(%q): %v", tc.expr, err)
			}
			if got := e.String(); got != tc.want {
				t.Fatalf("String() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParseExprSyntaxError(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		expr string
	}{
		"空":            {expr: "  "},
		"閉じ括弧が無い":      {expr: "(カツ OR カレー AND おいしい"},
		"開き括弧が無い":      {expr: "カツ OR カレー) AND おいしい"},
		"末尾に演算子":       {expr: "カツ OR"},
		"先頭に演算子":       {expr: "AND カツ"},
		"演算子が続く":       {expr: "カツ OR OR カレー"},
		"語を演算子なしで並べる":  {expr: "カツ カレー"},
		"空の括弧":         {expr: "カツ AND ()"},
		"括弧の後に演算子なしで語": {expr: "(カツ OR カレー) おいしい"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if _, err := search.ParseExpr(tc.expr); !errors.Is(err, search.ErrExprSyntax) {
				t.Fatalf("ParseExpr(%q) error = %v, want ErrExprSyntax", tc.expr, err)
			}
		})
	}
}

func mustExpr(t *testing.T, s string) *search.Expr {
	t.Helper()
	e, err := search.ParseExpr(s)
	if err != nil {
		t.Fatalf("ParseExpr(%q): %v", s, err)
	}
	return e
}

func TestRankExprTutorial(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	testCases := map[string]struct {
		expr string
		want []string
	}{
		"(カツ OR カレー) AND おいしいはおいしいを含む文書 1 と 2（文書は 1 から数える）": {expr: "(カツ OR カレー) AND おいしい", want: []string{"d0", "d1"}},
		"括弧が無ければカツ OR (カレー AND おいしい) で文書 3 も入る":             {expr: "カツ OR カレー AND おいしい", want: []string{"d0", "d1", "d2"}},
		"カツ AND (ライス OR スパイス) は文書 3 だけ":                     {expr: "カツ AND (ライス OR スパイス)", want: []string{"d2"}},
		"(スパイス OR カツ) AND (辛い OR ライス) は文書 2 と 3":            {expr: "(スパイス OR カツ) AND (辛い OR ライス)", want: []string{"d1", "d2"}},
		"索引に無い語だけの AND は候補が無い":                              {expr: "ナン AND カレー", want: []string{}},
		"索引に無い語を OR で足しても候補は変わらない":                          {expr: "ナン OR スパイス", want: []string{"d1"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			res := ix.RankQuery(search.Query{Expr: mustExpr(t, tc.expr)}, -1)
			got := hitDocIDs(res.Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("docs = %v, want %v", got, tc.want)
			}
			if res.Scored != len(tc.want) {
				t.Fatalf("Scored = %d, want %d", res.Scored, len(tc.want))
			}
		})
	}
}

// TestRankExprScoresAllWords は式の候補の点数が、式の語すべてを OR で並べたクエリの点数と同じであることを確かめる。
func TestRankExprScoresAllWords(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	e := mustExpr(t, "(カツ OR スパイス) AND (カレー OR ライス)")
	or := map[string]float64{}
	for _, h := range ix.RankQuery(search.Query{Text: strings.Join(e.Words(), " ")}, -1).Hits {
		or[h.Doc.ID] = h.Score
	}
	hits := ix.RankQuery(search.Query{Expr: e}, -1).Hits
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	for _, h := range hits {
		if s, ok := or[h.Doc.ID]; !ok || s != h.Score {
			t.Fatalf("%s: expr score %v, OR score %v (ok=%v)", h.Doc.ID, h.Score, s, ok)
		}
	}
}

// randomExpr は語を w00001 から w00040 に限った式と、同じ式を文書の語の集合で直接判定する関数を作る。
func randomExpr(r *rand.Rand, depth int) (string, func(map[string]bool) bool) {
	if depth == 0 || r.IntN(3) == 0 {
		w := fmt.Sprintf("w%05d", 1+r.IntN(40))
		return w, func(has map[string]bool) bool { return has[w] }
	}
	n := 2 + r.IntN(2)
	texts := make([]string, n)
	preds := make([]func(map[string]bool) bool, n)
	for i := range n {
		texts[i], preds[i] = randomExpr(r, depth-1)
	}
	if r.IntN(2) == 0 {
		return "(" + strings.Join(texts, " AND ") + ")", func(has map[string]bool) bool {
			return !slices.ContainsFunc(preds, func(p func(map[string]bool) bool) bool { return !p(has) })
		}
	}
	return "(" + strings.Join(texts, " OR ") + ")", func(has map[string]bool) bool {
		return slices.ContainsFunc(preds, func(p func(map[string]bool) bool) bool { return p(has) })
	}
}

// TestRankExprMatchesNaive はでたらめに組んだ式の候補が、文書を 1 件ずつ調べた結果と集合として一致することを確かめる。
func TestRankExprMatchesNaive(t *testing.T) {
	t.Parallel()
	ds := randomDocs(2_000)
	words := make([]map[string]bool, len(ds))
	for i, d := range ds {
		words[i] = map[string]bool{}
		for w := range strings.FieldsSeq(d.Content) {
			words[i][w] = true
		}
	}
	r := rand.New(rand.NewPCG(3, 5))
	type exprCase struct {
		text string
		pred func(map[string]bool) bool
	}
	exprs := make([]exprCase, 50)
	for i := range exprs {
		exprs[i].text, exprs[i].pred = randomExpr(r, 3)
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
			for _, ec := range exprs {
				want := []string{}
				for i, d := range ds {
					if ec.pred(words[i]) {
						want = append(want, d.Title)
					}
				}
				got := titles(ix.RankQuery(search.Query{Expr: mustExpr(t, ec.text)}, -1).Hits)
				slices.Sort(got)
				slices.Sort(want)
				if !slices.Equal(got, want) {
					t.Fatalf("%s: got %d docs, want %d", ec.text, len(got), len(want))
				}
			}
		})
	}
}

func TestRankExprFields(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{ID: "a", Title: "カツ", Content: "おいしい"},
		{ID: "b", Title: "おいしい", Content: "カレー"},
		{ID: "c", Title: "ライス", Content: "カツ おいしい"},
	}
	ix := search.New(ds, search.WithAnalyzer(search.NewAnalyzer(search.WithWhitespaceTokenizer())))
	testCases := map[string]struct {
		fields []search.Field
		want   []string
	}{
		"項目を絞らなければタイトルと本文のどちらでもよい": {want: []string{"a", "b", "c"}},
		"本文に絞ると本文だけで式を満たす文書を残す":    {fields: []search.Field{search.FieldContent}, want: []string{"c"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := hitDocIDs(ix.RankQuery(search.Query{Expr: mustExpr(t, "(カツ OR カレー) AND おいしい"), Fields: tc.fields}, -1).Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("docs = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestShardedRankExpr(t *testing.T) {
	t.Parallel()
	ds := randomDocs(1_000)
	e := mustExpr(t, "(w00001 OR w00030) AND w00007")
	one := search.New(ds).RankQuery(search.Query{Expr: e}, 10).Hits
	sh := search.NewSharded(ds, 4, search.HashRouter(4))
	sh.Type = search.DFSQueryThenFetch
	got := sh.RankQuery(search.Query{Expr: e}, 10).Hits
	if !slices.Equal(titles(got), titles(one)) {
		t.Fatalf("sharded = %v, single = %v", titles(got), titles(one))
	}
}
