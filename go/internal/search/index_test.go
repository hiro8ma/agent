package search_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func docs(contents ...string) []search.Doc {
	ds := make([]search.Doc, len(contents))
	for i, c := range contents {
		ds[i] = search.Doc{Title: fmt.Sprintf("d%d", i), Content: c}
	}
	return ds
}

func titles(hits []search.Hit) []string {
	ts := make([]string, len(hits))
	for i, h := range hits {
		ts[i] = h.Doc.Title
	}
	return ts
}

func TestTokenize(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		text string
		want []string
	}{
		"英数字は小文字の単語になる":         {text: "Cloud Run, Go1.27!", want: []string{"cloud", "run", "go1", "27"}},
		"日本語は文字の bigram になる":    {text: "締め日", want: []string{"締め", "め日"}},
		"日本語 1 文字は unigram になる": {text: "日", want: []string{"日"}},
		"英語と日本語の境目で切れる":         {text: "Goを使う", want: []string{"go", "を使", "使う"}},
		"長音記号はカタカナとつながる":        {text: "サーバー", want: []string{"サー", "ーバ", "バー"}},
		"記号と空白だけなら語は無い":         {text: " 、。!? ", want: nil},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.Tokenize(tc.text); !slices.Equal(got, tc.want) {
				t.Fatalf("Tokenize(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestRankScoresOnlyCandidates(t *testing.T) {
	t.Parallel()
	ix := search.New(docs(
		"go backend service",
		"next frontend",
		"cloud run infra",
		"go cloud run",
		"expense report deadline",
	))
	testCases := map[string]struct {
		query      string
		wantScored int
		wantTitles []string
	}{
		"1 語なら含む文書だけを採点する":         {query: "go", wantScored: 2, wantTitles: []string{"d0", "d3"}},
		"2 語なら postings の和集合を採点する": {query: "go infra", wantScored: 3, wantTitles: []string{"d2", "d0", "d3"}},
		"どの文書にも無い語なら採点しない":         {query: "kubernetes", wantScored: 0, wantTitles: []string{}},
		"空のクエリなら採点しない":             {query: "", wantScored: 0, wantTitles: []string{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			res := ix.Rank(tc.query, 10)
			if res.Scored != tc.wantScored {
				t.Fatalf("Scored = %d, want %d", res.Scored, tc.wantScored)
			}
			if got := titles(res.Hits); !slices.Equal(got, tc.wantTitles) {
				t.Fatalf("titles = %v, want %v", got, tc.wantTitles)
			}
		})
	}
}

func TestRankBM25(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		docs    []search.Doc
		query   string
		wantTop string
	}{
		"珍しい語を 1 回含む文書がありふれた語を 2 回含む文書より上": {
			docs: docs(
				"common common filler",
				"rare filler filler",
				"common other words", "common other words", "common other words", "common other words",
				"common other words", "common other words", "common other words", "common other words",
			),
			query:   "common rare",
			wantTop: "d1",
		},
		"同じ長さなら語を多く含む文書が上": {
			docs:    docs("go java java java", "go go go java"),
			query:   "go",
			wantTop: "d1",
		},
		"同じ出現回数なら短い文書が上": {
			docs:    docs("go x x x x x x x x x", "go x"),
			query:   "go",
			wantTop: "d1",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			res := search.New(tc.docs).Rank(tc.query, 1)
			if len(res.Hits) != 1 || res.Hits[0].Doc.Title != tc.wantTop {
				t.Fatalf("top = %v, want %s", titles(res.Hits), tc.wantTop)
			}
		})
	}
}

func TestSearchJapanese(t *testing.T) {
	t.Parallel()
	ix := search.New([]search.Doc{
		{Title: "経費精算の締め日", Content: "経費精算の申請締め日は毎月25日。締め日を過ぎた申請は翌月精算になる。"},
		{Title: "リモートワーク規程", Content: "リモートワークは週3日まで。コアタイム 11:00-15:00 は接続必須。"},
		{Title: "注文キャンセルポリシー", Content: "出荷前の注文はキャンセル可能。出荷後は返品扱いとなり、返送料は顧客負担。"},
		{Title: "技術選定ガイドライン", Content: "バックエンドは Go を第一候補とする。インフラは Cloud Run を標準とする。"},
	})
	testCases := map[string]struct {
		query   string
		wantTop string
	}{
		"語順が違っても bigram でつながる": {query: "締め日の経費", wantTop: "経費精算の締め日"},
		"カタカナ語で引ける":            {query: "リモート勤務", wantTop: "リモートワーク規程"},
		"日本語と英語が混ざったクエリで引ける":   {query: "Cloud Run の標準", wantTop: "技術選定ガイドライン"},
		"文書に無い日本語は候補に入らない":     {query: "有給休暇", wantTop: ""},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := ix.Search(t.Context(), tc.query, 1)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			top := ""
			if len(got) > 0 {
				top = got[0].Title
			}
			if top != tc.wantTop {
				t.Fatalf("top = %q, want %q", top, tc.wantTop)
			}
		})
	}
}

func TestSearchLimitAndContext(t *testing.T) {
	t.Parallel()
	ix := search.New(docs("go a", "go b", "go c"))
	got, err := ix.Search(t.Context(), "go", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := ix.Search(ctx, "go", 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
