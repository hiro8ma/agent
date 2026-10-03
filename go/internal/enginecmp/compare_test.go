package enginecmp_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

func TestKenpouHasEveryArticleInOrder(t *testing.T) {
	t.Parallel()
	articles, err := enginecmp.Kenpou()
	if err != nil {
		t.Fatal(err)
	}
	if len(articles) != 103 {
		t.Fatalf("条の数 = %d, want 103", len(articles))
	}
	testCases := map[string]struct {
		number int
		prefix string
	}{
		"第一条は天皇の地位":      {number: 1, prefix: "第一条　天皇は、日本国の象徴であり"},
		"第二十六条は 2 項とも残る": {number: 26, prefix: "第二十六条　すべて国民は、法律の定めるところにより、その能力に応じて、ひとしく教育を受ける権利を有する。\nすべて国民は"},
		"第七条の号は号の番号を付ける": {number: 7, prefix: "第七条　天皇は、内閣の助言と承認により、国民のために、左の国事に関する行為を行ふ。\n一　憲法改正"},
		"最後は第百三条":        {number: 103, prefix: "第百三条　この憲法施行の際"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := articles[tc.number-1]; !strings.HasPrefix(got, tc.prefix) {
				t.Errorf("第%d条 = %q, want 接頭辞 %q", tc.number, got, tc.prefix)
			}
		})
	}
}

func TestLocalEngineOnKenpou(t *testing.T) {
	t.Parallel()
	articles, err := enginecmp.Kenpou()
	if err != nil {
		t.Fatal(err)
	}
	testCases := map[string]struct {
		ranking search.Ranking
		query   string
		want    []int
	}{
		"BM25 は「すべて国民」を 2 回含む第二十六条を最上位にする": {
			ranking: search.RankingBM25, query: "すべて国民", want: []int{26, 25, 13},
		},
		"規則による順位は同点を番号の小さい順に並べる": {
			ranking: search.RankingBucket, query: "すべて国民", want: []int{13, 14, 25},
		},
		"bigram は日本国の「本国」にも一致する": {
			ranking: search.RankingBM25, query: "本国", want: []int{1, 10, 98},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			e := &enginecmp.LocalEngine{Ranking: tc.ranking}
			if err := e.Index(t.Context(), articles); err != nil {
				t.Fatal(err)
			}
			hits, err := e.Search(t.Context(), tc.query, 3)
			if err != nil {
				t.Fatal(err)
			}
			if got := ids(hits); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("上位 = %v, want %v", got, tc.want)
			}
		})
	}
}

type stubEngine struct {
	indexErr error
	indexed  []string
}

func (*stubEngine) Name() string { return "stub" }

func (s *stubEngine) Index(_ context.Context, texts []string) error {
	s.indexed = texts
	return s.indexErr
}

func (s *stubEngine) Search(_ context.Context, query string, limit int) ([]enginecmp.Hit, error) {
	return []enginecmp.Hit{{ID: len(query), Score: float64(limit)}}, nil
}

func TestCompare(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		indexErr error
		wantErr  bool
	}{
		"索引してからクエリの順に結果を集める":    {},
		"索引に失敗したらエンジンの名前を付けて返す": {indexErr: errors.New("boom"), wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			e := &stubEngine{indexErr: tc.indexErr}
			reports, err := enginecmp.Compare(t.Context(), []enginecmp.Engine{e}, []string{"a", "b"}, []string{"x", "yy"}, 5)
			if tc.wantErr {
				if !errors.Is(err, tc.indexErr) || !strings.HasPrefix(err.Error(), "stub: ") {
					t.Fatalf("err = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(e.indexed, []string{"a", "b"}) {
				t.Errorf("索引した文書 = %v", e.indexed)
			}
			want := [][]enginecmp.Hit{{{ID: 1, Score: 5}}, {{ID: 2, Score: 5}}}
			if len(reports) != 1 || reports[0].Engine != "stub" || !reflect.DeepEqual(reports[0].Hits, want) {
				t.Errorf("reports = %+v", reports)
			}
		})
	}
}

func TestLocalEngineSearchBeforeIndex(t *testing.T) {
	t.Parallel()
	if _, err := (&enginecmp.LocalEngine{}).Search(t.Context(), "国民", 3); err == nil {
		t.Error("Index の前の Search がエラーにならない")
	}
}

func ids(hits []enginecmp.Hit) []int {
	out := make([]int, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}
