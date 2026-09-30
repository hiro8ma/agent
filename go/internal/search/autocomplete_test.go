package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// curryIndex は語を空白で区切った文書で、文書頻度がカレー 6、カツ 5、カレーライス 4、カツカレー 3、カレーパン 2、カツ丼 1 になる。
func curryIndex() *search.Index {
	all := []string{"カレー", "カツ", "カレーライス", "カツカレー", "カレーパン", "カツ丼"}
	var ds []search.Doc
	for n := len(all); n >= 1; n-- {
		content := ""
		for _, w := range all[:n] {
			content += w + " "
		}
		ds = append(ds, search.Doc{Content: content})
	}
	return search.New(ds, search.WithAnalyzer(search.NewAnalyzer(search.WithWhitespaceTokenizer())))
}

func TestCompleterCurry(t *testing.T) {
	t.Parallel()
	ix := curryIndex()
	testCases := map[string]struct {
		typed  string
		opt    search.Autocomplete
		want   []string
		counts []int
	}{
		"カ は 6 語を文書頻度の順に返す": {
			typed: "カ", want: []string{"カレー", "カツ", "カレーライス", "カツカレー", "カレーパン", "カツ丼"}, counts: []int{6, 5, 4, 3, 2, 1},
		},
		"カレ はカツカレーを含めない":          {typed: "カレ", want: []string{"カレー", "カレーライス", "カレーパン"}},
		"カレー は入力した語そのものも候補にする":    {typed: "カレー", want: []string{"カレー", "カレーライス", "カレーパン"}},
		"カツ はカツで始まる語だけを返す":        {typed: "カツ", want: []string{"カツ", "カツカレー", "カツ丼"}},
		"K が 2 なら上位 2 件":          {typed: "カ", opt: search.Autocomplete{K: 2}, want: []string{"カレー", "カツ"}},
		"前の語は補完せずそのまま残す":          {typed: "カツ カレ", want: []string{"カツ カレー", "カツ カレーライス", "カツ カレーパン"}},
		"前の語が接頭辞でも広げない":           {typed: "カ カレーパ", want: []string{"カ カレーパン"}},
		"空白で終われば最後の語も入力済みとして返さない": {typed: "カレー ", want: nil},
		"一致する語が無ければ返さない":          {typed: "ラ", want: nil},
		"検索ログの回数を渡すとその順に並べ 0 回は語の順にする": {
			typed: "カ", opt: search.Autocomplete{Counts: map[string]int{"カレーパン": 100, "カツ丼": 50}},
			want: []string{"カレーパン", "カツ丼", "カツ", "カツカレー", "カレー", "カレーライス"}, counts: []int{100, 50, 0, 0, 0, 0},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			for _, noCache := range []bool{false, true} {
				opt := tc.opt
				opt.NoCache = noCache
				var got []string
				var counts []int
				for _, c := range search.NewCompleter(ix, opt).Complete(tc.typed) {
					got = append(got, c.Text)
					counts = append(counts, c.Count)
				}
				if !slices.Equal(got, tc.want) || (tc.counts != nil && !slices.Equal(counts, tc.counts)) {
					t.Fatalf("NoCache %v: %v %v, want %v %v", noCache, got, counts, tc.want, tc.counts)
				}
			}
		})
	}
}
