package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestAnalyzeEnglish(t *testing.T) {
	t.Parallel()
	english := []search.AnalyzerOption{search.WithEnglish()}
	cities := []search.AnalyzerOption{search.WithEnglish(), search.WithPhrases("New York", "San Francisco", "United States of America")}
	testCases := map[string]struct {
		opts []search.AnalyzerOption
		text string
		want []string
	}{
		"複数語を 1 つの索引語にする":       {opts: cities, text: "I live in New York", want: []string{"i", "live", "in", "new_york"}},
		"複数語は大文字小文字を区別しない":      {opts: cities, text: "NEW york", want: []string{"new_york"}},
		"複数語は空白とハイフンの揺れを許す":     {opts: cities, text: "San  Francisco-based", want: []string{"san_francisco", "based"}},
		"複数語は除去語を含んでもつなぐ":       {opts: cities, text: "United States of America", want: []string{"united_states_of_america"}},
		"句読点をまたぐ語はつながない":        {opts: cities, text: "new. York", want: []string{"new", "york"}},
		"複数語は短縮形の設定が無くても使える":    {opts: []search.AnalyzerOption{search.WithPhrases("New York")}, text: "I'm in New York", want: []string{"i", "m", "in", "new_york"}},
		"短縮形を展開する":              {opts: english, text: "I'm sure it's done", want: []string{"i", "am", "sure", "it", "is", "done"}},
		"辞書に無い n't も not に展開する": {opts: english, text: "shouldn't", want: []string{"should", "not"}},
		"所有格の 's を落とす":          {opts: english, text: "Mike's book", want: []string{"mike", "book"}},
		"右シングル引用符の所有格も落とす":      {opts: english, text: "Mike’s book", want: []string{"mike", "book"}},
		"先頭のエリジオンを落とす":          {opts: english, text: "l'histoire d'Artagnan", want: []string{"histoire", "artagnan"}},
		"辞書に無いアポストロフィでは分ける":     {opts: english, text: "O'Brien", want: []string{"o", "brien"}},
		"語の後ろの引用符は区切り":          {opts: english, text: "'students' books'", want: []string{"students", "books"}},
		"IPv4 アドレスを 1 つにする":     {opts: english, text: "ping 192.168.0.1.", want: []string{"ping", "192.168.0.1"}},
		"範囲外の数を含むものは IPv4 にしない": {opts: english, text: "192.168.0.256", want: []string{"192", "168", "0", "256"}},
		"5 つ並んだ数は IPv4 にしない":    {opts: english, text: "1.2.3.4.5", want: []string{"1", "2", "3", "4", "5"}},
		"ドメイン名を 1 つにする":         {opts: english, text: "see Google.com, docs.go.dev", want: []string{"see", "google.com", "docs.go.dev"}},
		"最後のラベルが数字ならドメイン名にしない":  {opts: english, text: "3.14 go1.27", want: []string{"3", "14", "go1", "27"}},
		"文末のピリオドはドメイン名にしない":     {opts: english, text: "hello. world", want: []string{"hello", "world"}},
		"1 文字とピリオドの繰り返しを略語にする":  {opts: english, text: "U.S.A. and e.g.", want: []string{"usa", "and", "eg"}},
		"略語は最後のピリオドが無くてもまとめる":   {opts: english, text: "U.S.A", want: []string{"usa"}},
		"日本語は bigram のまま":       {opts: cities, text: "東京とNew York", want: []string{"東京", "京と", "new_york"}},
		"語幹化は英小文字だけの索引語にかける":    {opts: append(slices.Clone(cities), search.WithStemming()), text: "walked in New York google.com", want: []string{"walk", "in", "new_york", "google.com"}},
		"除去語は語幹化の前の形で照合する":      {opts: []search.AnalyzerOption{search.WithStemming(), search.WithStopWords("was")}, text: "it was", want: []string{"it"}},
		"見出し語化は後に渡したものが勝つ":      {opts: []search.AnalyzerOption{search.WithStemming(), search.WithLemmatization()}, text: "children saw", want: []string{"child", "see"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.NewAnalyzer(tc.opts...).Analyze(tc.text); !slices.Equal(got, tc.want) {
				t.Fatalf("Analyze(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestAnalyzeDefaultIgnoresEnglishRules(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		text string
		want []string
	}{
		"複数語":     {text: "I live in New York", want: []string{"i", "live", "in", "new", "york"}},
		"短縮形と所有格": {text: "I'm sure it's Mike's book", want: []string{"i", "m", "sure", "it", "s", "mike", "s", "book"}},
		"エリジオン":   {text: "l'histoire", want: []string{"l", "histoire"}},
		"略語":      {text: "U.S.A.", want: []string{"u", "s", "a"}},
		"ドメイン名":   {text: "google.com", want: []string{"google", "com"}},
		"IPv4":    {text: "192.168.0.1", want: []string{"192", "168", "0", "1"}},
		"語形変化":    {text: "walks walked walking", want: []string{"walks", "walked", "walking"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.NewAnalyzer().Analyze(tc.text); !slices.Equal(got, tc.want) {
				t.Fatalf("Analyze(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestPhraseSameForDocsAndQueries(t *testing.T) {
	t.Parallel()
	a := search.NewAnalyzer(search.WithEnglish(), search.WithPhrases("New York"), search.WithStopWords(search.DefaultStopWords()...))
	ix := search.New(docs("I love NEW YORK pizza", "a new cafe in York", "pizza in Naples"), search.WithAnalyzer(a))
	testCases := map[string]struct {
		query  search.Query
		wantTo []string
	}{
		"クエリの New York も 1 つの索引語になり、新しい York の文書には当たらない": {
			query: search.Query{Text: "new york"}, wantTo: []string{"d0"},
		},
		"1 つの索引語として位置を数えるのでフレーズで次の語と並ぶ": {
			query: search.Query{Text: "New-York pizza", Phrase: true}, wantTo: []string{"d0"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := titles(ix.RankQuery(tc.query, 10).Hits); !slices.Equal(got, tc.wantTo) {
				t.Fatalf("hits = %q, want %q", got, tc.wantTo)
			}
		})
	}
	if got := search.QueryTerms(ix, "New York"); !slices.Equal(got, []string{"new_york"}) {
		t.Fatalf("QueryTerms = %q, want [new_york]", got)
	}
	ny, pizza := search.PostingsOf(ix, "new_york"), search.PostingsOf(ix, "pizza")
	if len(ny) != 1 || len(pizza) != 2 || pizza[0].Positions[0] != ny[0].Positions[0]+1 {
		t.Fatalf("new_york = %+v, pizza = %+v, want pizza right after new_york", ny, pizza)
	}
}
