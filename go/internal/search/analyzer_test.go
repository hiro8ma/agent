package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestNormalize(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		text string
		want string
	}{
		"全角英数字は半角の小文字になる":             {text: "Ｇｏ１．２７", want: "go1.27"},
		"半角カナは全角カナになる":                {text: "ｶﾘｰ", want: "カリー"},
		"半角カナの濁点は 1 文字にまとまる":          {text: "ｻｰﾊﾞｰ", want: "サーバ"},
		"4 文字以上のカタカナ語は末尾の長音を落とす":      {text: "コンピューター", want: "コンピュータ"},
		"3 文字のカタカナ語は末尾の長音を残す":         {text: "カレー", want: "カレー"},
		"語の途中の長音は残す":                  {text: "サーバーは", want: "サーバは"},
		"カタカナの直後のダッシュ類は長音になる":         {text: "サ−バ", want: "サーバ"},
		"カタカナの直後の ASCII のハイフンは区切りのまま": {text: "テスト-ケース", want: "テスト-ケース"},
		"ひらがなの後の長音は変えない":              {text: "すごーい", want: "すごーい"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.Normalize(tc.text); got != tc.want {
				t.Fatalf("Normalize(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestAnalyze(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		opts []search.AnalyzerOption
		text string
		want []string
	}{
		"既定では Tokenize と同じ": {
			text: "日本のｶﾘｰ",
			want: search.Tokenize("日本のｶﾘｰ"),
		},
		"正規化してから bigram にする": {
			opts: []search.AnalyzerOption{search.WithNormalization()},
			text: "日本のｶﾘｰ",
			want: []string{"日本", "本の", "のカ", "カリ", "リー"},
		},
		"助詞だけの bigram と英語の一般語を落とす": {
			opts: []search.AnalyzerOption{search.WithStopWords(search.DefaultStopWords()...)},
			text: "京都には the Go",
			want: []string{"京都", "都に", "go"},
		},
		"辞書の見出しを置き換えてから bigram にする": {
			opts: []search.AnalyzerOption{search.WithSynonyms(map[string]string{"カリー": "カレー"})},
			text: "カリー",
			want: []string{"カレ", "レー"},
		},
		"辞書は長い見出しを優先する": {
			opts: []search.AnalyzerOption{search.WithSynonyms(map[string]string{"カリー": "カレー", "カリーパン": "カレーパン", "パン": "麺麭"})},
			text: "カリーパン",
			want: []string{"カレ", "レー", "ーパ", "パン"},
		},
		"辞書の見出しも正規化してから照合する": {
			opts: []search.AnalyzerOption{search.WithNormalization(), search.WithSynonyms(map[string]string{"ｶﾘｰ": "カレー"})},
			text: "カリー",
			want: []string{"カレ", "レー"},
		},
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

func TestRankWithAnalyzer(t *testing.T) {
	t.Parallel()
	curry := []search.Doc{
		{Title: "日本のカレー", Content: "日本のカレーは小麦粉でとろみをつけたルーが特徴です"},
		{Title: "インドのカリー", Content: "インドのカリーはスパイスが特徴です"},
		{Title: "日本の寿司", Content: "日本の寿司は酢飯が特徴です"},
	}
	standard := []search.AnalyzerOption{search.WithNormalization(), search.WithStopWords(search.DefaultStopWords()...)}
	testCases := map[string]struct {
		docs       []search.Doc
		opts       []search.AnalyzerOption
		query      string
		wantScored int
		wantTop    string
	}{
		"辞書が無ければカリーと書いた文書が上": {
			docs: curry, opts: standard, query: "日本のカリー 特徴", wantScored: 3, wantTop: "インドのカリー",
		},
		"辞書でカリーをカレーに寄せると日本のカレーが上": {
			docs:  curry,
			opts:  append(slices.Clone(standard), search.WithSynonyms(map[string]string{"カリー": "カレー"})),
			query: "日本のカリー 特徴", wantScored: 3, wantTop: "日本のカレー",
		},
		"正規化が無ければ半角カナのクエリはカタカナの文書に当たらない": {
			docs: docs("サーバーの監視", "ネットワーク"), query: "ｻｰﾊﾞｰ", wantScored: 0, wantTop: "",
		},
		"正規化すれば半角カナのクエリがカタカナの文書に当たる": {
			docs: docs("サーバーの監視", "ネットワーク"), opts: standard, query: "ｻｰﾊﾞｰ", wantScored: 1, wantTop: "d0",
		},
		"除去が無ければ助詞だけの bigram で候補が増える": {
			docs: docs("京都には寺が多い", "大阪には城がある", "奈良の鹿"), query: "京都には", wantScored: 2, wantTop: "d0",
		},
		"除去すれば助詞だけの bigram では候補にならない": {
			docs: docs("京都には寺が多い", "大阪には城がある", "奈良の鹿"), opts: standard, query: "京都には", wantScored: 1, wantTop: "d0",
		},
		"除去すれば英語の一般語では候補にならない": {
			docs: docs("the cloud", "go run"), opts: standard, query: "the go", wantScored: 1, wantTop: "d1",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix := search.New(tc.docs, search.WithAnalyzer(search.NewAnalyzer(tc.opts...)))
			res := ix.Rank(tc.query, 1)
			if res.Scored != tc.wantScored {
				t.Fatalf("Scored = %d, want %d", res.Scored, tc.wantScored)
			}
			top := ""
			if len(res.Hits) > 0 {
				top = res.Hits[0].Doc.Title
			}
			if top != tc.wantTop {
				t.Fatalf("top = %q, want %q", top, tc.wantTop)
			}
		})
	}
}
