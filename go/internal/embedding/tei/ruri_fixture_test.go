package tei_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
)

const (
	ruriModel   = "cl-nagoya/ruri-v3-310m"
	ruriImage   = "ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4"
	fixturePath = "testdata/ruri_kenpou.json"
)

// tutorialSentences は教材が sentence-transformers で求めたコサインの例。接頭辞は付けない。
var tutorialSentences = []string{"カレーライスはおいしい", "カレーのお店", "転置インデックスは便利"}

// sparsePairs の Words は手で語に分けたもの。語の単位の索引語として空白で区切る。
var sparsePairs = []struct{ A, B, WordsA, WordsB string }{
	{A: "カレーライス", B: "カレー", WordsA: "カレーライス", WordsB: "カレー"},
	{A: "カレーライスはおいしい", B: "カレーのお店", WordsA: "カレーライス は おいしい", WordsB: "カレー の お店"},
	{A: "カレーライスはおいしい", B: "転置インデックスは便利", WordsA: "カレーライス は おいしい", WordsB: "転置 インデックス は 便利"},
	{A: "インド料理のお店", B: "キーマカレーの専門店", WordsA: "インド 料理 の お店", WordsB: "キーマカレー の 専門店"},
	{A: "ねこ", B: "猫", WordsA: "ねこ", WordsB: "猫"},
	{A: "自動車", B: "車", WordsA: "自動車", WordsB: "車"},
	{A: "猫はかわいい", B: "アクビをする猫", WordsA: "猫 は かわいい", WordsB: "アクビ を する 猫"},
}

// unrelatedSentences は互いに話題の重ならない短い文。異方性を見るときに憲法の条文と混ぜる。
var unrelatedSentences = []string{
	"今日は朝から雨が降っている",
	"株価が大きく下がった",
	"富士山は日本一高い山だ",
	"量子コンピュータの研究が進む",
	"サッカーの試合は延長戦になった",
	"冷蔵庫に牛乳が残っている",
	"電車が遅れて会議に遅刻した",
	"バッハのフーガを練習する",
	"火星探査機が着陸した",
	"ラーメンは塩味が好きだ",
	"新しいスマートフォンを買った",
	"川で魚を釣った",
	"図書館で本を借りた",
	"猫はかわいい",
	"インド料理のお店",
	"転置インデックスは便利",
}

// searchQueries の Relevant は手で付けた正解の条の番号。条文と同じ語を避けた言い方を多く含める。
var searchQueries = []struct {
	Text     string
	Relevant []int
}{
	{Text: "差別されない権利", Relevant: []int{14, 44}},
	{Text: "宗教の自由", Relevant: []int{20}},
	{Text: "戦争をしない", Relevant: []int{9}},
	{Text: "裁判を受ける権利", Relevant: []int{32, 37}},
	{Text: "学校に通う権利", Relevant: []int{26}},
	{Text: "税金を払う義務", Relevant: []int{30}},
	{Text: "働く権利", Relevant: []int{27}},
	{Text: "表現の自由", Relevant: []int{21}},
	{Text: "好きな仕事を選べる", Relevant: []int{22}},
	{Text: "結婚は二人の合意で決まる", Relevant: []int{24}},
}

type fixtureVector struct {
	Text   string    `json:"text"`
	Values []float32 `json:"values"`
}

// fixtureQuery の Prefixed は RuriQueryPrefix を付けたもの、Plain は付けないもの。Elasticsearch は記録した kuromoji + BM25 の上位の条の番号。
type fixtureQuery struct {
	Text          string    `json:"text"`
	Prefixed      []float32 `json:"prefixed"`
	Plain         []float32 `json:"plain"`
	Elasticsearch []int     `json:"elasticsearch,omitempty"`
}

// ruriFixture の Articles は enginecmp.Kenpou と同じ順。Sentences は接頭辞なしで、教材の例 / 対の両側 / 無関係な文を重複なく持つ。
type ruriFixture struct {
	Model            string          `json:"model"`
	Image            string          `json:"image"`
	Dimensions       int             `json:"dimensions"`
	ArticlesPrefixed [][]float32     `json:"articles_prefixed"`
	ArticlesPlain    [][]float32     `json:"articles_plain"`
	Queries          []fixtureQuery  `json:"queries"`
	Sentences        []fixtureVector `json:"sentences"`
}

func fixtureSentenceTexts() []string {
	var out []string
	seen := map[string]bool{}
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	for _, s := range tutorialSentences {
		add(s)
	}
	for _, p := range sparsePairs {
		add(p.A)
		add(p.B)
	}
	for _, s := range unrelatedSentences {
		add(s)
	}
	return out
}

func loadFixture() (ruriFixture, []string, error) {
	var f ruriFixture
	b, err := os.ReadFile(filepath.FromSlash(fixturePath))
	if err != nil {
		return f, nil, err
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return f, nil, err
	}
	articles, err := enginecmp.Kenpou()
	if err != nil {
		return f, nil, err
	}
	if len(f.ArticlesPrefixed) != len(articles) || len(f.ArticlesPlain) != len(articles) {
		return f, nil, fmt.Errorf("fixture has %d/%d articles, want %d", len(f.ArticlesPrefixed), len(f.ArticlesPlain), len(articles))
	}
	if len(f.Queries) != len(searchQueries) {
		return f, nil, fmt.Errorf("fixture has %d queries, want %d", len(f.Queries), len(searchQueries))
	}
	for i, q := range searchQueries {
		if f.Queries[i].Text != q.Text {
			return f, nil, fmt.Errorf("fixture query %d is %q, want %q", i, f.Queries[i].Text, q.Text)
		}
	}
	return f, articles, nil
}

func (f ruriFixture) sentence(text string) []float32 {
	for _, s := range f.Sentences {
		if s.Text == text {
			return s.Values
		}
	}
	panic(fmt.Sprintf("fixture has no sentence %q", text))
}

// roundVector は保存する JSON を小さくする。正規化したベクトルの成分を小数 4 桁に丸めても、コサインの誤差は 1e-4 程度に収まる。
func roundVector(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(math.Round(float64(x)*1e4) / 1e4)
	}
	return out
}

func roundVectors(vs [][]float32) [][]float32 {
	out := make([][]float32, len(vs))
	for i, v := range vs {
		out[i] = roundVector(v)
	}
	return out
}
