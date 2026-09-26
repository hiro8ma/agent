package search_test

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

var morphologyDocs = []string{
	"A child learns to read with picture books",
	"Parenting tips for a young child",
	"The children played soccer after school",
	"She walks her dog every morning",
	"He walked ten miles along the river",
	"Walking improves heart health",
	"The river flooded the valley",
	"Mice live in the old barn",
	"A mouse ran across the kitchen floor",
	"The cat caught a bird",
	"We visited NEW YORK last summer",
	"New-York City has many museums",
	"A new cafe opened in York",
	"The new museum opened in the city",
	"The university announced a research center",
	"Universities compete for international students",
	"The universe is expanding according to astronomers",
	"Galaxies fill the observable universe",
	"The organization hired more staff",
	"Organizations must protect personal data",
	"The heart is a vital organ",
	"Organ donation saves lives",
	"Today's news covers the election",
	"News reporters interviewed the mayor",
	"A new phone was released today",
	"I saw a movie yesterday",
	"We will see the concert tonight",
	"The carpenter used a saw to cut the wood",
	"Studies show that sleep improves memory",
	"She studied biology at college",
	"Running every day builds stamina",
	"He runs a small bakery",
}

var morphologyQueries = []struct {
	text     string
	relevant []int
}{
	{text: "child", relevant: []int{0, 1, 2}},
	{text: "walk", relevant: []int{3, 4, 5}},
	{text: "mouse", relevant: []int{7, 8}},
	{text: "new york", relevant: []int{10, 11}},
	{text: "universe", relevant: []int{16, 17}},
	{text: "university", relevant: []int{14, 15}},
	{text: "organization", relevant: []int{18, 19}},
	{text: "news", relevant: []int{22, 23}},
	{text: "see", relevant: []int{25, 26}},
	{text: "study", relevant: []int{28, 29}},
	{text: "run", relevant: []int{30, 31}},
}

func morphologyConfigs() map[string][]search.AnalyzerOption {
	stop := search.WithStopWords(search.DefaultStopWords()...)
	english := func(opts ...search.AnalyzerOption) []search.AnalyzerOption {
		return append([]search.AnalyzerOption{search.WithEnglish(), search.WithPhrases("New York"), stop}, opts...)
	}
	return map[string][]search.AnalyzerOption{
		"none":  {stop},
		"en":    english(),
		"stem":  english(search.WithStemming()),
		"lemma": english(search.WithLemmatization()),
	}
}

type morphologyScore struct {
	recall, precision float64
	top               []int
}

func scoreMorphology(t *testing.T, opts []search.AnalyzerOption) []morphologyScore {
	t.Helper()
	ds := make([]search.Doc, len(morphologyDocs))
	for i, c := range morphologyDocs {
		ds[i] = search.Doc{Title: fmt.Sprint(i), Content: c}
	}
	ix := search.New(ds, search.WithAnalyzer(search.NewAnalyzer(opts...)))
	scores := make([]morphologyScore, len(morphologyQueries))
	for qi, q := range morphologyQueries {
		var s morphologyScore
		hit := 0
		for _, h := range ix.Rank(q.text, 3).Hits {
			var id int
			if _, err := fmt.Sscan(h.Doc.Title, &id); err != nil {
				t.Fatal(err)
			}
			s.top = append(s.top, id)
			if slices.Contains(q.relevant, id) {
				hit++
			}
		}
		s.recall = float64(hit) / float64(len(q.relevant))
		if len(s.top) > 0 {
			s.precision = float64(hit) / float64(len(s.top))
		}
		scores[qi] = s
	}
	return scores
}

func mean(scores []morphologyScore, f func(morphologyScore) float64) float64 {
	sum := 0.0
	for _, s := range scores {
		sum += f(s)
	}
	return sum / float64(len(scores))
}

func TestMorphologyRetrieval(t *testing.T) {
	t.Parallel()
	configs := morphologyConfigs()
	results := make(map[string][]morphologyScore, len(configs))
	for name, opts := range configs {
		results[name] = scoreMorphology(t, opts)
	}
	names := []string{"none", "en", "stem", "lemma"}
	var b strings.Builder
	fmt.Fprintf(&b, "\n| query | %s |\n", strings.Join(names, " | "))
	for qi, q := range morphologyQueries {
		cells := make([]string, len(names))
		for i, n := range names {
			s := results[n][qi]
			cells[i] = fmt.Sprintf("R %.2f / P %.2f %v", s.recall, s.precision, s.top)
		}
		fmt.Fprintf(&b, "| %s | %s |\n", q.text, strings.Join(cells, " | "))
	}
	for _, n := range names {
		fmt.Fprintf(&b, "%s: recall@3 %.3f precision@3 %.3f\n", n, mean(results[n], func(s morphologyScore) float64 { return s.recall }), mean(results[n], func(s morphologyScore) float64 { return s.precision }))
	}
	t.Log(b.String())

	recall := func(n string) float64 { return mean(results[n], func(s morphologyScore) float64 { return s.recall }) }
	precisionOf := func(n, query string) float64 {
		for qi, q := range morphologyQueries {
			if q.text == query {
				return results[n][qi].precision
			}
		}
		t.Fatalf("no query %q", query)
		return 0
	}
	testCases := map[string]struct {
		ok bool
	}{
		"語幹化で recall が上がる":                  {ok: recall("stem") > recall("en")},
		"見出し語化で recall が上がる":                {ok: recall("lemma") > recall("en")},
		"複数語の辞書で new york の precision が上がる": {ok: precisionOf("en", "new york") > precisionOf("none", "new york")},
		"語幹化は universe に university を混ぜる":   {ok: precisionOf("stem", "universe") < precisionOf("lemma", "universe")},
		"見出し語化は see に名詞の saw を混ぜる":          {ok: precisionOf("lemma", "see") < 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if !tc.ok {
				t.Fatal("not satisfied")
			}
		})
	}
}

func BenchmarkAnalyzeEnglish(b *testing.B) {
	for _, name := range []string{"none", "en", "stem", "lemma"} {
		a := search.NewAnalyzer(morphologyConfigs()[name]...)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for _, d := range morphologyDocs {
					a.Analyze(d)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*len(morphologyDocs)), "ns/doc")
		})
	}
}
