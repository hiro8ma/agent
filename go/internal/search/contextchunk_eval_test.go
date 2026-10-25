package search_test

import (
	"cmp"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/embedding/tei"
	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

var chunkKs = [...]int{5, 10, 20}

// retrieveFunc は k 件を求められたときに返す項目の番号を順位の順に返す。節を丸ごと返す方式は k 件より多く返す。
type retrieveFunc func(q ruleQuery, k int) []int

type chunkScore struct {
	returned, recall, depRecall, precision [len(chunkKs)]float64
}

// scoreChunks は recall と precision をクエリで平均する。依存 recall は主語を前に頼る項目だけで数え、その項目を正解に持つクエリで平均する。
func scoreChunks(c ruleCorpus, qs []ruleQuery, retrieve retrieveFunc) chunkScore {
	var s chunkScore
	for ki, k := range chunkKs {
		var depQueries int
		for _, q := range qs {
			got := retrieve(q, k)
			var hit, depHit, deps int
			for _, id := range got {
				if slices.Contains(q.relevant, id) {
					hit++
					if c.items[id].dependent {
						depHit++
					}
				}
			}
			for _, id := range q.relevant {
				if c.items[id].dependent {
					deps++
				}
			}
			s.returned[ki] += float64(len(got))
			s.recall[ki] += float64(hit) / float64(len(q.relevant))
			if len(got) > 0 {
				s.precision[ki] += float64(hit) / float64(len(got))
			}
			if deps > 0 {
				s.depRecall[ki] += float64(depHit) / float64(deps)
				depQueries++
			}
		}
		s.returned[ki] /= float64(len(qs))
		s.recall[ki] /= float64(len(qs))
		s.precision[ki] /= float64(len(qs))
		s.depRecall[ki] /= float64(max(depQueries, 1))
	}
	return s
}

func formatCount(v float64) string {
	if v == math.Trunc(v) {
		return strconv.Itoa(int(v))
	}
	return fmt.Sprintf("%.1f", v)
}

func (s chunkScore) row(name string) string {
	cells := []string{name, fmt.Sprintf("%s / %s / %s", formatCount(s.returned[0]), formatCount(s.returned[1]), formatCount(s.returned[2]))}
	for _, m := range [][len(chunkKs)]float64{s.recall, s.depRecall, s.precision} {
		for _, v := range m {
			cells = append(cells, fmt.Sprintf("%.3f", v))
		}
	}
	return "| " + strings.Join(cells, " | ") + " |"
}

const chunkTableHeader = "| 方式 | 返した項目数 @5 / @10 / @20 | recall@5 | @10 | @20 | 依存 recall@5 | @10 | @20 | precision@5 | @10 | @20 |\n" +
	"| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

func docIndex(t testing.TB, d search.Doc) int {
	t.Helper()
	i, err := strconv.Atoi(d.ID)
	if err != nil {
		t.Fatal(err)
	}
	return i
}

func numberedDocs(texts []string) []search.Doc {
	docs := make([]search.Doc, len(texts))
	for i, s := range texts {
		docs[i] = search.Doc{ID: strconv.Itoa(i), Content: s}
	}
	return docs
}

func bm25Chunks(t testing.TB, texts []string) retrieveFunc {
	ix := search.New(numberedDocs(texts))
	return func(q ruleQuery, k int) []int {
		hits := ix.Rank(q.text, k).Hits
		out := make([]int, len(hits))
		for i, h := range hits {
			out[i] = docIndex(t, h.Doc)
		}
		return out
	}
}

// bm25Sections は節を順位の順に丸ごと足し、返す項目が k 件以上になったところで止める。
func bm25Sections(t testing.TB, c ruleCorpus) retrieveFunc {
	ix := search.New(numberedDocs(c.sectionTexts()))
	return func(q ruleQuery, k int) []int {
		var out []int
		for _, h := range ix.Rank(q.text, -1).Hits {
			s := c.sections[docIndex(t, h.Doc)]
			out = append(out, span(s.start, s.end)...)
			if len(out) >= k {
				break
			}
		}
		return out
	}
}

func allItems(c ruleCorpus) retrieveFunc {
	all := span(0, len(c.items))
	return func(ruleQuery, int) []int { return all }
}

// idealItems は主語を前に頼る正解、残りの正解、不正解の順に k 件を返す。recall、依存 recall、precision の上限になる。
func idealItems(c ruleCorpus) retrieveFunc {
	return func(q ruleQuery, k int) []int {
		var deps, leads, rest []int
		for _, id := range q.relevant {
			if c.items[id].dependent {
				deps = append(deps, id)
			} else {
				leads = append(leads, id)
			}
		}
		for i := range c.items {
			if !slices.Contains(q.relevant, i) {
				rest = append(rest, i)
			}
		}
		out := slices.Concat(deps, leads, rest)
		return out[:min(k, len(out))]
	}
}

type querySet struct {
	name    string
	queries []ruleQuery
}

func ruleQuerySets(c ruleCorpus) []querySet {
	return []querySet{
		{name: "話題のクエリ（正解は節の全項目）", queries: c.topicQueries()},
		{name: "主題のクエリ（正解は主項目と後続の項目）", queries: c.groupQueries()},
	}
}

func logRuleCorpusStats(t *testing.T, c ruleCorpus) {
	t.Helper()
	var deps int
	sizes := make([]int, len(c.sections))
	for i, s := range c.sections {
		sizes[i] = s.end - s.start
		for _, it := range c.items[s.start:s.end] {
			if it.dependent {
				deps++
			}
		}
	}
	t.Logf("節 %d、項目 %d（節あたり %d〜%d）、主語を前に頼る項目 %d（%.1f%%）", len(c.sections), len(c.items), slices.Min(sizes), slices.Max(sizes), deps, 100*float64(deps)/float64(len(c.items)))
	for _, qs := range ruleQuerySets(c) {
		var rel, dep int
		for _, q := range qs.queries {
			rel += len(q.relevant)
			for _, id := range q.relevant {
				if c.items[id].dependent {
					dep++
				}
			}
		}
		n := float64(len(qs.queries))
		t.Logf("%s: %d 件、正解は平均 %.1f 件（うち主語を前に頼る項目 %.1f 件）", qs.name, len(qs.queries), float64(rel)/n, float64(dep)/n)
	}
}

// TestContextChunkReport は主語を前の項目に頼る項目の取りこぼしを、BM25 で4つのチャンクの作り方について測り、表の行を記録する。
//
//	go test -run TestContextChunkReport -v ./internal/search
func TestContextChunkReport(t *testing.T) {
	t.Parallel()
	c := newRuleCorpus(ruleTopics, ruleSeed)
	logRuleCorpusStats(t, c)
	methods := []struct {
		name     string
		retrieve retrieveFunc
	}{
		{name: "BM25 " + chunkItem.String(), retrieve: bm25Chunks(t, c.chunks(chunkItem))},
		{name: "BM25 " + chunkHeading.String(), retrieve: bm25Chunks(t, c.chunks(chunkHeading))},
		{name: "BM25 " + chunkPrevious.String(), retrieve: bm25Chunks(t, c.chunks(chunkPrevious))},
		{name: "BM25 d 節を丸ごと返す", retrieve: bm25Sections(t, c)},
		{name: "全文（全項目を返す）", retrieve: allItems(c)},
		{name: "上限（正解を先に並べる）", retrieve: idealItems(c)},
	}
	for _, qs := range ruleQuerySets(c) {
		rows := []string{qs.name, chunkTableHeader}
		scores := make(map[string]chunkScore, len(methods))
		for _, m := range methods {
			s := scoreChunks(c, qs.queries, m.retrieve)
			scores[m.name] = s
			rows = append(rows, s.row(m.name))
		}
		t.Log("\n" + strings.Join(rows, "\n"))
		a, b := scores["BM25 "+chunkItem.String()], scores["BM25 "+chunkHeading.String()]
		if last := len(chunkKs) - 1; a.depRecall[last] >= b.depRecall[last] {
			t.Errorf("%s: dependent recall@%d of a = %.3f, want below b %.3f", qs.name, chunkKs[last], a.depRecall[last], b.depRecall[last])
		}
	}
}

// TestLiveContextChunkRuri は同じデータを、手元の TEI で動かす Ruri v3 の埋め込みで a〜c について測る。起動の手順は internal/embedding/tei/README.md にある。
//
//	TEI_LIVE=1 go test -run TestLiveContextChunkRuri -v ./internal/search
func TestLiveContextChunkRuri(t *testing.T) {
	if os.Getenv("TEI_LIVE") != "1" {
		t.Skip("TEI_LIVE=1 で起動済みの TEI に繋ぐ")
	}
	const ruriModel = "cl-nagoya/ruri-v3-310m"
	client := tei.New(cmp.Or(os.Getenv("TEI_URL"), "http://localhost:8080"), tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
	ctx := t.Context()
	if err := enginecmp.WaitReady(ctx, 5*time.Minute, client.Health); err != nil {
		t.Fatal(err)
	}
	info, err := client.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if info.ModelID != ruriModel {
		t.Fatalf("TEI serves %s, want %s", info.ModelID, ruriModel)
	}
	c := newRuleCorpus(ruleTopics, ruleSeed)
	logRuleCorpusStats(t, c)
	sets := ruleQuerySets(c)
	var texts []string
	for _, qs := range sets {
		for _, q := range qs.queries {
			texts = append(texts, q.text)
		}
	}
	start := time.Now()
	qvecs, err := client.EmbedQueries(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	queryVec := make(map[string][]float32, len(texts))
	for i, s := range texts {
		queryVec[s] = qvecs[i]
	}
	modes := []chunkMode{chunkItem, chunkHeading, chunkPrevious}
	rows := make(map[string][]string, len(sets))
	for _, mode := range modes {
		chunks := c.chunks(mode)
		dvecs, err := client.EmbedDocuments(ctx, chunks)
		if err != nil {
			t.Fatal(err)
		}
		flat, err := search.NewFlat(numberedDocs(chunks), dvecs, nil)
		if err != nil {
			t.Fatal(err)
		}
		retrieve := func(q ruleQuery, k int) []int {
			hits, err := flat.SearchVector(queryVec[q.text], k)
			if err != nil {
				t.Fatal(err)
			}
			out := make([]int, len(hits))
			for i, h := range hits {
				out[i] = h.ID
			}
			return out
		}
		for _, qs := range sets {
			rows[qs.name] = append(rows[qs.name], scoreChunks(c, qs.queries, retrieve).row("Ruri "+mode.String()))
		}
	}
	t.Logf("クエリ %d 件と項目 %d 件 x 3通りの埋め込みに %s", len(texts), len(c.items), time.Since(start).Round(time.Second))
	for _, qs := range sets {
		t.Log("\n" + strings.Join(append([]string{qs.name, chunkTableHeader}, rows[qs.name]...), "\n"))
	}
}
