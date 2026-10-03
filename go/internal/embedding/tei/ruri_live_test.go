package tei_test

import (
	"cmp"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/embedding/tei"
	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

// liveClient は TEI_LIVE=1 のときだけ、起動済みの TEI（既定 http://localhost:8080）に繋ぐ。起動の手順は README.md にある。
func liveClient(t *testing.T) *tei.Client {
	t.Helper()
	if os.Getenv("TEI_LIVE") != "1" {
		t.Skip("TEI_LIVE=1 で起動済みの TEI に繋ぐ")
	}
	c := tei.New(cmp.Or(os.Getenv("TEI_URL"), "http://localhost:8080"), tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
	if err := enginecmp.WaitReady(t.Context(), 5*time.Minute, c.Health); err != nil {
		t.Fatal(err)
	}
	info, err := c.Info(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if info.ModelID != ruriModel {
		t.Fatalf("TEI serves %s, want %s", info.ModelID, ruriModel)
	}
	return c
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// 教材と Ruri のモデルカードは sentence-transformers（PyTorch）で求めた値。TEI は candle で計算するので、浮動小数の誤差の分だけずれる。
func TestLiveTutorialCosine(t *testing.T) {
	c := liveClient(t)
	v, err := c.EmbedTexts(t.Context(), tutorialSentences)
	if err != nil {
		t.Fatal(err)
	}
	const tol = 0.002
	testCases := map[string]struct {
		a, b int
		want float64
	}{
		"カレーライスはおいしい と カレーのお店":      {a: 0, b: 1, want: 0.9154},
		"カレーライスはおいしい と 転置インデックスは便利": {a: 0, b: 2, want: 0.7853},
	}
	for tn, tc := range testCases {
		got := cosine(v[tc.a], v[tc.b])
		t.Logf("%s: %.6f（教材 %.4f）", tn, got, tc.want)
		if math.Abs(got-tc.want) > tol {
			t.Errorf("%s = %.4f, want %.4f ± %.3f", tn, got, tc.want, tol)
		}
	}

	card := []string{
		"川べりでサーフボードを持った人たちがいます",
		"サーファーたちが川べりに立っています",
		tei.RuriTopicPrefix + "瑠璃色のサーファー",
		tei.RuriQueryPrefix + "瑠璃色はどんな色？",
		tei.RuriDocumentPrefix + "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。",
	}
	wantCard := [][]float64{
		{1.0000, 0.9603, 0.8157, 0.7074, 0.6916},
		{0.9603, 1.0000, 0.8192, 0.7014, 0.6819},
		{0.8157, 0.8192, 1.0000, 0.8701, 0.8470},
		{0.7074, 0.7014, 0.8701, 1.0000, 0.9746},
		{0.6916, 0.6819, 0.8470, 0.9746, 1.0000},
	}
	cv, err := c.EmbedTexts(t.Context(), card)
	if err != nil {
		t.Fatal(err)
	}
	for i := range cv {
		for j := range cv {
			if got := cosine(cv[i], cv[j]); math.Abs(got-wantCard[i][j]) > tol {
				t.Errorf("モデルカードの (%d, %d) = %.4f, want %.4f", i, j, got, wantCard[i][j])
			}
		}
	}
}

func durationStats(ds []time.Duration) (median, p95, mean time.Duration) {
	s := slices.Clone(ds)
	slices.Sort(s)
	var sum time.Duration
	for _, d := range s {
		sum += d
	}
	return s[len(s)/2], s[(len(s)*95+99)/100-1], sum / time.Duration(len(s))
}

// TestLiveLatency は CPU で文書とクエリを 1 件ずつ埋め込む時間と、送る件数を変えて条文の索引を作る時間を記録する。
func TestLiveLatency(t *testing.T) {
	c := liveClient(t)
	ctx := t.Context()
	articles, err := enginecmp.Kenpou()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.EmbedDocuments(ctx, articles[:4]); err != nil {
		t.Fatal(err)
	}

	perDoc := make([]time.Duration, len(articles))
	for i, a := range articles {
		start := time.Now()
		if _, err := c.EmbedDocuments(ctx, []string{a}); err != nil {
			t.Fatal(err)
		}
		perDoc[i] = time.Since(start)
	}
	med, p95, mean := durationStats(perDoc)
	t.Logf("文書 1 件ずつ: 中央値 %v, p95 %v, 平均 %v", med, p95, mean)

	perQuery := make([]time.Duration, len(searchQueries))
	for i, q := range searchQueries {
		start := time.Now()
		if _, err := c.Embed(ctx, q.Text); err != nil {
			t.Fatal(err)
		}
		perQuery[i] = time.Since(start)
	}
	med, p95, mean = durationStats(perQuery)
	t.Logf("クエリ 1 件ずつ: 中央値 %v, p95 %v, 平均 %v", med, p95, mean)

	docs := make([]search.Doc, len(articles))
	for i, a := range articles {
		docs[i] = search.Doc{ID: strconv.Itoa(i + 1), Content: a}
	}
	base := cmp.Or(os.Getenv("TEI_URL"), "http://localhost:8080")
	for _, n := range []int{1, 4, 32} {
		bc := tei.New(base, tei.WithBatchSize(n), tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
		start := time.Now()
		vecs, err := bc.EmbedDocuments(ctx, articles)
		if err != nil {
			t.Fatal(err)
		}
		embedded := time.Since(start)
		if _, err := search.NewFlat(docs, vecs, bc); err != nil {
			t.Fatal(err)
		}
		t.Logf("索引 %d 条（%d 件ずつ送る）: 埋め込み %v（1 条あたり %v）, 全体 %v", len(articles), n, embedded, embedded/time.Duration(len(articles)), time.Since(start))
	}
	bm25 := time.Now()
	search.New(docs)
	t.Logf("比較: BM25 の索引 %v", time.Since(bm25))
}

// TestLiveWriteRuriFixture は条文、クエリ、例文の Ruri の埋め込みを testdata に保存する。TEI_ES_URL があれば Elasticsearch（kuromoji）の上位 10 件も記録する。
//
//	TEI_LIVE=1 TEI_UPDATE_EMBEDDINGS=1 go test -run LiveWriteRuriFixture ./internal/embedding/tei/
func TestLiveWriteRuriFixture(t *testing.T) {
	if os.Getenv("TEI_UPDATE_EMBEDDINGS") != "1" {
		t.Skip("TEI_UPDATE_EMBEDDINGS=1 で testdata を作り直す")
	}
	c := liveClient(t)
	ctx := t.Context()
	articles, err := enginecmp.Kenpou()
	if err != nil {
		t.Fatal(err)
	}
	fx := ruriFixture{Model: ruriModel, Image: ruriImage}
	prefixed, err := c.EmbedDocuments(ctx, articles)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := c.EmbedTexts(ctx, articles)
	if err != nil {
		t.Fatal(err)
	}
	fx.Dimensions = len(prefixed[0])
	fx.ArticlesPrefixed, fx.ArticlesPlain = roundVectors(prefixed), roundVectors(plain)

	texts := make([]string, len(searchQueries))
	for i, q := range searchQueries {
		texts[i] = q.Text
	}
	qp, err := c.EmbedQueries(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	qn, err := c.EmbedTexts(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	es := elasticsearchTop(t, articles, texts)
	for i, s := range texts {
		q := fixtureQuery{Text: s, Prefixed: roundVector(qp[i]), Plain: roundVector(qn[i])}
		if es != nil {
			q.Elasticsearch = es[i]
		}
		fx.Queries = append(fx.Queries, q)
	}

	sentences := fixtureSentenceTexts()
	sv, err := c.EmbedTexts(ctx, sentences)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range sentences {
		fx.Sentences = append(fx.Sentences, fixtureVector{Text: s, Values: roundVector(sv[i])})
	}

	b, err := json.Marshal(fx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.FromSlash(fixturePath), append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d bytes", fixturePath, len(b)+1)
}

func elasticsearchTop(t *testing.T, articles, queries []string) [][]int {
	t.Helper()
	url := os.Getenv("TEI_ES_URL")
	if url == "" {
		return nil
	}
	ctx := t.Context()
	es := enginecmp.NewElasticsearch(url, "kenpou_ruri", nil)
	if err := enginecmp.WaitReady(ctx, 2*time.Minute, es.Ping); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := es.DeleteIndex(context.WithoutCancel(ctx)); err != nil {
			t.Error(err)
		}
	})
	reports, err := enginecmp.Compare(ctx, []enginecmp.Engine{enginecmp.ElasticsearchEngine{Client: es}}, articles, queries, 10)
	if err != nil {
		t.Fatal(err)
	}
	out := make([][]int, len(queries))
	for i, hits := range reports[0].Hits {
		for _, h := range hits {
			out[i] = append(out[i], h.ID)
		}
	}
	return out
}
