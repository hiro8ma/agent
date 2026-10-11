package tei_test

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/embedding/tei"
	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

const (
	defaultRerankModel = "hotchpotch/japanese-reranker-xsmall-v2"
	rerankFixturePath  = "testdata/rerank_kenpou.json"
)

// rerankDepths は Ruri から交差エンコーダに渡す候補の数。
var rerankDepths = []int{20, 50}

// rerankMethods の順で testdata に保存し、表に出す。Key は保存する JSON の名前。
var rerankMethods = []struct{ Key, Name string }{
	{Key: "ruri", Name: "Ruri（双方向エンコーダ）"},
	{Key: "cross_top20", Name: fmt.Sprintf("Ruri の上位%d件 + 交差エンコーダ", rerankDepths[0])},
	{Key: "cross_top50", Name: fmt.Sprintf("Ruri の上位%d件 + 交差エンコーダ", rerankDepths[1])},
	{Key: "maxsim", Name: "Ruri のトークンで MaxSim（全トークン）"},
	{Key: "maxsim_inner", Name: "Ruri のトークンで MaxSim（<s> と </s> を除く）"},
}

// rerankFixture の Top は rerankMethods と同じ順に、各クエリの上位10件の条の番号を持つ。
type rerankFixture struct {
	EmbeddingModel string             `json:"embedding_model"`
	RerankerModel  string             `json:"reranker_model"`
	Image          string             `json:"image"`
	Depths         []int              `json:"depths"`
	Queries        []string           `json:"queries"`
	Methods        []rerankFixtureRow `json:"methods"`
}

type rerankFixtureRow struct {
	Key  string  `json:"key"`
	Hit3 float64 `json:"hit3"`
	Top  [][]int `json:"top"`
}

// liveReranker は TEI_LIVE=1 のときだけ、交差エンコーダを読み込んだ TEI（既定 http://localhost:8081）に繋ぐ。
func liveReranker(t *testing.T) (*tei.Client, string) {
	t.Helper()
	if os.Getenv("TEI_LIVE") != "1" {
		t.Skip("TEI_LIVE=1 で起動済みの TEI に繋ぐ")
	}
	c := tei.New(cmp.Or(os.Getenv("TEI_RERANK_URL"), "http://localhost:8081"))
	if err := enginecmp.WaitReady(t.Context(), 5*time.Minute, c.Health); err != nil {
		t.Fatal(err)
	}
	info, err := c.Info(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	model := cmp.Or(os.Getenv("TEI_RERANK_MODEL"), defaultRerankModel)
	if info.ModelID != model {
		t.Fatalf("TEI serves %s, want %s", info.ModelID, model)
	}
	return c, model
}

// tokenRetriever は文書のトークンのベクトルを前もって持ち、クエリだけを TEI でトークンのベクトルにして MaxSim で全件と比べる。dropEnds なら両側の先頭と末尾のトークンを除く。
type tokenRetriever struct {
	c        *tei.Client
	docs     []search.Doc
	vecs     [][][]float32
	dropEnds bool
}

func (r tokenRetriever) Search(ctx context.Context, query string, limit int) ([]search.Doc, error) {
	q, err := r.c.EmbedTokens(ctx, query)
	if err != nil {
		return nil, err
	}
	if r.dropEnds {
		q = search.DropEnds(q)
	}
	hits := search.RankMaxSim(q, r.vecs, limit)
	out := make([]search.Doc, len(hits))
	for i, h := range hits {
		out[i] = r.docs[h.ID]
	}
	return out, nil
}

// TestLiveRerankKenpou は日本国憲法の103条で、Ruri だけ、Ruri の上位20件と50件を交差エンコーダで並べ直したもの、Ruri のトークンの MaxSim を比べる。
// 時間はクエリを TEI に送ってから上位10件が決まるまで。文書の埋め込みは前もって求めておくので含めない。
//
//	TEI_LIVE=1 go test -run LiveRerankKenpou -v ./internal/embedding/tei/
//	TEI_LIVE=1 TEI_UPDATE_EMBEDDINGS=1 go test -run LiveRerankKenpou ./internal/embedding/tei/
func TestLiveRerankKenpou(t *testing.T) {
	rr, rerankModel := liveReranker(t)
	c := liveClient(t)
	ctx := t.Context()
	fx, articles := mustFixture(t)
	docs := make([]search.Doc, len(articles))
	for i, a := range articles {
		docs[i] = search.Doc{ID: strconv.Itoa(i + 1), Content: a}
	}
	ruri, err := search.NewFlat(docs, fx.ArticlesPrefixed, c)
	if err != nil {
		t.Fatal(err)
	}
	cross := func(depth int) search.Retriever {
		r, err := search.NewCrossRerank(ruri, rr, search.CrossRerankConfig{Depth: depth})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	start := time.Now()
	tokens, err := c.EmbedDocumentTokens(ctx, articles)
	if err != nil {
		t.Fatal(err)
	}
	var n int
	for _, d := range tokens {
		n += len(d)
	}
	t.Logf("条文のトークンのベクトル: %d 条、%d トークン、%v", len(tokens), n, time.Since(start))
	inner := make([][][]float32, len(tokens))
	for i, d := range tokens {
		inner[i] = search.DropEnds(d)
	}
	retrievers := map[string]search.Retriever{
		"ruri":         ruri,
		"cross_top20":  cross(rerankDepths[0]),
		"cross_top50":  cross(rerankDepths[1]),
		"maxsim":       tokenRetriever{c: c, docs: docs, vecs: tokens},
		"maxsim_inner": tokenRetriever{c: c, docs: docs, vecs: inner, dropEnds: true},
	}
	for _, q := range searchQueries[:2] {
		for _, r := range retrievers {
			if _, err := r.Search(ctx, q.Text, 10); err != nil {
				t.Fatal(err)
			}
		}
	}

	out := rerankFixture{
		EmbeddingModel: ruriModel,
		RerankerModel:  rerankModel,
		Image:          ruriImage,
		Depths:         rerankDepths,
	}
	for _, q := range searchQueries {
		out.Queries = append(out.Queries, q.Text)
	}
	var table, top strings.Builder
	table.WriteString("\n| 方式 | hit@1 | hit@3 | MRR@10 | 1クエリの時間（中央値） | p95 |\n|---|---|---|---|---|---|\n")
	baseline := evaluate(fixtureRanking(t, fx, docs))
	fmt.Fprintf(&table, "| Ruri（保存した埋め込み） | %.2f | %.2f | %.3f | - | - |\n", baseline.hit1, baseline.hit3, baseline.mrr)
	for _, m := range rerankMethods {
		var ranked [][]int
		times := make([]time.Duration, len(searchQueries))
		for qi, q := range searchQueries {
			start := time.Now()
			got, err := retrievers[m.Key].Search(ctx, q.Text, 10)
			if err != nil {
				t.Fatalf("%s: %v", m.Name, err)
			}
			times[qi] = time.Since(start)
			ranked = append(ranked, docIDs(t, got))
		}
		r := evaluate(ranked)
		med, p95, _ := durationStats(times)
		fmt.Fprintf(&table, "| %s | %.2f | %.2f | %.3f | %v | %v |\n", m.Name, r.hit1, r.hit3, r.mrr, med.Round(time.Millisecond), p95.Round(time.Millisecond))
		out.Methods = append(out.Methods, rerankFixtureRow{Key: m.Key, Hit3: r.hit3, Top: ranked})
	}
	top.WriteString("\n| クエリ | 正解 |")
	for _, m := range rerankMethods {
		top.WriteString(" " + m.Name + " |")
	}
	top.WriteString("\n|---|---|" + strings.Repeat("---|", len(rerankMethods)) + "\n")
	for qi, q := range searchQueries {
		fmt.Fprintf(&top, "| %s | %v |", q.Text, q.Relevant)
		for _, m := range out.Methods {
			fmt.Fprintf(&top, " %v |", m.Top[qi][:min(3, len(m.Top[qi]))])
		}
		top.WriteString("\n")
	}
	t.Log(table.String())
	t.Log(top.String())

	if os.Getenv("TEI_UPDATE_EMBEDDINGS") != "1" {
		return
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.FromSlash(rerankFixturePath), append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d bytes", rerankFixturePath, len(b)+1)
}

// fixtureRanking は保存したクエリと条文の埋め込みで、各クエリの上位10件の条の番号を返す。
func fixtureRanking(t *testing.T, fx ruriFixture, docs []search.Doc) [][]int {
	t.Helper()
	e := fixedEmbedder{}
	for _, q := range fx.Queries {
		e[q.Text] = q.Prefixed
	}
	f, err := search.NewFlat(docs, fx.ArticlesPrefixed, e)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]int
	for _, q := range searchQueries {
		got, err := f.Search(t.Context(), q.Text, 10)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, docIDs(t, got))
	}
	return out
}
