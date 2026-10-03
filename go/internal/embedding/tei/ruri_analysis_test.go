package tei_test

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// 保存した Ruri の埋め込みだけを使い、TEI にも外部の API にも繋がない。

func mustFixture(t *testing.T) (ruriFixture, []string) {
	t.Helper()
	fx, articles, err := loadFixture()
	if err != nil {
		t.Fatal(err)
	}
	return fx, articles
}

func TestRuriTutorialCosineFromFixture(t *testing.T) {
	t.Parallel()
	fx, _ := mustFixture(t)
	testCases := map[string]struct {
		a, b string
		want float64
	}{
		"意味の近い文": {a: "カレーライスはおいしい", b: "カレーのお店", want: 0.9154},
		"意味の遠い文": {a: "カレーライスはおいしい", b: "転置インデックスは便利", want: 0.7853},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := cosine(fx.sentence(tc.a), fx.sentence(tc.b))
			if math.Abs(got-tc.want) > 0.002 {
				t.Errorf("cos(%s, %s) = %.4f, want %.4f", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// tfidfCosine は内部の検索エンジンの既定（TF は 1 + log10(f)、IDF は log2(1 + N/DF)）で疎なベクトルを作り、コサインを返す。DF は corpus で数える。
func tfidfCosine(a, b []string, corpus [][]string) float64 {
	df := map[string]int{}
	for _, terms := range corpus {
		seen := map[string]bool{}
		for _, w := range terms {
			if !seen[w] {
				seen[w] = true
				df[w]++
			}
		}
	}
	vec := func(terms []string) map[string]float64 {
		tf := map[string]float64{}
		for _, w := range terms {
			tf[w]++
		}
		out := map[string]float64{}
		for w, f := range tf {
			out[w] = (1 + math.Log10(f)) * math.Log2(1+float64(len(corpus))/float64(df[w]))
		}
		return out
	}
	va, vb := vec(a), vec(b)
	var ab, aa, bb float64
	for w, x := range va {
		ab += x * vb[w]
		aa += x * x
	}
	for _, y := range vb {
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}

func TestSparseVersusDense(t *testing.T) {
	t.Parallel()
	fx, _ := mustFixture(t)
	bigram := search.NewAnalyzer()
	var bigramCorpus, wordCorpus [][]string
	for _, p := range sparsePairs {
		bigramCorpus = append(bigramCorpus, bigram.Analyze(p.A), bigram.Analyze(p.B))
		wordCorpus = append(wordCorpus, strings.Fields(p.WordsA), strings.Fields(p.WordsB))
	}
	var b strings.Builder
	b.WriteString("\n| A | B | bigram TF-IDF | 語 TF-IDF | Ruri |\n|---|---|---|---|---|\n")
	got := map[string][3]float64{}
	for i, p := range sparsePairs {
		bi := tfidfCosine(bigramCorpus[2*i], bigramCorpus[2*i+1], bigramCorpus)
		wd := tfidfCosine(wordCorpus[2*i], wordCorpus[2*i+1], wordCorpus)
		dn := cosine(fx.sentence(p.A), fx.sentence(p.B))
		got[p.A+"/"+p.B] = [3]float64{bi, wd, dn}
		fmt.Fprintf(&b, "| %s | %s | %.4f | %.4f | %.4f |\n", p.A, p.B, bi, wd, dn)
	}
	t.Log(b.String())

	curry := got["カレーライス/カレー"]
	if curry[1] != 0 {
		t.Errorf("語の単位のカレーライスとカレーのコサイン = %.4f, want 0（索引語が別の次元になる）", curry[1])
	}
	if curry[0] <= 0 {
		t.Errorf("bigram のカレーライスとカレーのコサイン = %.4f, want > 0（カレ / レー を共有する）", curry[0])
	}
	if cat := got["ねこ/猫"]; cat[0] != 0 || cat[1] != 0 {
		t.Errorf("ねこ と 猫 の疎なコサイン = %.4f / %.4f, want 0", cat[0], cat[1])
	}
	near, far := got["カレーライスはおいしい/カレーのお店"][2], got["カレーライスはおいしい/転置インデックスは便利"][2]
	if near <= far {
		t.Errorf("Ruri: 近い対 %.4f <= 遠い対 %.4f", near, far)
	}
}

type cosineStats struct {
	N                                 int
	Mean, Std, Min, P5, P50, P95, Max float64
}

func (s cosineStats) String() string {
	return fmt.Sprintf("%d 対 | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f", s.N, s.Mean, s.Std, s.Min, s.P5, s.P50, s.P95, s.Max)
}

func pairwiseStats(vs [][]float64) cosineStats {
	var cs []float64
	for i := range vs {
		for j := i + 1; j < len(vs); j++ {
			cs = append(cs, dot64(vs[i], vs[j]))
		}
	}
	slices.Sort(cs)
	var sum, sq float64
	for _, c := range cs {
		sum += c
	}
	mean := sum / float64(len(cs))
	for _, c := range cs {
		sq += (c - mean) * (c - mean)
	}
	q := func(p float64) float64 { return cs[int(p*float64(len(cs)-1))] }
	return cosineStats{N: len(cs), Mean: mean, Std: math.Sqrt(sq / float64(len(cs))), Min: cs[0], P5: q(0.05), P50: q(0.5), P95: q(0.95), Max: cs[len(cs)-1]}
}

func dot64(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func unit(v []float64) []float64 {
	n := math.Sqrt(dot64(v, v))
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

func toUnit64(vs [][]float32) [][]float64 {
	out := make([][]float64, len(vs))
	for i, v := range vs {
		f := make([]float64, len(v))
		for j, x := range v {
			f[j] = float64(x)
		}
		out[i] = unit(f)
	}
	return out
}

// postProcessor は文書の集合から求めた平均と主成分を、文書にもクエリにも同じようにかける。
type postProcessor struct {
	mean []float64
	pcs  [][]float64
}

// fitPostProcessor は平均を引き、d > 0 なら上位 d 個の主成分も取り除く（all-but-the-top）。主成分は N×N のグラム行列の冪乗法で求める。
func fitPostProcessor(vs [][]float64, d int) postProcessor {
	dim := len(vs[0])
	mean := make([]float64, dim)
	for _, v := range vs {
		for i, x := range v {
			mean[i] += x / float64(len(vs))
		}
	}
	p := postProcessor{mean: mean}
	x := make([][]float64, len(vs))
	for i, v := range vs {
		x[i] = sub(v, mean)
	}
	for range d {
		gram := make([][]float64, len(x))
		for i := range x {
			gram[i] = make([]float64, len(x))
			for j := range x {
				gram[i][j] = dot64(x[i], x[j])
			}
		}
		u := make([]float64, len(x))
		for i := range u {
			u[i] = 1 + float64(i%7)
		}
		for range 300 {
			next := make([]float64, len(x))
			for i := range gram {
				next[i] = dot64(gram[i], u)
			}
			u = unit(next)
		}
		pc := make([]float64, dim)
		for i, row := range x {
			for k, v := range row {
				pc[k] += u[i] * v
			}
		}
		pc = unit(pc)
		p.pcs = append(p.pcs, pc)
		for i := range x {
			x[i] = removeComponent(x[i], pc)
		}
	}
	return p
}

func sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func removeComponent(v, pc []float64) []float64 {
	c := dot64(v, pc)
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] - c*pc[i]
	}
	return out
}

func (p postProcessor) apply(v []float64) []float64 {
	out := sub(v, p.mean)
	for _, pc := range p.pcs {
		out = removeComponent(out, pc)
	}
	return unit(out)
}

func (p postProcessor) applyAll(vs [][]float64) [][]float64 {
	out := make([][]float64, len(vs))
	for i, v := range vs {
		out[i] = p.apply(v)
	}
	return out
}

// abttComponents は Mu と Viswanath の all-but-the-top が目安にする次元数 / 100。
const abttComponents = 7

func TestAnisotropy(t *testing.T) {
	t.Parallel()
	fx, _ := mustFixture(t)
	var short [][]float32
	for _, s := range unrelatedSentences {
		short = append(short, fx.sentence(s))
	}
	// 16 件では主成分を 7 個除くと残りの自由度の半分を消すので、主成分の除去は件数の多い集合だけにかける。
	sets := []struct {
		name string
		vecs [][]float64
		abtt bool
	}{
		{name: "無関係な短文 16 件", vecs: toUnit64(short)},
		{name: "条文 103 + 短文 16", vecs: toUnit64(append(slices.Clone(fx.ArticlesPlain), short...)), abtt: true},
	}
	var b strings.Builder
	b.WriteString("\n| 集合 | 処理 | 対の数 | 平均 | 標準偏差 | 最小 | p5 | 中央値 | p95 | 最大 |\n|---|---|---|---|---|---|---|---|---|---|\n")
	for _, s := range sets {
		raw := pairwiseStats(s.vecs)
		centered := pairwiseStats(fitPostProcessor(s.vecs, 0).applyAll(s.vecs))
		fmt.Fprintf(&b, "| %s | そのまま | %s |\n| %s | 平均を引く | %s |\n", s.name, raw, s.name, centered)
		if s.abtt {
			abtt := pairwiseStats(fitPostProcessor(s.vecs, abttComponents).applyAll(s.vecs))
			fmt.Fprintf(&b, "| %s | 平均 + 上位 %d 主成分を除く | %s |\n", s.name, abttComponents, abtt)
		}
		if raw.Min < 0.3 {
			t.Errorf("%s: 生のコサインの最小 = %.3f, want >= 0.3（狭い円錐に集まる）", s.name, raw.Min)
		}
		if math.Abs(centered.Mean) >= raw.Mean/4 {
			t.Errorf("%s: 平均を引いた後の平均 = %.3f, want 生の %.3f より十分 0 に近い", s.name, centered.Mean, raw.Mean)
		}
	}
	t.Log(b.String())
}

// fixedEmbedder は保存したクエリのベクトルを返す。
type fixedEmbedder map[string][]float32

func (e fixedEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v, ok := e[text]
	if !ok {
		return nil, fmt.Errorf("no stored vector for %q", text)
	}
	return v, nil
}

type metrics struct{ hit1, hit3, mrr float64 }

// evaluate は各クエリの上位 10 件の条の番号から hit@1 / hit@3 / MRR@10 の平均を求める。
func evaluate(ranked [][]int) metrics {
	var m metrics
	for i, ids := range ranked {
		rel := searchQueries[i].Relevant
		for r, id := range ids[:min(10, len(ids))] {
			if slices.Contains(rel, id) {
				if r == 0 {
					m.hit1++
				}
				if r < 3 {
					m.hit3++
				}
				m.mrr += 1 / float64(r+1)
				break
			}
		}
	}
	n := float64(len(ranked))
	return metrics{hit1: m.hit1 / n, hit3: m.hit3 / n, mrr: m.mrr / n}
}

func docIDs(t *testing.T, docs []search.Doc) []int {
	t.Helper()
	out := make([]int, len(docs))
	for i, d := range docs {
		id, err := strconv.Atoi(d.ID)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = id
	}
	return out
}

func toFloat32(vs [][]float64) [][]float32 {
	out := make([][]float32, len(vs))
	for i, v := range vs {
		out[i] = make([]float32, len(v))
		for j, x := range v {
			out[i][j] = float32(x)
		}
	}
	return out
}

func TestConstitutionSearch(t *testing.T) {
	t.Parallel()
	fx, articles := mustFixture(t)
	ctx := t.Context()
	docs := make([]search.Doc, len(articles))
	for i, a := range articles {
		docs[i] = search.Doc{ID: strconv.Itoa(i + 1), Content: a}
	}
	bm25 := search.New(docs)

	queryVecs := func(prefixed bool) fixedEmbedder {
		e := fixedEmbedder{}
		for _, q := range fx.Queries {
			if prefixed {
				e[q.Text] = q.Prefixed
			} else {
				e[q.Text] = q.Plain
			}
		}
		return e
	}
	flat := func(docVecs [][]float32, e search.Embedder) *search.Flat {
		f, err := search.NewFlat(docs, docVecs, e)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	ruri := flat(fx.ArticlesPrefixed, queryVecs(true))

	// 平均と主成分は文書の側だけから求め、クエリにも同じ変換をかける。
	post := func(d int) *search.Flat {
		p := fitPostProcessor(toUnit64(fx.ArticlesPrefixed), d)
		e := fixedEmbedder{}
		for _, q := range fx.Queries {
			e[q.Text] = toFloat32([][]float64{p.apply(toUnit64([][]float32{q.Prefixed})[0])})[0]
		}
		return flat(toFloat32(p.applyAll(toUnit64(fx.ArticlesPrefixed))), e)
	}

	methods := []struct {
		name string
		r    search.Retriever
	}{
		{name: "BM25（bigram）", r: bm25},
		{name: "Ruri（両方に接頭辞）", r: ruri},
		{name: "RRF（BM25 + Ruri）", r: search.Hybrid{Retrievers: []search.Retriever{bm25, ruri}, Depth: 20}},
		{name: "Ruri（接頭辞なし）", r: flat(fx.ArticlesPlain, queryVecs(false))},
		{name: "Ruri（クエリだけ接頭辞）", r: flat(fx.ArticlesPlain, queryVecs(true))},
		{name: "Ruri（文書だけ接頭辞）", r: flat(fx.ArticlesPrefixed, queryVecs(false))},
		{name: "Ruri（平均を引く）", r: post(0)},
		{name: fmt.Sprintf("Ruri（平均 + 上位 %d 主成分を除く）", abttComponents), r: post(abttComponents)},
	}
	results := map[string]metrics{}
	var top, table strings.Builder
	top.WriteString("\n| クエリ | 正解 |")
	for _, m := range methods[:3] {
		top.WriteString(" " + m.name + " |")
	}
	top.WriteString(" Elasticsearch（kuromoji, 記録） |\n|---|---|---|---|---|---|\n")
	rankings := make([][][]int, len(methods))
	for mi, m := range methods {
		for _, q := range searchQueries {
			got, err := m.r.Search(ctx, q.Text, 10)
			if err != nil {
				t.Fatalf("%s: %v", m.name, err)
			}
			rankings[mi] = append(rankings[mi], docIDs(t, got))
		}
	}
	var es [][]int
	for _, q := range fx.Queries {
		es = append(es, q.Elasticsearch)
	}
	top3 := func(ids []int) string {
		if len(ids) == 0 {
			return "-"
		}
		return fmt.Sprint(ids[:min(3, len(ids))])
	}
	for qi, q := range searchQueries {
		fmt.Fprintf(&top, "| %s | %v | %s | %s | %s | %s |\n", q.Text, q.Relevant, top3(rankings[0][qi]), top3(rankings[1][qi]), top3(rankings[2][qi]), top3(es[qi]))
	}
	table.WriteString("\n| 方式 | hit@1 | hit@3 | MRR@10 |\n|---|---|---|---|\n")
	for mi, m := range methods {
		r := evaluate(rankings[mi])
		results[m.name] = r
		fmt.Fprintf(&table, "| %s | %.2f | %.2f | %.3f |\n", m.name, r.hit1, r.hit3, r.mrr)
	}
	if es[0] != nil {
		r := evaluate(es)
		results["Elasticsearch"] = r
		fmt.Fprintf(&table, "| Elasticsearch（kuromoji, 記録） | %.2f | %.2f | %.3f |\n", r.hit1, r.hit3, r.mrr)
	}
	var changed strings.Builder
	for mi := 6; mi < len(methods); mi++ {
		for qi, q := range searchQueries {
			if a, b := top3(rankings[1][qi]), top3(rankings[mi][qi]); a != b {
				fmt.Fprintf(&changed, "\n%s: %s の上位 3 件 %s → %s", methods[mi].name, q.Text, a, b)
			}
		}
	}
	t.Log(top.String())
	t.Log(table.String())
	t.Log("後処理で上位 3 件が変わったクエリ" + changed.String())

	if bm, rr := results["BM25（bigram）"], results["Ruri（両方に接頭辞）"]; rr.mrr <= bm.mrr {
		t.Errorf("Ruri の MRR %.3f <= BM25 の %.3f（言い換えのクエリでベクトルが勝つはず）", rr.mrr, bm.mrr)
	}
}

// TestPrefixEffect は接頭辞の組み合わせごとに、正解の条の全 103 条中の順位、正解と全条の平均とのコサインの差、
// その差を全条の標準偏差で割った値、正解と正解以外で最も近い条との差を比べる。コサインの絶対値はモデルの中でしか比べられないので、標準偏差で割った値も見る。
func TestPrefixEffect(t *testing.T) {
	t.Parallel()
	fx, _ := mustFixture(t)
	variants := []struct {
		name     string
		docs     [][]float32
		prefixed bool
	}{
		{name: "両方に接頭辞", docs: fx.ArticlesPrefixed, prefixed: true},
		{name: "接頭辞なし", docs: fx.ArticlesPlain},
		{name: "クエリだけ", docs: fx.ArticlesPlain, prefixed: true},
		{name: "文書だけ", docs: fx.ArticlesPrefixed},
	}
	var b strings.Builder
	b.WriteString("\n| 接頭辞 | 正解の順位の平均 | 正解のコサイン | 全条の平均 | 差 | 差 / 標準偏差 | 正解 - 次点 |\n|---|---|---|---|---|---|---|\n")
	for _, v := range variants {
		var rankSum, relSum, meanSum, zSum, gapSum float64
		for qi, q := range searchQueries {
			qv := fx.Queries[qi].Plain
			if v.prefixed {
				qv = fx.Queries[qi].Prefixed
			}
			scores := make([]float64, len(v.docs))
			var mean, sq float64
			for i, d := range v.docs {
				scores[i] = cosine(qv, d)
				mean += scores[i] / float64(len(v.docs))
			}
			for _, sc := range scores {
				sq += (sc - mean) * (sc - mean)
			}
			std := math.Sqrt(sq / float64(len(scores)))
			best, bestRank, other := 0.0, len(v.docs)+1, -1.0
			for i, sc := range scores {
				if slices.Contains(q.Relevant, i+1) {
					rank := 1
					for _, x := range scores {
						if x > sc {
							rank++
						}
					}
					if rank < bestRank {
						best, bestRank = sc, rank
					}
				} else {
					other = max(other, sc)
				}
			}
			rankSum += float64(bestRank)
			relSum += best
			meanSum += mean
			zSum += (best - mean) / std
			gapSum += best - other
		}
		n := float64(len(searchQueries))
		fmt.Fprintf(&b, "| %s | %.2f | %.3f | %.3f | %.3f | %.2f | %+.3f |\n", v.name, rankSum/n, relSum/n, meanSum/n, (relSum-meanSum)/n, zSum/n, gapSum/n)
		if rankSum/n > 1.5 {
			t.Errorf("%s: 正解の順位の平均 = %.2f, want <= 1.5", v.name, rankSum/n)
		}
	}
	t.Log(b.String())
}
