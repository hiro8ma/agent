package ann

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// 評価は時間がかかるので ANN_EVAL を付けたときだけ走らせる。件数は ANN_EVAL_N（カンマ区切り、既定 100000）、クエリ数は ANN_EVAL_Q（既定 1000）。
//
//	ANN_EVAL=1 ANN_EVAL_N=100000,1000000 go test -run 'TestReport' -v -timeout 0 ./internal/ann
func evalEnabled(t *testing.T) {
	t.Helper()
	if os.Getenv("ANN_EVAL") == "" {
		t.Skip("ANN_EVAL is not set")
	}
}

func envInts(t *testing.T, key string, def []int) []int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	var out []int
	for f := range strings.SplitSeq(v, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			t.Fatalf("%s: %v", key, err)
		}
		out = append(out, n)
	}
	return out
}

const (
	evalDim      = 128
	evalK        = 20
	evalClusters = 1000
	evalNoise    = 1.0
)

type evalSet struct {
	name    string
	data    Matrix
	queries [][]float32
	truth   [][]Neighbor
	exact   *Exact
}

func rows(m Matrix) [][]float32 {
	out := make([][]float32, m.N)
	for i := range m.N {
		out[i] = m.Row(i)
	}
	return out
}

func makeSets(t *testing.T, n, nq int) []*evalSet {
	t.Helper()
	sets := []*evalSet{
		{name: "正規乱数", data: RandomNormal(n, evalDim, 1), queries: rows(RandomNormal(nq, evalDim, 2))},
		{name: "混合ガウス", data: GaussianMixture(n, evalDim, evalClusters, evalNoise, 1, 2), queries: rows(GaussianMixture(nq, evalDim, evalClusters, evalNoise, 1, 3))},
	}
	for _, s := range sets {
		s.exact = NewExact(s.data)
		start := time.Now()
		s.truth = s.exact.SearchBatchParallel(s.queries, evalK)
		t.Logf("%s N=%d: ground truth for %d queries in %v", s.name, n, nq, time.Since(start).Round(time.Millisecond))
	}
	return sets
}

// measure は 1 goroutine でクエリを順に投げ、平均の recall@K と 1 クエリあたりの時間を返す。
func (s *evalSet) measure(search func(q []float32) []Neighbor) (recall float64, per time.Duration) {
	start := time.Now()
	res := make([][]Neighbor, len(s.queries))
	for i, q := range s.queries {
		res[i] = search(q)
	}
	per = time.Since(start) / time.Duration(len(s.queries))
	for i, r := range res {
		recall += Recall(r, s.truth[i])
	}
	return recall / float64(len(res)), per
}

type table struct {
	t      *testing.T
	header string
}

func newTable(t *testing.T, cols ...string) *table {
	t.Helper()
	tb := &table{t: t, header: "| " + strings.Join(cols, " | ") + " |"}
	t.Log(tb.header)
	t.Log("|" + strings.Repeat(" --- |", len(cols)))
	return tb
}

func (tb *table) row(cells ...any) {
	parts := make([]string, len(cells))
	for i, c := range cells {
		switch v := c.(type) {
		case float64:
			parts[i] = strconv.FormatFloat(v, 'f', 3, 64)
		case time.Duration:
			parts[i] = fmtDur(v)
		default:
			parts[i] = fmt.Sprint(v)
		}
	}
	tb.t.Log("| " + strings.Join(parts, " | ") + " |")
}

func fmtDur(d time.Duration) string {
	switch {
	case d >= time.Second:
		return fmt.Sprintf("%.2f s", d.Seconds())
	case d >= time.Millisecond:
		return fmt.Sprintf("%.2f ms", float64(d)/1e6)
	}
	return fmt.Sprintf("%.1f µs", float64(d)/1e3)
}

func qps(d time.Duration) string { return fmt.Sprintf("%.0f", 1/d.Seconds()) }

// TestReportExact は教材の NumPy の実験（N=100 万、d=128、K=20、正規乱数）を Go で再現する。
func TestReportExact(t *testing.T) {
	evalEnabled(t)
	const n, nq = 1_000_000, 32
	data := RandomNormal(n, evalDim, 1)
	qs := rows(RandomNormal(nq, evalDim, 2))
	start := time.Now()
	e := NewExact(data)
	t.Logf("||x||² for %d rows in %v", n, time.Since(start).Round(time.Millisecond))
	buf := make([]Neighbor, n)
	tb := newTable(t, "方式", "1 クエリあたり")
	median := func(ds []time.Duration) time.Duration {
		slices.Sort(ds)
		return ds[len(ds)/2]
	}
	timeIt := func(name string, reps int, f func()) {
		f()
		ds := make([]time.Duration, reps)
		for i := range ds {
			start := time.Now()
			f()
			ds[i] = time.Since(start)
		}
		tb.row(name, median(ds))
	}
	timeIt("ヒープ（並べた k 件）", nq, func() { e.Search(qs[0], evalK) })
	timeIt("quickselect（argpartition 相当、順序なし）", nq, func() { e.Partition(qs[0], evalK, buf) })
	timeIt("ヒープ + ノルムなし（内積だけ）", nq, func() { e.scanNoNorm(qs[0], evalK) })
	timeBatch := func(name string, f func([][]float32, int) [][]Neighbor) {
		f(qs, evalK)
		ds := make([]time.Duration, 7)
		for i := range ds {
			start := time.Now()
			f(qs, evalK)
			ds[i] = time.Since(start) / nq
		}
		tb.row(name, median(ds))
	}
	timeBatch(fmt.Sprintf("バッチ %d 本（1 goroutine）", nq), e.SearchBatch)
	timeBatch(fmt.Sprintf("バッチ %d 本（全コア）", nq), e.SearchBatchParallel)
	timeBatch("1 本ずつ（全コア）", func(qs [][]float32, k int) [][]Neighbor {
		for i := range qs {
			e.SearchBatchParallel(qs[i:i+1], k)
		}
		return nil
	})
}

// scanNoNorm は ||x||² を読まずに内積だけで順位を付ける。ノルムの配列を読む分の時間を測る比較用。
func (e *Exact) scanNoNorm(q []float32, k int) []Neighbor {
	t := newKBest(k)
	for i := range e.m.N {
		t.push(Neighbor{ID: int32(i), Dist: -2 * dot(q, e.m.Row(i))})
	}
	return t.sorted()
}

// TestReportMethods は方式ごとに recall@20 と速さ、1 件あたりのメモリ、構築時間を表にする。
func TestReportMethods(t *testing.T) {
	evalEnabled(t)
	nq := envInts(t, "ANN_EVAL_Q", []int{1000})[0]
	for _, n := range envInts(t, "ANN_EVAL_N", []int{100_000}) {
		for _, s := range makeSets(t, n, nq) {
			t.Logf("### %s N=%d d=%d K=%d クエリ %d", s.name, n, evalDim, evalK, nq)
			reportSet(t, s, n)
		}
	}
}

func reportSet(t *testing.T, s *evalSet, n int) {
	tb := newTable(t, "方式", "パラメータ", "recall@20", "1 クエリ", "QPS", "距離計算/クエリ", "B/件", "構築")
	r, per := s.measure(func(q []float32) []Neighbor { return s.exact.Search(q, evalK) })
	tb.row("全探索", "-", r, per, qps(per), n, s.exact.BytesPerVector(), "-")

	nlist := int(math.Round(math.Sqrt(float64(n))))
	start := time.Now()
	ivf := NewIVFFlat(s.data, IVFConfig{NList: nlist, Seed: 1})
	build := time.Since(start)
	for _, p := range []int{1, 4, 16, 64} {
		ix := ivf.WithNProbe(p)
		evals := 0
		r, per := s.measure(func(q []float32) []Neighbor {
			ns, st := ix.SearchStats(q, evalK)
			evals += st.Vectors + st.Centroids
			return ns
		})
		tb.row("IVF-Flat", fmt.Sprintf("nlist=%d nprobe=%d", nlist, p), r, per, qps(per), evals/len(s.queries), ix.BytesPerVector(), build)
	}

	if s.name == "混合ガウス" {
		start = time.Now()
		six, err := search.NewIVF(make([]search.Doc, n), rows(s.data), nil, search.IVFConfig{NList: nlist, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		build = time.Since(start)
		for _, p := range []int{1, 4, 16, 64} {
			ix := six.WithNProbe(p)
			r, per := s.measure(func(q []float32) []Neighbor {
				hits, _ := ix.SearchVector(q, evalK)
				out := make([]Neighbor, len(hits))
				for i, h := range hits {
					out[i] = Neighbor{ID: int32(h.ID), Dist: -h.Score}
				}
				return out
			})
			tb.row("search.IVF（球面 k-means）", fmt.Sprintf("nlist=%d nprobe=%d", nlist, p), r, per, qps(per), "-", 4*evalDim+4, build)
		}
	}

	for _, m := range []int{8, 16, 32} {
		start = time.Now()
		pq, err := TrainPQ(s.data, PQConfig{M: m, K: 256, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		px := NewPQIndex(pq, s.data)
		build = time.Since(start)
		r, per := s.measure(func(q []float32) []Neighbor { return px.Search(q, evalK) })
		tb.row("PQ（ADC）", fmt.Sprintf("M=%d K=256", m), r, per, qps(per), n, px.BytesPerVector(), build)
		for _, f := range []int{2, 4, 10, 50} {
			r, per := s.measure(func(q []float32) []Neighbor { return px.SearchRefine(q, evalK, f*evalK, s.exact) })
			tb.row("PQ + 再ランキング", fmt.Sprintf("M=%d 候補 %d×K", m, f), r, per, qps(per), fmt.Sprintf("%d+%d", n, f*evalK), fmt.Sprintf("%d+%d", px.BytesPerVector(), s.exact.BytesPerVector()), build)
		}
	}

	for _, raw := range []bool{false, true} {
		start = time.Now()
		ip, err := NewIVFPQ(s.data, IVFPQConfig{IVF: IVFConfig{NList: nlist, Seed: 1}, PQ: PQConfig{M: 16, K: 256, Seed: 1}, EncodeRaw: raw})
		if err != nil {
			t.Fatal(err)
		}
		build = time.Since(start)
		name := "IVF-PQ（残差）"
		if raw {
			name = "IVF-PQ（元のベクトル）"
		}
		for _, p := range []int{4, 16, 64} {
			ix := ip.WithNProbe(p)
			evals := 0
			r, per := s.measure(func(q []float32) []Neighbor {
				ns, st := ix.SearchStats(q, evalK)
				evals += st.Vectors
				return ns
			})
			tb.row(name, fmt.Sprintf("M=16 nprobe=%d", p), r, per, qps(per), evals/len(s.queries), ix.BytesPerVector(), build)
			if !raw {
				r, per := s.measure(func(q []float32) []Neighbor { return ix.SearchRefine(q, evalK, 10*evalK, s.exact) })
				tb.row(name+" + 再ランキング", fmt.Sprintf("M=16 nprobe=%d 候補 10×K", p), r, per, qps(per), "-", fmt.Sprintf("%d+%d", ix.BytesPerVector(), s.exact.BytesPerVector()), build)
			}
		}
	}

	for _, b := range []int{64, 128, 256, 512} {
		start = time.Now()
		l, err := NewLSH(s.data, b, 1)
		if err != nil {
			t.Fatal(err)
		}
		build = time.Since(start)
		for _, c := range []int{100, 1000, 5000} {
			r, per := s.measure(func(q []float32) []Neighbor { return l.Search(q, evalK, c, s.exact) })
			tb.row("LSH（ハミング全走査 + 再ランキング）", fmt.Sprintf("bits=%d 候補 %d", b, c), r, per, qps(per), c, fmt.Sprintf("%d+%d", l.BytesPerVector(), s.exact.BytesPerVector()), build)
		}
	}

	for _, cfg := range []LSHIndexConfig{
		{Bits: 8, Tables: 1}, {Bits: 8, Tables: 4}, {Bits: 12, Tables: 4}, {Bits: 12, Tables: 8}, {Bits: 16, Tables: 8},
	} {
		start = time.Now()
		cfg.Seed = 1
		lx, err := NewLSHIndex(s.data, cfg)
		if err != nil {
			t.Fatal(err)
		}
		build = time.Since(start)
		for _, p := range []int{1, 1 + cfg.Bits/2, 1 + cfg.Bits} {
			ix := lx.WithProbes(p, 0)
			evals := 0
			r, per := s.measure(func(q []float32) []Neighbor {
				ns, st := ix.Search(q, evalK, s.exact)
				evals += st.Vectors
				return ns
			})
			tb.row("LSH 転置（マルチプローブ）", fmt.Sprintf("b=%d 表 %d 探すバケット %d/表", cfg.Bits, cfg.Tables, p), r, per, qps(per), evals/len(s.queries), fmt.Sprintf("%d+%d", ix.BytesPerVector(), s.exact.BytesPerVector()), build)
		}
	}

	efC, flats := 200, []bool{false, true}
	if n > 200_000 {
		// 構築が 1 スレッドで件数にほぼ比例して重くなるので、100 万件では efC を下げ、NSW を省く。
		efC, flats = 100, []bool{false}
	}
	for _, flat := range flats {
		start = time.Now()
		h := NewHNSW(s.data, HNSWConfig{M: 16, EfConstruction: efC, Seed: 1, Flat: flat})
		build = time.Since(start)
		name := "HNSW"
		if flat {
			name = "NSW（1 層）"
		}
		for _, ef := range []int{20, 40, 80, 160, 320} {
			ix := h.WithEfSearch(ef)
			evals := 0
			r, per := s.measure(func(q []float32) []Neighbor {
				ns, st := ix.SearchStats(q, evalK)
				evals += st.Evals
				return ns
			})
			tb.row(name, fmt.Sprintf("M=16 efC=%d efSearch=%d", efC, ef), r, per, qps(per), evals/len(s.queries), fmt.Sprintf("%.0f", h.BytesPerVector()), build)
		}
	}
}

// TestReportGraphs は 1 層の NSW、正確な k 近傍グラフ、HNSW で、ホップ数と距離計算の数と recall を N ごとに比べる。
// 128 次元の混合ガウスと、直径が伸びやすい 4 次元の一様乱数の 2 つで測る。
func TestReportGraphs(t *testing.T) {
	evalEnabled(t)
	nq := 200
	tb := newTable(t, "データ", "N", "グラフ", "探索", "recall", "ホップ", "距離計算", "構築")
	for _, n := range envInts(t, "ANN_EVAL_GRAPH_N", []int{10_000, 100_000}) {
		if n <= 200_000 {
			reportGraphs(t, tb, "混合ガウス d=128", GaussianMixture(n, evalDim, evalClusters, evalNoise, 1, 2), rows(GaussianMixture(nq, evalDim, evalClusters, evalNoise, 1, 3)))
		}
		reportGraphs(t, tb, "一様 d=4", Uniform(n, 4, 1), rows(Uniform(nq, 4, 2)))
	}
}

func reportGraphs(t *testing.T, tb *table, name string, data Matrix, qs [][]float32) {
	n, nq := data.N, len(qs)
	e := NewExact(data)
	truth := e.SearchBatchParallel(qs, evalK)
	type graph struct {
		name  string
		h     *HNSW
		build time.Duration
	}
	var gs []graph
	for _, flat := range []bool{false, true} {
		start := time.Now()
		h := NewHNSW(data, HNSWConfig{M: 16, EfConstruction: 100, Seed: 1, Flat: flat})
		gname := "HNSW"
		if flat {
			gname = "NSW（1 層）"
		}
		gs = append(gs, graph{gname, h, time.Since(start)})
		if !flat {
			strip := h.WithEfSearch(64)
			strip.maxLevel = 0
			gs = append(gs, graph{"HNSW の層 0 だけ", strip, 0})
			t.Logf("%s N=%d HNSW 最上層ごとの点の数: %v", name, n, h.LevelCounts())
		}
	}
	if n <= 100_000 {
		start := time.Now()
		gs = append(gs, graph{"正確な 16 近傍グラフ", NewKNNGraph(data, 16), time.Since(start)})
	}
	for _, g := range gs {
		hit, hops, evals := 0, 0, 0
		for i, q := range qs {
			nb, st := g.h.Greedy(q)
			if nb.ID == truth[i][0].ID {
				hit++
			}
			hops += st.Hops
			evals += st.Evals
		}
		tb.row(name, n, g.name, "貪欲（候補 1 つ）recall@1", float64(hit)/float64(nq), float64(hops)/float64(nq), evals/nq, g.build)
		for _, ef := range []int{20, 80} {
			ix := g.h.WithEfSearch(ef)
			var recall float64
			hops, evals = 0, 0
			for i, q := range qs {
				ns, st := ix.SearchStats(q, evalK)
				recall += Recall(ns, truth[i])
				hops += st.Hops
				evals += st.Evals
			}
			tb.row(name, n, g.name, fmt.Sprintf("優先度付きキュー ef=%d recall@20", ef), recall/float64(nq), float64(hops)/float64(nq), evals/nq, g.build)
		}
	}
}

// TestReportLevels は最上層ごとの点の数を、幾何分布の期待値 N(1-1/M)M^-l と並べる。
func TestReportLevels(t *testing.T) {
	evalEnabled(t)
	const n, m = 1_000_000, 16
	counts := make([]int, hnswMaxLevel+1)
	for _, l := range sampleLevels(n, m, 1) {
		counts[l]++
	}
	tb := newTable(t, "最上層 l", "点の数", "期待値", "前の層との比")
	for l, c := range counts {
		if c == 0 && l > 0 {
			break
		}
		ratio := "-"
		if l > 0 && c > 0 {
			ratio = fmt.Sprintf("1/%.1f", float64(counts[l-1])/float64(c))
		}
		tb.row(l, c, fmt.Sprintf("%.1f", n*(1-1.0/m)*math.Pow(m, -float64(l))), ratio)
	}
}

// TestReportSimHash は角度 θ の組でビットが一致する割合を、超平面の枚数ごとに測って 1-θ/π と比べる。
func TestReportSimHash(t *testing.T) {
	evalEnabled(t)
	tb := newTable(t, "θ", "1-θ/π", "実測（256 枚 × 100 組の平均）", "差", "実測（65536 枚 × 1 組）", "差")
	for _, deg := range []float64{10, 45, 90, 135} {
		want := 1 - deg/180
		var avg float64
		for s := range 100 {
			avg += simHashAgreement(evalDim, deg, 256, uint64(s+1))
		}
		avg /= 100
		big := simHashAgreement(evalDim, deg, 65536, 7)
		tb.row(fmt.Sprintf("%.0f°", deg), want, avg, avg-want, big, big-want)
	}
}

// TestReportBinary は 256 次元の fp32 を 256 ビットの符号にしたときのメモリ、全走査の時間、再ランキングの recall を比べる。
func TestReportBinary(t *testing.T) {
	evalEnabled(t)
	const dim, nq = 256, 200
	for _, n := range envInts(t, "ANN_EVAL_N", []int{100_000}) {
		for _, ds := range []struct {
			name string
			data Matrix
			qs   [][]float32
		}{
			{"正規乱数", RandomNormal(n, dim, 1), rows(RandomNormal(nq, dim, 2))},
			{"混合ガウス", GaussianMixture(n, dim, evalClusters, evalNoise, 1, 2), rows(GaussianMixture(nq, dim, evalClusters, evalNoise, 1, 3))},
		} {
			e := NewExact(ds.data)
			s := &evalSet{data: ds.data, queries: ds.qs, exact: e, truth: e.SearchBatchParallel(ds.qs, evalK)}
			sign, err := NewLSHPlanes(ds.data, Identity(dim))
			if err != nil {
				t.Fatal(err)
			}
			hyper, err := NewLSH(ds.data, dim, 1)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("### %s N=%d d=%d: fp32 %d B/件、256 ビット %d B/件（1/%d）、全件 %.1f MB → %.1f MB",
				ds.name, n, dim, 4*dim, sign.BytesPerVector(), 4*dim/sign.BytesPerVector(), float64(4*dim*n)/1e6, float64(sign.BytesPerVector()*n)/1e6)
			tb := newTable(t, "方式", "recall@20", "1 クエリ")
			r, per := s.measure(func(q []float32) []Neighbor { return e.Search(q, evalK) })
			tb.row("fp32 の L2 全走査", r, per)
			qsig := make([]uint64, sign.Words)
			r, per = s.measure(func(q []float32) []Neighbor {
				sign.Signature(q, qsig)
				t := newKBest(evalK)
				for i := range n {
					t.push(Neighbor{ID: int32(i), Dist: float32(Hamming(qsig, sign.sig(i)))})
				}
				return t.sorted()
			})
			tb.row("符号 256 ビットのハミング全走査（並べ直しなし）", r, per)
			for _, l := range []struct {
				name string
				l    *LSH
			}{{"符号", sign}, {"SimHash", hyper}} {
				for _, c := range []int{100, 400, 2000} {
					r, per := s.measure(func(q []float32) []Neighbor { return l.l.Search(q, evalK, c, e) })
					tb.row(fmt.Sprintf("%s 256 ビットで %d 件 → fp32 で並べ直し", l.name, c), r, per)
				}
			}
		}
	}
}

// TestReportIVFCount は 10 億件・k=65536・nprobe=4（1 クラスタ平均 15,258 件、約 6 万件の距離計算）を 100 万件に縮め、
// 1 クラスタの平均をそろえた nlist=64 で、実際に距離を計算した件数を数える。
func TestReportIVFCount(t *testing.T) {
	evalEnabled(t)
	const n, nlist, nprobe, nq = 1_000_000, 64, 4, 200
	tb := newTable(t, "データ", "nlist", "平均 件/クラスタ", "最小〜最大", "nprobe", "計算した件数の平均", "最小〜最大", "中心との距離", "recall@20")
	for _, ds := range []struct {
		name string
		data Matrix
		qs   [][]float32
	}{
		{"正規乱数", RandomNormal(n, evalDim, 1), rows(RandomNormal(nq, evalDim, 2))},
		{"混合ガウス", GaussianMixture(n, evalDim, evalClusters, evalNoise, 1, 2), rows(GaussianMixture(nq, evalDim, evalClusters, evalNoise, 1, 3))},
	} {
		e := NewExact(ds.data)
		truth := e.SearchBatchParallel(ds.qs, evalK)
		ix := NewIVFFlat(ds.data, IVFConfig{NList: nlist, NProbe: nprobe, Seed: 1})
		sizes := ix.ListSizes()
		lo, hi := n, 0
		for _, sz := range sizes {
			lo, hi = min(lo, sz), max(hi, sz)
		}
		total, mn, mx, cents := 0, n, 0, 0
		var recall float64
		for i, q := range ds.qs {
			ns, st := ix.SearchStats(q, evalK)
			total += st.Vectors
			mn, mx = min(mn, st.Vectors), max(mx, st.Vectors)
			cents = st.Centroids
			recall += Recall(ns, truth[i])
		}
		tb.row(ds.name, nlist, n/nlist, fmt.Sprintf("%d〜%d", lo, hi), nprobe, total/nq, fmt.Sprintf("%d〜%d", mn, mx), cents, recall/nq)
	}
}

// heapDelta は build の前後で GC を走らせ、build が返した値が保持しているヒープの増分を返す。
func heapDelta(build func() any) (any, uint64) {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	v := build()
	runtime.GC()
	runtime.ReadMemStats(&after)
	return v, after.HeapAlloc - before.HeapAlloc
}

// TestReportMemory は索引が保持するヒープを実測し、1 件あたりのバイト数を計算値と並べる。
func TestReportMemory(t *testing.T) {
	evalEnabled(t)
	const n = 1_000_000
	tb := newTable(t, "対象", "N", "ヒープの増分", "実測 B/件", "計算 B/件")
	var data Matrix
	_, d := heapDelta(func() any { data = RandomNormal(n, evalDim, 1); return data })
	tb.row("fp32 の元のベクトル（d=128）", n, fmt.Sprintf("%.1f MB", float64(d)/1e6), fmt.Sprintf("%.1f", float64(d)/n), 4*evalDim)
	pq, err := TrainPQ(data, PQConfig{M: 8, K: 256, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	v, d := heapDelta(func() any { return NewPQIndex(pq, data) })
	tb.row("PQ の符号（M=8 K=256）", n, fmt.Sprintf("%.1f MB", float64(d)/1e6), fmt.Sprintf("%.1f", float64(d)/n), v.(*PQIndex).BytesPerVector())
	t.Logf("PQ の中心（M×K×Sub×4）: %d B", 4*len(pq.Codebooks))
	v, d = heapDelta(func() any { l, _ := NewLSH(data, 256, 1); return l })
	tb.row("SimHash 256 ビット", n, fmt.Sprintf("%.1f MB", float64(d)/1e6), fmt.Sprintf("%.1f", float64(d)/n), v.(*LSH).BytesPerVector())
	const hn = 100_000
	small := Matrix{N: hn, Dim: evalDim, Data: data.Data[:hn*evalDim]}
	v, d = heapDelta(func() any { return NewHNSW(small, HNSWConfig{M: 16, EfConstruction: 40, Seed: 1}) })
	tb.row("HNSW のグラフ（M=16、ベクトル本体を除く）", hn, fmt.Sprintf("%.1f MB", float64(d)/1e6), fmt.Sprintf("%.1f", float64(d)/hn), fmt.Sprintf("%.1f", float64(v.(*HNSW).Bytes())/hn))
	runtime.KeepAlive(data)
}

// TestReportFilter は条件で絞った HNSW の検索で、efSearch 件を集めてから絞る場合と探索中に絞る場合を、許可する点の選び方 3 通りで比べる。
func TestReportFilter(t *testing.T) {
	evalEnabled(t)
	const n, nq, k = 100_000, 200, 10
	data, labels := gaussianMixture(n, evalDim, evalClusters, evalNoise, 1, 2)
	qm, qlabels := gaussianMixture(nq, evalDim, evalClusters, evalNoise, 1, 3)
	qs := rows(qm)
	e := NewExact(data)
	start := time.Now()
	h := NewHNSW(data, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})
	t.Logf("HNSW M=16 efC=200 N=%d d=%d: built in %v", n, evalDim, time.Since(start).Round(time.Second))
	members := make([][]int32, evalClusters)
	for id, c := range labels {
		members[c] = append(members[c], int32(id))
	}
	tb := newTable(t, "許可する点", "選択率", "efSearch", "min(K, ef×s)", "二項分布の期待値", "後で絞る 件数", "recall@10", "1 クエリ", "距離計算", "探索中に絞る 件数", "recall@10", "1 クエリ", "距離計算", "絞った集合の全探索")
	allow := make([]bool, n)
	var ids []int32
	for _, mode := range []struct {
		name               string
		byCluster, withOwn bool
	}{
		{name: "一様"},
		{name: "塊（クエリの塊を含む）", byCluster: true, withOwn: true},
		{name: "塊（クエリの塊を含まない）", byCluster: true},
	} {
		for _, sel := range []float64{0.5, 0.1, 0.01, 0.001} {
			r := newRand(7)
			uniform := r.Perm(n)[:int(math.Round(sel*n))]
			picks := make([][]int, nq)
			want := int(math.Round(sel * evalClusters))
			for i := range picks {
				if !mode.byCluster {
					continue
				}
				own := int(qlabels[i])
				if mode.withOwn {
					picks[i] = append(picks[i], own)
				}
				for _, c := range r.Perm(evalClusters) {
					if len(picks[i]) == want {
						break
					}
					if c != own {
						picks[i] = append(picks[i], c)
					}
				}
			}
			setAllow := func(i int) {
				clear(allow)
				ids = ids[:0]
				if mode.byCluster {
					for _, c := range picks[i] {
						ids = append(ids, members[c]...)
					}
				} else {
					for _, id := range uniform {
						ids = append(ids, int32(id))
					}
				}
				for _, id := range ids {
					allow[id] = true
				}
			}
			truth := make([][]Neighbor, nq)
			var exactDur time.Duration
			allowed := 0
			for i, q := range qs {
				setAllow(i)
				allowed += len(ids)
				start := time.Now()
				truth[i] = e.Refine(q, ids, k)
				exactDur += time.Since(start)
			}
			t.Logf("%s 選択率 %g%%: 許可した点の平均 %.0f 件", mode.name, sel*100, float64(allowed)/nq)
			// 方式ごとに全クエリを流す。同じクエリを続けて投げると、後の方式が前の方式で読んだベクトルをキャッシュから読めて速く見える。
			pass := func(ix *HNSW, search func(*HNSW, []float32, int, []bool) ([]Neighbor, HNSWStats)) []any {
				var count, recall float64
				var dur time.Duration
				evals := 0
				for i, q := range qs {
					setAllow(i)
					start := time.Now()
					ns, st := search(ix, q, k, allow)
					dur += time.Since(start)
					count += float64(len(ns))
					recall += Recall(ns, truth[i])
					evals += st.Evals
				}
				return []any{fmt.Sprintf("%.2f", count/nq), recall / nq, dur / nq, evals / nq}
			}
			for _, ef := range []int{40, 100, 400, 1000} {
				ix := h.WithEfSearch(ef)
				cells := []any{mode.name, fmt.Sprintf("%g%%", sel*100), ef, fmt.Sprintf("%.1f", min(k, float64(ef)*sel)), fmt.Sprintf("%.2f", expectedKept(ef, k, sel))}
				cells = append(cells, pass(ix, (*HNSW).SearchPostFilter)...)
				cells = append(cells, pass(ix, (*HNSW).SearchFiltered)...)
				tb.row(append(cells, exactDur/nq)...)
			}
		}
	}
}

// expectedKept は ef 件の候補がそれぞれ確率 s で許可されるときに残る min(k, X) の期待値。X は二項分布 B(ef, s) に従う。
func expectedKept(ef, k int, s float64) float64 {
	lgEf, _ := math.Lgamma(float64(ef + 1))
	var sum, below float64
	for x := range min(k, ef+1) {
		lgX, _ := math.Lgamma(float64(x + 1))
		lgRest, _ := math.Lgamma(float64(ef - x + 1))
		p := math.Exp(lgEf - lgX - lgRest + float64(x)*math.Log(s) + float64(ef-x)*math.Log1p(-s))
		sum += float64(x) * p
		below += p
	}
	return sum + float64(k)*(1-below)
}

// TestReportMultiVector は ColBERT 型の多ベクトル文書で、全文書の MaxSim と、トークンの近傍から候補の文書を集めて MaxSim で並べ直す近似を比べる。
func TestReportMultiVector(t *testing.T) {
	evalEnabled(t)
	const k, nq = 10, 200
	cfg := TopicConfig{Docs: 10_000, Dim: 64, NumTopics: 100, MinTokens: 20, MaxTokens: 60, MinTopics: 2, MaxTopics: 4, Noise: 1, Seed: 1, SampleSeed: 2}
	docs := TopicMultiVector(cfg)
	cfg.Docs, cfg.MinTokens, cfg.MaxTokens, cfg.MinTopics, cfg.MaxTopics, cfg.SampleSeed = nq, 4, 8, 1, 3, 3
	queries := TopicMultiVector(cfg)
	t.Logf("文書 %d 件、トークン %d 個（平均 %.1f 個/文書）、d=%d、話題 %d、クエリ %d 件（トークン %d 個）",
		docs.Docs(), docs.Tokens.N, float64(docs.Tokens.N)/float64(docs.Docs()), docs.Tokens.Dim, cfg.NumTopics, nq, queries.Tokens.N)

	start := time.Now()
	truth := make([][]Neighbor, nq)
	for i := range nq {
		truth[i] = docs.SearchMaxSim(queries.Doc(i), k)
	}
	exactPer := time.Since(start) / nq

	kToks := []int{10, 50, 200, 1000}
	// ranks[i][r][j] は真の上位 r+1 番目の文書について、クエリのトークン j に最も近いその文書のトークンが、全トークンの中で何位か。
	ranks := make([][][]int, nq)
	parallelRange(nq, func(qlo, qhi int) {
		sims := make([]float32, docs.Tokens.N)
		for i := qlo; i < qhi; i++ {
			q := queries.Doc(i)
			ranks[i] = make([][]int, len(truth[i]))
			for j := range q.N {
				for tok := range docs.Tokens.N {
					sims[tok] = dot(q.Row(j), docs.Tokens.Row(tok))
				}
				for r, nb := range truth[i] {
					best := slices.Max(sims[docs.Start[nb.ID]:docs.Start[nb.ID+1]])
					rank := 1
					for _, s := range sims {
						if s > best {
							rank++
						}
					}
					ranks[i][r] = append(ranks[i][r], rank)
				}
			}
		}
	})
	bestRank := func(i, r int) int { return slices.Min(ranks[i][r]) }
	median := func(xs []int) int {
		s := slices.Clone(xs)
		slices.Sort(s)
		return s[len(s)/2]
	}

	start = time.Now()
	hn := NewHNSW(docs.Tokens, HNSWConfig{M: 16, EfConstruction: 100, Seed: 1})
	t.Logf("トークンの HNSW M=16 efC=100: built in %v", time.Since(start).Round(time.Second))
	tb := newTable(t, "方式", "K'", "recall@10", "候補の文書数", "1 クエリ", "取りこぼし/クエリ", "うち全クエリトークンで K' 位より下", "取りこぼした文書の最良順位（中央値）")
	tb.row("全文書の MaxSim", "-", 1.0, docs.Docs(), exactPer, "-", "-", "-")
	for _, ix := range []struct {
		name string
		s    TokenSearcher
	}{
		{name: "トークンを全探索", s: NewExact(docs.Tokens)},
		{name: "トークンを HNSW（ef=max(64, K')）", s: hn},
	} {
		for _, kTok := range kToks {
			res := make([][]Neighbor, nq)
			cands := 0
			start := time.Now()
			for i := range nq {
				var c int
				res[i], c = docs.SearchCandidates(queries.Doc(i), k, kTok, ix.s)
				cands += c
			}
			per := time.Since(start) / nq
			var recall float64
			missed, below := 0, 0
			var missedRanks []int
			for i := range nq {
				recall += Recall(res[i], truth[i])
				for r, nb := range truth[i] {
					if slices.ContainsFunc(res[i], func(g Neighbor) bool { return g.ID == nb.ID }) {
						continue
					}
					missed++
					b := bestRank(i, r)
					missedRanks = append(missedRanks, b)
					if b > kTok {
						below++
					}
				}
			}
			share, med := "-", "-"
			if missed > 0 {
				share, med = fmt.Sprintf("%.2f", float64(below)/float64(missed)), fmt.Sprint(median(missedRanks))
			}
			tb.row(ix.name, kTok, recall/nq, fmt.Sprintf("%.0f", float64(cands)/nq), per, fmt.Sprintf("%.2f", float64(missed)/nq), share, med)
		}
	}

	tb = newTable(t, "K'", "真の上位 10 件のうち全クエリトークンで K' 位より下", "その文書の真の順位（平均）", "トークンごとの順位 最良 / 中央値 / 最悪", "K' 位以内に入る文書の真の順位（平均）", "トークンごとの順位 最良 / 中央値 / 最悪")
	for _, kTok := range kToks {
		type group struct {
			pos              float64
			best, mid, worst []int
		}
		var in, out group
		for i := range nq {
			for r, rs := range ranks[i] {
				g := &in
				if bestRank(i, r) > kTok {
					g = &out
				}
				g.pos += float64(r + 1)
				g.best = append(g.best, bestRank(i, r))
				g.mid = append(g.mid, median(rs))
				g.worst = append(g.worst, slices.Max(rs))
			}
		}
		cells := func(g group) []any {
			if len(g.best) == 0 {
				return []any{"-", "-"}
			}
			return []any{fmt.Sprintf("%.1f", g.pos/float64(len(g.best))), fmt.Sprintf("%d / %d / %d", median(g.best), median(g.mid), median(g.worst))}
		}
		row := []any{kTok, fmt.Sprintf("%.3f", float64(len(out.best))/float64(len(in.best)+len(out.best)))}
		row = append(row, cells(out)...)
		tb.row(append(row, cells(in)...)...)
	}
}
