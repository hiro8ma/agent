package ann

import (
	"math"
	"slices"
	"strconv"
	"testing"
)

// naiveKNN は float64 で全件の距離を取り、近い順に並べた正解を返す。
func naiveKNN(m Matrix, q []float32, k int) []Neighbor {
	all := make([]Neighbor, m.N)
	dists := make([]float64, m.N)
	for i := range m.N {
		var s float64
		for j, v := range m.Row(i) {
			d := float64(q[j]) - float64(v)
			s += d * d
		}
		dists[i] = s
		all[i] = Neighbor{ID: int32(i), Dist: float32(s)}
	}
	slices.SortFunc(all, compareNeighbor)
	return all[:min(k, m.N)]
}

func trueDist(m Matrix, q []float32, id int32) float64 {
	var s float64
	for j, v := range m.Row(int(id)) {
		d := float64(q[j]) - float64(v)
		s += d * d
	}
	return s
}

// assertKNN は got の各点の正確な距離が正解の k 番目以下で、Dist が正確な距離に近いことを確かめる。
// 境目で同じ距離に並ぶ点は fp32 の丸めで入れ替わりうるので、ID の一致ではなく距離で判定する。
func assertKNN(t *testing.T, m Matrix, q []float32, got []Neighbor, k int) {
	t.Helper()
	want := naiveKNN(m, q, k)
	if len(got) != len(want) {
		t.Fatalf("got %d neighbors, want %d", len(got), len(want))
	}
	kth := float64(want[len(want)-1].Dist)
	seen := map[int32]bool{}
	for _, n := range got {
		if seen[n.ID] {
			t.Fatalf("duplicate id %d", n.ID)
		}
		seen[n.ID] = true
		d := trueDist(m, q, n.ID)
		if d > kth*(1+1e-5)+1e-5 {
			t.Errorf("id %d at distance %g is farther than the k-th %g", n.ID, d, kth)
		}
		if math.Abs(float64(n.Dist)-d) > 1e-3*(1+d) {
			t.Errorf("id %d reports %g, true distance %g", n.ID, n.Dist, d)
		}
	}
}

func TestExactMatchesNaive(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		n, dim, k int
	}{
		"k=1 で最近傍だけを返す":      {n: 500, dim: 16, k: 1},
		"k=20 で教材と同じ件数を返す":   {n: 2000, dim: 128, k: 20},
		"k が件数を超えたら全件を返す":    {n: 30, dim: 8, k: 50},
		"次元が 4 の倍数でなくても端を足す": {n: 300, dim: 13, k: 5},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			m := RandomNormal(tc.n, tc.dim, 7)
			qs := RandomNormal(5, tc.dim, 8)
			e := NewExact(m)
			queries := make([][]float32, qs.N)
			for i := range qs.N {
				queries[i] = qs.Row(i)
			}
			batch := e.SearchBatch(queries, tc.k)
			par := e.SearchBatchParallel(queries, tc.k)
			for i, q := range queries {
				assertKNN(t, m, q, e.Search(q, tc.k), tc.k)
				part := e.Partition(q, tc.k, nil)
				slices.SortFunc(part, compareNeighbor)
				assertKNN(t, m, q, part, tc.k)
				assertKNN(t, m, q, batch[i], tc.k)
				assertKNN(t, m, q, par[i], tc.k)
			}
		})
	}
}

// ||q||² + ||x||² - 2q·x から ||x||² を外すと、長さの違う点の順位が崩れる。長さ 1 にそろえた点なら ||x||² は定数なので崩れない。
func TestNormTermRequiredForUnnormalizedData(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		data       Matrix
		wantBroken bool
	}{
		"正規分布の点は長さがばらつくので順位が崩れる":     {data: RandomNormal(3000, 32, 1), wantBroken: true},
		"長さ 1 の点は ||x||² が同じなので崩れない": {data: GaussianMixture(3000, 32, 20, 0.5, 1, 2), wantBroken: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			qs := RandomNormal(20, tc.data.Dim, 3)
			e := NewExact(tc.data)
			removed := &Exact{m: e.m, norms: make([]float32, len(e.norms))}
			var full, cut float64
			for i := range qs.N {
				want := naiveKNN(tc.data, qs.Row(i), 20)
				full += Recall(e.Search(qs.Row(i), 20), want)
				cut += Recall(removed.Search(qs.Row(i), 20), want)
			}
			full /= float64(qs.N)
			cut /= float64(qs.N)
			if full < 0.999 {
				t.Errorf("with the norm term recall = %.3f, want 1", full)
			}
			if tc.wantBroken && cut > 0.5 {
				t.Errorf("without the norm term recall = %.3f, want it to break below 0.5", cut)
			}
			if !tc.wantBroken && cut < 0.999 {
				t.Errorf("without the norm term recall = %.3f, want 1 on unit vectors", cut)
			}
		})
	}
}

func TestSelectKPutsSmallestFirst(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		n, k int
		dup  bool
	}{
		"k=0 は何もしない":        {n: 100, k: 0},
		"k=1 で最小を先頭に置く":     {n: 100, k: 1},
		"半分で分ける":            {n: 1001, k: 500},
		"k=n は全件":           {n: 50, k: 50},
		"同じ距離が多くても ID で決まる": {n: 500, k: 37, dup: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := newRand(uint64(tc.n + tc.k))
			ns := make([]Neighbor, tc.n)
			for i := range ns {
				d := r.Float32()
				if tc.dup {
					d = float32(r.IntN(5))
				}
				ns[i] = Neighbor{ID: int32(i), Dist: d}
			}
			want := slices.Clone(ns)
			slices.SortFunc(want, compareNeighbor)
			selectK(ns, tc.k)
			got := slices.Clone(ns[:tc.k])
			slices.SortFunc(got, compareNeighbor)
			if !slices.Equal(got, want[:tc.k]) {
				t.Errorf("selectK(%d) first k differ from sorted prefix", tc.k)
			}
		})
	}
}

// 12 次元を 3 次元ずつ 4 ブロックに分け、各ブロックで 4 個の中心から最も近いものを選ぶと符号が (2,2,3,3) になる。
func TestPQEncodeTwelveDimsIntoFourBlocks(t *testing.T) {
	t.Parallel()
	books := make([]float32, 0, 4*4*3)
	for m := range 4 {
		for c := range 4 {
			v := float32(10*m + c)
			books = append(books, v, v, v)
		}
	}
	pq, err := NewPQ(12, 4, 4, books)
	if err != nil {
		t.Fatal(err)
	}
	v := []float32{2.1, 1.9, 2.0, 12.2, 11.8, 12.1, 23.0, 22.9, 23.1, 33.2, 32.8, 33.0}
	code := make([]uint8, pq.M)
	pq.Encode(v, code)
	if want := []uint8{2, 2, 3, 3}; !slices.Equal(code, want) {
		t.Fatalf("code = %v, want %v", code, want)
	}
	dec := make([]float32, 12)
	pq.Decode(code, dec)
	want := []float32{2, 2, 2, 12, 12, 12, 23, 23, 23, 33, 33, 33}
	if !slices.Equal(dec, want) {
		t.Errorf("decoded = %v, want %v", dec, want)
	}
	if pq.M != 4 || pq.Sub != 3 || len(code) != 4 {
		t.Errorf("shape M=%d Sub=%d len(code)=%d, want 4, 3, 4", pq.M, pq.Sub, len(code))
	}
}

// 距離表の手計算の例。q を 4 次元ずつ 3 ブロックに分け、第 1 ブロックと中心 c_{1,1} の距離の 2 乗が 55.23 になる。
// 割り当て (2,2,3)（1 始まり）のベクトルの近似距離は D_{1,2}+D_{2,2}+D_{3,3} で、復元したベクトルとの距離に一致する。
func TestPQDistanceTableHandExample(t *testing.T) {
	t.Parallel()
	q := []float32{6.1, 7.9, 6.1, 5.4, 7.0, 4.0, 9.7, 4.3, 5.5, 0.9, 0.5, 1.7}
	books := []float32{
		6.2, 2.0, 4.0, 9.4, 1, 1, 1, 1, 5, 5, 5, 5,
		3, 3, 3, 3, 7.1, 4.2, 9.0, 4.0, 0, 0, 0, 0,
		9, 9, 9, 9, 1, 1, 1, 1, 5.0, 1.0, 0.0, 2.0,
	}
	pq, err := NewPQ(12, 3, 3, books)
	if err != nil {
		t.Fatal(err)
	}
	tab := make([]float32, pq.M*pq.K)
	pq.Table(q, tab)
	d := func(m, c int) float32 { return tab[(m-1)*pq.K+(c-1)] }
	if got := d(1, 1); math.Abs(float64(got)-55.23) > 1e-4 {
		t.Errorf("D_{1,1} = %v, want 55.23", got)
	}
	code := []uint8{1, 1, 2}
	got := pq.adcDistance(tab, code)
	want := d(1, 2) + d(2, 2) + d(3, 3)
	if got != want {
		t.Errorf("ADC = %v, want D_{1,2}+D_{2,2}+D_{3,3} = %v", got, want)
	}
	dec := make([]float32, 12)
	pq.Decode(code, dec)
	if direct := l2sq(q, dec); math.Abs(float64(direct-got)) > 1e-4 {
		t.Errorf("ADC = %v, distance to decoded vector = %v", got, direct)
	}
}

func TestNewPQRejectsBadShapes(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		dim, m, k, books int
	}{
		"次元がブロック数で割り切れない":     {dim: 10, m: 3, k: 4, books: 0},
		"中心が 1 バイトに収まらない":     {dim: 8, m: 2, k: 300, books: 2 * 300 * 4},
		"中心の値の数が形と合わない":       {dim: 8, m: 2, k: 4, books: 5},
		"ブロック数が 0 以下では分けられない": {dim: 8, m: 0, k: 4, books: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if _, err := NewPQ(tc.dim, tc.m, tc.k, make([]float32, tc.books)); err == nil {
				t.Error("NewPQ succeeded, want an error")
			}
		})
	}
}

func TestPQADCEqualsDistanceToDecoded(t *testing.T) {
	t.Parallel()
	data := RandomNormal(2000, 32, 1)
	pq, err := TrainPQ(data, PQConfig{M: 8, K: 16, Iterations: 5, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	x := NewPQIndex(pq, data)
	q := RandomNormal(1, 32, 9).Row(0)
	tab := make([]float32, pq.M*pq.K)
	pq.Table(q, tab)
	dec := make([]float32, 32)
	for i := range 50 {
		pq.Decode(x.Code(i), dec)
		if a, b := pq.adcDistance(tab, x.Code(i)), l2sq(q, dec); math.Abs(float64(a-b)) > 1e-3*(1+float64(b)) {
			t.Fatalf("row %d: ADC %v, decoded distance %v", i, a, b)
		}
	}
	if got := x.BytesPerVector(); got != 8 {
		t.Errorf("BytesPerVector = %d, want 8", got)
	}
}

func TestPQRefineRaisesRecall(t *testing.T) {
	t.Parallel()
	data := GaussianMixture(4000, 32, 30, 0.6, 1, 2)
	qs := GaussianMixture(30, 32, 30, 0.6, 1, 3)
	e := NewExact(data)
	pq, err := TrainPQ(data, PQConfig{M: 4, K: 64, Iterations: 8, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	x := NewPQIndex(pq, data)
	var plain, refined float64
	for i := range qs.N {
		want := e.Search(qs.Row(i), 10)
		plain += Recall(x.Search(qs.Row(i), 10), want)
		refined += Recall(x.SearchRefine(qs.Row(i), 10, 100, e), want)
	}
	if refined <= plain {
		t.Errorf("recall with refine %.3f, without %.3f, want refine to be higher", refined/30, plain/30)
	}
}

// pqSearchDirect は距離表を使わず、符号ごとに中心との距離を計算し直す。距離表を外したときの比較用。
func pqSearchDirect(x *PQIndex, q []float32, k int) []Neighbor {
	p := x.pq
	t := newKBest(min(k, x.n))
	for i := range x.n {
		var d float32
		for m, c := range x.Code(i) {
			d += l2sq(q[m*p.Sub:(m+1)*p.Sub], p.centroid(m, int(c)))
		}
		t.push(Neighbor{ID: int32(i), Dist: d})
	}
	return t.sorted()
}

func TestPQDirectMatchesTable(t *testing.T) {
	t.Parallel()
	data := RandomNormal(1500, 32, 4)
	pq, err := TrainPQ(data, PQConfig{M: 8, K: 32, Iterations: 4, Seed: 5})
	if err != nil {
		t.Fatal(err)
	}
	x := NewPQIndex(pq, data)
	q := RandomNormal(1, 32, 6).Row(0)
	if r := Recall(pqSearchDirect(x, q, 20), x.Search(q, 20)); r < 0.95 {
		t.Errorf("direct and table searches agree on %.2f of top-20, want them to match", r)
	}
}

func TestTrainingIsDeterministic(t *testing.T) {
	t.Parallel()
	data := GaussianMixture(3000, 16, 10, 0.5, 1, 2)
	testCases := map[string]func() any{
		"PQ の中心": func() any {
			pq, err := TrainPQ(data, PQConfig{M: 4, K: 16, Seed: 9})
			if err != nil {
				t.Fatal(err)
			}
			return pq.Codebooks
		},
		"IVF の割り当て": func() any { return NewIVFFlat(data, IVFConfig{NList: 20, Seed: 9}).ids },
		"LSH の符号":   func() any { l, _ := NewLSH(data, 128, 9); return l.sigs },
		"HNSW の辺": func() any {
			h := NewHNSW(Matrix{N: 800, Dim: data.Dim, Data: data.Data[:800*data.Dim]}, HNSWConfig{M: 8, EfConstruction: 40, Seed: 9})
			return [3]any{h.layer0, h.upper, h.entry}
		},
	}
	for tn, build := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			a, b := build(), build()
			if !deepEqual(a, b) {
				t.Errorf("two builds with the same seed differ")
			}
		})
	}
}

func deepEqual(a, b any) bool {
	switch x := a.(type) {
	case []float32:
		return slices.Equal(x, b.([]float32))
	case []int32:
		return slices.Equal(x, b.([]int32))
	case []uint64:
		return slices.Equal(x, b.([]uint64))
	case [3]any:
		y := b.([3]any)
		if !slices.Equal(x[0].([]int32), y[0].([]int32)) || x[2] != y[2] {
			return false
		}
		return slices.EqualFunc(x[1].([][]int32), y[1].([][]int32), slices.Equal[[]int32])
	}
	return false
}

func TestIVFFlat(t *testing.T) {
	t.Parallel()
	data := GaussianMixture(5000, 32, 40, 0.6, 1, 2)
	qs := GaussianMixture(20, 32, 40, 0.6, 1, 3)
	e := NewExact(data)
	ix := NewIVFFlat(data, IVFConfig{NList: 50, Seed: 1})
	sizes := ix.ListSizes()
	testCases := map[string]struct {
		nprobe    int
		minRecall float64
	}{
		"全クラスタを探せば全探索と同じ": {nprobe: 50, minRecall: 1},
		"8 クラスタでも大半を拾う":   {nprobe: 8, minRecall: 0.8},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			s := ix.WithNProbe(tc.nprobe)
			var recall float64
			for i := range qs.N {
				q := qs.Row(i)
				got, st := s.SearchStats(q, 20)
				recall += Recall(got, e.Search(q, 20))
				want := 0
				for _, l := range ix.probe(q, tc.nprobe) {
					want += sizes[l.ID]
				}
				if st.Vectors != want || st.Centroids != 50 {
					t.Errorf("stats = %+v, want %d vectors and 50 centroids", st, want)
				}
			}
			if recall /= float64(qs.N); recall < tc.minRecall {
				t.Errorf("recall = %.3f, want at least %.2f", recall, tc.minRecall)
			}
		})
	}
}

func TestIVFPQResidualBeatsRaw(t *testing.T) {
	t.Parallel()
	data := GaussianMixture(6000, 32, 40, 0.5, 1, 2)
	qs := GaussianMixture(40, 32, 40, 0.5, 1, 3)
	e := NewExact(data)
	recall := func(raw bool) float64 {
		ix, err := NewIVFPQ(data, IVFPQConfig{IVF: IVFConfig{NList: 40, NProbe: 8, Seed: 1}, PQ: PQConfig{M: 4, K: 64, Seed: 1}, EncodeRaw: raw})
		if err != nil {
			t.Fatal(err)
		}
		var r float64
		for i := range qs.N {
			r += Recall(ix.Search(qs.Row(i), 10), e.Search(qs.Row(i), 10))
		}
		return r / float64(qs.N)
	}
	res, raw := recall(false), recall(true)
	if res <= raw {
		t.Errorf("residual recall %.3f, raw recall %.3f, want residual to be higher", res, raw)
	}
}

func parseBits(t *testing.T, s string) []uint64 {
	t.Helper()
	v, err := strconv.ParseUint(s, 2, 64)
	if err != nil {
		t.Fatal(err)
	}
	return []uint64{v}
}

func TestHamming(t *testing.T) {
	t.Parallel()
	ones := []uint64{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	testCases := map[string]struct {
		a, b []uint64
		want int
	}{
		"教材の例 1011001 と 1001101 は 2 ビット違う": {a: parseBits(t, "1011001"), b: parseBits(t, "1001101"), want: 2},
		"同じ符号は 0":             {a: ones, b: ones, want: 0},
		"256 ビットすべて反転すると 256": {a: ones, b: make([]uint64, 4), want: 256},
		"語をまたいで数える":           {a: []uint64{1, 0, 1 << 63, 3}, b: []uint64{0, 0, 0, 0}, want: 4},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := Hamming(tc.a, tc.b); got != tc.want {
				t.Errorf("Hamming = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestHammingMatchesBitByBit(t *testing.T) {
	t.Parallel()
	r := newRand(1)
	for range 200 {
		a, b := make([]uint64, 4), make([]uint64, 4)
		for i := range a {
			a[i], b[i] = r.Uint64(), r.Uint64()
		}
		want := 0
		for i := range 256 {
			if (a[i/64]>>(i%64))&1 != (b[i/64]>>(i%64))&1 {
				want++
			}
		}
		if got := Hamming(a, b); got != want {
			t.Fatalf("Hamming = %d, bit by bit = %d", got, want)
		}
	}
}

func TestSignBitsAreSigns(t *testing.T) {
	t.Parallel()
	data := RandomNormal(10, 130, 1)
	l, err := NewLSHPlanes(data, Identity(130))
	if err != nil {
		t.Fatal(err)
	}
	if l.Words != 3 || l.BytesPerVector() != 24 {
		t.Fatalf("words = %d bytes = %d, want 3 and 24", l.Words, l.BytesPerVector())
	}
	for i := range data.N {
		for j, v := range data.Row(i) {
			if got, want := (l.sig(i)[j/64]>>(j%64))&1 == 1, v > 0; got != want {
				t.Fatalf("row %d bit %d = %v, value %v", i, j, got, v)
			}
		}
	}
}

func TestLSHCandidatesAreHammingNearest(t *testing.T) {
	t.Parallel()
	data := RandomNormal(3000, 32, 1)
	testCases := map[string]struct {
		bits, c int
	}{
		"64 ビットで 100 件":   {bits: 64, c: 100},
		"256 ビットで 1000 件": {bits: 256, c: 1000},
		"件数を超える候補は全件に丸める": {bits: 64, c: 5000},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			l, err := NewLSH(data, tc.bits, 2)
			if err != nil {
				t.Fatal(err)
			}
			q := RandomNormal(1, 32, 3).Row(0)
			qs := make([]uint64, l.Words)
			l.Signature(q, qs)
			cands := l.Candidates(q, tc.c)
			if len(cands) != min(tc.c, data.N) {
				t.Fatalf("got %d candidates, want %d", len(cands), min(tc.c, data.N))
			}
			in := make([]bool, data.N)
			worst := 0
			for _, id := range cands {
				in[id] = true
				worst = max(worst, Hamming(qs, l.sig(int(id))))
			}
			for i := range data.N {
				if !in[i] && Hamming(qs, l.sig(i)) < worst {
					t.Fatalf("row %d at distance %d was left out while %d was kept", i, Hamming(qs, l.sig(i)), worst)
				}
			}
		})
	}
}

// anglePair は 2 本の単位ベクトルを角度 θ で作る。正規分布の超平面は向きが一様なので、どの平面に置いても確率は同じ。
func anglePair(dim int, deg float64, r uint64) Matrix {
	basis := RandomNormal(2, dim, r)
	u, w := basis.Row(0), basis.Row(1)
	normalizeRow(u)
	p := dot(u, w)
	for i := range w {
		w[i] -= p * u[i]
	}
	normalizeRow(w)
	th := deg * math.Pi / 180
	data := make([]float32, 2*dim)
	for i := range dim {
		data[i] = u[i]
		data[dim+i] = float32(math.Cos(th))*u[i] + float32(math.Sin(th))*w[i]
	}
	return Matrix{N: 2, Dim: dim, Data: data}
}

func normalizeRow(v []float32) {
	inv := float32(1 / math.Sqrt(float64(dot(v, v))))
	for i := range v {
		v[i] *= inv
	}
}

// SimHashAgreement は角度 θ の組で 1 ビットが一致する割合を、nbits 枚の超平面で測る。
func simHashAgreement(dim int, deg float64, nbits int, seed uint64) float64 {
	pair := anglePair(dim, deg, seed)
	l, _ := NewLSH(pair, nbits, seed+100)
	return 1 - float64(Hamming(l.sig(0), l.sig(1)))/float64(nbits)
}

func TestSimHashAgreementIsOneMinusThetaOverPi(t *testing.T) {
	t.Parallel()
	testCases := map[string]float64{
		"10 度": 10, "45 度": 45, "90 度": 90, "135 度": 135,
	}
	for tn, deg := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := simHashAgreement(64, deg, 8192, uint64(deg))
			want := 1 - deg/180
			// 8192 ビットの標準誤差は最大 0.0055 なので、約 4 倍の幅で判定する。
			if math.Abs(got-want) > 0.02 {
				t.Errorf("agreement = %.4f, want 1-θ/π = %.4f", got, want)
			}
		})
	}
}

func TestLSHIndexMoreProbesFindMore(t *testing.T) {
	t.Parallel()
	data := GaussianMixture(4000, 32, 30, 0.6, 1, 2)
	qs := GaussianMixture(20, 32, 30, 0.6, 1, 3)
	e := NewExact(data)
	ix, err := NewLSHIndex(data, LSHIndexConfig{Bits: 8, Tables: 2, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	prevCand, prevRecall := -1, -1.0
	for _, probes := range []int{1, 3, 9} {
		s := ix.WithProbes(probes, 0)
		cand, recall := 0, 0.0
		for i := range qs.N {
			got, st := s.Search(qs.Row(i), 10, e)
			cand += st.Vectors
			recall += Recall(got, e.Search(qs.Row(i), 10))
		}
		if cand < prevCand || recall < prevRecall {
			t.Errorf("probes=%d: candidates %d recall %.2f dropped from %d, %.2f", probes, cand, recall, prevCand, prevRecall)
		}
		prevCand, prevRecall = cand, recall
	}
	_, buckets := ix.WithProbes(1, data.N).Candidates(qs.Row(0))
	if buckets != 2*9 {
		t.Errorf("MinCandidates above N probed %d buckets, want all %d single-bit neighbours in both tables", buckets, 2*9)
	}
}

func TestSampleLevelsAreGeometric(t *testing.T) {
	t.Parallel()
	const n, m = 200_000, 16
	counts := make([]int, hnswMaxLevel+1)
	for _, l := range sampleLevels(n, m, 1) {
		counts[l]++
	}
	for l := range 3 {
		want := n * (1 - 1.0/m) * math.Pow(m, -float64(l))
		if got := float64(counts[l]); math.Abs(got-want) > 5*math.Sqrt(want) {
			t.Errorf("level %d has %d nodes, want about %.0f", l, counts[l], want)
		}
	}
}

func hnswSet() (Matrix, Matrix, *Exact) {
	data := GaussianMixture(3000, 32, 30, 0.6, 1, 2)
	qs := GaussianMixture(40, 32, 30, 0.6, 1, 3)
	return data, qs, NewExact(data)
}

func TestHNSWRecallOnClusteredSet(t *testing.T) {
	t.Parallel()
	data, qs, e := hnswSet()
	testCases := map[string]struct {
		cfg       HNSWConfig
		minRecall float64
	}{
		"ヒューリスティックで辺を選ぶ": {cfg: HNSWConfig{M: 12, EfConstruction: 80, EfSearch: 64, Seed: 1}, minRecall: 0.95},
		"近い順に辺を選ぶ":       {cfg: HNSWConfig{M: 12, EfConstruction: 80, EfSearch: 64, Seed: 1, SimpleNeighbors: true}, minRecall: 0.9},
		"階層なしの NSW":      {cfg: HNSWConfig{M: 12, EfConstruction: 80, EfSearch: 64, Seed: 1, Flat: true}, minRecall: 0.9},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			h := NewHNSW(data, tc.cfg)
			var recall float64
			for i := range qs.N {
				recall += Recall(h.Search(qs.Row(i), 10), e.Search(qs.Row(i), 10))
			}
			if recall /= float64(qs.N); recall < tc.minRecall {
				t.Errorf("recall@10 = %.3f, want at least %.2f", recall, tc.minRecall)
			}
		})
	}
}

// 上の層を外して層 0 の入口から直接探すと、入口から近傍までを層 0 の短い辺でたどるので、ホップと距離計算が増える。
// 128 次元ではグラフの直径がもともと数ホップで差が出にくいので、直径が伸びる 4 次元の一様乱数で確かめる。
func TestHNSWUpperLayersShortenTheWalk(t *testing.T) {
	t.Parallel()
	data, qs := Uniform(5000, 4, 1), Uniform(100, 4, 2)
	h := NewHNSW(data, HNSWConfig{M: 8, EfConstruction: 64, Seed: 1})
	if h.MaxLevel() == 0 {
		t.Fatal("graph has no upper layer to remove")
	}
	flat := h.WithEfSearch(64)
	flat.maxLevel = 0
	var with, without HNSWStats
	for i := range qs.N {
		_, a := h.Greedy(qs.Row(i))
		_, b := flat.Greedy(qs.Row(i))
		with.Hops += a.Hops
		with.Evals += a.Evals
		without.Hops += b.Hops
		without.Evals += b.Evals
	}
	if without.Hops <= with.Hops || without.Evals <= with.Evals {
		t.Errorf("greedy without upper layers %+v, with %+v, want more hops and evaluations without", without, with)
	}
}

func TestHNSWBeamBeatsGreedy(t *testing.T) {
	t.Parallel()
	data, qs, e := hnswSet()
	h := NewHNSW(data, HNSWConfig{M: 8, EfConstruction: 64, EfSearch: 32, Seed: 1})
	greedy, beam := 0, 0
	for i := range qs.N {
		want := e.Search(qs.Row(i), 1)[0].ID
		if g, _ := h.Greedy(qs.Row(i)); g.ID == want {
			greedy++
		}
		if h.Search(qs.Row(i), 1)[0].ID == want {
			beam++
		}
	}
	if beam < greedy || beam < qs.N*9/10 {
		t.Errorf("recall@1 beam %d/%d, greedy %d/%d, want beam high and not below greedy", beam, qs.N, greedy, qs.N)
	}
}

func TestKNNGraphLinksExactNeighbors(t *testing.T) {
	t.Parallel()
	data := RandomNormal(400, 8, 1)
	g := NewKNNGraph(data, 5)
	for i := range data.N {
		ns := g.neighbors(int32(i), 0)
		want := naiveKNN(data, data.Row(i), 6)
		if len(ns) != 5 || slices.Contains(ns, int32(i)) {
			t.Fatalf("node %d links %v, want 5 others", i, ns)
		}
		for _, id := range ns {
			if !slices.ContainsFunc(want, func(n Neighbor) bool { return n.ID == id }) {
				t.Fatalf("node %d links %d which is not among its 5 nearest", i, id)
			}
		}
	}
}
