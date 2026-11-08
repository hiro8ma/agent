package dedup_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"

	"github.com/hiro8ma/agent/go/internal/crawl"
	"github.com/hiro8ma/agent/go/internal/dedup"
)

func TestText(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		page string
		want string
	}{
		"タグを外して空白をまとめる":          {page: "<p>今日は\n\n  <b>晴れ</b>   です</p>", want: "今日は 晴れ です"},
		"script と style の中身を捨てる": {page: "<style>p{color:red}</style><p>本文</p><script>var x = 1;</script>", want: "本文"},
		"文字参照を戻す":                {page: "<p>A &amp; B</p>", want: "A & B"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := dedup.Text([]byte(tc.page)); got != tc.want {
				t.Errorf("Text = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFingerprint(t *testing.T) {
	t.Parallel()
	base := "<html><body><h1>お知らせ</h1><p>新しい記事を公開しました</p></body></html>"
	testCases := map[string]struct {
		other string
		same  bool
	}{
		"タグと空白だけが違えば同じ":  {other: "<div>\n  <h2>お知らせ</h2>\n<span>新しい記事を公開しました</span>\n</div>", same: true},
		"script を足しても同じ": {other: base + "<script>track()</script>", same: true},
		"本文の1文字が違えば違う":   {other: "<h1>お知らせ</h1><p>新しい記事を公開しました!</p>", same: false},
		"語の区切りが違えば違う":    {other: "<h1>お知</h1><p>らせ新しい記事を公開しました</p>", same: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := dedup.Fingerprint([]byte(base)) == dedup.Fingerprint([]byte(tc.other)); got != tc.same {
				t.Errorf("同じハッシュか = %v, want %v", got, tc.same)
			}
		})
	}
}

func TestExact(t *testing.T) {
	t.Parallel()
	e := dedup.NewExact(crawl.Normalizer{DropParams: []string{"utm_*"}})
	type add struct {
		url, page string
		kind      dedup.Kind
		first     string
	}
	steps := []add{
		{url: "https://example.com/a", page: "<p>記事 A</p>", kind: dedup.New, first: "https://example.com/a"},
		{url: "https://EXAMPLE.com/a?utm_source=mail#top", page: "<p>記事 A（別の版）</p>", kind: dedup.SameURL, first: "https://example.com/a"},
		{url: "https://example.com/print/a", page: "<div>記事\nA</div>", kind: dedup.SameContent, first: "https://example.com/a"},
		{url: "https://example.com/b", page: "<p>記事 B</p>", kind: dedup.New, first: "https://example.com/b"},
	}
	for i, s := range steps {
		kind, first, err := e.Add(s.url, []byte(s.page))
		if err != nil {
			t.Fatalf("%d: %v", i, err)
		}
		if kind != s.kind || first != s.first {
			t.Errorf("%d: Add(%q) = %v, %q, want %v, %q", i, s.url, kind, first, s.kind, s.first)
		}
	}
}

func TestBloomParams(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		m        int
		p        float64
		wantBits uint64
		wantK    int
	}{
		"1,000件で1%":   {m: 1_000, p: 0.01, wantBits: 9_586, wantK: 7},
		"1,000件で0.1%": {m: 1_000, p: 0.001, wantBits: 14_378, wantK: 10},
		"1件で50%":      {m: 1, p: 0.5, wantBits: 2, wantK: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			bits, k := dedup.BloomParams(tc.m, tc.p)
			if bits != tc.wantBits || k != tc.wantK {
				t.Errorf("BloomParams(%d, %v) = %d, %d, want %d, %d", tc.m, tc.p, bits, k, tc.wantBits, tc.wantK)
			}
			if got := dedup.FalsePositiveRate(bits, k, tc.m); got > tc.p*1.1 {
				t.Errorf("決めた N と k の偽陽性率 %.5f が目標 %v を10%%より超える", got, tc.p)
			}
		})
	}
}

func TestFalsePositiveRateFormula(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		bits uint64
		k, m int
		want float64
	}{
		"k=1": {bits: 10, k: 1, m: 1, want: 1 - math.Exp(-0.1)},
		"k=2": {bits: 100, k: 2, m: 10, want: math.Pow(1-math.Exp(-0.2), 2)},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := dedup.FalsePositiveRate(tc.bits, tc.k, tc.m); math.Abs(got-tc.want) > 1e-15 {
				t.Errorf("FalsePositiveRate = %v, want %v", got, tc.want)
			}
		})
	}
}

// 入れる値と問い合わせる値は、重ならないように別の範囲から引く。
func fillBloom(bits uint64, k, m int, seed uint64) (*dedup.Bloom, []uint64) {
	r := rand.New(rand.NewPCG(seed, 1))
	b := dedup.NewBloom(bits, k)
	in := make([]uint64, m)
	for i := range in {
		in[i] = r.Uint64() >> 1
		b.Add(in[i])
	}
	return b, in
}

func TestBloomNoFalseNegatives(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		bitsPerElement uint64
		k              int
	}{
		"1要素4ビットで k=3":  {bitsPerElement: 4, k: 3},
		"1要素10ビットで k=7": {bitsPerElement: 10, k: 7},
		"1要素16ビットで k=1": {bitsPerElement: 16, k: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			const m = 20_000
			b, in := fillBloom(tc.bitsPerElement*m, tc.k, m, 3)
			for _, x := range in {
				if !b.Contains(x) {
					t.Fatalf("入れた %d を含まないと答えた", x)
				}
			}
		})
	}
}

// 許す差は式の値の相対15%。問い合わせ20万件で、偽陽性の件数の標準偏差は期待値の平方根なので、0.4%で800件のとき相対3.5%。
func TestBloomFalsePositiveRateNearFormula(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		bitsPerElement uint64
		k              int
	}{
		"1要素8ビットで最適の k=6":    {bitsPerElement: 8, k: 6},
		"1要素10ビットで最適の k=7":   {bitsPerElement: 10, k: 7},
		"1要素10ビットで少ない k=2":   {bitsPerElement: 10, k: 2},
		"1要素16ビットで多すぎる k=20": {bitsPerElement: 16, k: 20},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			const m, queries = 10_000, 200_000
			bits := tc.bitsPerElement * m
			b, _ := fillBloom(bits, tc.k, m, 5)
			r := rand.New(rand.NewPCG(9, 9))
			fp := 0
			for range queries {
				if b.Contains(r.Uint64() | 1<<63) {
					fp++
				}
			}
			got := float64(fp) / queries
			want := dedup.FalsePositiveRate(bits, tc.k, m)
			if math.Abs(got-want) > 0.15*want {
				t.Errorf("偽陽性率 %.5f, 式 %.5f", got, want)
			}
		})
	}
}

// 1人あたり数十件の既読を入れる小さなフィルタを多数作り、平均の偽陽性率を見る。50万件の問い合わせで、0.1%の偽陽性の件数の標準偏差は期待値の4.5%。
func TestSmallBloomFalsePositiveRateNearFormula(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		m int
		p float64
	}{
		"50件で0.1%":  {m: 50, p: 0.001},
		"50件で1%":    {m: 50, p: 0.01},
		"200件で0.1%": {m: 200, p: 0.001},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			const filters, queries = 5_000, 100
			bits, k := dedup.BloomParams(tc.m, tc.p)
			r := rand.New(rand.NewPCG(13, uint64(tc.m)))
			fp := 0
			for range filters {
				b := dedup.NewBloom(bits, k)
				for range tc.m {
					b.Add(r.Uint64() >> 1)
				}
				for range queries {
					if b.Contains(r.Uint64() | 1<<63) {
						fp++
					}
				}
			}
			got := float64(fp) / (filters * queries)
			want := dedup.FalsePositiveRate(bits, k, tc.m)
			if math.Abs(got-want) > 0.15*want {
				t.Errorf("N=%d、k=%d の偽陽性率 %.5f, 式 %.5f", bits, k, got, want)
			}
		})
	}
}

func TestShingles(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		got  []string
		want []string
	}{
		"文字の2-グラムは重なりを含めて前から並べる": {got: dedup.CharShingles("abcab", 2), want: []string{"ab", "bc", "ca", "ab"}},
		"日本語は rune で数える":         {got: dedup.CharShingles("東京都庁", 2), want: []string{"東京", "京都", "都庁"}},
		"k 文字に満たなければ全体を1つにする":    {got: dedup.CharShingles("東", 3), want: []string{"東"}},
		"空の文字列は断片を作らない":          {got: dedup.CharShingles("", 3), want: nil},
		"単語の2-グラム":               {got: dedup.WordShingles("the cat sat on the mat", 2), want: []string{"the cat", "cat sat", "sat on", "on the", "the mat"}},
		"単語は空白の並びで区切る":           {got: dedup.WordShingles("a  b\nc", 3), want: []string{"a b c"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if !slices.Equal(tc.got, tc.want) {
				t.Errorf("= %q, want %q", tc.got, tc.want)
			}
		})
	}
}

func TestJaccard(t *testing.T) {
	t.Parallel()
	set := func(items ...string) dedup.Set { return dedup.NewSet(items) }
	testCases := map[string]struct {
		a, b dedup.Set
		want float64
	}{
		"重複を除いた {ab bc ca} と {ab bc cd} は2/4": {a: dedup.NewSet(dedup.CharShingles("abcab", 2)), b: set("ab", "bc", "cd"), want: 0.5},
		"あいうえお と あいうかお の2-グラムは2/6":            {a: dedup.NewSet(dedup.CharShingles("あいうえお", 2)), b: dedup.NewSet(dedup.CharShingles("あいうかお", 2)), want: 1.0 / 3},
		"単語の2-グラムで1つだけ共通なら1/5":                {a: dedup.NewSet(dedup.WordShingles("a b c d", 2)), b: dedup.NewSet(dedup.WordShingles("a b x y", 2)), want: 1.0 / 5},
		"同じ集合は1":   {a: set("x", "y"), b: set("y", "x"), want: 1},
		"共通が無ければ0": {a: set("x"), b: set("y"), want: 0},
		"両方が空なら1":  {a: set(), b: set(), want: 1},
		"片方だけ空なら0": {a: set("x"), b: set(), want: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := dedup.Jaccard(tc.a, tc.b); math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("Jaccard = %v, want %v", got, tc.want)
			}
		})
	}
}

// pairWithJaccard はジャッカード係数がちょうど shared / (2n − shared) になる2つの集合を、label ごとに別の要素で作る。
func pairWithJaccard(n, shared int, label int) (dedup.Set, dedup.Set) {
	a := make([]string, 0, n)
	b := make([]string, 0, n)
	for i := range n {
		a = append(a, fmt.Sprintf("%d-a%d", label, i))
		if i < shared {
			b = append(b, fmt.Sprintf("%d-a%d", label, i))
		} else {
			b = append(b, fmt.Sprintf("%d-b%d", label, i))
		}
	}
	return dedup.NewSet(a), dedup.NewSet(b)
}

func TestMinHashApproachesJaccard(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		shared int
	}{
		"J=0.2": {shared: 100},
		"J=0.5": {shared: 200},
		"J=0.8": {shared: 267},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			const n, pairs = 300, 300
			mh := dedup.NewMinHasher(256, 1)
			prevRMS := math.Inf(1)
			for _, k := range []int{16, 64, 256} {
				var sq float64
				var j float64
				for p := range pairs {
					a, b := pairWithJaccard(n, tc.shared, p)
					j = dedup.Jaccard(a, b)
					est := dedup.EstimateJaccard(mh.Signature(a)[:k], mh.Signature(b)[:k])
					sq += (est - j) * (est - j)
				}
				rms := math.Sqrt(sq / pairs)
				theory := math.Sqrt(j * (1 - j) / float64(k))
				if rms < 0.75*theory || rms > 1.25*theory {
					t.Errorf("K=%d: 誤差の二乗平均の平方根 %.4f が理論の標準偏差 %.4f から25%%より外れる", k, rms, theory)
				}
				if rms >= prevRMS {
					t.Errorf("K=%d の誤差 %.4f が、小さい K の %.4f より小さくならない", k, rms, prevRMS)
				}
				prevRMS = rms
			}
		})
	}
}

func TestMinHashExtremes(t *testing.T) {
	t.Parallel()
	mh := dedup.NewMinHasher(128, 7)
	a, b := pairWithJaccard(200, 200, 0)
	if got := dedup.EstimateJaccard(mh.Signature(a), mh.Signature(b)); got != 1 {
		t.Errorf("同じ集合の推定 = %v, want 1", got)
	}
	c, d := pairWithJaccard(200, 0, 1)
	if got := dedup.EstimateJaccard(mh.Signature(c), mh.Signature(d)); got > 0.02 {
		t.Errorf("共通の無い集合の推定 = %v", got)
	}
}

// 各位置の値は集合の要素のハッシュの最小値なので、要素を足すと値は下がるか変わらない。2つの集合の和の署名は、署名の位置ごとの小さい方になる。
func TestMinHashSignatureOfUnion(t *testing.T) {
	t.Parallel()
	mh := dedup.NewMinHasher(64, 3)
	a, b := pairWithJaccard(50, 10, 0)
	union := dedup.Set{}
	for x := range a {
		union[x] = struct{}{}
	}
	for x := range b {
		union[x] = struct{}{}
	}
	sa, sb, su := mh.Signature(a), mh.Signature(b), mh.Signature(union)
	for i := range su {
		if su[i] != min(sa[i], sb[i]) {
			t.Fatalf("位置 %d: 和の署名 %d が、2つの署名の小さい方 %d と違う", i, su[i], min(sa[i], sb[i]))
		}
	}
}

func TestLSHCandidateProbability(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lsh  dedup.LSH
		j    float64
		want float64
	}{
		"b=16 r=4 で J=0.5":  {lsh: dedup.LSH{Bands: 16, Rows: 4}, j: 0.5, want: 1 - math.Pow(1-0.0625, 16)},
		"b=64 r=1 で J=0.05": {lsh: dedup.LSH{Bands: 64, Rows: 1}, j: 0.05, want: 1 - math.Pow(0.95, 64)},
		"J=0なら0":            {lsh: dedup.LSH{Bands: 16, Rows: 4}, j: 0, want: 0},
		"J=1なら1":            {lsh: dedup.LSH{Bands: 16, Rows: 4}, j: 1, want: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := tc.lsh.CandidateProbability(tc.j); math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("CandidateProbability = %v, want %v", got, tc.want)
			}
		})
	}
}

// 許す差は0.05。1,500組の割合の標準偏差は0.013以下。
func TestLSHCandidateRateNearFormula(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lsh    dedup.LSH
		shared int
	}{
		"b=16 r=4 で J=0.33":  {lsh: dedup.LSH{Bands: 16, Rows: 4}, shared: 50},
		"b=16 r=4 で J=0.5":   {lsh: dedup.LSH{Bands: 16, Rows: 4}, shared: 67},
		"b=16 r=4 で J=0.67":  {lsh: dedup.LSH{Bands: 16, Rows: 4}, shared: 80},
		"b=8 r=8 で J=0.82":   {lsh: dedup.LSH{Bands: 8, Rows: 8}, shared: 90},
		"b=64 r=1 で J=0.01":  {lsh: dedup.LSH{Bands: 64, Rows: 1}, shared: 2},
		"b=256 r=1 で J=0.01": {lsh: dedup.LSH{Bands: 256, Rows: 1}, shared: 2},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			const n, pairs = 100, 1_500
			mh := dedup.NewMinHasher(tc.lsh.Bands*tc.lsh.Rows, 11)
			hit := 0
			var j float64
			for p := range pairs {
				a, b := pairWithJaccard(n, tc.shared, p)
				j = dedup.Jaccard(a, b)
				if tc.lsh.Match(mh.Signature(a), mh.Signature(b)) {
					hit++
				}
			}
			got := float64(hit) / pairs
			if want := tc.lsh.CandidateProbability(j); math.Abs(got-want) > 0.05 {
				t.Errorf("J=%.3f の候補の割合 %.3f, 式 %.3f", j, got, want)
			}
		})
	}
}

func TestLSHCandidatesMatchesAllPairs(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lsh dedup.LSH
	}{
		"帯を4行ずつ": {lsh: dedup.LSH{Bands: 8, Rows: 4}},
		"帯を1行ずつ": {lsh: dedup.LSH{Bands: 32, Rows: 1}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			mh := dedup.NewMinHasher(32, 5)
			var sigs [][]uint64
			for d := range 60 {
				a, b := pairWithJaccard(20, d%20, d/3)
				sigs = append(sigs, mh.Signature(a), mh.Signature(b))
			}
			want := map[[2]int]bool{}
			for i := range sigs {
				for j := i + 1; j < len(sigs); j++ {
					if tc.lsh.Match(sigs[i], sigs[j]) {
						want[[2]int{i, j}] = true
					}
				}
			}
			calls := 0
			got := map[[2]int]bool{}
			tc.lsh.Candidates(sigs, func(i, j int) {
				calls++
				got[[2]int{i, j}] = true
			})
			if calls != len(got) {
				t.Errorf("同じ組を2回以上渡した: %d 回で %d 組", calls, len(got))
			}
			if len(want) == 0 || len(got) != len(want) {
				t.Fatalf("候補 %d 組, 全組の照合 %d 組", len(got), len(want))
			}
			for p := range want {
				if !got[p] {
					t.Errorf("組 %v が候補に無い", p)
				}
			}
		})
	}
}

func TestKey(t *testing.T) {
	t.Parallel()
	seen := map[uint64]string{}
	for i := range 10_000 {
		s := "article-" + strconv.Itoa(i)
		k := dedup.Key([]byte(s))
		if prev, ok := seen[k]; ok {
			t.Fatalf("%q と %q が同じ値 %d", prev, s, k)
		}
		seen[k] = s
	}
}
