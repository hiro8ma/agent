// dedupbench は、ブルームフィルタの偽陽性率とメモリ、既読の除外の大きさ、MinHash の推定の誤差、LSH の候補の数、2回目からキャッシュに入れる効果を、種で固定した合成のデータで測る。
// 文書は、かなを並べた架空の語から作る。ネットワークには出ない。
package main

import (
	"container/list"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/dedup"
)

func main() {
	var (
		seed    = flag.Uint64("seed", 1, "乱数の種")
		docs    = flag.Int("docs", 10_000, "表 G の文書の数（半分が元の文書、半分がその編集）")
		shingle = flag.Int("shingle", 5, "文字の k-グラムの k")
	)
	flag.Parse()
	start := time.Now()
	w := os.Stdout
	bloomTable(w, *seed)
	readHistory(w, *seed)
	corpus := newCorpus(*seed)
	minhashTable(w, corpus, *shingle, *seed)
	lshTable(w, corpus, *shingle, *docs, *seed)
	cacheTable(w, *seed)
	fmt.Fprintf(w, "\n所要時間は%.2f秒\n", time.Since(start).Seconds())
}

func bloomTable(w io.Writer, seed uint64) {
	const queries = 1_000_000
	fmt.Fprintf(w, "## 表 D ブルームフィルタ\n\n")
	fmt.Fprintf(w, "- m 個の乱数を入れ、入れていない%d個を問い合わせて偽陽性率を数える（種%d）。k は1、最適、最適の2倍\n", queries, seed)
	fmt.Fprintf(w, "- 式は (1−e^(−km/N))^k、最適の k は (N/m) ln 2 の四捨五入\n\n")
	fmt.Fprintln(w, "| m | 1要素のビット | N | k | 式の偽陽性率 | 実測の偽陽性率 | メモリ |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|")
	for _, m := range []int{10_000, 1_000_000} {
		for _, bpe := range []uint64{4, 8, 10, 16} {
			bits := bpe * uint64(m)
			opt := dedup.OptimalK(bits, m)
			for _, k := range []int{1, opt, 2 * opt} {
				r := rand.New(rand.NewPCG(seed, uint64(m)))
				b := dedup.NewBloom(bits, k)
				for range m {
					b.Add(r.Uint64() >> 1)
				}
				fp := 0
				for range queries {
					if b.Contains(r.Uint64() | 1<<63) {
						fp++
					}
				}
				label := fmt.Sprint(k)
				if k == opt {
					label += "（最適）"
				}
				fmt.Fprintf(w, "| %d | %d | %d | %s | %.5f | %.5f | %s |\n",
					m, bpe, bits, label, dedup.FalsePositiveRate(bits, k, m), float64(fp)/queries, size(b.Bytes()))
			}
		}
	}
}

func readHistory(w io.Writer, seed uint64) {
	const users, candidates = 2_000, 100
	fmt.Fprintf(w, "\n## 表 E 既読の除外の大きさ\n\n")
	fmt.Fprintf(w, "- 利用者ごとに既読の記事 ID（8バイト）を n 件入れたフィルタを作り、未読の候補%d件を問い合わせる。%d人で数える（種%d）\n", candidates, users, seed)
	fmt.Fprintf(w, "- フィルタのバイト数は64ビットの語に切り上げた実際の大きさ。誤って外す件数は、候補%d件のうち偽陽性で既読とみなす件数の平均\n\n", candidates)
	fmt.Fprintln(w, "| 既読の件数 | 目標の偽陽性率 | N | k | フィルタのバイト数 | ID をそのまま持つバイト数 | 比 | 式の偽陽性率 | 実測の偽陽性率 | 候補100件で誤って外す件数 | 100万人のフィルタ | 100万人の ID |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|---|---|---|")
	for _, n := range []int{50, 200, 1_000} {
		for _, p := range []float64{0.01, 0.001} {
			bits, k := dedup.BloomParams(n, p)
			r := rand.New(rand.NewPCG(seed, uint64(n)))
			fp := 0
			var bytes int
			for range users {
				b := dedup.NewBloom(bits, k)
				for range n {
					b.Add(r.Uint64() >> 1)
				}
				for range candidates {
					if b.Contains(r.Uint64() | 1<<63) {
						fp++
					}
				}
				bytes = b.Bytes()
			}
			rate := float64(fp) / (users * candidates)
			raw := 8 * n
			fmt.Fprintf(w, "| %d | %.1f%% | %d | %d | %d | %d | %.2f | %.5f | %.5f | %.3f | %s | %s |\n",
				n, p*100, bits, k, bytes, raw, float64(bytes)/float64(raw), dedup.FalsePositiveRate(bits, k, n), rate, rate*candidates,
				size(bytes*1_000_000), size(raw*1_000_000))
		}
	}
}

// corpus は合成の文書を作る。語はかなを1〜4文字並べた架空のもので、使う頻度は語の順位の逆数に比例させる。
type corpus struct {
	vocab []string
	cdf   []float64
	rng   *rand.Rand
}

var kana = []rune("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ")

func newCorpus(seed uint64) *corpus {
	r := rand.New(rand.NewPCG(seed, 3))
	c := &corpus{rng: r}
	seen := map[string]bool{}
	for len(c.vocab) < 5_000 {
		n := 1 + r.IntN(4)
		var b strings.Builder
		for range n {
			b.WriteRune(kana[r.IntN(len(kana))])
		}
		if s := b.String(); !seen[s] {
			seen[s] = true
			c.vocab = append(c.vocab, s)
		}
	}
	c.cdf = zipfCDF(len(c.vocab), 1.0)
	return c
}

// document は8文の文書を作る。1文は6〜14語で、句点で終える。
func (c *corpus) document() string {
	var b strings.Builder
	for range 8 {
		for range 6 + c.rng.IntN(9) {
			b.WriteString(c.vocab[sample(c.cdf, c.rng)])
		}
		b.WriteString("。")
	}
	return b.String()
}

// edit は句点以外の文字を、それぞれ確率 p で別のかなに置き換える。
func (c *corpus) edit(doc string, p float64) string {
	runes := []rune(doc)
	for i, ch := range runes {
		if ch != '。' && c.rng.Float64() < p {
			for {
				repl := kana[c.rng.IntN(len(kana))]
				if repl != ch {
					runes[i] = repl
					break
				}
			}
		}
	}
	return string(runes)
}

func zipfCDF(n int, s float64) []float64 {
	cdf := make([]float64, n)
	var sum float64
	for i := range cdf {
		sum += math.Pow(float64(i+1), -s)
		cdf[i] = sum
	}
	for i := range cdf {
		cdf[i] /= sum
	}
	return cdf
}

func sample(cdf []float64, r *rand.Rand) int {
	i, _ := slices.BinarySearch(cdf, r.Float64())
	return min(i, len(cdf)-1)
}

func shingleSet(doc string, k int) dedup.Set { return dedup.NewSet(dedup.CharShingles(doc, k)) }

func minhashTable(w io.Writer, c *corpus, k int, seed uint64) {
	const pairs = 500
	rates := []float64{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3}
	ks := []int{16, 64, 256}
	mh := dedup.NewMinHasher(256, seed)
	fmt.Fprintf(w, "\n## 表 F MinHash\n\n")
	fmt.Fprintf(w, "- 合成の文書（架空の語5,000個から8文、平均%d文字ほど）%d件に、句点以外の文字を割合 p で別のかなに置き換える編集を入れ、元の文書と組にする（種%d）\n", avgLen(c, 200), pairs, seed)
	fmt.Fprintf(w, "- 断片は文字の%d-グラム。誤差は推定 − 厳密な J。理論の標準偏差は √(J(1−J)/K) を組で二乗平均した値\n\n", k)
	fmt.Fprintln(w, "| 置き換えの割合 p | 厳密な J の平均 | K | 誤差の平均 | 誤差の95%の幅 | 誤差の標準偏差 | 理論の標準偏差 |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|")
	for _, p := range rates {
		js := make([]float64, pairs)
		sigs := make([][2][]uint64, pairs)
		for i := range pairs {
			doc := c.document()
			a, b := shingleSet(doc, k), shingleSet(c.edit(doc, p), k)
			js[i] = dedup.Jaccard(a, b)
			sigs[i] = [2][]uint64{mh.Signature(a), mh.Signature(b)}
		}
		for _, kk := range ks {
			errs := make([]float64, pairs)
			var sum, sq, theory float64
			for i := range pairs {
				e := dedup.EstimateJaccard(sigs[i][0][:kk], sigs[i][1][:kk]) - js[i]
				errs[i] = e
				sum += e
				sq += e * e
				theory += js[i] * (1 - js[i]) / float64(kk)
			}
			slices.Sort(errs)
			mean := sum / pairs
			fmt.Fprintf(w, "| %.3f | %.3f | %d | %+.4f | %+.3f〜%+.3f | %.4f | %.4f |\n",
				p, average(js), kk, mean, errs[pairs*25/1000], errs[pairs*975/1000-1], math.Sqrt(sq/pairs-mean*mean), math.Sqrt(theory/pairs))
		}
	}
}

func avgLen(c *corpus, n int) int {
	r := *c.rng
	tmp := &corpus{vocab: c.vocab, cdf: c.cdf, rng: &r}
	total := 0
	for range n {
		total += len([]rune(tmp.document()))
	}
	return total / n
}

func average(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

var lshConfigs = []struct {
	name string
	lsh  dedup.LSH
}{
	{name: "帯 b=16、r=4", lsh: dedup.LSH{Bands: 16, Rows: 4}},
	{name: "帯 b=8、r=8", lsh: dedup.LSH{Bands: 8, Rows: 8}},
	{name: "帯 b=32、r=8", lsh: dedup.LSH{Bands: 32, Rows: 8}},
	{name: "次元ごとの転置 m=64", lsh: dedup.LSH{Bands: 64, Rows: 1}},
	{name: "次元ごとの転置 m=256", lsh: dedup.LSH{Bands: 256, Rows: 1}},
}

func lshTable(w io.Writer, c *corpus, k, nDocs int, seed uint64) {
	mh := dedup.NewMinHasher(256, seed+1)
	fmt.Fprintf(w, "\n## 表 G LSH\n\n")
	fmt.Fprintf(w, "- 署名は K=256 で作り、先頭の b × r 個を使う。次元ごとの転置は、次元 i の値ごとに文書の一覧を持ち、どれか1次元で一致すれば候補にする方式で、b=m、r=1 に当たる（式は 1−(1−J)^m）\n")
	fmt.Fprintf(w, "- 断片は文字の%d-グラム（種%d）\n\n", k, seed)

	const edited, unrelated = 3_000, 2_000
	type pair struct {
		j    float64
		a, b []uint64
	}
	var pairs []pair
	var unrelatedSets [][2]dedup.Set
	for i := range edited + unrelated {
		doc := c.document()
		other := c.document()
		if i < edited {
			other = c.edit(doc, 0.5*c.rng.Float64())
		}
		a, b := shingleSet(doc, k), shingleSet(other, k)
		if i >= edited {
			unrelatedSets = append(unrelatedSets, [2]dedup.Set{a, b})
		}
		pairs = append(pairs, pair{j: dedup.Jaccard(a, b), a: mh.Signature(a), b: mh.Signature(b)})
	}
	fmt.Fprintf(w, "### ジャッカード係数ごとの候補になる割合\n\n")
	fmt.Fprintf(w, "- 元の文書と、割合 p（0〜0.5の一様）で置き換えた文書の組%d個と、関係の無い文書の組%d個。J を0.1刻みで分け、各欄は 式 / 実測\n\n", edited, unrelated)
	var header strings.Builder
	header.WriteString("| J | 組 |")
	for _, cfg := range lshConfigs {
		header.WriteString(" " + cfg.name + " |")
	}
	fmt.Fprintln(w, header.String())
	fmt.Fprintln(w, "|---|---|"+strings.Repeat("---|", len(lshConfigs)))
	for bin := range 10 {
		lo, hi := float64(bin)/10, float64(bin+1)/10
		var in []pair
		for _, p := range pairs {
			if p.j >= lo && (p.j < hi || (bin == 9 && p.j <= 1)) {
				in = append(in, p)
			}
		}
		if len(in) == 0 {
			continue
		}
		var row strings.Builder
		fmt.Fprintf(&row, "| %.1f〜%.1f | %d |", lo, hi, len(in))
		for _, cfg := range lshConfigs {
			var formula float64
			hit := 0
			for _, p := range in {
				formula += cfg.lsh.CandidateProbability(p.j)
				if cfg.lsh.Match(p.a, p.b) {
					hit++
				}
			}
			fmt.Fprintf(&row, " %.3f / %.3f |", formula/float64(len(in)), float64(hit)/float64(len(in)))
		}
		fmt.Fprintln(w, row.String())
	}
	var unrelatedJ float64
	for _, p := range pairs[edited:] {
		unrelatedJ += p.j
	}
	lo, hi := math.Inf(1), 0.0
	for s := range uint64(10) {
		other := dedup.NewMinHasher(64, seed*1_000+s)
		var match float64
		for _, sets := range unrelatedSets {
			match += dedup.EstimateJaccard(other.Signature(sets[0]), other.Signature(sets[1]))
		}
		match /= float64(len(unrelatedSets))
		lo, hi = min(lo, match), max(hi, match)
	}
	fmt.Fprintf(w, "\n- 関係の無い組%d個の厳密な J の平均は%.4fで、1次元あたりの一致の割合は、署名の種を10通り変えると%.4f〜%.4fに動いた。どの組も同じ頻出の断片で一致するので、組を増やしても一致の割合のばらつきは減らない。J が0.1未満の行の 式 と 実測 の差は、この種によるずれを含む\n", len(unrelatedSets), unrelatedJ/float64(len(unrelatedSets)), lo, hi)

	half := nDocs / 2
	sets := make([]dedup.Set, nDocs)
	js := make([]float64, half)
	for i := range half {
		doc := c.document()
		sets[i] = shingleSet(doc, k)
		sets[half+i] = shingleSet(c.edit(doc, 0.3*c.rng.Float64()), k)
		js[i] = dedup.Jaccard(sets[i], sets[half+i])
	}
	sigs := make([][]uint64, nDocs)
	for i, s := range sets {
		sigs[i] = mh.Signature(s)
	}
	var bg []float64
	for range 2_000 {
		i, j := c.rng.IntN(nDocs), c.rng.IntN(nDocs)
		if i%half != j%half {
			bg = append(bg, dedup.Jaccard(sets[i], sets[j]))
		}
	}
	slices.Sort(bg)
	near := 0
	for _, j := range js {
		if j >= 0.5 {
			near++
		}
	}
	all := nDocs * (nDocs - 1) / 2
	fmt.Fprintf(w, "\n### %d件の文書の候補の組\n\n", nDocs)
	fmt.Fprintf(w, "- 元の文書%d件と、それぞれを割合 p（0〜0.3の一様）で置き換えた%d件。全部の組は%d組\n", half, half, all)
	fmt.Fprintf(w, "- 作った組のうち J ≥ 0.5は%d組。関係の無い組%d個の J は平均%.4f、99パーセンタイル%.4f、最大%.4f\n\n", near, len(bg), average(bg), bg[len(bg)*99/100], bg[len(bg)-1])
	fmt.Fprintln(w, "| 方式 | b | r | 使う次元 | 候補の組 | 全部の組に対する割合 | J ≥ 0.5の組のうち候補になった割合 | 候補のうち作った組の割合 |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|")
	for _, cfg := range lshConfigs {
		count, nearFound, made := 0, 0, 0
		cfg.lsh.Candidates(sigs, func(i, j int) {
			count++
			if j == i+half {
				made++
				if js[i] >= 0.5 {
					nearFound++
				}
			}
		})
		fmt.Fprintf(w, "| %s | %d | %d | %d | %d | %.4f%% | %.4f | %.4f |\n",
			cfg.name, cfg.lsh.Bands, cfg.lsh.Rows, cfg.lsh.Bands*cfg.lsh.Rows, count, 100*float64(count)/float64(all),
			float64(nearFound)/float64(near), float64(made)/float64(max(count, 1)))
	}
}

type lru struct {
	cap   int
	order *list.List
	items map[uint64]*list.Element
}

func newLRU(capacity int) *lru {
	return &lru{cap: capacity, order: list.New(), items: map[uint64]*list.Element{}}
}

func (c *lru) get(key uint64) bool {
	if e, ok := c.items[key]; ok {
		c.order.MoveToFront(e)
		return true
	}
	return false
}

func (c *lru) put(key uint64) {
	if c.order.Len() >= c.cap {
		old := c.order.Back()
		c.order.Remove(old)
		delete(c.items, old.Value.(uint64))
	}
	c.items[key] = c.order.PushFront(key)
}

func cacheTable(w io.Writer, seed uint64) {
	const catalog, requests, zipfS, fpTarget = 100_000, 2_000_000, 0.9, 0.01
	cdf := zipfCDF(catalog, zipfS)
	fmt.Fprintf(w, "\n## 表 H 2回目でキャッシュ\n\n")
	fmt.Fprintf(w, "- %d個の対象から指数%.1fの Zipf で引いた要求%d回に、1回しか来ない対象を割合 q で混ぜる（種%d）\n", catalog, zipfS, requests, seed)
	fmt.Fprintf(w, "- キャッシュは LRU。2回目から入れる方は、見た対象をブルームフィルタ（要求の回数と同じ個数を入れて偽陽性率%.0f%%になる大きさ）に入れ、フィルタに既にあった対象だけをキャッシュに入れる。フィルタは途中で空にしない\n\n", fpTarget*100)
	fmt.Fprintln(w, "| 1回だけの対象の割合 q | キャッシュの大きさ | 入れ方 | ヒット率 | キャッシュへの書き込み | 1回だけの対象の書き込み | 書き込み1回あたりのヒット | フィルタのメモリ |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|")
	for _, q := range []float64{0.25, 0.5} {
		r := rand.New(rand.NewPCG(seed, uint64(q*100)))
		stream := make([]uint64, requests)
		next := uint64(1 << 40)
		for i := range stream {
			if r.Float64() < q {
				stream[i] = next
				next++
			} else {
				stream[i] = uint64(sample(cdf, r))
			}
		}
		for _, capacity := range []int{1_000, 10_000} {
			for _, second := range []bool{false, true} {
				cache := newLRU(capacity)
				var seen *dedup.Bloom
				memory := "なし"
				if second {
					bits, k := dedup.BloomParams(requests, fpTarget)
					seen = dedup.NewBloom(bits, k)
					memory = size(seen.Bytes())
				}
				hits, writes, oneHitWrites := 0, 0, 0
				for _, key := range stream {
					if cache.get(key) {
						hits++
						continue
					}
					if second {
						if !seen.Contains(key) {
							seen.Add(key)
							continue
						}
					}
					cache.put(key)
					writes++
					if key >= 1<<40 {
						oneHitWrites++
					}
				}
				policy := "1回目から"
				if second {
					policy = "2回目から"
				}
				fmt.Fprintf(w, "| %.2f | %d | %s | %.4f | %d | %d | %.2f | %s |\n",
					q, capacity, policy, float64(hits)/requests, writes, oneHitWrites, float64(hits)/float64(writes), memory)
			}
		}
	}
}

func size(bytes int) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.2fGiB", float64(bytes)/(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.2fMiB", float64(bytes)/(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.1fKiB", float64(bytes)/(1<<10))
	}
	return fmt.Sprintf("%dB", bytes)
}
