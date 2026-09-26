package search

import (
	"cmp"
	"fmt"
	"math/bits"
	"math/rand/v2"
	"slices"
	"testing"
)

var memorySizes = []int{4 << 10, 32 << 10, 256 << 10, 2 << 20, 16 << 20, 128 << 20, 512 << 20}

// shuffledIndexes は 0 から n-1 を並べ替えた列を返す。r.Perm は []int を作り、大きな n で倍のメモリを使うので int32 のまま混ぜる。
func shuffledIndexes(r *rand.Rand, n int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32(i)
	}
	r.Shuffle(n, func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p
}

// cycle は next[i] をたどると全要素を 1 周する並び（Sattolo の方法）を作る。次に読む場所が前の読み出しの値で決まるので、読み出しが重ならない。
func cycle(r *rand.Rand, n int) []int32 {
	next := make([]int32, n)
	for i := range next {
		next[i] = int32(i)
	}
	for i := n - 1; i > 0; i-- {
		j := r.IntN(i)
		next[i], next[j] = next[j], next[i]
	}
	return next
}

var sink int64

// BenchmarkMemoryRead は int32 の配列を、順に読む / 事前に作った並べ替えの順に読む / 読んだ値を次の添字にしてたどる、の 3 通りで読み、1 要素あたりの時間を配列の大きさごとに出す。
//
//	go test -run '^$' -bench MemoryRead ./internal/search/
func BenchmarkMemoryRead(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, size := range memorySizes {
		n := size / 4
		a := make([]int32, n)
		for i := range a {
			a[i] = int32(i)
		}
		perm := shuffledIndexes(r, n)
		next := cycle(r, n)
		modes := []struct {
			name string
			f    func() int64
		}{
			{name: "sequential", f: func() int64 {
				var s int64
				for _, x := range a {
					s += int64(x)
				}
				return s
			}},
			{name: "permuted", f: func() int64 {
				var s int64
				for _, i := range perm {
					s += int64(a[i])
				}
				return s
			}},
			{name: "chase", f: func() int64 {
				var i int32
				for range n {
					i = next[i]
				}
				return int64(i)
			}},
		}
		for _, m := range modes {
			b.Run(fmt.Sprintf("size=%s/%s", byteSize(size), m.name), func(b *testing.B) {
				b.SetBytes(int64(size))
				for b.Loop() {
					sink += m.f()
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/elem")
			})
		}
	}
}

func byteSize(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%dMB", n>>20)
	default:
		return fmt.Sprintf("%dKB", n>>10)
	}
}

// BenchmarkIntersectLayout は同じ長さの 2 つの postings の共通部分を、長い方を前から読む 2 つのポインタと、短い方の要素ごとに長い方を二分探索で引く版で比べる。
// 二分探索は短い方を昇順に引く版と、でたらめな順に引く版を置く。昇順なら続けて引く探索の経路がほぼ重なり、キャッシュに残る。
// ns/elem は短い方の 1 要素あたり、ns/read は長い方の要素を 1 つ読むあたり（2 つのポインタは 2 本の長さの和、二分探索は log2 の回数）。
//
//	go test -run '^$' -bench IntersectLayout ./internal/search/
func BenchmarkIntersectLayout(b *testing.B) {
	for _, n := range []int{1 << 10, 1 << 14, 1 << 18, 1 << 22} {
		r := rand.New(rand.NewPCG(3, uint64(n)))
		a, ps := gappedIDs(r, n, 2*n), postingsFor(gappedIDs(r, n, 2*n))
		buf := make([]int32, n)
		probes := bits.Len(uint(n))
		shuffled := slices.Clone(a)
		r.Shuffle(n, func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		search := func(ids []int32) int {
			found := 0
			for _, id := range ids {
				if _, ok := slices.BinarySearchFunc(ps, id, func(p posting, id int32) int { return cmp.Compare(p.doc, id) }); ok {
					found++
				}
			}
			return found
		}
		methods := []struct {
			name  string
			reads int
			f     func() int
		}{
			{name: "merge", reads: 2 * n, f: func() int {
				copy(buf, a)
				return len(intersectMerge(buf, ps, selectFields(nil), true))
			}},
			{name: "binary", reads: probes * n, f: func() int { return search(a) }},
			{name: "binary-shuffled", reads: probes * n, f: func() int { return search(shuffled) }},
		}
		for _, m := range methods {
			b.Run(fmt.Sprintf("len=%d/postings=%s/%s", n, byteSize(n*32), m.name), func(b *testing.B) {
				var found int
				for b.Loop() {
					found = m.f()
				}
				per := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
				b.ReportMetric(per/float64(n), "ns/elem")
				b.ReportMetric(per/float64(m.reads), "ns/read")
				b.ReportMetric(float64(found), "found")
			})
		}
	}
}
