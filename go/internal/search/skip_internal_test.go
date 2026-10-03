package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
	"time"
)

// countMerge は intersectMerge と同じ手順で、文書番号を比べた回数を数える。
func countMerge(ids []int32, ps []posting) ([]int32, int) {
	out := ids[:0]
	n, i, j := 0, 0, 0
	for i < len(ids) && j < len(ps) {
		n++
		switch a, b := ids[i], ps[j].doc; {
		case a < b:
			i++
		case a > b:
			j++
		default:
			out = append(out, a)
			i++
			j++
		}
	}
	return out, n
}

// countGallop は intersectGallop と同じ手順で、間隔を倍にする比較と二分探索の比較を数える。
func countGallop(ids []int32, ps []posting) ([]int32, int) {
	out := ids[:0]
	n, j := 0, 0
	for _, a := range ids {
		if j >= len(ps) {
			break
		}
		lo, step := j, 1
		for lo+step < len(ps) {
			n++
			if ps[lo+step].doc >= a {
				break
			}
			lo += step
			step *= 2
		}
		hi := min(lo+step+1, len(ps))
		l, r := lo, hi
		for l < r {
			n++
			m := int(uint(l+r) >> 1)
			if ps[m].doc < a {
				l = m + 1
			} else {
				r = m
			}
		}
		j = l
		if j < len(ps) {
			n++
			if ps[j].doc == a {
				out = append(out, a)
				j++
			}
		}
	}
	return out, n
}

// TestSkipPointersTutorial は教材の例を再現する。便利の postings 17 件に √17 の 4 件おきの飛び先 1→9→30→39→51 を持たせ、転置インデックス [6,40,41] との AND を取る。
// 40 を探すときに 9 から 30、39 と飛び、7 から 39 の手前までの 8 回の比較を飛び先の 3 回の比較で済ませる。
func TestSkipPointersTutorial(t *testing.T) {
	t.Parallel()
	benri := []int32{1, 5, 6, 7, 9, 13, 20, 25, 30, 31, 32, 35, 39, 40, 45, 50, 51}
	tenchi := []int32{6, 40, 41}
	sp := newSkipPostings(postingsFor(benri), sqrtInterval(len(benri)))
	if want := []int32{1, 9, 30, 39, 51}; sp.interval != 4 || !slices.Equal(sp.skips, want) {
		t.Fatalf("interval = %d, skips = %v, want 4, %v", sp.interval, sp.skips, want)
	}
	got, withSkips := intersectSkip(slices.Clone(tenchi), sp, selectFields(nil), true)
	merged, plain := countMerge(slices.Clone(tenchi), postingsFor(benri))
	if want := []int32{6, 40}; !slices.Equal(got, want) || !slices.Equal(merged, want) {
		t.Fatalf("skip = %v, merge = %v, want %v", got, merged, want)
	}
	if withSkips != 13 || plain != 15 {
		t.Errorf("比較の回数 = 飛び先あり %d / なし %d, want 13 / 15", withSkips, plain)
	}
}

func TestIntersectSkipMatchesMerge(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		short, long, interval int
	}{
		"同じ長さで √n おき":       {short: 500, long: 500, interval: -1},
		"比 100 で √n おき":     {short: 50, long: 5_000, interval: -1},
		"比 100 で 8 件おき":     {short: 50, long: 5_000, interval: 8},
		"比 1000 で 256 件おき":  {short: 10, long: 10_000, interval: 256},
		"間隔 1 は飛び先を持たない":    {short: 100, long: 1_000, interval: 1},
		"長い方より大きい値だけの短いリスト": {short: 3, long: 100, interval: 10},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := rand.New(rand.NewPCG(3, uint64(tc.long+tc.interval)))
			a, b := sortedIDs(r, tc.short, tc.long*3), sortedIDs(r, tc.long, tc.long*3)
			iv := tc.interval
			if iv < 0 {
				iv = sqrtInterval(tc.long)
			}
			want := naiveIntersect(a, b)
			got, _ := intersectSkip(slices.Clone(a), newSkipPostings(postingsFor(b), iv), selectFields(nil), true)
			if !slices.Equal(got, want) {
				t.Fatalf("got %d ids, want %d", len(got), len(want))
			}
			if g, _ := countGallop(slices.Clone(a), postingsFor(b)); !slices.Equal(g, want) {
				t.Fatalf("countGallop: got %d ids, want %d", len(g), len(want))
			}
		})
	}
}

func TestIntersectSkipFiltersByField(t *testing.T) {
	t.Parallel()
	ps := []posting{{doc: 1, split: 1, pos: []int32{0}}, {doc: 2, split: 0, pos: []int32{3}}}
	got, _ := intersectSkip([]int32{1, 2}, newSkipPostings(ps, 2), selectFields([]Field{FieldContent}), false)
	if !slices.Equal(got, []int32{2}) {
		t.Fatalf("got %v, want [2]", got)
	}
}

type andMethod struct {
	name  string
	run   func(ids []int32) []int32
	count func(ids []int32) int
}

func andMethods(ps []posting) []andMethod {
	methods := []andMethod{
		{
			name: "merge", run: func(ids []int32) []int32 { return intersectMerge(ids, ps, selectFields(nil), true) },
			count: func(ids []int32) int { _, n := countMerge(ids, ps); return n },
		},
		{
			name: "gallop", run: func(ids []int32) []int32 { return intersectGallop(ids, ps, selectFields(nil), true) },
			count: func(ids []int32) int { _, n := countGallop(ids, ps); return n },
		},
	}
	for _, iv := range []int{sqrtInterval(len(ps)), 8, 64, 512} {
		sp := newSkipPostings(ps, iv)
		name := fmt.Sprintf("skip-%d", iv)
		if iv == sqrtInterval(len(ps)) {
			name = fmt.Sprintf("skip-sqrt(%d)", iv)
		}
		methods = append(methods, andMethod{
			name:  name,
			run:   func(ids []int32) []int32 { out, _ := intersectSkip(ids, sp, selectFields(nil), true); return out },
			count: func(ids []int32) int { _, n := intersectSkip(ids, sp, selectFields(nil), true); return n },
		})
	}
	return methods
}

// TestSkipComparisons は短いリストを 1,000 件に固定し、長さの比ごとに方法ごとの比較の回数と時間を出し、結果が一致することを確かめる。
//
//	go test -run TestSkipComparisons -v ./internal/search/
func TestSkipComparisons(t *testing.T) {
	t.Parallel()
	const short = 1_000
	for _, ratio := range []int{1, 10, 100, 1_000} {
		long := short * ratio
		r := rand.New(rand.NewPCG(uint64(ratio), 9))
		a, ps := sortedIDs(r, short, long*2), postingsFor(sortedIDs(r, long, long*2))
		want := naiveIntersectSorted(a, ps)
		buf := make([]int32, short)
		for _, m := range andMethods(ps) {
			copy(buf, a)
			if got := m.run(buf); !slices.Equal(got, want) {
				t.Fatalf("ratio=%d %s: got %d ids, want %d", ratio, m.name, len(got), len(want))
			}
			copy(buf, a)
			n := m.count(buf)
			const reps = 20
			start := time.Now()
			for range reps {
				copy(buf, a)
				m.run(buf)
			}
			t.Logf("比 %4d %-16s 比較 %9d 回 %8.1fµs 共通 %d", ratio, m.name, n, float64(time.Since(start).Microseconds())/reps, len(want))
		}
	}
}

func naiveIntersectSorted(a []int32, ps []posting) []int32 {
	out := []int32{}
	for _, x := range a {
		if _, ok := slices.BinarySearchFunc(ps, x, func(p posting, id int32) int { return int(p.doc - id) }); ok {
			out = append(out, x)
		}
	}
	return out
}

// BenchmarkIntersectSkip は短いリストを 1,000 件に固定し、長さの比 1 / 10 / 100 / 1000 で、2 つのポインタ、galloping、飛び先の間隔ごとの AND を比べる。
//
//	go test -run '^$' -bench IntersectSkip ./internal/search/
func BenchmarkIntersectSkip(b *testing.B) {
	const short = 1_000
	for _, ratio := range []int{1, 10, 100, 1_000} {
		long := short * ratio
		r := rand.New(rand.NewPCG(uint64(ratio), 9))
		a, ps := sortedIDs(r, short, long*2), postingsFor(sortedIDs(r, long, long*2))
		buf := make([]int32, short)
		for _, m := range andMethods(ps) {
			copy(buf, a)
			n := m.count(buf)
			b.Run(fmt.Sprintf("ratio=%d/%s", ratio, m.name), func(b *testing.B) {
				for b.Loop() {
					copy(buf, a)
					m.run(buf)
				}
				b.ReportMetric(float64(n), "cmp/op")
			})
		}
	}
}
