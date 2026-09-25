package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

func postingsFor(ids []int32) []posting {
	ps := make([]posting, len(ids))
	for i, id := range ids {
		ps[i] = posting{doc: id}
	}
	return ps
}

// sortedIDs は [0, universe) から n 個を重複なく選んで昇順に並べる。
func sortedIDs(r *rand.Rand, n, universe int) []int32 {
	perm := r.Perm(universe)[:n]
	ids := make([]int32, n)
	for i, v := range perm {
		ids[i] = int32(v)
	}
	slices.Sort(ids)
	return ids
}

func naiveIntersect(a, b []int32) []int32 {
	out := []int32{}
	for _, x := range a {
		if slices.Contains(b, x) {
			out = append(out, x)
		}
	}
	return out
}

func TestIntersect(t *testing.T) {
	t.Parallel()
	funcs := map[string]func([]int32, []posting, fieldSet, bool) []int32{
		"merge":  intersectMerge,
		"gallop": intersectGallop,
	}
	testCases := map[string]struct {
		a, b []int32
		want []int32
	}{
		"共通の要素だけを昇順で返す":        {a: []int32{1, 3, 5, 7}, b: []int32{2, 3, 4, 7, 9}, want: []int32{3, 7}},
		"長い方の末尾より先の要素は落とす":     {a: []int32{3, 100}, b: []int32{1, 2, 3, 4, 5, 6, 7, 8}, want: []int32{3}},
		"長い方の先頭と末尾でも見つける":      {a: []int32{0, 99}, b: []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 99}, want: []int32{0, 99}},
		"共通の要素が無ければ空":          {a: []int32{1, 2}, b: []int32{3, 4}, want: []int32{}},
		"片方が空なら空":              {a: []int32{}, b: []int32{1}, want: []int32{}},
		"倍々に進めた区間の端にある要素も見つける": {a: []int32{4, 8, 16}, b: []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, want: []int32{4, 8, 16}},
	}
	for fn, f := range funcs {
		for tn, tc := range testCases {
			t.Run(fn+"/"+tn, func(t *testing.T) {
				t.Parallel()
				got := f(slices.Clone(tc.a), postingsFor(tc.b), selectFields(nil), true)
				if !slices.Equal(got, tc.want) {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			})
		}
	}
}

func TestIntersectMatchesNaiveOnRandomLists(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		short, long int
	}{
		"同じ長さ":      {short: 500, long: 500},
		"長さの比が 16":  {short: 100, long: 1_600},
		"長さの比が 500": {short: 10, long: 5_000},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := rand.New(rand.NewPCG(1, uint64(tc.long)))
			a, b := sortedIDs(r, tc.short, 10_000), sortedIDs(r, tc.long, 10_000)
			want := naiveIntersect(a, b)
			for name, f := range map[string]func([]int32, []posting, fieldSet, bool) []int32{"merge": intersectMerge, "gallop": intersectGallop} {
				if got := f(slices.Clone(a), postingsFor(b), selectFields(nil), true); !slices.Equal(got, want) {
					t.Fatalf("%s: got %d ids, want %d", name, len(got), len(want))
				}
			}
		})
	}
}

func TestIntersectFiltersByField(t *testing.T) {
	t.Parallel()
	ps := []posting{{doc: 1, split: 1, pos: []int32{0}}, {doc: 2, split: 0, pos: []int32{3}}}
	sel := selectFields([]Field{FieldContent})
	for name, f := range map[string]func([]int32, []posting, fieldSet, bool) []int32{"merge": intersectMerge, "gallop": intersectGallop} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := f([]int32{1, 2}, ps, sel, false); !slices.Equal(got, []int32{2}) {
				t.Fatalf("got %v, want [2]", got)
			}
		})
	}
}

func TestUseGallop(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		short, long, ratio int
		want               bool
	}{
		"長さが同じなら 2 つのポインタ":        {short: 1_000, long: 1_000, ratio: defaultGallopRatio, want: false},
		"比が閾値より小さければ 2 つのポインタ":    {short: 1_000, long: defaultGallopRatio*1_000 - 1, ratio: defaultGallopRatio, want: false},
		"比が閾値以上なら galloping":      {short: 1_000, long: defaultGallopRatio * 1_000, ratio: defaultGallopRatio, want: true},
		"比が 0 なら常に 2 つのポインタ":      {short: 1, long: 1_000_000, ratio: 0, want: false},
		"短い方が 1 件なら長い方が長ければ切り替える": {short: 1, long: 100, ratio: defaultGallopRatio, want: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := useGallop(tc.short, tc.long, tc.ratio); got != tc.want {
				t.Fatalf("useGallop(%d, %d, %d) = %v, want %v", tc.short, tc.long, tc.ratio, got, tc.want)
			}
		})
	}
}

// BenchmarkIntersect は短い方の長さと、長い方との長さの比ごとに、2 つのポインタと galloping の時間を比べる。
//
//	go test -run '^$' -bench Intersect ./internal/search/
func BenchmarkIntersect(b *testing.B) {
	for _, short := range []int{100, 1_000} {
		for _, ratio := range []int{1, 4, 8, 16, 20, 32, 64, 1024} {
			long := short * ratio
			r := rand.New(rand.NewPCG(uint64(ratio), uint64(short)))
			a, ps := sortedIDs(r, short, long*2), postingsFor(sortedIDs(r, long, long*2))
			buf := make([]int32, short)
			for _, m := range []struct {
				name string
				f    func([]int32, []posting, fieldSet, bool) []int32
			}{{name: "merge", f: intersectMerge}, {name: "gallop", f: intersectGallop}} {
				b.Run(fmt.Sprintf("short=%d/ratio=%d/%s", short, ratio, m.name), func(b *testing.B) {
					for b.Loop() {
						copy(buf, a)
						m.f(buf, ps, selectFields(nil), true)
					}
				})
			}
		}
	}
}
