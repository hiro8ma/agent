package crawl_test

import (
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func TestFrontierOneFetchPerHost(t *testing.T) {
	t.Parallel()
	f := crawl.NewFrontier(crawl.FrontierConfig{Politeness: 10})
	for _, u := range []string{"http://example.com/1", "http://example.com/2", "http://example.com/3"} {
		if res := f.Add(u, 0); res != crawl.Added {
			t.Fatalf("Add(%q) = %v", u, res)
		}
	}
	first, ok := f.Next(t0)
	if !ok || first.URL != "http://example.com/1" {
		t.Fatalf("Next = %+v, %v", first, ok)
	}
	if it, ok := f.Next(t0.Add(time.Hour)); ok {
		t.Fatalf("取得中のホストから2件目を出した: %+v", it)
	}
	if wake, ok := f.Wake(); ok {
		t.Fatalf("取得中のホストの Wake = %v", wake)
	}
	f.Done("example.com", t0, t0.Add(time.Second))
	if it, ok := f.Next(t0.Add(11*time.Second - time.Nanosecond)); ok {
		t.Fatalf("待ちの終わる前に出した: %+v", it)
	}
	if it, ok := f.Next(t0.Add(11 * time.Second)); !ok || it.URL != "http://example.com/2" {
		t.Fatalf("Next = %+v, %v", it, ok)
	}
}

func TestFrontierWaitIsKTimesFetch(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		k     float64
		fetch time.Duration
		want  time.Duration
	}{
		"k=0なら終わった時刻にすぐ取ってよい": {k: 0, fetch: 200 * time.Millisecond, want: 0},
		"k=1なら取得と同じだけ待つ":      {k: 1, fetch: 200 * time.Millisecond, want: 200 * time.Millisecond},
		"k=10なら取得の10倍を待つ":     {k: 10, fetch: 200 * time.Millisecond, want: 2 * time.Second},
		"遅い応答ほど長く待つ":          {k: 10, fetch: 3 * time.Second, want: 30 * time.Second},
		"k は小数でもよい":           {k: 2.5, fetch: 400 * time.Millisecond, want: time.Second},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			f := crawl.NewFrontier(crawl.FrontierConfig{Politeness: tc.k})
			f.Add("http://example.com/1", 0)
			f.Add("http://example.com/2", 0)
			start := t0.Add(time.Minute)
			if _, ok := f.Next(start); !ok {
				t.Fatal("1件目を出さない")
			}
			end := start.Add(tc.fetch)
			f.Done("example.com", start, end)
			wake, ok := f.Wake()
			if !ok {
				t.Fatal("Wake が false")
			}
			if got := wake.Sub(end); got != tc.want {
				t.Errorf("待ち = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestFrontierOrdersHostsByReadyTime(t *testing.T) {
	t.Parallel()
	f := crawl.NewFrontier(crawl.FrontierConfig{Politeness: 1})
	for _, u := range []string{"http://a.example.com/1", "http://a.example.com/2", "http://b.example.com/1", "http://b.example.com/2", "http://c.example.com/1"} {
		f.Add(u, 0)
	}
	var order []string
	for range 3 {
		it, ok := f.Next(t0)
		if !ok {
			t.Fatal("同じ時刻に3つのホストから出さない")
		}
		order = append(order, it.URL)
	}
	if want := []string{"http://a.example.com/1", "http://b.example.com/1", "http://c.example.com/1"}; !equalStrings(order, want) {
		t.Fatalf("同じ時刻のホストの順 = %v, want %v", order, want)
	}
	f.Done("a.example.com", t0, t0.Add(5*time.Second))
	f.Done("b.example.com", t0, t0.Add(time.Second))
	f.Done("c.example.com", t0, t0.Add(time.Second))
	got := []string{}
	for {
		it, ok := f.Next(t0.Add(time.Minute))
		if !ok {
			break
		}
		got = append(got, it.URL)
	}
	if want := []string{"http://b.example.com/2", "http://a.example.com/2"}; !equalStrings(got, want) {
		t.Fatalf("取ってよい時刻の順 = %v, want %v", got, want)
	}
}

func TestFrontierLimits(t *testing.T) {
	t.Parallel()
	type add struct {
		url   string
		depth int
		want  crawl.AddResult
	}
	testCases := map[string]struct {
		cfg  crawl.FrontierConfig
		adds []add
	}{
		"深さの上限を超えたら受け付けない": {
			cfg: crawl.FrontierConfig{MaxDepth: 2},
			adds: []add{
				{url: "http://example.com/a", depth: 2, want: crawl.Added},
				{url: "http://example.com/b", depth: 3, want: crawl.TooDeep},
			},
		},
		"深すぎて断った URL は浅い段から来れば受け付ける": {
			cfg: crawl.FrontierConfig{MaxDepth: 2},
			adds: []add{
				{url: "http://example.com/b", depth: 3, want: crawl.TooDeep},
				{url: "http://example.com/b", depth: 1, want: crawl.Added},
			},
		},
		"ホストごとの件数の上限を超えたら受け付けない": {
			cfg: crawl.FrontierConfig{MaxPerHost: 2},
			adds: []add{
				{url: "http://example.com/1", want: crawl.Added},
				{url: "http://example.com/2", want: crawl.Added},
				{url: "http://example.com/3", want: crawl.HostFull},
				{url: "http://other.example.com/1", want: crawl.Added},
			},
		},
		"同じ URL は重複": {
			cfg: crawl.FrontierConfig{},
			adds: []add{
				{url: "http://example.com/1", want: crawl.Added},
				{url: "http://example.com/1", depth: 1, want: crawl.Duplicate},
			},
		},
		"同じ区間の繰り返しを上限まで受け付け、超えたら罠": {
			cfg: crawl.FrontierConfig{MaxSegmentRepeat: 2},
			adds: []add{
				{url: "http://example.com/loop/loop/", want: crawl.Added},
				{url: "http://example.com/loop/loop/loop/", want: crawl.Trap},
				{url: "http://example.com/a/b/a/b/a/b", want: crawl.Trap},
				{url: "http://example.com/2026/11/11/", want: crawl.Added},
			},
		},
		"host の無い URL は不正": {
			cfg:  crawl.FrontierConfig{},
			adds: []add{{url: "/relative", want: crawl.Invalid}},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			f := crawl.NewFrontier(tc.cfg)
			for _, a := range tc.adds {
				if got := f.Add(a.url, a.depth); got != a.want {
					t.Errorf("Add(%q, %d) = %v, want %v", a.url, a.depth, got, a.want)
				}
			}
		})
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
