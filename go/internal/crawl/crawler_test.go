package crawl_test

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

// fakeClock は After で待つ代わりに時刻を進める。取得の時間は fakeTransport が進める。
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) After(d time.Duration) <-chan time.Time {
	ch := make(chan time.Time, 1)
	ch <- c.advance(d)
	return ch
}

func (c *fakeClock) advance(d time.Duration) time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	return c.now
}

type request struct {
	host, path string
	start, end time.Time
}

// fakeTransport は httptest のサーバーへ送り、応答を返す前に、path ごとに決めた取得の時間だけ時計を進める。
type fakeTransport struct {
	base  http.RoundTripper
	clock *fakeClock
	delay func(path string) time.Duration

	mu   sync.Mutex
	reqs []request
}

func (tr *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	start := tr.clock.Now()
	resp, err := tr.base.RoundTrip(req)
	end := tr.clock.advance(tr.delay(req.URL.Path))
	tr.mu.Lock()
	tr.reqs = append(tr.reqs, request{host: req.URL.Host, path: req.URL.RequestURI(), start: start, end: end})
	tr.mu.Unlock()
	return resp, err
}

// site は1つの httptest のサーバーで、Host の見出しごとに別のサイトとして答える。
type site struct {
	srv  *httptest.Server
	slow time.Duration

	mu          sync.Mutex
	log         []string
	inFlight    map[string]int
	maxPerHost  map[string]int
	total       int
	maxInFlight int
}

func newSite(t *testing.T, slow time.Duration) *site {
	t.Helper()
	s := &site{slow: slow, inFlight: map[string]int{}, maxPerHost: map[string]int{}}
	s.srv = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.srv.Close)
	return s
}

// transport は宛先の名前によらず httptest のサーバーにつなぐ。example.com などの名前を使っても外へは出ない。
func (s *site) transport() *http.Transport {
	addr := s.srv.Listener.Addr().String()
	return &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, network, addr)
	}}
}

func (s *site) requests(host string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, l := range s.log {
		if h, p, _ := strings.Cut(l, " "); h == host {
			out = append(out, p)
		}
	}
	return out
}

func page(links ...string) string {
	var b strings.Builder
	b.WriteString("<!doctype html><html><body>")
	for _, l := range links {
		fmt.Fprintf(&b, `<a href="%s">link</a>`, l)
	}
	b.WriteString("</body></html>")
	return b.String()
}

func (s *site) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.log = append(s.log, r.Host+" "+r.URL.RequestURI())
	s.inFlight[r.Host]++
	s.maxPerHost[r.Host] = max(s.maxPerHost[r.Host], s.inFlight[r.Host])
	s.total++
	s.maxInFlight = max(s.maxInFlight, s.total)
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.inFlight[r.Host]--
		s.total--
		s.mu.Unlock()
	}()

	path := r.URL.Path
	switch r.Host {
	case "a.example.com":
		s.serveA(w, path)
	case "b.example.com":
		switch path {
		case "/robots.txt":
			http.NotFound(w, r)
		case "/":
			fmt.Fprint(w, page("/b1.html", "/b2.html"))
		case "/b1.html":
			fmt.Fprint(w, page("/b3.html"))
		case "/b2.html", "/b3.html":
			fmt.Fprint(w, page())
		default:
			http.NotFound(w, r)
		}
	case "c.example.com":
		if path == "/robots.txt" {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, page("/c1.html"))
	default:
		if strings.HasPrefix(r.Host, "slow") {
			time.Sleep(s.slow)
			if path == "/" {
				fmt.Fprint(w, page("/1.html", "/2.html", "/3.html", "/4.html", "/5.html"))
				return
			}
			fmt.Fprint(w, page())
			return
		}
		http.NotFound(w, r)
	}
}

func (s *site) serveA(w http.ResponseWriter, path string) {
	switch {
	case path == "/robots.txt":
		fmt.Fprint(w, "User-agent: *\nDisallow: /private/\nAllow: /private/open.html\n")
	case path == "/":
		fmt.Fprint(w, page(
			"/articles/1.html",
			"/articles/1.html?utm_source=news&utm_medium=mail",
			"articles/1.html#comments",
			"/private/secret.html",
			"/private/open.html",
			"/calendar/2026-01",
			"/loop/",
			"/gone.html",
			"/busy.html",
			"/slow.html",
			"http://b.example.com/",
			"http://c.example.com/",
			"mailto:info@example.com",
		))
	case path == "/articles/1.html":
		fmt.Fprint(w, page("/articles/2.html", "../"))
	case path == "/articles/2.html", path == "/private/open.html", path == "/slow.html":
		fmt.Fprint(w, page())
	case strings.HasPrefix(path, "/calendar/"):
		var y, m int
		if _, err := fmt.Sscanf(strings.TrimPrefix(path, "/calendar/"), "%d-%d", &y, &m); err != nil {
			http.NotFound(w, nil)
			return
		}
		next := time.Date(y, time.Month(m+1), 1, 0, 0, 0, 0, time.UTC)
		fmt.Fprint(w, page(next.Format("/calendar/2006-01")))
	case strings.HasPrefix(path, "/loop/"):
		fmt.Fprint(w, page("loop/"))
	case path == "/gone.html":
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, page("/never-404.html"))
	case path == "/busy.html":
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, page("/never-503.html"))
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

type fakeEnv struct {
	clock *fakeClock
	tr    *fakeTransport
}

func newFakeEnv(s *site, delay func(string) time.Duration) fakeEnv {
	clock := &fakeClock{now: t0}
	return fakeEnv{clock: clock, tr: &fakeTransport{base: s.transport(), clock: clock, delay: delay}}
}

func constDelay(d time.Duration) func(string) time.Duration {
	return func(string) time.Duration { return d }
}

func pageStatus(rep crawl.Report) map[string]int {
	out := map[string]int{}
	for _, p := range rep.Pages {
		out[p.URL] = p.Status
	}
	return out
}

func TestCrawlSite(t *testing.T) {
	t.Parallel()
	s := newSite(t, 0)
	env := newFakeEnv(s, constDelay(100*time.Millisecond))
	c := crawl.New(crawl.Config{
		Agent:      "testbot",
		Frontier:   &crawl.FrontierConfig{Politeness: 1, MaxDepth: 6, MaxPerHost: 50, MaxSegmentRepeat: 2},
		Normalizer: crawl.Normalizer{DropParams: []string{"utm_*"}},
		Client:     &http.Client{Transport: env.tr},
		Clock:      env.clock,
	})
	rep := c.Run(t.Context(), "http://a.example.com/")

	if rep.Stop != crawl.StopFrontierEmpty {
		t.Errorf("Stop = %q", rep.Stop)
	}
	a := s.requests("a.example.com")
	count := func(path string) int {
		n := 0
		for _, p := range a {
			if p == path {
				n++
			}
		}
		return n
	}
	testCases := map[string]struct {
		path string
		want int
	}{
		"robots.txt を1回取る": {path: "/robots.txt", want: 1},
		"計測用のパラメータと # だけが違う重複は1回だけ取る": {path: "/articles/1.html", want: 1},
		"Disallow のディレクトリの中を取らない":     {path: "/private/secret.html", want: 0},
		"長い Allow で許可された URL は取る":     {path: "/private/open.html", want: 1},
		"相対リンクが重なった2段目までは取る":          {path: "/loop/loop/", want: 1},
		"同じ区間の3回目は罠として取らない":           {path: "/loop/loop/loop/", want: 0},
		"深さ6のカレンダーは取る":                {path: "/calendar/2026-06", want: 1},
		"深さ7のカレンダーは取らない":              {path: "/calendar/2026-07", want: 0},
		"404のページのリンクはたどらない":           {path: "/never-404.html", want: 0},
		"503のページのリンクはたどらない":           {path: "/never-503.html", want: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := count(tc.path); got != tc.want {
				t.Errorf("%s を取った回数 = %d, want %d", tc.path, got, tc.want)
			}
		})
	}

	if got, want := s.requests("b.example.com"), []string{"/robots.txt", "/", "/b1.html", "/b2.html", "/b3.html"}; !equalStrings(got, want) {
		t.Errorf("robots.txt が404のホストへの要求 = %v, want %v", got, want)
	}
	if got, want := s.requests("c.example.com"), []string{"/robots.txt"}; !equalStrings(got, want) {
		t.Errorf("robots.txt が503のホストへの要求 = %v, want %v", got, want)
	}
	if !slices.Contains(rep.Blocked, "http://a.example.com/private/secret.html") || !slices.Contains(rep.Blocked, "http://c.example.com/") {
		t.Errorf("Blocked = %v", rep.Blocked)
	}
	status := pageStatus(rep)
	if status["http://a.example.com/gone.html"] != http.StatusNotFound || status["http://a.example.com/busy.html"] != http.StatusServiceUnavailable {
		t.Errorf("404 と 503 の記録 = %d, %d", status["http://a.example.com/gone.html"], status["http://a.example.com/busy.html"])
	}
	if rep.RobotsStatus["b.example.com"] != http.StatusNotFound || rep.RobotsStatus["c.example.com"] != http.StatusServiceUnavailable {
		t.Errorf("RobotsStatus = %v", rep.RobotsStatus)
	}
	if rep.Rejected[crawl.Trap] == 0 || rep.Rejected[crawl.TooDeep] == 0 || rep.Rejected[crawl.Duplicate] == 0 {
		t.Errorf("Rejected = %v", rep.Rejected)
	}
}

func TestCrawlPoliteWait(t *testing.T) {
	t.Parallel()
	delays := map[string]time.Duration{
		"/robots.txt": 50 * time.Millisecond,
		"/":           300 * time.Millisecond,
		"/b1.html":    2 * time.Second,
		"/b2.html":    100 * time.Millisecond,
		"/b3.html":    700 * time.Millisecond,
	}
	testCases := map[string]struct {
		frontier *crawl.FrontierConfig
		wantK    time.Duration
	}{
		"k=3を渡せば取得の3倍を待つ":    {frontier: &crawl.FrontierConfig{Politeness: 3}, wantK: 3},
		"規則を渡さなければ取得の10倍を待つ": {frontier: nil, wantK: 10},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			s := newSite(t, 0)
			env := newFakeEnv(s, func(p string) time.Duration { return delays[p] })
			c := crawl.New(crawl.Config{
				Agent:       "testbot",
				MaxInFlight: 4,
				Frontier:    tc.frontier,
				Client:      &http.Client{Transport: env.tr},
				Clock:       env.clock,
			})
			c.Run(t.Context(), "http://b.example.com/")
			reqs := env.tr.reqs
			if len(reqs) != 5 {
				t.Fatalf("要求 = %d 件, want 5", len(reqs))
			}
			for i := 1; i < len(reqs); i++ {
				prev, cur := reqs[i-1], reqs[i]
				if got, want := cur.start.Sub(prev.end), tc.wantK*prev.end.Sub(prev.start); got != want {
					t.Errorf("%s の後の待ち = %v, want %v（取得 %v）", prev.path, got, want, prev.end.Sub(prev.start))
				}
			}
		})
	}
}

func TestCrawlOneConnectionPerHost(t *testing.T) {
	t.Parallel()
	s := newSite(t, 15*time.Millisecond)
	c := crawl.New(crawl.Config{
		Agent:       "testbot",
		MaxInFlight: 3,
		Frontier:    &crawl.FrontierConfig{Politeness: 0},
		Client:      &http.Client{Transport: s.transport()},
	})
	rep := c.Run(t.Context(), "http://slow1.example.com/", "http://slow2.example.com/", "http://slow3.example.com/")
	if len(rep.Pages) != 18 {
		t.Fatalf("取ったページ = %d, want 18", len(rep.Pages))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for host, n := range s.maxPerHost {
		if n != 1 {
			t.Errorf("%s への同時の要求 = %d, want 1", host, n)
		}
	}
	if s.maxInFlight < 2 {
		t.Errorf("ホストをまたいだ同時の要求 = %d。並行に取れていないので、同じホストへの同時の要求を見分けられない", s.maxInFlight)
	}
}

func TestCrawlStops(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		seed        string
		maxPages    int
		maxDuration time.Duration
		canceled    bool
		wantStop    crawl.StopReason
		wantPages   int
	}{
		"件数の上限で止める":        {seed: "http://a.example.com/calendar/2026-01", maxPages: 3, wantStop: crawl.StopMaxPages, wantPages: 3},
		"時間の上限より後は新しく取らない": {seed: "http://a.example.com/calendar/2026-01", maxDuration: 5 * time.Second, wantStop: crawl.StopMaxDuration, wantPages: 2},
		"フロンティアが空になったら止める": {seed: "http://b.example.com/", wantStop: crawl.StopFrontierEmpty, wantPages: 4},
		"取り消されていたら1件も取らない": {seed: "http://b.example.com/", canceled: true, wantStop: crawl.StopCanceled, wantPages: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			s := newSite(t, 0)
			env := newFakeEnv(s, constDelay(time.Second))
			c := crawl.New(crawl.Config{
				Agent:       "testbot",
				MaxPages:    tc.maxPages,
				MaxDuration: tc.maxDuration,
				Frontier:    &crawl.FrontierConfig{Politeness: 1},
				Client:      &http.Client{Transport: env.tr},
				Clock:       env.clock,
			})
			ctx := t.Context()
			if tc.canceled {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			rep := c.Run(ctx, tc.seed)
			if rep.Stop != tc.wantStop || len(rep.Pages) != tc.wantPages {
				t.Errorf("Stop = %q, ページ = %d, want %q, %d", rep.Stop, len(rep.Pages), tc.wantStop, tc.wantPages)
			}
		})
	}
}

func TestCrawlPerHostLimitStopsTrap(t *testing.T) {
	t.Parallel()
	s := newSite(t, 0)
	env := newFakeEnv(s, constDelay(10*time.Millisecond))
	c := crawl.New(crawl.Config{
		Agent:    "testbot",
		MaxPages: 100,
		Frontier: &crawl.FrontierConfig{Politeness: 1, MaxPerHost: 5},
		Client:   &http.Client{Transport: env.tr},
		Clock:    env.clock,
	})
	rep := c.Run(t.Context(), "http://a.example.com/calendar/2026-01")
	if got := len(s.requests("a.example.com")); got != 5 {
		t.Errorf("深さの上限が無いカレンダーへの要求 = %d, want 5（robots.txt を含む）", got)
	}
	if rep.Stop != crawl.StopFrontierEmpty || rep.Rejected[crawl.HostFull] != 1 {
		t.Errorf("Stop = %q, Rejected = %v", rep.Stop, rep.Rejected)
	}
}

func TestExtractLinks(t *testing.T) {
	t.Parallel()
	base := mustParse(t, "http://example.com/dir/page.html")
	testCases := map[string]struct {
		body string
		want []string
	}{
		"相対 URL を解決し、http と https だけを残す": {
			body: `<a href="a.html">1</a><a href="/b">2</a><a href="../c?x=1#f">3</a>` +
				`<a href="https://other.example.org/d">4</a><a href="mailto:x@example.com">5</a>` +
				`<a href="javascript:void(0)">6</a><a name="top">7</a><link href="/style.css"><A HREF="E.html">8</A>`,
			want: []string{
				"http://example.com/dir/a.html",
				"http://example.com/b",
				"http://example.com/c?x=1#f",
				"https://other.example.org/d",
				"http://example.com/dir/E.html",
			},
		},
		"<base href> があればそれで解決する": {
			body: `<head><base href="http://example.com/other/"></head><a href="x.html">1</a>`,
			want: []string{"http://example.com/other/x.html"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := crawl.ExtractLinks(base, []byte(tc.body)); !equalStrings(got, tc.want) {
				t.Errorf("ExtractLinks = %v, want %v", got, tc.want)
			}
		})
	}
}

func mustParse(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u
}
