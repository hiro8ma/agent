package crawl

import (
	"bytes"
	"context"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/html"
)

const (
	// RFC 9309 は robots.txt を少なくとも 500 KiB まで読むよう求める。
	maxRobotsBytes = 500 << 10
	maxPageBytes   = 2 << 20
	maxRedirects   = 5
)

// Clock は時刻と待ちを差し替えるための時計。
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

// SystemClock は実際の時計。
type SystemClock struct{}

func (SystemClock) Now() time.Time                         { return time.Now() }
func (SystemClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Config はクローラーの設定。
type Config struct {
	// Agent は robots.txt のグループを選ぶ名前で、User-Agent の見出しにも使う。
	Agent string
	// MaxPages は取得するページの件数の上限で、robots.txt は数えない。0なら上限なし。
	MaxPages int
	// MaxDuration は新しい取得を始めてよい時間の上限。0なら上限なし。
	MaxDuration time.Duration
	// MaxInFlight はホストをまたいで同時に取得する件数の上限。0なら1。
	MaxInFlight int
	// Frontier は待ち行列の規則。nil なら DefaultFrontierConfig。robots.txt もホストごとの件数に数える。
	Frontier   *FrontierConfig
	Normalizer Normalizer
	Client     *http.Client
	Clock      Clock
}

// Page は取得したページの1件。Status は到達できなければ StatusUnreachable。
type Page struct {
	URL    string
	Depth  int
	Status int
	Links  int
	Start  time.Time
	End    time.Time
	Err    error
}

// StopReason は取得を終えた理由。
type StopReason string

const (
	StopFrontierEmpty StopReason = "フロンティアが空"
	StopMaxPages      StopReason = "件数の上限"
	StopMaxDuration   StopReason = "時間の上限"
	StopCanceled      StopReason = "取り消し"
)

// Report はクロールの結果。
type Report struct {
	// Pages は取得を終えた順のページ。
	Pages []Page
	// RobotsStatus はホストごとの robots.txt の取得の状態。
	RobotsStatus map[string]int
	// Blocked は robots.txt で禁止されて取らなかった URL を、見つけた順に1回ずつ並べる。
	Blocked []string
	// Rejected は待ち行列が受け付けなかった URL の件数を理由ごとに数える。
	Rejected map[AddResult]int
	Stop     StopReason
}

// Crawler は robots.txt に従い、ホストごとの待ち行列の規則で取得する。
type Crawler struct {
	cfg          Config
	pageClient   *http.Client
	robotsClient *http.Client
}

func New(cfg Config) *Crawler {
	if cfg.Client == nil {
		cfg.Client = http.DefaultClient
	}
	if cfg.Clock == nil {
		cfg.Clock = SystemClock{}
	}
	if cfg.MaxInFlight <= 0 {
		cfg.MaxInFlight = 1
	}
	if cfg.Frontier == nil {
		def := DefaultFrontierConfig()
		cfg.Frontier = &def
	}
	page := *cfg.Client
	// リダイレクトの先も待ち行列と robots.txt を通すため、ページでは自動で辿らない。
	page.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	robots := *cfg.Client
	robots.CheckRedirect = func(_ *http.Request, via []*http.Request) error {
		if len(via) >= maxRedirects {
			return http.ErrUseLastResponse
		}
		return nil
	}
	return &Crawler{cfg: cfg, pageClient: &page, robotsClient: &robots}
}

type hostRobots struct {
	url   string
	rules *Robots
}

type fetchResult struct {
	item   Item
	robots bool
	page   Page
	body   []byte
	links  []string
}

type run struct {
	c        *Crawler
	f        *Frontier
	robots   map[string]*hostRobots
	blocked  map[string]bool
	report   Report
	results  chan fetchResult
	inFlight int
	pages    int
}

// Run は seeds から取得を始め、件数か時間の上限に達するか、待ち行列が空になるまで続ける。取得中のものは終わるまで待つ。
func (c *Crawler) Run(ctx context.Context, seeds ...string) Report {
	r := &run{
		c:       c,
		f:       NewFrontier(*c.cfg.Frontier),
		robots:  map[string]*hostRobots{},
		blocked: map[string]bool{},
		report:  Report{RobotsStatus: map[string]int{}, Rejected: map[AddResult]int{}},
		results: make(chan fetchResult, c.cfg.MaxInFlight),
	}
	for _, s := range seeds {
		r.add(s, 0)
	}
	r.report.Stop = r.loop(ctx)
	return r.report
}

func (r *run) loop(ctx context.Context) StopReason {
	cfg := r.c.cfg
	var deadline time.Time
	if cfg.MaxDuration > 0 {
		deadline = cfg.Clock.Now().Add(cfg.MaxDuration)
	}
	var stop StopReason
	for {
		now := cfg.Clock.Now()
		if stop == "" {
			stop = r.stopReason(ctx, now, deadline)
		}
		for stop == "" && r.inFlight < cfg.MaxInFlight {
			it, ok := r.f.Next(now)
			if !ok {
				break
			}
			r.dispatch(ctx, it, now)
			stop = r.stopReason(ctx, now, deadline)
		}
		var timer <-chan time.Time
		if stop == "" && r.inFlight < cfg.MaxInFlight {
			if wake, ok := r.f.Wake(); ok {
				d := wake.Sub(now)
				if !deadline.IsZero() {
					d = min(d, deadline.Sub(now))
				}
				if d <= 0 {
					continue
				}
				timer = cfg.Clock.After(d)
			}
		}
		if r.inFlight == 0 && timer == nil {
			if stop == "" {
				stop = StopFrontierEmpty
			}
			return stop
		}
		var canceled <-chan struct{}
		if stop == "" {
			canceled = ctx.Done()
		}
		select {
		case res := <-r.results:
			r.inFlight--
			r.handle(res)
		case <-timer:
		case <-canceled:
		}
	}
}

func (r *run) stopReason(ctx context.Context, now, deadline time.Time) StopReason {
	switch {
	case ctx.Err() != nil:
		return StopCanceled
	case r.c.cfg.MaxPages > 0 && r.pages >= r.c.cfg.MaxPages:
		return StopMaxPages
	case !deadline.IsZero() && !now.Before(deadline):
		return StopMaxDuration
	}
	return ""
}

// dispatch は取り出した URL を取りに行く。robots.txt で禁止されていれば取らずに、待ちを足さずにホストを空ける。
func (r *run) dispatch(ctx context.Context, it Item, now time.Time) {
	hr := r.robots[it.Host]
	isRobots := hr.rules == nil && it.URL == hr.url
	if !isRobots {
		u, err := url.Parse(it.URL)
		if err != nil || hr.rules == nil || !hr.rules.Allowed(r.c.cfg.Agent, requestPath(u)) {
			r.block(it.URL)
			r.f.Done(it.Host, now, now)
			return
		}
		r.pages++
	}
	r.inFlight++
	go func() {
		r.results <- r.c.fetch(ctx, it, isRobots)
	}()
}

func (r *run) handle(res fetchResult) {
	r.f.Done(res.item.Host, res.page.Start, res.page.End)
	if res.robots {
		r.robots[res.item.Host].rules = RobotsFromFetch(res.page.Status, res.body)
		r.report.RobotsStatus[res.item.Host] = res.page.Status
		return
	}
	r.report.Pages = append(r.report.Pages, res.page)
	for _, link := range res.links {
		r.add(link, res.item.Depth+1)
	}
}

// add は URL を正規化して待ち行列に入れる。初めてのホストなら、その robots.txt を先に入れる。
func (r *run) add(raw string, depth int) {
	norm, err := r.c.cfg.Normalizer.Normalize(raw)
	if err != nil {
		r.report.Rejected[Invalid]++
		return
	}
	u, err := url.Parse(norm)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		r.report.Rejected[Invalid]++
		return
	}
	hr := r.robots[u.Host]
	switch {
	case hr == nil:
		hr = &hostRobots{url: u.Scheme + "://" + u.Host + "/robots.txt"}
		r.robots[u.Host] = hr
		r.f.Add(hr.url, 0)
	case hr.rules != nil && !hr.rules.Allowed(r.c.cfg.Agent, requestPath(u)):
		r.block(norm)
		return
	}
	if res := r.f.Add(norm, depth); res != Added {
		r.report.Rejected[res]++
	}
}

func (r *run) block(u string) {
	if !r.blocked[u] {
		r.blocked[u] = true
		r.report.Blocked = append(r.report.Blocked, u)
	}
}

func requestPath(u *url.URL) string {
	p := u.EscapedPath()
	if p == "" {
		p = "/"
	}
	if u.RawQuery != "" {
		p += "?" + u.RawQuery
	}
	return p
}

func (c *Crawler) fetch(ctx context.Context, it Item, isRobots bool) fetchResult {
	res := fetchResult{item: it, robots: isRobots, page: Page{URL: it.URL, Depth: it.Depth}}
	client, limit := c.pageClient, int64(maxPageBytes)
	if isRobots {
		client, limit = c.robotsClient, maxRobotsBytes
	}
	res.page.Start = c.cfg.Clock.Now()
	resp, err := c.get(ctx, client, it.URL, limit)
	res.page.End = c.cfg.Clock.Now()
	res.page.Err = err
	res.page.Status = resp.status
	res.body = resp.body
	if isRobots {
		return res
	}
	switch {
	case resp.location != "":
		res.links = []string{resp.location}
	case resp.status >= 200 && resp.status < 300 && isHTML(resp.contentType):
		res.links = ExtractLinks(resp.url, resp.body)
	}
	res.page.Links = len(res.links)
	return res
}

type response struct {
	status      int
	contentType string
	location    string
	url         *url.URL
	body        []byte
}

func (c *Crawler) get(ctx context.Context, client *http.Client, rawURL string, limit int64) (response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return response{}, err
	}
	req.Header.Set("User-Agent", c.cfg.Agent)
	resp, err := client.Do(req)
	if err != nil {
		return response{}, err
	}
	defer resp.Body.Close()
	out := response{status: resp.StatusCode, contentType: resp.Header.Get("Content-Type"), url: resp.Request.URL}
	if resp.StatusCode >= 300 && resp.StatusCode < 400 {
		if loc, err := resp.Location(); err == nil {
			out.location = loc.String()
		}
	}
	out.body, err = io.ReadAll(io.LimitReader(resp.Body, limit))
	return out, err
}

func isHTML(contentType string) bool {
	if contentType == "" {
		return true
	}
	mt, _, err := mime.ParseMediaType(contentType)
	return err == nil && (mt == "text/html" || mt == "application/xhtml+xml")
}

// ExtractLinks は HTML の <a href> を base（<base href> があればそれ）で絶対 URL に直して、出てきた順に返す。http と https 以外は捨てる。
func ExtractLinks(base *url.URL, body []byte) []string {
	z := html.NewTokenizer(bytes.NewReader(body))
	var links []string
	baseSet := false
	for {
		tt := z.Next()
		if tt == html.ErrorToken {
			return links
		}
		if tt != html.StartTagToken && tt != html.SelfClosingTagToken {
			continue
		}
		name, hasAttr := z.TagName()
		tag := string(name)
		if !hasAttr || (tag != "a" && tag != "base") {
			continue
		}
		href, ok := attr(z, "href")
		if !ok {
			continue
		}
		ref, err := url.Parse(strings.TrimSpace(href))
		if err != nil {
			continue
		}
		abs := base.ResolveReference(ref)
		if tag == "base" {
			if !baseSet {
				base, baseSet = abs, true
			}
			continue
		}
		if abs.Scheme == "http" || abs.Scheme == "https" {
			links = append(links, abs.String())
		}
	}
}

func attr(z *html.Tokenizer, name string) (string, bool) {
	for {
		key, val, more := z.TagAttr()
		if string(key) == name {
			return string(val), true
		}
		if !more {
			return "", false
		}
	}
}
