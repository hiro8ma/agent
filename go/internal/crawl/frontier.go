package crawl

import (
	"container/heap"
	"net/url"
	"strings"
	"time"
)

// DefaultPoliteness は、同じホストへ次を取りに行くまでに、前の取得にかかった時間の何倍を待つかの既定。
const DefaultPoliteness = 10

// FrontierConfig は待ち行列の規則。上限の0は上限なしを表す。
type FrontierConfig struct {
	// Politeness は k。同じホストの次の取得は、前の取得が終わった時刻 + k × その取得にかかった時間より後にする。
	Politeness float64
	// MaxDepth は種からたどったリンクの段数の上限。
	MaxDepth int
	// MaxPerHost はホストごとに受け付ける URL の件数の上限。
	MaxPerHost int
	// MaxSegmentRepeat は、パスの同じ区間が出てよい回数。超えた URL は罠とみなす。
	MaxSegmentRepeat int
}

// DefaultFrontierConfig は k を DefaultPoliteness にした既定の規則を返す。
func DefaultFrontierConfig() FrontierConfig {
	return FrontierConfig{Politeness: DefaultPoliteness, MaxDepth: 16, MaxPerHost: 10_000, MaxSegmentRepeat: 2}
}

// Item は待ち行列の1件。
type Item struct {
	URL   string
	Host  string
	Depth int
}

// AddResult は Add が URL を受け付けたか、受け付けなかった理由。
type AddResult int

const (
	Added AddResult = iota
	Duplicate
	TooDeep
	HostFull
	Trap
	Invalid
)

func (r AddResult) String() string {
	return [...]string{"受け付けた", "重複", "深すぎる", "ホストの上限", "罠", "URL が不正"}[r]
}

type hostQueue struct {
	name    string
	seq     int
	items   []Item
	readyAt time.Time
	busy    bool
	count   int
	index   int
}

// Frontier はホストごとの待ち行列。同じホストへは同時に1件だけを出し、ホストをまたいでは取ってよい時刻の早い順に出す。
type Frontier struct {
	cfg   FrontierConfig
	seen  map[string]bool
	hosts map[string]*hostQueue
	ready hostHeap
}

func NewFrontier(cfg FrontierConfig) *Frontier {
	return &Frontier{cfg: cfg, seen: map[string]bool{}, hosts: map[string]*hostQueue{}}
}

// Add は正規化済みの URL を depth の段として受け付ける。受け付けなかった URL は既出として覚えないので、後で浅い段から来れば受け付ける。
func (f *Frontier) Add(rawURL string, depth int) AddResult {
	if f.seen[rawURL] {
		return Duplicate
	}
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return Invalid
	}
	if f.cfg.MaxDepth > 0 && depth > f.cfg.MaxDepth {
		return TooDeep
	}
	if f.cfg.MaxSegmentRepeat > 0 && repeatsSegment(u.EscapedPath(), f.cfg.MaxSegmentRepeat) {
		return Trap
	}
	h := f.hosts[u.Host]
	if h == nil {
		h = &hostQueue{name: u.Host, seq: len(f.hosts), index: -1}
		f.hosts[u.Host] = h
	}
	if f.cfg.MaxPerHost > 0 && h.count >= f.cfg.MaxPerHost {
		return HostFull
	}
	f.seen[rawURL] = true
	h.count++
	h.items = append(h.items, Item{URL: rawURL, Host: u.Host, Depth: depth})
	if !h.busy && h.index < 0 {
		heap.Push(&f.ready, h)
	}
	return Added
}

// repeatsSegment はパスのどれかの区間が limit 回を超えて出るかを返す。/a/b/a/b/a/b や、相対リンクが重なった /x/x/x/ を拾う。
func repeatsSegment(path string, limit int) bool {
	counts := map[string]int{}
	for seg := range strings.SplitSeq(path, "/") {
		if seg == "" {
			continue
		}
		counts[seg]++
		if counts[seg] > limit {
			return true
		}
	}
	return false
}

// Next は now の時点で取ってよい URL を1件返し、そのホストを取得中にする。取得が終わったら Done を呼ぶ。
func (f *Frontier) Next(now time.Time) (Item, bool) {
	if len(f.ready) == 0 || f.ready[0].readyAt.After(now) {
		return Item{}, false
	}
	h := heap.Pop(&f.ready).(*hostQueue)
	it := h.items[0]
	h.items = h.items[1:]
	h.busy = true
	return it, true
}

// Wake は取得中でないホストのうち、最も早く取ってよくなる時刻を返す。そのようなホストが無ければ false。
func (f *Frontier) Wake() (time.Time, bool) {
	if len(f.ready) == 0 {
		return time.Time{}, false
	}
	return f.ready[0].readyAt, true
}

// Done は host の取得が start に始まり end に終わったことを記録し、次に取ってよい時刻を end + k × (end − start) にする。
func (f *Frontier) Done(host string, start, end time.Time) {
	h := f.hosts[host]
	if h == nil || !h.busy {
		return
	}
	h.busy = false
	h.readyAt = end.Add(time.Duration(f.cfg.Politeness * float64(end.Sub(start))))
	if len(h.items) > 0 {
		heap.Push(&f.ready, h)
	}
}

type hostHeap []*hostQueue

func (h hostHeap) Len() int { return len(h) }

func (h hostHeap) Less(i, j int) bool {
	if !h[i].readyAt.Equal(h[j].readyAt) {
		return h[i].readyAt.Before(h[j].readyAt)
	}
	return h[i].seq < h[j].seq
}

func (h hostHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *hostHeap) Push(x any) {
	q := x.(*hostQueue)
	q.index = len(*h)
	*h = append(*h, q)
}

func (h *hostHeap) Pop() any {
	old := *h
	q := old[len(old)-1]
	old[len(old)-1] = nil
	q.index = -1
	*h = old[:len(old)-1]
	return q
}
