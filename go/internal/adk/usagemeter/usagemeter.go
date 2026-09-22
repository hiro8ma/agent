// Package usagemeter はモデルの呼び出しのトークン数を、エージェントとモデルの組ごとに 1 日単位で数えるプラグイン。
//
// ADK Go（v2.2.0）はトークン数をトレースのスパンの属性に書くが、メトリクスとしては出さない。
// ここではプロセスの中で数え、書き出しは呼び出し側が間隔を空けてまとめて行う。
package usagemeter

import (
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
)

// Key は集計の単位。
type Key struct {
	Agent string
	Model string
}

// Usage はトークン数の合計。Prompt はキャッシュから読んだ分（Cached）を含む。
type Usage struct {
	Calls         int64
	Prompt        int64
	Cached        int64
	ToolUsePrompt int64
	Output        int64
	Thoughts      int64
}

// Price は 100 万トークンあたりの単価。通貨は呼び出し側で揃える。思考のトークンは出力の単価で数える。
type Price struct {
	Input       float64
	CachedInput float64
	Output      float64
}

// Cost は u を p の単価で計算する。
func (u Usage) Cost(p Price) float64 {
	uncached := u.Prompt - u.Cached + u.ToolUsePrompt
	return (float64(uncached)*p.Input + float64(u.Cached)*p.CachedInput + float64(u.Output+u.Thoughts)*p.Output) / 1e6
}

// Meter は 1 日分の使用量を持つ。日付は loc で区切る。
type Meter struct {
	mu      sync.Mutex
	prices  map[string]Price
	budget  float64
	loc     *time.Location
	now     func() time.Time
	day     string
	usage   map[Key]Usage
	pending map[string]string
}

// New は単価の表と 1 日の予算で Meter を作る。
func New(prices map[string]Price, dailyBudget float64, loc *time.Location) *Meter {
	return &Meter{prices: prices, budget: dailyBudget, loc: loc, now: time.Now, usage: map[Key]Usage{}, pending: map[string]string{}}
}

// SetClock は時刻の取り方を差し替える。テストで日付の切り替わりを確かめるため。
func (m *Meter) SetClock(now func() time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.now = now
}

// Plugin はランナーに渡すプラグインを返す。
func (m *Meter) Plugin() (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "usagemeter",
		BeforeModelCallback: func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			m.mu.Lock()
			defer m.mu.Unlock()
			m.pending[ctx.InvocationID()+"/"+ctx.AgentName()] = req.Model
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			// ストリーミングでは途中の応答にも使用量が付き、最後の集約した応答が同じ値を繰り返す。
			if resp != nil && resp.Partial {
				return nil, nil
			}
			m.mu.Lock()
			defer m.mu.Unlock()
			call := ctx.InvocationID() + "/" + ctx.AgentName()
			k := Key{Agent: ctx.AgentName(), Model: m.pending[call]}
			delete(m.pending, call)
			if resp == nil || resp.UsageMetadata == nil {
				return nil, nil
			}
			m.rollover()
			u := resp.UsageMetadata
			cur := m.usage[k]
			cur.Calls++
			cur.Prompt += int64(u.PromptTokenCount)
			cur.Cached += int64(u.CachedContentTokenCount)
			cur.ToolUsePrompt += int64(u.ToolUsePromptTokenCount)
			cur.Output += int64(u.CandidatesTokenCount)
			cur.Thoughts += int64(u.ThoughtsTokenCount)
			m.usage[k] = cur
			return nil, nil
		},
	})
}

func (m *Meter) rollover() {
	day := m.now().In(m.loc).Format(time.DateOnly)
	if day != m.day {
		m.day = day
		m.usage = map[Key]Usage{}
	}
}

// Report はその日の集計。
type Report struct {
	Day   string
	Usage map[Key]Usage
	Cost  float64
	// BudgetRatio は Cost を 1 日の予算で割った値。予算が 0 なら 0。
	BudgetRatio float64
	// Unpriced は単価の表に無いモデル。そのモデルの分は Cost に入っていない。
	Unpriced []string
}

// Report はいまの日付の集計を返す。日付が変わっていれば空の集計を返す。
func (m *Meter) Report() Report {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rollover()
	r := Report{Day: m.day, Usage: make(map[Key]Usage, len(m.usage))}
	for k, u := range m.usage {
		r.Usage[k] = u
		p, ok := m.prices[k.Model]
		if !ok {
			if !slices.Contains(r.Unpriced, k.Model) {
				r.Unpriced = append(r.Unpriced, k.Model)
			}
			continue
		}
		r.Cost += u.Cost(p)
	}
	slices.Sort(r.Unpriced)
	if m.budget > 0 {
		r.BudgetRatio = r.Cost / m.budget
	}
	return r
}
