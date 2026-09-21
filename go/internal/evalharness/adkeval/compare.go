package adkeval

import (
	"context"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strings"
	"text/tabwriter"
	"time"
)

// Thresholds は合格ライン。
type Thresholds struct {
	Trajectory float64
	Response   float64
}

// CaseResult は 1 件のケースを 1 つのエージェントで流した結果。
type CaseResult struct {
	Target        string       `json:"target"`
	EvalID        string       `json:"eval_id"`
	Trajectory    float64      `json:"trajectory"`
	HasTrajectory bool         `json:"has_trajectory"`
	Response      float64      `json:"response"`
	HasResponse   bool         `json:"has_response"`
	Turns         int          `json:"turns"`
	ReachedLimit  bool         `json:"reached_limit,omitempty"`
	Violations    []string     `json:"violations,omitempty"`
	Error         string       `json:"error,omitempty"`
	Actual        []Invocation `json:"actual,omitempty"`
}

// Passed は閾値をすべて満たし、実行に失敗しておらず、禁止した内容を出していないかを返す。
// 利用者役との対話が上限まで伸びたケースも落とす。目的を果たせずに堂々巡りしている。
func (r CaseResult) Passed(t Thresholds) bool {
	if r.Error != "" || r.ReachedLimit || len(r.Violations) > 0 {
		return false
	}
	if r.HasTrajectory && r.Trajectory < t.Trajectory {
		return false
	}
	return !r.HasResponse || r.Response >= t.Response
}

// Target は比べる対象のエージェント。
type Target struct {
	Name  string
	Agent Agent
}

// Options は Compare の設定。
type Options struct {
	Match  MatchType
	UserID string
	// Pace はケースの間に空ける時間。無料枠の毎分の上限に当たらないようにする。
	Pace  time.Duration
	Cases []string // 空なら全件
	// Simulator は conversation_scenario のケースで利用者役を演じる。nil ならそのケースは失敗にする。
	Simulator      Simulator
	MaxInvocations int
	// Forbidden はエージェントの応答に出てはいけない内容（アンチゴール）。1 つでも当たれば不合格。
	Forbidden []*regexp.Regexp
}

// Compare は同じ評価セットを各エージェントに流して採点する。
// 実行に失敗したケースは採点せずに Error に残し、他のケースは続ける。
func Compare(ctx context.Context, set *EvalSet, targets []Target, opts Options) []CaseResult {
	if opts.UserID == "" {
		opts.UserID = "eval_user"
	}
	var results []CaseResult
	first := true
	for _, c := range set.EvalCases {
		if len(opts.Cases) > 0 && !slices.Contains(opts.Cases, c.EvalID) {
			continue
		}
		for _, t := range targets {
			if !first && opts.Pace > 0 {
				select {
				case <-ctx.Done():
					return results
				case <-time.After(opts.Pace):
				}
			}
			first = false
			results = append(results, runCase(ctx, t, c, opts))
		}
	}
	return results
}

func runCase(ctx context.Context, t Target, c EvalCase, opts Options) CaseResult {
	r := CaseResult{Target: t.Name, EvalID: c.EvalID}
	var actual []Invocation
	var err error
	if sc := c.ConversationScenario; sc != nil {
		if opts.Simulator == nil {
			r.Error = "conversation_scenario のケースに利用者役（Simulator）が無い"
			return r
		}
		actual, r.ReachedLimit, err = Simulate(ctx, t.Agent, opts.Simulator, opts.UserID, *sc, opts.MaxInvocations)
	} else {
		actual, err = Infer(ctx, t.Agent, opts.UserID, c)
	}
	r.Actual, r.Turns = actual, len(actual)
	r.Violations = violations(actual, opts.Forbidden)
	if err != nil {
		r.Error = err.Error()
		return r
	}
	if c.ConversationScenario == nil {
		r.Trajectory, r.HasTrajectory = TrajectoryScore(actual, c.Conversation, opts.Match), true
		r.Response, r.HasResponse = ResponseScore(actual, c.Conversation)
	}
	return r
}

func violations(actual []Invocation, forbidden []*regexp.Regexp) []string {
	var out []string
	for i, inv := range actual {
		text := inv.FinalResponse.Text()
		for _, re := range forbidden {
			if re.MatchString(text) {
				out = append(out, fmt.Sprintf("%d ターン目が %s に当たった", i+1, re))
			}
		}
	}
	return out
}

// AllPassed は全ケースが閾値を満たしたかを返す。CI の終了コードに使う。
func AllPassed(results []CaseResult, t Thresholds) bool {
	for _, r := range results {
		if !r.Passed(t) {
			return false
		}
	}
	return len(results) > 0
}

// WriteTable はケースごとの結果を表にする。
func WriteTable(w io.Writer, results []CaseResult, t Thresholds) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "case\ttarget\tturns\ttrajectory\tresponse\tresult\tnote")
	for _, r := range results {
		traj, resp := "-", "-"
		if r.HasTrajectory {
			traj = fmt.Sprintf("%.2f", r.Trajectory)
		}
		if r.HasResponse {
			resp = fmt.Sprintf("%.2f", r.Response)
		}
		mark := "PASS"
		if !r.Passed(t) {
			mark = "FAIL"
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\t%s\t%s\t%s\n", r.EvalID, r.Target, r.Turns, traj, resp, mark, noteOf(r))
	}
	return tw.Flush()
}

func noteOf(r CaseResult) string {
	switch {
	case r.Error != "":
		return truncate(r.Error, 60)
	case len(r.Violations) > 0:
		return truncate(strings.Join(r.Violations, " / "), 60)
	case r.ReachedLimit:
		return "利用者役が終える前に上限に達した"
	}
	var calls []string
	for _, inv := range r.Actual {
		for _, u := range inv.IntermediateData.ToolUses {
			calls = append(calls, fmt.Sprintf("%s%v", u.Name, u.Args))
		}
	}
	return strings.Join(calls, " ")
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
