package adkeval

import (
	"context"
	"fmt"
	"io"
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
	Target      string       `json:"target"`
	EvalID      string       `json:"eval_id"`
	Trajectory  float64      `json:"trajectory"`
	Response    float64      `json:"response"`
	HasResponse bool         `json:"has_response"`
	Error       string       `json:"error,omitempty"`
	Actual      []Invocation `json:"actual,omitempty"`
}

// Passed は閾値をすべて満たし、実行に失敗していないかを返す。
func (r CaseResult) Passed(t Thresholds) bool {
	if r.Error != "" || r.Trajectory < t.Trajectory {
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
			r := CaseResult{Target: t.Name, EvalID: c.EvalID}
			actual, err := Infer(ctx, t.Agent, opts.UserID, c)
			if err != nil {
				r.Error = err.Error()
				results = append(results, r)
				continue
			}
			r.Actual = actual
			r.Trajectory = TrajectoryScore(actual, c.Conversation, opts.Match)
			r.Response, r.HasResponse = ResponseScore(actual, c.Conversation)
			results = append(results, r)
		}
	}
	return results
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
	fmt.Fprintln(tw, "case\ttarget\ttrajectory\tresponse\tresult\tnote")
	for _, r := range results {
		resp := "-"
		if r.HasResponse {
			resp = fmt.Sprintf("%.2f", r.Response)
		}
		mark := "PASS"
		if !r.Passed(t) {
			mark = "FAIL"
		}
		fmt.Fprintf(tw, "%s\t%s\t%.2f\t%s\t%s\t%s\n", r.EvalID, r.Target, r.Trajectory, resp, mark, noteOf(r))
	}
	return tw.Flush()
}

func noteOf(r CaseResult) string {
	if r.Error != "" {
		return truncate(r.Error, 60)
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
