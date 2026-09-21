package adkeval

import (
	"encoding/json"
	"fmt"
	"os"
)

// Regression はベースラインから悪化した 1 件。
type Regression struct {
	Key    string `json:"key"`
	Reason string `json:"reason"`
}

// LoadResults は以前に書き出した結果を読む。
func LoadResults(path string) ([]CaseResult, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // ベースラインのパスは実行する人が指定する
	if err != nil {
		return nil, fmt.Errorf("adkeval: ベースラインの読み込み: %w", err)
	}
	var out []CaseResult
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("adkeval: ベースラインの解析: %w", err)
	}
	return out, nil
}

// Regressions は同じ対象と同じケースを比べ、合格から不合格に変わったものと、
// スコアが maxDrop を超えて下がったものを返す。ベースラインに無いケースは比べない。
func Regressions(baseline, current []CaseResult, th Thresholds, maxDrop float64) []Regression {
	before := make(map[string]CaseResult, len(baseline))
	for _, r := range baseline {
		before[r.Target+"/"+r.EvalID] = r
	}
	var out []Regression
	for _, now := range current {
		key := now.Target + "/" + now.EvalID
		was, ok := before[key]
		if !ok {
			continue
		}
		if was.Passed(th) && !now.Passed(th) {
			out = append(out, Regression{Key: key, Reason: "合格から不合格に変わった"})
		}
		if was.HasTrajectory && now.HasTrajectory && was.Trajectory-now.Trajectory > maxDrop {
			out = append(out, Regression{Key: key, Reason: fmt.Sprintf("trajectory が %.2f から %.2f に下がった", was.Trajectory, now.Trajectory)})
		}
		if was.HasResponse && now.HasResponse && was.Response-now.Response > maxDrop {
			out = append(out, Regression{Key: key, Reason: fmt.Sprintf("response が %.2f から %.2f に下がった", was.Response, now.Response)})
		}
	}
	return out
}
