package evalharness

import (
	"fmt"
	"sort"
	"strings"
)

// Reproducibility は同じ入力を繰り返した実行の一致度を測る。
//
// 一致は 1 つの数字にならない。
// 出力の文字列は違っても、呼んだツールの列は同じことがある。
// どの層で一致を求めるかは用途で決まるため、層ごとに出す。
//
//	tools   ツール呼び出しの列（名前と引数）
//	names   ツール名の列だけ（引数は問わない）
//	steps   ステップ数
//	output  出力文字列の完全一致
//
// 総合点は tools を使う。出力の文字列一致は言い換えで落ち、
// ステップ数だけでは中身が違っても一致する。
func Reproducibility(runs []Trajectory, threshold float64) AgentScore {
	s := AgentScore{Metric: "再現性", Threshold: threshold}
	if len(runs) < 2 {
		s.Reason = fmt.Sprintf("実行が %d 件。2 件以上が要る", len(runs))
		return s
	}

	layers := []struct {
		name string
		key  func(Trajectory) string
	}{
		{"ツール呼び出し（引数まで）", toolKey},
		{"ツール名の列", nameKey},
		{"ステップ数", func(t Trajectory) string { return fmt.Sprint(len(t.Steps)) }},
		{"出力の完全一致", func(t Trajectory) string { return strings.TrimSpace(t.Output) }},
	}

	s.Scored = true
	for i, l := range layers {
		rate, top := agreement(runs, l.key)
		s.Details = append(s.Details,
			fmt.Sprintf("%s: 一致率 %.2f（最多 %d/%d）", l.name, rate, top, len(runs)))
		if i == 0 {
			s.Value = rate
		}
	}
	s.Passed = s.Value >= threshold
	if s.Value < 1 {
		s.Reason = "実行ごとに手順が違う。温度・Seed・構造化出力を確かめる"
	}
	return s
}

// agreement は最頻の鍵が占める割合と、その件数を返す。
func agreement(runs []Trajectory, key func(Trajectory) string) (float64, int) {
	count := map[string]int{}
	for _, r := range runs {
		count[key(r)]++
	}
	top := 0
	for _, c := range count {
		if c > top {
			top = c
		}
	}
	return float64(top) / float64(len(runs)), top
}

func toolKey(t Trajectory) string {
	var b strings.Builder
	for _, st := range t.Steps {
		for _, c := range st.ToolCalls {
			b.WriteString(c.Name)
			b.WriteByte('(')
			keys := make([]string, 0, len(c.Args))
			for k := range c.Args {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Fprintf(&b, "%s=%v,", k, c.Args[k])
			}
			b.WriteString(");")
		}
	}
	return b.String()
}

func nameKey(t Trajectory) string {
	var names []string
	for _, st := range t.Steps {
		for _, c := range st.ToolCalls {
			names = append(names, c.Name)
		}
	}
	return strings.Join(names, ">")
}
