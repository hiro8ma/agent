package evalharness

import (
	"strings"
	"testing"
)

func repTraj(output string, calls ...ToolCall) Trajectory {
	steps := make([]Step, 0, len(calls))
	for _, c := range calls {
		steps = append(steps, Step{ToolCalls: []ToolCall{c}})
	}
	return Trajectory{Input: "同じ入力", Output: output, Steps: steps}
}

func TestReproducibilityNeedsTwoRuns(t *testing.T) {
	s := Reproducibility([]Trajectory{repTraj("a")}, 0.8)
	if s.Scored {
		t.Error("1 件で採点が成立した")
	}
	if !strings.Contains(s.Reason, "2 件以上") {
		t.Errorf("理由が出ない: %q", s.Reason)
	}
}

// 手順が同じで文言だけ違う場合、層によって一致率が分かれる。
func TestReproducibilityLayersDiverge(t *testing.T) {
	call := ToolCall{Name: "search", Args: map[string]any{"q": "休暇"}}
	runs := []Trajectory{
		repTraj("休暇は年 20 日です", call),
		repTraj("年 20 日の休暇があります", call),
		repTraj("休暇: 20 日/年", call),
	}
	s := Reproducibility(runs, 0.8)
	if !s.Passed {
		t.Errorf("手順が同じなのに不合格: %v", s.Value)
	}
	if s.Value != 1 {
		t.Errorf("ツール一致率が %.2f。1.00 を期待", s.Value)
	}
	joined := strings.Join(s.Details, "\n")
	t.Log("\n" + joined)
	if !strings.Contains(joined, "出力の完全一致: 一致率 0.33") {
		t.Errorf("出力の一致率が層として出ていない:\n%s", joined)
	}
}

// 引数だけが揺れる場合、名前の列は一致してもツール一致率は落ちる。
func TestReproducibilityDetectsArgDrift(t *testing.T) {
	runs := []Trajectory{
		repTraj("同じ", ToolCall{Name: "search", Args: map[string]any{"q": "休暇"}}),
		repTraj("同じ", ToolCall{Name: "search", Args: map[string]any{"q": "有給休暇"}}),
		repTraj("同じ", ToolCall{Name: "search", Args: map[string]any{"q": "休暇制度"}}),
	}
	s := Reproducibility(runs, 0.8)
	if s.Passed {
		t.Error("引数が毎回違うのに合格した")
	}
	joined := strings.Join(s.Details, "\n")
	t.Log("\n" + joined)
	if !strings.Contains(joined, "ツール名の列: 一致率 1.00") {
		t.Errorf("名前の列は一致するはず:\n%s", joined)
	}
}

func TestReproducibilityIdenticalRuns(t *testing.T) {
	call := ToolCall{Name: "search", Args: map[string]any{"q": "休暇"}}
	runs := []Trajectory{repTraj("同じ", call), repTraj("同じ", call), repTraj("同じ", call)}
	s := Reproducibility(runs, 1.0)
	if !s.Passed || s.Value != 1 {
		t.Errorf("完全に同じ実行で %.2f", s.Value)
	}
	if s.Reason != "" {
		t.Errorf("一致しているのに理由が出た: %q", s.Reason)
	}
}
