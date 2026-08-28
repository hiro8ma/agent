package evalharness

import (
	"strings"
	"testing"
	"time"
)

var specs = []ToolSpec{
	{
		Name:     "get_exchange_rate",
		Required: []string{"from", "to"},
		Optional: []string{"date"},
		Enum:     map[string][]string{"from": {"USD", "EUR", "JPY"}, "to": {"USD", "EUR", "JPY"}},
	},
	{Name: "search_knowledge", Required: []string{"query"}, Optional: []string{"limit"}},
}

func call(name string, args map[string]any) ToolCall { return ToolCall{Name: name, Args: args} }

func traj(calls ...ToolCall) Trajectory {
	steps := make([]Step, len(calls))
	for i, c := range calls {
		steps[i] = Step{ToolCalls: []ToolCall{c}}
	}
	return Trajectory{Steps: steps}
}

func TestToolCallAccuracy(t *testing.T) {
	tests := []struct {
		name      string
		traj      Trajectory
		want      float64
		wantInDet string
	}{
		{
			// 教材の正例。
			name: "定義どおりなら 1.0",
			traj: traj(call("get_exchange_rate", map[string]any{"from": "USD", "to": "EUR"})),
			want: 1.0,
		},
		{
			// 教材の誤例。必須引数が足りない。
			name:      "必須引数が無ければ 0",
			traj:      traj(call("get_exchange_rate", map[string]any{"currency": "EUR"})),
			want:      0.0,
			wantInDet: "必須引数 from が無い",
		},
		{
			name:      "定義に無いツールは 0",
			traj:      traj(call("get_weather", map[string]any{"city": "Tokyo"})),
			want:      0.0,
			wantInDet: "定義に無いツール",
		},
		{
			name: "任意引数は減点しない",
			traj: traj(call("get_exchange_rate",
				map[string]any{"from": "USD", "to": "JPY", "date": "2026-08-28"})),
			want: 1.0,
		},
		{
			// 定義に無い引数は、モデルが引数名を取り違えた兆候になる。
			name: "未定義の引数は減点する",
			traj: traj(call("get_exchange_rate",
				map[string]any{"from": "USD", "to": "EUR", "currency": "EUR"})),
			want:      0.0,
			wantInDet: "未定義の引数 currency",
		},
		{
			// 値の候補が決まっている引数は値まで見る。
			name:      "候補にない値は減点する",
			traj:      traj(call("get_exchange_rate", map[string]any{"from": "USD", "to": "GBP"})),
			want:      0.0,
			wantInDet: `to の値 "GBP" が候補にない`,
		},
		{
			name: "2 件中 1 件が正しければ 0.5",
			traj: traj(
				call("get_exchange_rate", map[string]any{"from": "USD", "to": "EUR"}),
				call("search_knowledge", map[string]any{}),
			),
			want:      0.5,
			wantInDet: "必須引数 query が無い",
		},
		{
			// 正しく呼んでも外部サービス側で失敗することがある。
			// 呼び出しの妥当性とは別軸だが、タスク解決に貢献していない点は同じ。
			name: "実行に失敗した呼び出しは減点する",
			traj: Trajectory{Steps: []Step{{ToolCalls: []ToolCall{
				{Name: "search_knowledge", Args: map[string]any{"query": "経費"}, Failed: true}}}}},
			want:      0.0,
			wantInDet: "実行に失敗",
		},
		{
			name: "呼び出しが無ければ満点",
			traj: Trajectory{Steps: []Step{{}}},
			want: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ToolCallAccuracy(tt.traj, specs, 0.9)
			if diff := got.Value - tt.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("スコア = %v, 期待 %v\n%s", got.Value, tt.want, got)
			}
			if tt.wantInDet != "" && !strings.Contains(strings.Join(got.Details, "\n"), tt.wantInDet) {
				t.Errorf("内訳に %q が含まれない:\n%s", tt.wantInDet, got)
			}
		})
	}
}

func TestStepEfficiency(t *testing.T) {
	tests := []struct {
		name    string
		steps   int
		optimal int
		want    float64
	}{
		{"最短なら 1.0", 2, 2, 1.0},
		{"倍かかれば 0.5", 4, 2, 0.5},
		{"最小より少なくても 1.0 で頭打ち", 1, 2, 1.0},
		{"1 ステップ想定で 4 ステップなら 0.25", 4, 1, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StepEfficiency(Trajectory{Steps: make([]Step, tt.steps)}, tt.optimal, 0.5)
			if diff := got.Value - tt.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("スコア = %v, 期待 %v", got.Value, tt.want)
			}
		})
	}

	// 同じ引数での再呼び出しは、前回の結果を使えていない兆候になる。
	t.Run("同一の呼び出しの繰り返しを検出する", func(t *testing.T) {
		same := map[string]any{"query": "経費精算の締め日"}
		got := StepEfficiency(traj(
			call("search_knowledge", same),
			call("search_knowledge", same),
		), 1, 0.1)

		if len(got.Details) != 1 {
			t.Fatalf("重複を検出していない: %+v", got.Details)
		}
		if !strings.Contains(got.Details[0], "繰り返している") {
			t.Errorf("内訳が伝わらない: %s", got.Details[0])
		}
	})

	// 引数が違えば別の問い合わせなので重複ではない。
	t.Run("引数が違えば重複としない", func(t *testing.T) {
		got := StepEfficiency(traj(
			call("search_knowledge", map[string]any{"query": "経費"}),
			call("search_knowledge", map[string]any{"query": "リモートワーク"}),
		), 2, 0.5)

		if len(got.Details) != 0 {
			t.Errorf("重複でないものを検出した: %+v", got.Details)
		}
	})

	// 引数の順序で署名が変わると、同じ呼び出しを別物と数えてしまう。
	t.Run("引数の順序で判定が変わらない", func(t *testing.T) {
		a := call("get_exchange_rate", map[string]any{"from": "USD", "to": "EUR"})
		b := call("get_exchange_rate", map[string]any{"to": "EUR", "from": "USD"})
		if a.signature() != b.signature() {
			t.Errorf("署名が一致しない:\n  %s\n  %s", a.signature(), b.signature())
		}
	})
}

func TestCostEfficiency(t *testing.T) {
	budget := CostBudget{MaxTokens: 1000, MaxToolCalls: 5, MaxDuration: 2 * time.Second}

	tests := []struct {
		name   string
		traj   Trajectory
		want   float64
		passed bool
	}{
		{
			name:   "半分の消費なら 0.5",
			traj:   Trajectory{Steps: []Step{{Tokens: 500, Duration: time.Second}}},
			want:   0.5,
			passed: true,
		},
		{
			// 平均を取ると 1 項目の超過が他の余裕で埋められてしまう。
			// 最も余裕の無い項目で決める。
			name: "1 項目が上限を超えたら 0",
			traj: Trajectory{Steps: []Step{{Tokens: 2000, Duration: time.Millisecond}}},
			want: 0, passed: false,
		},
		{
			name:   "未使用に近ければ 1 に近い",
			traj:   Trajectory{Steps: []Step{{Tokens: 10, Duration: time.Millisecond}}},
			want:   0.99,
			passed: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CostEfficiency(tt.traj, budget, 0.2)
			if diff := got.Value - tt.want; diff > 0.01 || diff < -0.01 {
				t.Errorf("スコア = %v, 期待 %v\n%s", got.Value, tt.want, got)
			}
			if got.Passed != tt.passed {
				t.Errorf("合否 = %v, 期待 %v", got.Passed, tt.passed)
			}
		})
	}

	t.Run("ツール呼び出し回数も上限に入る", func(t *testing.T) {
		var steps []Step
		for i := 0; i < 6; i++ {
			steps = append(steps, Step{Tokens: 1,
				ToolCalls: []ToolCall{call("search_knowledge", map[string]any{"query": "x"})}})
		}
		got := CostEfficiency(Trajectory{Steps: steps}, budget, 0.2)
		if got.Passed {
			t.Errorf("6 回で上限 5 回を超えているのに合格した\n%s", got)
		}
		if !strings.Contains(strings.Join(got.Details, "\n"), "上限超過") {
			t.Errorf("超過した項目が分からない:\n%s", got)
		}
	})

	t.Run("上限が無ければ採点しない", func(t *testing.T) {
		got := CostEfficiency(Trajectory{Steps: []Step{{Tokens: 999999}}}, CostBudget{}, 0.5)
		if got.Scored {
			t.Errorf("上限未設定なのに採点した: %s", got)
		}
	})

	// 実行されていない軌跡は採点対象外にする。
	// 0 点として扱うと「ステップを足したら点が上がる」形になり、指標として成り立たない。
	t.Run("空の軌跡は採点しない", func(t *testing.T) {
		for _, got := range []AgentScore{
			ToolCallAccuracy(Trajectory{}, specs, 0.9),
			StepEfficiency(Trajectory{}, 1, 0.5),
			CostEfficiency(Trajectory{}, budget, 0.2),
		} {
			if got.Scored || got.Passed {
				t.Errorf("%s: 空の軌跡を採点した: %s", got.Metric, got)
			}
			if !strings.Contains(got.Reason, "実行されていない") {
				t.Errorf("%s: 理由が伝わらない: %s", got.Metric, got.Reason)
			}
		}
	})
}

// 3 指標がすべて 0..1 で 1 が良い向きに揃っていることを確かめる。
// 向きが混ざると、平均を取った瞬間に意味が壊れる。
func TestScoreDirectionIsConsistent(t *testing.T) {
	good := Trajectory{Steps: []Step{{
		Tokens: 10, Duration: time.Millisecond,
		ToolCalls: []ToolCall{call("search_knowledge", map[string]any{"query": "経費"})}}}}
	bad := Trajectory{Steps: []Step{
		{Tokens: 5000, Duration: time.Minute,
			ToolCalls: []ToolCall{call("unknown_tool", map[string]any{})}},
		{Tokens: 5000, ToolCalls: []ToolCall{call("unknown_tool", map[string]any{})}},
	}}
	budget := CostBudget{MaxTokens: 1000, MaxToolCalls: 5, MaxDuration: 2 * time.Second}

	for _, m := range []struct {
		name        string
		goodV, badV float64
	}{
		{"ToolCallAccuracy",
			ToolCallAccuracy(good, specs, 0.9).Value, ToolCallAccuracy(bad, specs, 0.9).Value},
		{"StepEfficiency",
			StepEfficiency(good, 1, 0.5).Value, StepEfficiency(bad, 1, 0.5).Value},
		{"CostEfficiency",
			CostEfficiency(good, budget, 0.2).Value, CostEfficiency(bad, budget, 0.2).Value},
	} {
		if m.goodV <= m.badV {
			t.Errorf("%s: 良い軌跡 %.2f が悪い軌跡 %.2f を上回らない。向きが逆の可能性がある",
				m.name, m.goodV, m.badV)
		}
		for _, v := range []float64{m.goodV, m.badV} {
			if v < 0 || v > 1 {
				t.Errorf("%s: スコア %.2f が 0..1 の範囲外", m.name, v)
			}
		}
	}
}
