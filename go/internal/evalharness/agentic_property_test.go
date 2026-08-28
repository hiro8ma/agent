package evalharness

import (
	"testing"
	"time"

	"pgregory.net/rapid"
)

// 例ベースのテストは「この軌跡ならこの点数」を確かめる。
// 特定の入力で期待値が合うことしか示せず、想定しなかった軌跡で崩れる。
//
// ここでは指標が満たすべき性質を書き、rapid に反例を探させる。
// 見つかった反例は縮約され、最小の再現例として報告される。
//
// 生成 AI 側の出力は確率的で例ベースの評価が効きにくいが、
// 指標そのものは決定的なので、性質を厳密に主張できる。

// 任意の軌跡を作る。指標が壊れる形を探させるため、極端な値も含める。
func genTrajectory(t *rapid.T) Trajectory {
	names := []string{"get_exchange_rate", "search_knowledge", "unknown_tool", ""}
	keys := []string{"from", "to", "date", "query", "limit", "currency", ""}

	genCall := rapid.Custom(func(t *rapid.T) ToolCall {
		args := map[string]any{}
		for _, k := range rapid.SliceOfN(rapid.SampledFrom(keys), 0, 4).Draw(t, "keys") {
			args[k] = rapid.SampledFrom([]string{"USD", "EUR", "GBP", "経費", ""}).Draw(t, "v")
		}
		return ToolCall{
			Name:   rapid.SampledFrom(names).Draw(t, "name"),
			Args:   args,
			Failed: rapid.Bool().Draw(t, "failed"),
		}
	})

	genStep := rapid.Custom(func(t *rapid.T) Step {
		return Step{
			ToolCalls: rapid.SliceOfN(genCall, 0, 3).Draw(t, "calls"),
			Tokens:    rapid.IntRange(0, 10000).Draw(t, "tokens"),
			Duration:  time.Duration(rapid.IntRange(0, 60000).Draw(t, "ms")) * time.Millisecond,
		}
	})

	return Trajectory{
		Input:  rapid.String().Draw(t, "input"),
		Output: rapid.String().Draw(t, "output"),
		Steps:  rapid.SliceOfN(genStep, 0, 8).Draw(t, "steps"),
	}
}

var budget = CostBudget{MaxTokens: 1000, MaxToolCalls: 5, MaxDuration: 2 * time.Second}

// どんな軌跡でもスコアは 0..1 に収まる。
//
// 範囲を外れると、複数指標をダッシュボードで並べた瞬間に平均が壊れる。
// 例ベースでは想定した軌跡しか確かめられない。
func TestPropertyScoresStayInRange(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		opt := rapid.IntRange(0, 10).Draw(t, "optimal")

		for _, s := range []AgentScore{
			ToolCallAccuracy(tr, specs, 0.9),
			StepEfficiency(tr, opt, 0.5),
			CostEfficiency(tr, budget, 0.2),
		} {
			if s.Value < 0 || s.Value > 1 {
				t.Fatalf("%s のスコアが範囲外: %v", s.Metric, s.Value)
			}
		}
	})
}

// 合否は必ずスコアと閾値の比較に一致する。
//
// ここがずれると、スコアを見て納得したのに合否が逆という状態になる。
func TestPropertyPassedMatchesThreshold(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		th := rapid.Float64Range(0, 1).Draw(t, "threshold")

		for _, s := range []AgentScore{
			ToolCallAccuracy(tr, specs, th),
			StepEfficiency(tr, rapid.IntRange(1, 10).Draw(t, "optimal"), th),
			CostEfficiency(tr, budget, th),
		} {
			// 採点が成立していないものは合否を持たない。
			// ここを区別しないと「採点しなかった」と「0 点だった」が混ざる。
			if !s.Scored {
				if s.Passed {
					t.Fatalf("%s: 採点していないのに合格になっている", s.Metric)
				}
				continue
			}
			if s.Passed != (s.Value >= s.Threshold) {
				t.Fatalf("%s: 合否 %v がスコア %.4f と閾値 %.4f に一致しない",
					s.Metric, s.Passed, s.Value, s.Threshold)
			}
		}
	})
}

// 定義に無いツールを足すと、ツール呼び出し精度は決して上がらない。
//
// 悪い呼び出しを足して点が上がる指標は、改善の方向を示せない。
func TestPropertyBadCallNeverImprovesAccuracy(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		if tr.Valid() != nil {
			t.Skip("採点対象外の軌跡")
		}
		before := ToolCallAccuracy(tr, specs, 0.9).Value

		tr.Steps = append(tr.Steps, Step{ToolCalls: []ToolCall{
			{Name: "definitely_not_defined", Args: map[string]any{"x": 1}}}})
		after := ToolCallAccuracy(tr, specs, 0.9).Value

		if after > before {
			t.Fatalf("定義に無いツールを足したのにスコアが %.4f → %.4f と上がった", before, after)
		}
	})
}

// ステップを足すと手順効率は決して上がらない。
//
// 空の軌跡は採点対象外なので除く。ここを含めると、
// 実行していない軌跡（採点不能）と 1 ステップの軌跡を比べることになり、
// 性質の主張として意味を成さない。
func TestPropertyMoreStepsNeverImprovesEfficiency(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		if tr.Valid() != nil {
			t.Skip("採点対象外の軌跡")
		}
		opt := rapid.IntRange(1, 10).Draw(t, "optimal")
		before := StepEfficiency(tr, opt, 0.5).Value

		tr.Steps = append(tr.Steps, Step{})
		after := StepEfficiency(tr, opt, 0.5).Value

		if after > before {
			t.Fatalf("ステップを足したのにスコアが %.4f → %.4f と上がった", before, after)
		}
	})
}

// 資源を余分に使うとコスト効率は決して上がらない。
func TestPropertyMoreCostNeverImprovesEfficiency(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		if tr.Valid() != nil {
			t.Skip("採点対象外の軌跡")
		}
		before := CostEfficiency(tr, budget, 0.2).Value

		tr.Steps = append(tr.Steps, Step{
			Tokens:   rapid.IntRange(1, 500).Draw(t, "extra_tokens"),
			Duration: time.Duration(rapid.IntRange(1, 1000).Draw(t, "extra_ms")) * time.Millisecond,
		})
		after := CostEfficiency(tr, budget, 0.2).Value

		if after > before {
			t.Fatalf("資源を余分に使ったのにスコアが %.4f → %.4f と上がった", before, after)
		}
	})
}

// 同じ軌跡を 2 回採点すれば同じ結果になる。
//
// map の反復順序に依存すると、同じ入力で結果が揺れる。
// 評価が揺れると、回帰なのか採点のばらつきなのか判別できなくなる。
func TestPropertyScoringIsDeterministic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tr := genTrajectory(t)
		opt := rapid.IntRange(1, 10).Draw(t, "optimal")

		for i := 0; i < 5; i++ {
			a := ToolCallAccuracy(tr, specs, 0.9)
			b := ToolCallAccuracy(tr, specs, 0.9)
			if a.Value != b.Value || a.Reason != b.Reason {
				t.Fatalf("同じ軌跡で結果が変わる: %v / %v", a, b)
			}
			if len(a.Details) != len(b.Details) {
				t.Fatalf("内訳の件数が変わる: %d / %d", len(a.Details), len(b.Details))
			}
			for j := range a.Details {
				if a.Details[j] != b.Details[j] {
					t.Fatalf("内訳の順序が安定しない:\n  %q\n  %q", a.Details[j], b.Details[j])
				}
			}
			if StepEfficiency(tr, opt, 0.5).Value != StepEfficiency(tr, opt, 0.5).Value {
				t.Fatal("StepEfficiency の結果が変わる")
			}
		}
	})
}

// 引数の並べ方を変えても呼び出しの同一性は変わらない。
//
// map は反復順序が保証されないため、署名を素朴に組み立てると
// 同じ呼び出しが別物として数えられ、重複の検出が漏れる。
func TestPropertySignatureIgnoresArgOrder(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		keys := rapid.SliceOfNDistinct(
			rapid.SampledFrom([]string{"a", "b", "c", "d", "e"}), 1, 5,
			func(s string) string { return s }).Draw(t, "keys")

		args := map[string]any{}
		for i, k := range keys {
			args[k] = i
		}
		copied := map[string]any{}
		for k, v := range args {
			copied[k] = v
		}

		a := ToolCall{Name: "t", Args: args}
		b := ToolCall{Name: "t", Args: copied}
		if a.signature() != b.signature() {
			t.Fatalf("同じ引数なのに署名が違う:\n  %s\n  %s", a.signature(), b.signature())
		}
	})
}

// 軌跡を連結すると、トークンとツール呼び出しの合計も連結される。
//
// 集計が壊れるとコスト評価が丸ごと信用できなくなる。
func TestPropertyTotalsAreAdditive(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := genTrajectory(t)
		b := genTrajectory(t)

		merged := Trajectory{Steps: append(append([]Step{}, a.Steps...), b.Steps...)}
		if got, want := merged.TotalTokens(), a.TotalTokens()+b.TotalTokens(); got != want {
			t.Fatalf("トークンの合計 = %d, 期待 %d", got, want)
		}
		if got, want := len(merged.ToolCalls()), len(a.ToolCalls())+len(b.ToolCalls()); got != want {
			t.Fatalf("呼び出し数 = %d, 期待 %d", got, want)
		}
		if got, want := merged.TotalDuration(), a.TotalDuration()+b.TotalDuration(); got != want {
			t.Fatalf("所要時間 = %v, 期待 %v", got, want)
		}
	})
}

// 引数の値が違えば署名も違う。
//
// 順序に依存しないことだけを主張しても、署名が引数を無視していれば
// その性質は成り立ってしまう。別の呼び出しを別物として数えられることは
// 別に主張する必要がある。実際に署名から値を落とす変異を入れたところ、
// 順序の性質だけでは捕まらなかった。
func TestPropertySignatureDistinguishesArgs(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		key := rapid.SampledFrom([]string{"from", "to", "query"}).Draw(t, "key")
		a := rapid.SampledFrom([]string{"USD", "EUR", "JPY", "経費"}).Draw(t, "a")
		b := rapid.SampledFrom([]string{"USD", "EUR", "JPY", "経費"}).Draw(t, "b")
		if a == b {
			t.Skip("同じ値では区別しようがない")
		}

		x := ToolCall{Name: "t", Args: map[string]any{key: a}}
		y := ToolCall{Name: "t", Args: map[string]any{key: b}}
		if x.signature() == y.signature() {
			t.Fatalf("引数が違うのに署名が同じ: %s（%s=%q と %q）", x.signature(), key, a, b)
		}
	})
}

// ツール名が違えば署名も違う。
func TestPropertySignatureDistinguishesNames(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		names := rapid.SliceOfNDistinct(
			rapid.SampledFrom([]string{"a", "b", "c"}), 2, 2,
			func(s string) string { return s }).Draw(t, "names")

		args := map[string]any{"k": "v"}
		x := ToolCall{Name: names[0], Args: args}
		y := ToolCall{Name: names[1], Args: args}
		if x.signature() == y.signature() {
			t.Fatalf("ツール名が違うのに署名が同じ: %s", x.signature())
		}
	})
}
