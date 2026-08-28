package evalharness

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// AgentScore は軌跡に対する採点。
//
// Value は他の指標と揃えて 0..1 で、1 が良い。教材が紹介する Azure AI の
// Task Adherence は 1〜5 の尺度になっているが、そのまま混ぜると
// 平均を取った瞬間に意味が壊れる。取り込むときは 0..1 に直す。
type AgentScore struct {
	Metric    string
	Value     float64
	Threshold float64
	Passed    bool
	Reason    string

	// Scored は採点が成立したかを表す。
	//
	// 採点できなかった場合と 0 点だった場合を同じ形で返すと、
	// 集計側で区別できない。CI で skip が緑になるのと同じ構造で、
	// 「評価しなかった」が「評価に通らなかった」と混ざる。
	// 実際に Passed と Value の関係が崩れ、性質ベースのテストで検出した。
	Scored bool
	// Details は内訳。集計値だけでは何を直せばよいか分からない。
	Details []string
}

func (s AgentScore) String() string {
	if !s.Scored {
		return fmt.Sprintf("%s: 採点なし\n  %s", s.Metric, s.Reason)
	}
	out := fmt.Sprintf("%s: %.2f (閾値 %.2f) %s",
		s.Metric, s.Value, s.Threshold, map[bool]string{true: "合格", false: "不合格"}[s.Passed])
	if s.Reason != "" {
		out += "\n  " + s.Reason
	}
	for _, d := range s.Details {
		out += "\n  - " + d
	}
	return out
}

// ---------- ツール呼び出し精度 ----------

// ToolCallAccuracy は各ツール呼び出しが定義に沿っているかを機械的に判定する。
//
// 教材が挙げる 3 つの判定基準のうち、2 つは定義との照合で決まる。
//
//	ツール定義に沿った正しいパラメータか  → specs と照合すれば決まる
//	呼び出しがタスクの文脈に即しているか  → 未定義のツールを呼んでいないかは決まる
//	呼び出し結果がタスク解決に貢献したか  → 実行の成否までは決まる
//
// 「文脈に即しているか」を厳密に測るには意味の判断が要るが、
// 実務で起きる失敗の多くは引数の不足・名前の間違い・未定義ツールの呼び出しで、
// これらは定義との照合で捕まる。LLM に投げるとコストと判定のばらつきが増える。
//
// スコアは 適切な呼び出し / 全呼び出し。呼び出しが 0 件なら 1 とする。
// 判定するものが無い以上、減点する根拠も無い。
func ToolCallAccuracy(t Trajectory, specs []ToolSpec, threshold float64) AgentScore {
	index := make(map[string]ToolSpec, len(specs))
	for _, s := range specs {
		index[s.Name] = s
	}

	if err := t.Valid(); err != nil {
		return AgentScore{Metric: "ToolCallAccuracy", Threshold: threshold,
			Reason: "採点できない: " + err.Error()}
	}

	calls := t.ToolCalls()
	if len(calls) == 0 {
		return AgentScore{Metric: "ToolCallAccuracy", Value: 1, Threshold: threshold,
			Passed: true, Scored: true, Reason: "ツール呼び出しが無いため満点"}
	}

	good := 0
	var details []string
	for i, c := range calls {
		if problems := checkCall(c, index); len(problems) == 0 {
			good++
		} else {
			details = append(details, fmt.Sprintf("%d 件目 %s: %s",
				i+1, c.Name, strings.Join(problems, " / ")))
		}
	}

	v := float64(good) / float64(len(calls))
	return AgentScore{
		Metric: "ToolCallAccuracy", Value: v, Threshold: threshold, Passed: v >= threshold, Scored: true,
		Reason:  fmt.Sprintf("%d 件中 %d 件が定義に沿っている", len(calls), good),
		Details: details,
	}
}

func checkCall(c ToolCall, index map[string]ToolSpec) []string {
	spec, ok := index[c.Name]
	if !ok {
		return []string{"定義に無いツール"}
	}

	allowed := make(map[string]bool, len(spec.Required)+len(spec.Optional))
	for _, k := range spec.Required {
		allowed[k] = true
	}
	for _, k := range spec.Optional {
		allowed[k] = true
	}

	var problems []string
	for _, k := range spec.Required {
		if _, ok := c.Args[k]; !ok {
			problems = append(problems, fmt.Sprintf("必須引数 %s が無い", k))
		}
	}

	// 未知の引数は順不同で出ると差分が読みにくいので並べる。
	var unknown []string
	for k := range c.Args {
		if !allowed[k] {
			unknown = append(unknown, k)
		}
	}
	sort.Strings(unknown)
	for _, k := range unknown {
		problems = append(problems, fmt.Sprintf("未定義の引数 %s", k))
	}

	// map の走査順は保証されない。順に並べないと、同じ呼び出しでも
	// 内訳の順序が実行ごとに変わり、採点が非決定的になる。
	enumKeys := make([]string, 0, len(spec.Enum))
	for k := range spec.Enum {
		enumKeys = append(enumKeys, k)
	}
	sort.Strings(enumKeys)

	for _, k := range enumKeys {
		cands := spec.Enum[k]
		v, ok := c.Args[k]
		if !ok {
			continue
		}
		s := fmt.Sprint(v)
		found := false
		for _, cand := range cands {
			if s == cand {
				found = true
				break
			}
		}
		if !found {
			problems = append(problems, fmt.Sprintf("%s の値 %q が候補にない", k, s))
		}
	}

	if c.Failed {
		problems = append(problems, "実行に失敗")
	}
	return problems
}

// ---------- ステップ効率 ----------

// StepEfficiency は手順の無駄を測る。
//
// スコアは 最小手数 / 実際の手数 で、1 が最短になる。
// 加えて同じ引数での重複呼び出しを検出する。同じ問いを 2 回投げるのは、
// 手数が増えるだけでなく前回の結果を使えていないことを示す。
//
// optimalSteps はそのタスクを解くのに要る最小のステップ数。
// テストケースごとに人が決める。自動で決められる値ではない。
// 空の軌跡は採点しない。ステップが 0 件なのは効率が悪いのではなく、
// タスクを実行していないことを表す。達成と効率を 1 つのスコアに混ぜると、
// 「ステップを足したら点が上がる」という指標として成り立たない状態になる。
// 実際そう書いていて、性質ベースのテストで検出した。
func StepEfficiency(t Trajectory, optimalSteps int, threshold float64) AgentScore {
	if optimalSteps <= 0 {
		optimalSteps = 1
	}
	if err := t.Valid(); err != nil {
		return AgentScore{Metric: "StepEfficiency", Threshold: threshold,
			Reason: "採点できない: " + err.Error()}
	}

	steps := len(t.Steps)
	v := float64(optimalSteps) / float64(steps)
	if v > 1 {
		// 最小手数より少ないのは、最小手数の見積もりが誤っているか、
		// 手順を飛ばして結論を出している。どちらも 1 で頭打ちにする。
		v = 1
	}

	seen := map[string]int{}
	var details []string
	for _, c := range t.ToolCalls() {
		sig := c.signature()
		seen[sig]++
		if seen[sig] == 2 {
			details = append(details, fmt.Sprintf("同一の呼び出しを繰り返している: %s", sig))
		}
	}
	sort.Strings(details)

	return AgentScore{
		Metric: "StepEfficiency", Value: v, Threshold: threshold, Passed: v >= threshold, Scored: true,
		Reason:  fmt.Sprintf("最小 %d ステップに対し実際は %d ステップ", optimalSteps, steps),
		Details: details,
	}
}

// ---------- コスト効率 ----------

// CostBudget は 1 タスクあたりの上限。
type CostBudget struct {
	MaxTokens    int
	MaxToolCalls int
	MaxDuration  time.Duration
}

// CostEfficiency は資源の消費を上限と比べる。
//
// スコアは各項目の余裕率のうち最も低いもの。平均にすると、
// 1 項目が上限を大きく超えていても他が余っていれば合格になる。
// 上限を超えた項目があれば、それがそのタスクの制約になる。
//
// 精度だけを追って資源を無視すると、精度は出るが本番に載らない実装ができる。
// 上限を先に決めて、超えたら落とす。
func CostEfficiency(t Trajectory, b CostBudget, threshold float64) AgentScore {
	if err := t.Valid(); err != nil {
		return AgentScore{Metric: "CostEfficiency", Threshold: threshold,
			Reason: "採点できない: " + err.Error()}
	}

	type item struct {
		name       string
		used, want float64
		unit       string
	}
	var items []item
	if b.MaxTokens > 0 {
		items = append(items, item{"トークン", float64(t.TotalTokens()), float64(b.MaxTokens), ""})
	}
	if b.MaxToolCalls > 0 {
		items = append(items, item{"ツール呼び出し", float64(len(t.ToolCalls())), float64(b.MaxToolCalls), " 回"})
	}
	if b.MaxDuration > 0 {
		items = append(items, item{"所要時間",
			float64(t.TotalDuration().Milliseconds()), float64(b.MaxDuration.Milliseconds()), " ms"})
	}
	if len(items) == 0 {
		return AgentScore{Metric: "CostEfficiency", Threshold: threshold,
			Reason: "上限が設定されていないため採点しない"}
	}

	worst := 1.0
	var details []string
	for _, it := range items {
		// 使い切って 0、未使用で 1 になる余裕率。
		v := 1 - it.used/it.want
		if v < 0 {
			v = 0
		}
		if v < worst {
			worst = v
		}
		mark := ""
		if it.used > it.want {
			mark = "  ← 上限超過"
		}
		details = append(details, fmt.Sprintf("%s %.0f%s / 上限 %.0f%s%s",
			it.name, it.used, it.unit, it.want, it.unit, mark))
	}

	return AgentScore{
		Metric: "CostEfficiency", Value: worst, Threshold: threshold, Passed: worst >= threshold, Scored: true,
		Reason:  "最も余裕の無い項目で採点する",
		Details: details,
	}
}
