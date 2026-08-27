package evalharness

import (
	"context"
	"fmt"
	"strings"
)

// TestCase は 1 件の評価対象。DeepEval の LLMTestCase に対応する。
type TestCase struct {
	Name           string
	Input          string
	ActualOutput   string
	ExpectedOutput string
	// Context は出力の根拠。RAG なら検索結果をそのまま入れる。
	Context []string
}

// Score は採点結果。
//
// 全指標で 1.0 が合格、0.0 が不合格に揃えてある。指標ごとに向きが違うと、
// 平均を取った瞬間に意味が壊れる。バイアスが強いほど品質が高い、という表ができる。
// DeepEval も 4.x でこの向きに統一した。
type Score struct {
	Metric    string
	Value     float64
	Threshold float64
	Passed    bool
	// Reason は減点された単位の理由。スコアだけでは何を直せばよいか分からない。
	Reason string
	// Verdicts は判定の内訳。集計値が正しく見えても内訳が壊れていることがある。
	Verdicts []Verdict
}

// Verdict は判定単位 1 つ分の結果。
type Verdict struct {
	Unit    string
	Verdict string
	Reason  string
	Good    bool
}

// Metric は採点方法。
type Metric interface {
	Name() string
	Threshold() float64
	Measure(ctx context.Context, llm LLM, tc TestCase) (Score, error)
}

// verdictMetric は DeepEval の指標に共通する骨格を実装する。
//
//  1. 判定単位に分解する（分解が要らない指標は directUnits を使う）
//  2. 1 単位ずつ yes / no で判定させる
//  3. 良い判定の割合を採点にする
type verdictMetric struct {
	name      string
	threshold float64

	// extractPrompt は判定単位を取り出すプロンプト。空文字なら分解しない。
	extractPrompt func(tc TestCase) string
	// directUnits は分解せずに単位が決まる場合に使う。Hallucination の文脈がこれ。
	directUnits func(tc TestCase) []string

	verdictPrompt func(tc TestCase, units []string) string

	// goodIsYes は yes と no のどちらを良い側に数えるか。
	// 関連性・忠実性は yes が良く、バイアス・毒性は no が良い。
	goodIsYes bool

	// requiresContext が true なら Context 未設定を設定漏れとして落とす。
	// 文脈が空だと判定単位が 0 件になり、採点は満点になる。
	// 「評価に通った」と「評価する対象が無かった」を取り違える事故を防ぐ。
	requiresContext bool
}

func (m *verdictMetric) Name() string       { return m.name }
func (m *verdictMetric) Threshold() float64 { return m.threshold }

func (m *verdictMetric) Measure(ctx context.Context, llm LLM, tc TestCase) (Score, error) {
	if tc.ActualOutput == "" {
		return Score{}, fmt.Errorf("%s: ActualOutput が空", m.name)
	}
	if m.requiresContext && len(tc.Context) == 0 {
		return Score{}, fmt.Errorf("%s: Context が空。この指標は根拠が無いと判定できない", m.name)
	}

	units, err := m.collectUnits(ctx, llm, tc)
	if err != nil {
		return Score{}, err
	}

	// 単位が 0 件なら満点。DeepEval も同じ扱いにしている。
	// 判定するものが無い以上、減点する根拠も無い。
	if len(units) == 0 {
		return Score{
			Metric: m.name, Value: 1, Threshold: m.threshold, Passed: true,
			Reason: "判定対象が抽出されなかったため満点",
		}, nil
	}

	var reply struct {
		Verdicts []struct {
			Verdict string `json:"verdict"`
			Reason  string `json:"reason"`
		} `json:"verdicts"`
	}
	if err := askJSON(ctx, llm, m.verdictPrompt(tc, units), &reply); err != nil {
		return Score{}, fmt.Errorf("%s の判定: %w", m.name, err)
	}
	if len(reply.Verdicts) != len(units) {
		return Score{}, fmt.Errorf("%s: 判定数 %d が単位数 %d と一致しない",
			m.name, len(reply.Verdicts), len(units))
	}

	// 悪い側の語に一致したときだけ減点する。判定が想定外の語で返っても
	// 良い側に倒れるため、判定の失敗が減点として現れない点は DeepEval と同じ。
	bad := "no"
	if !m.goodIsYes {
		bad = "yes"
	}

	verdicts := make([]Verdict, len(units))
	good := 0
	var reasons []string
	for i, v := range reply.Verdicts {
		isGood := strings.ToLower(strings.TrimSpace(v.Verdict)) != bad
		if isGood {
			good++
		} else if r := strings.TrimSpace(v.Reason); r != "" {
			reasons = append(reasons, r)
		}
		verdicts[i] = Verdict{Unit: units[i], Verdict: v.Verdict, Reason: v.Reason, Good: isGood}
	}

	value := float64(good) / float64(len(units))
	return Score{
		Metric:    m.name,
		Value:     value,
		Threshold: m.threshold,
		Passed:    value >= m.threshold,
		Reason:    strings.Join(reasons, " / "),
		Verdicts:  verdicts,
	}, nil
}

func (m *verdictMetric) collectUnits(ctx context.Context, llm LLM, tc TestCase) ([]string, error) {
	if m.directUnits != nil {
		return m.directUnits(tc), nil
	}

	var reply struct {
		Units []string `json:"units"`
	}
	if err := askJSON(ctx, llm, m.extractPrompt(tc), &reply); err != nil {
		return nil, fmt.Errorf("%s の分解: %w", m.name, err)
	}

	out := make([]string, 0, len(reply.Units))
	for _, u := range reply.Units {
		if u = strings.TrimSpace(u); u != "" {
			out = append(out, u)
		}
	}
	return out, nil
}

func numbered(items []string) string {
	var b strings.Builder
	for i, s := range items {
		fmt.Fprintf(&b, "%d. %s\n", i+1, s)
	}
	return b.String()
}

// verdictInstruction は判定パートの共通指示。
// 単位と同じ件数・同じ順で返させないと、判定と単位の対応が取れなくなる。
func verdictInstruction(n int, question string) string {
	return fmt.Sprintf(`上の %d 件それぞれについて、次の問いに "yes" か "no" で答えてください。
問い: %s

必ず %d 件、上と同じ順で、次の JSON 形式で返してください。
{"verdicts": [{"verdict": "yes または no", "reason": "理由を日本語で 1 文"}]}`, n, question, n)
}
