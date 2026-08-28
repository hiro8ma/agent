package evalharness

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Trajectory は 1 タスクの実行記録。
//
// 生成 AI の評価は最終出力だけを見れば足りるが、エージェントの評価はそれでは足りない。
// 正しい答えに偶然たどり着いた実行と、筋の通った手順でたどり着いた実行を区別できない。
// 前者は次の入力で崩れる。
//
// 出力に至るまでの手順・使った道具・消費した資源をまとめて記録する。
type Trajectory struct {
	Input  string
	Output string
	Steps  []Step
}

// Step は 1 ターン分の記録。LLM 呼び出し 1 回とそれに伴うツール実行にあたる。
type Step struct {
	ToolCalls []ToolCall
	Tokens    int
	Duration  time.Duration
}

// ToolCall はツール呼び出し 1 回。
type ToolCall struct {
	Name string
	Args map[string]any
	// Failed はツールの実行が失敗したことを表す。
	// 呼び出しの妥当性とは別に数える。正しいツールを正しい引数で呼んでも
	// 外部サービス側の都合で失敗することがある。
	Failed bool
}

// ToolSpec はツールの定義。引数の妥当性はここと照合して機械的に判定する。
//
// 教材はツール呼び出し精度を LLM ベースの評価器で測る形で紹介しているが、
// 「ツール定義に沿った正しいパラメータを与えているか」は定義との照合で決まる。
// LLM に投げると、コストと判定のばらつきを増やしたうえで精度が落ちる。
type ToolSpec struct {
	Name     string
	Required []string
	// Optional に無く Required にも無い引数は未知の引数として減点する。
	Optional []string
	// Enum は値の候補が決まっている引数。空なら値は検査しない。
	Enum map[string][]string
}

// Valid は軌跡が採点の対象になるかを返す。
//
// ステップが 1 つも無い軌跡はタスクを実行していない。これは効率や精度の問題ではなく、
// 達成の問題になる。効率の指標に混ぜると「ステップを足したら点が上がる」形になり、
// 改善の方向を示せなくなる。採点の前に切り分ける。
func (t Trajectory) Valid() error {
	if len(t.Steps) == 0 {
		return errors.New("ステップが 1 つも記録されていない。タスクが実行されていない")
	}
	return nil
}

// ToolCalls は軌跡に含まれる全ツール呼び出しを順に返す。
func (t Trajectory) ToolCalls() []ToolCall {
	var out []ToolCall
	for _, s := range t.Steps {
		out = append(out, s.ToolCalls...)
	}
	return out
}

// TotalTokens は消費したトークンの合計。
func (t Trajectory) TotalTokens() int {
	n := 0
	for _, s := range t.Steps {
		n += s.Tokens
	}
	return n
}

// TotalDuration は所要時間の合計。
func (t Trajectory) TotalDuration() time.Duration {
	var d time.Duration
	for _, s := range t.Steps {
		d += s.Duration
	}
	return d
}

// signature は引数まで含めた呼び出しの同一性を表す。重複の検出に使う。
func (c ToolCall) signature() string {
	keys := make([]string, 0, len(c.Args))
	for k := range c.Args {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	b.WriteString(c.Name)
	b.WriteString("(")
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%s=%v", k, c.Args[k])
	}
	b.WriteString(")")
	return b.String()
}
