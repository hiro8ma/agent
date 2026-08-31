// Package guardrail は ADK のコールバック機構でガードレールを組む。
//
// 教材はコールバックを 4 点に置く。
//
//	before_model  LLM へ送る前
//	after_model   LLM の出力を受けた後
//	before_tool   ツール実行の前
//	after_tool    ツール実行の後
//
// 4 点に分かれているのは、止めたい理由が段ごとに違うため。
// 送ってはいけない入力と、返してはいけない出力と、
// 呼んではいけないツールは、別の判断になる。
//
// ここで作るのは「失敗を大きな音にする」機構になる。
// 今週の実装で繰り返し踏んだのは、
// 例外が出ないまま誤った結果が最後まで流れる形だった。
//
//	状態の鍵を宣言し忘れて値が黙って捨てられる
//	判定の約束と解釈がずれて自己修正が常に False になる
//	検索が 0 件でも回数だけ数えて通る
//
// どれも「動いている」ように見える。
// 検査を通り道に置くと、通り道を通った回数と止めた回数が残る。
package guardrail

import (
	"fmt"
	"strings"
	"sync"
)

// Verdict は 1 件の検査の結果。
type Verdict struct {
	// Stage は検査を置いた段。before_model など。
	Stage string
	// Rule は当たった規則の名前。
	Rule string
	// Blocked は処理を止めたか。
	Blocked bool
	// Detail は何を見て判断したか。
	Detail string
}

// Log は検査の記録。
//
// 止めた件数だけでなく、通した件数も数える。
// 止めた数だけを見ると、検査が動いていないのと
// 「止めるものが無かった」の区別がつかない。
type Log struct {
	mu       sync.Mutex
	verdicts []Verdict
	passed   map[string]int
}

// NewLog は空の記録を返す。
func NewLog() *Log { return &Log{passed: map[string]int{}} }
func (l *Log) add(v Verdict) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if v.Blocked {
		l.verdicts = append(l.verdicts, v)
		return
	}
	l.passed[v.Stage]++
}

// Blocked は止めた検査を返す。
func (l *Log) Blocked() []Verdict {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]Verdict(nil), l.verdicts...)
}

// Passed は段ごとの通過件数を返す。
func (l *Log) Passed() map[string]int {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[string]int, len(l.passed))
	for k, v := range l.passed {
		out[k] = v
	}
	return out
}

// Summary は人が読む形にまとめる。
func (l *Log) Summary() string {
	var b strings.Builder
	p := l.Passed()
	for _, stage := range []string{"before_model", "after_model", "before_tool", "after_tool"} {
		fmt.Fprintf(&b, "  %-14s 通過 %d\n", stage, p[stage])
	}
	for _, v := range l.Blocked() {
		fmt.Fprintf(&b, "  止めた %-14s %s: %s\n", v.Stage, v.Rule, v.Detail)
	}
	return b.String()
}

// empty は結果が実質空かを判定する。
//
// keys を指定するとその鍵だけを見る。
// 指定が無ければ、値がすべて空とみなせるかを見る。
func empty(result map[string]any, keys []string) bool {
	if result == nil {
		return true
	}
	check := keys
	if len(check) == 0 {
		for k := range result {
			check = append(check, k)
		}
	}
	if len(check) == 0 {
		return true
	}
	for _, k := range check {
		switch v := result[k].(type) {
		case nil:
		case string:
			if strings.TrimSpace(v) != "" {
				return false
			}
		case []any:
			if len(v) > 0 {
				return false
			}
		case map[string]any:
			if len(v) > 0 {
				return false
			}
		default:
			return false
		}
	}
	return true
}
