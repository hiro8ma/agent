// Package guardrail は LLM 呼び出しとツール実行の前後に検査を置く。
//
// 検査は 4 種類。止めたい理由が段ごとに違うため分かれている。
//
//	before_model  送ってはいけない入力
//	after_model   返してはいけない出力
//	before_tool   引数が揃っていない呼び出し
//	after_tool    空を成功として扱わない
//
// 規則はこのパッケージに 1 か所で書き、
// 適用を adk.go と genkit.go で分ける。
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
// 通した件数も数える。止めた数だけでは、
// 検査が動いていないことと止めるものが無かったことを区別できない。
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
// keys を指定するとその鍵だけを、無指定なら全ての値を見る。
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
