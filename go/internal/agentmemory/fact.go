// Package agentmemory は長期記憶へ書く事実を検証する。
//
// 記憶は 1 回の誤りが繰り返し読まれる。
// 推論 1 回で終わる誤りと違い、書いた後の全ての読み出しに乗る。
package agentmemory

import (
	"fmt"
	"strings"
	"time"
)

// Fact は長期記憶に置く 1 件。
type Fact struct {
	Subject   string
	Statement string
	// Source は何を根拠に書いたか。空の Fact は検証できない。
	Source string
	// Confidence は 0..1。断定と推測を同じ形で保存しない。
	Confidence float64
	WrittenAt  time.Time
	// TTL は失効までの期間。0 なら失効しない。
	TTL time.Duration
}

func (f Fact) expired(now time.Time) bool {
	return f.TTL > 0 && now.Sub(f.WrittenAt) >= f.TTL
}

func (f Fact) String() string {
	return fmt.Sprintf("%s: %s (確度 %.2f, 出典 %s)", f.Subject, f.Statement, f.Confidence, f.Source)
}

// Rule は Fact を検査する。問題があれば理由を返す。
type Rule func(Fact) error

// RequireSource は根拠の無い Fact を拒む。
func RequireSource(f Fact) error {
	if strings.TrimSpace(f.Source) == "" {
		return fmt.Errorf("出典が無い")
	}
	return nil
}

// MinConfidence は確度が閾値に満たない Fact を拒む。
func MinConfidence(min float64) Rule {
	return func(f Fact) error {
		if f.Confidence < min {
			return fmt.Errorf("確度 %.2f が閾値 %.2f 未満", f.Confidence, min)
		}
		return nil
	}
}

// RejectHedged は推測の語を含む文を拒む。
//
// 推測を断定として保存すると、読み出す側で区別できなくなる。
func RejectHedged(f Fact) error {
	for _, w := range []string{"かもしれない", "おそらく", "たぶん", "と思われる", "推測"} {
		if strings.Contains(f.Statement, w) {
			return fmt.Errorf("推測の語 %q を含む", w)
		}
	}
	return nil
}

// NotExpired は失効した Fact を拒む。読み出し側で使う。
func NotExpired(now func() time.Time) Rule {
	return func(f Fact) error {
		if f.expired(now()) {
			return fmt.Errorf("失効している（TTL %v）", f.TTL)
		}
		return nil
	}
}
