// Package hitl はツール実行の前に人の承認を挟む。
//
// ガードレールは条件に当たった行動を止める。HITL は判断を人へ回す。
// 止めるか通すかを事前に決められない行動に使う。
package hitl

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
)

// Decision は 1 回の判断の結果。
type Decision int

const (
	// Allow は承認なしで実行してよい。
	Allow Decision = iota
	// Ask は人の承認を求める。
	Ask
	// Deny は承認を求めずに拒む。
	Deny
)

func (d Decision) String() string {
	switch d {
	case Ask:
		return "承認を求める"
	case Deny:
		return "拒む"
	default:
		return "そのまま実行"
	}
}

// Policy は引数から判断を返す。
type Policy func(args map[string]any) (Decision, string)

// Threshold は数値の引数が上限を超えたら承認を求める。
//
// 上限は 2 つ持つ。ask を超えたら人へ回し、deny を超えたら拒む。
// 1 つだと「大きすぎる」と「確認が要る」が同じ扱いになる。
func Threshold(key string, ask, deny float64, unit string) Policy {
	return func(args map[string]any) (Decision, string) {
		v, ok := toFloat(args[key])
		if !ok {
			return Allow, ""
		}
		switch {
		case v > deny:
			return Deny, fmt.Sprintf("%s が %g%s。上限 %g%s を超える", key, v, unit, deny, unit)
		case v > ask:
			return Ask, fmt.Sprintf("%s が %g%s。承認の要る %g%s を超える", key, v, unit, ask, unit)
		}
		return Allow, ""
	}
}

// MatchArg は引数の値が一覧に含まれたら承認を求める。
func MatchArg(key string, values ...string) Policy {
	set := make(map[string]struct{}, len(values))
	for _, v := range values {
		set[strings.ToLower(v)] = struct{}{}
	}
	return func(args map[string]any) (Decision, string) {
		s, _ := args[key].(string)
		if _, hit := set[strings.ToLower(strings.TrimSpace(s))]; hit {
			return Ask, fmt.Sprintf("%s が %q。実行してよいか", key, s)
		}
		return Allow, ""
	}
}

// Any は最も重い判断を返す。Deny > Ask > Allow の順になる。
func Any(policies ...Policy) Policy {
	return func(args map[string]any) (Decision, string) {
		out, reason := Allow, ""
		for _, p := range policies {
			d, r := p(args)
			if d > out {
				out, reason = d, r
			}
		}
		return out, reason
	}
}

// Gate はツール実行の前に判断を挟む。
//
// 承認済みなら通し、未承認なら確認を要求して実行を止める。
// 承認の有無は ctx.ToolConfirmation() で分かる。
func Gate(ctx agent.Context, args map[string]any, policy Policy) (proceed bool, result map[string]any, err error) {
	decision, reason := policy(args)

	switch decision {
	case Deny:
		return false, map[string]any{"status": "denied", "reason": reason}, nil
	case Allow:
		return true, nil, nil
	}

	if c := ctx.ToolConfirmation(); c != nil {
		if c.Confirmed {
			return true, nil, nil
		}
		return false, map[string]any{"status": "rejected", "reason": "利用者が承認しなかった"}, nil
	}

	if err := ctx.RequestConfirmation(reason, args); err != nil {
		return false, nil, fmt.Errorf("request confirmation: %w", err)
	}
	return false, map[string]any{"status": "pending", "reason": reason}, nil
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}
