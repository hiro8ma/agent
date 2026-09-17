// Package hitl はツール実行の前に人の承認を挟む。
//
// ガードレールは条件に当たった行動を止める。HITL は判断を人へ回す。
// 止めるか通すかを事前に決められない行動に使う。
//
// 書き口は 4 つあるが、下にある機構は 2 つだけになる。
//
//	ToolConfirmation 系
//	  functiontool.Config.RequireConfirmation   2 値。追加コード不要
//	  hitl.Gate                                 3 値。理由も返す
//
//	LongRunning 系
//	  functiontool.Config.IsLongRunning         自分で承認基盤へ投げる
//	  hitl.ApprovalNode                         Workflow の中断・再開に乗る
//
// 機構が違うのは、止まる場所と再開の経路が違うため。
// ToolConfirmation はツール呼び出し 1 回の中で完結する。
// LongRunning はイベントに LongRunningToolIDs を載せ、
// その ID への応答で再開する。Workflow の永続化に乗るのは後者になる。
//
// 選ぶ順は 3 つの問いで決まる。
//
//  1. 聞かずに拒む条件があるか
//     ある → Gate 以上。RequireConfirmation は bool しか返せない。
//     実測でも 1 億円の返金が確認要求になり、承認されれば通った
//  2. 1 回の対話で終わるか
//     終わらない → ApprovalNode。承認待ちが Workflow の状態として残る
//  3. 承認を外部へ投げるか
//     投げる → IsLongRunning。Slack へ通知して依頼 ID を返し、
//     後から届く応答で再開する
//
// ToolConfirmation は ADK v2.2.0 では experimental（default_on=True）。
// Gate を選んでも実験的な依存は避けられない。避けるなら LongRunning 系へ。
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
