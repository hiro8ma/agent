package adkeval

import (
	"context"
	"fmt"
)

// Infer はケースの発話を順に流し、ターンごとの実際の Invocation を返す。
//
// 2 ターン目以降の履歴には、期待する応答ではなくエージェント自身の応答が入る。
// Python の adk eval と同じで、1 ターン目で外れると後のターンも外れやすい。
func Infer(ctx context.Context, agent Agent, userID string, c EvalCase) ([]Invocation, error) {
	sessionID, err := agent.NewSession(ctx, userID)
	if err != nil {
		return nil, err
	}
	actual := make([]Invocation, 0, len(c.Conversation))
	for i, turn := range c.Conversation {
		events, err := agent.Run(ctx, userID, sessionID, turn.UserContent)
		if err != nil {
			return nil, fmt.Errorf("%s の %d ターン目: %w", c.EvalID, i+1, err)
		}
		inv, err := fromEvents(turn.UserContent, events)
		if err != nil {
			return nil, fmt.Errorf("%s の %d ターン目: %w", c.EvalID, i+1, err)
		}
		actual = append(actual, inv)
	}
	return actual, nil
}

// fromEvents はイベント列から、ツール呼び出しと最終応答を取り出す。
// 最終応答は、利用者以外が書いた、関数呼び出しを含まない最後のテキスト。
//
// errorCode の付いたイベントがあれば採点せずにエラーにする。
// ガードレールがモデルの失敗を決まった文に置き換えると、中身だけでは普通の応答と見分けられない。
func fromEvents(user Content, events []Event) (Invocation, error) {
	inv := Invocation{UserContent: user}
	for _, ev := range events {
		if ev.ErrorCode != "" {
			return inv, fmt.Errorf("エージェントが errorCode %s を返した: %s", ev.ErrorCode, ev.ErrorMessage)
		}
		if ev.Content == nil || ev.Partial || ev.Author == "user" {
			continue
		}
		hasCall := false
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				hasCall = true
				inv.IntermediateData.ToolUses = append(inv.IntermediateData.ToolUses, *p.FunctionCall)
			}
		}
		if !hasCall && ev.Content.Text() != "" {
			c := *ev.Content
			inv.FinalResponse = &c
		}
	}
	return inv, nil
}
