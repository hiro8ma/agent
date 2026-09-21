package adkeval

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hiro8ma/agent/go/internal/evalharness"
)

// DefaultMaxInvocations は利用者役との対話の上限。ADK の LlmBackedUserSimulator の既定と同じ 20。
// 最初の発話も 1 回と数える。
const DefaultMaxInvocations = 20

// Simulator は会話の続きで、利用者役の次の発話を決める。
type Simulator interface {
	// Next は次の発話を返す。会話を終えるなら done を true にする。
	Next(ctx context.Context, sc Scenario, history []Invocation) (message string, done bool, err error)
}

// LLMSimulator は言語モデルに利用者役を演じさせる。
type LLMSimulator struct {
	LLM evalharness.LLM
}

type nextTurn struct {
	Finished bool   `json:"finished"`
	Message  string `json:"message"`
}

func (s *LLMSimulator) Next(ctx context.Context, sc Scenario, history []Invocation) (string, bool, error) {
	raw, err := s.LLM.GenerateJSON(ctx, simulatorPrompt(sc, history))
	if err != nil {
		return "", false, fmt.Errorf("adkeval: 利用者役の生成: %w", err)
	}
	var next nextTurn
	if err := json.Unmarshal([]byte(raw), &next); err != nil {
		return "", false, fmt.Errorf("adkeval: 利用者役の応答が JSON でない: %w", err)
	}
	if !next.Finished && strings.TrimSpace(next.Message) == "" {
		return "", false, fmt.Errorf("adkeval: 利用者役が発話も終了もしなかった")
	}
	return next.Message, next.Finished, nil
}

func simulatorPrompt(sc Scenario, history []Invocation) string {
	var b strings.Builder
	b.WriteString("あなたはエージェントと話す利用者を演じる。エージェント自身として答えてはいけない。\n\n")
	b.WriteString("## 会話の計画\n" + sc.ConversationPlan + "\n\n")
	if p := sc.UserPersona; p != nil {
		b.WriteString("## あなたの人物像\n" + p.Description + "\n")
		for _, bh := range p.Behaviors {
			b.WriteString("- " + bh.Name + " " + bh.Description + "\n")
			for _, ins := range bh.BehaviorInstructions {
				b.WriteString("  - " + ins + "\n")
			}
		}
		b.WriteString("\n")
	}
	b.WriteString("## ここまでの会話\n")
	for _, inv := range history {
		b.WriteString("利用者: " + inv.UserContent.Text() + "\n")
		b.WriteString("エージェント: " + inv.FinalResponse.Text() + "\n")
	}
	b.WriteString("\n計画をすべて終えたか、これ以上続けても進まないなら finished を true にする。\n")
	b.WriteString(`次の形の JSON だけを返す。{"finished": false, "message": "次の発話"}`)
	return b.String()
}

// Simulate は台本の最初の発話から始め、利用者役が終えるか上限に達するまで対話を回す。
// 上限に達したら reachedLimit を true にする。
func Simulate(ctx context.Context, agent Agent, sim Simulator, userID string, sc Scenario, maxInvocations int) (actual []Invocation, reachedLimit bool, err error) {
	if maxInvocations <= 0 {
		maxInvocations = DefaultMaxInvocations
	}
	sessionID, err := agent.NewSession(ctx, userID)
	if err != nil {
		return nil, false, err
	}
	message := sc.StartingPrompt
	for len(actual) < maxInvocations {
		user := Content{Role: "user", Parts: []Part{{Text: message}}}
		events, err := agent.Run(ctx, userID, sessionID, user)
		if err != nil {
			return actual, false, fmt.Errorf("%d ターン目: %w", len(actual)+1, err)
		}
		inv, err := fromEvents(user, events)
		if err != nil {
			return actual, false, fmt.Errorf("%d ターン目: %w", len(actual)+1, err)
		}
		actual = append(actual, inv)

		next, done, err := sim.Next(ctx, sc, actual)
		if err != nil {
			return actual, false, err
		}
		if done {
			return actual, false, nil
		}
		message = next
	}
	return actual, true, nil
}
