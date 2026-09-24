package travelplanner

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const (
	skillsDir       = "../../travel/skills"
	budgetSkill     = "budget-allocation"
	budgetSkillBody = "予算が足りないときに削る順番"
)

// budgetCall は予算の生成へ届いた 1 回分の入力。
type budgetCall struct {
	system   string
	contents string
	tools    []string
}

// skillModel は予算の生成で use_skill が使えるなら 1 回呼び、その結果を読んでから予算を返す。
type skillModel struct {
	mu    sync.Mutex
	calls []budgetCall
}

func (s *skillModel) generate(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	var call budgetCall
	var system, contents strings.Builder
	for _, m := range req.Messages {
		if m.Role == ai.RoleSystem {
			system.WriteString(m.Text())
			continue
		}
		for _, p := range m.Content {
			contents.WriteString(p.Text)
			if p.ToolResponse != nil {
				fmt.Fprint(&contents, p.ToolResponse.Output)
			}
		}
	}
	for _, t := range req.Tools {
		call.tools = append(call.tools, t.Name)
	}
	call.system, call.contents = system.String(), contents.String()
	if !strings.Contains(call.system, roleBudget) {
		return textResponse("MARK"), nil
	}

	s.mu.Lock()
	s.calls = append(s.calls, call)
	s.mu.Unlock()
	last := req.Messages[len(req.Messages)-1]
	if slices.Contains(call.tools, "use_skill") && last.Role != ai.RoleTool {
		return &ai.ModelResponse{
			Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "use_skill", Input: map[string]any{"skillName": budgetSkill}}),
			}},
			FinishReason: ai.FinishReasonStop,
		}, nil
	}
	return textResponse(budgetJSON), nil
}

func TestBudgetGenerationLoadsSkillOnlyWhenConfigured(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		skillsDir     string
		wantCalls     int
		wantMetadata  bool
		wantBodyAfter bool
	}{
		"スキルを渡すと、メタデータだけが入り、use_skill の後の呼び出しに本文が入る": {
			skillsDir: skillsDir, wantCalls: 2, wantMetadata: true, wantBodyAfter: true,
		},
		"スキルを渡さないと、メタデータも本文も入らず 1 回で終わる": {
			skillsDir: "", wantCalls: 1, wantMetadata: false, wantBodyAfter: false,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			m := &skillModel{}
			g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/skill"))
			genkit.DefineModel(g, "test/skill", &ai.ModelOptions{
				Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true, Constrained: ai.ConstrainedSupportAll},
			}, m.generate)
			plan, err := DefineFlow(g, tc.skillsDir).Run(t.Context(), TripRequest{Request: "京都に1泊2日、予算3万円"})
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			if plan.Budget.TotalYen != 40000 {
				t.Fatalf("予算が返らない: %+v", plan.Budget)
			}

			if len(m.calls) != tc.wantCalls {
				t.Fatalf("予算の生成の呼び出し = %d, 期待 %d", len(m.calls), tc.wantCalls)
			}
			first, last := m.calls[0], m.calls[len(m.calls)-1]
			if got := strings.Contains(first.system, budgetSkill); got != tc.wantMetadata {
				t.Errorf("最初の呼び出しのスキルのメタデータ = %v, 期待 %v: %s", got, tc.wantMetadata, first.system)
			}
			if got := slices.Contains(first.tools, "use_skill"); got != tc.wantMetadata {
				t.Errorf("use_skill の有無 = %v, 期待 %v: %v", got, tc.wantMetadata, first.tools)
			}
			if strings.Contains(first.system+first.contents, budgetSkillBody) {
				t.Error("use_skill の前に SKILL.md の本文が入った")
			}
			if got := strings.Contains(last.contents, budgetSkillBody); got != tc.wantBodyAfter {
				t.Errorf("最後の呼び出しの SKILL.md の本文 = %v, 期待 %v", got, tc.wantBodyAfter)
			}
		})
	}
}
