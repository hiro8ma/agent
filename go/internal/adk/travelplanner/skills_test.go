package travelplanner

import (
	"context"
	"encoding/json"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/travel"
)

const (
	skillsDir       = "../../travel/skills"
	budgetSkill     = "budget-allocation"
	budgetSkillBody = "予算が足りないときに削る順番"
)

// budgetCall は予算の担当へ届いた 1 回分の入力。
type budgetCall struct {
	system   string
	contents string
	tools    []string
}

// skillModel は予算の担当で load_skill が使えるなら 1 回呼び、その結果を読んでから予算を返す。
type skillModel struct {
	mu    sync.Mutex
	calls []budgetCall
}

func (*skillModel) Name() string { return "skill" }

func (s *skillModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	call := readCall(req)
	reply := genai.NewContentFromText("MARK", genai.RoleModel)
	if strings.Contains(call.system, "旅行予算の計算担当") {
		s.mu.Lock()
		s.calls = append(s.calls, call)
		s.mu.Unlock()
		reply = genai.NewContentFromText(budgetJSON, genai.RoleModel)
		if slices.Contains(call.tools, "load_skill") && !strings.Contains(call.contents, `"skill_name"`) {
			reply = genai.NewContentFromFunctionCall("load_skill", map[string]any{"name": budgetSkill}, genai.RoleModel)
		}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: reply, TurnComplete: true}, nil)
	}
}

func readCall(req *model.LLMRequest) budgetCall {
	var call budgetCall
	var sys, contents strings.Builder
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, p := range req.Config.SystemInstruction.Parts {
			sys.WriteString(p.Text)
		}
	}
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			contents.WriteString(p.Text)
			if p.FunctionResponse != nil {
				b, _ := json.Marshal(p.FunctionResponse.Response)
				contents.Write(b)
			}
		}
	}
	if req.Config != nil {
		for _, t := range req.Config.Tools {
			for _, fd := range t.FunctionDeclarations {
				call.tools = append(call.tools, fd.Name)
			}
		}
	}
	call.system, call.contents = sys.String(), contents.String()
	return call
}

func TestBudgetReporterLoadsSkillOnlyWhenConfigured(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		withSkills    bool
		wantCalls     int
		wantMetadata  bool
		wantBodyAfter bool
	}{
		"スキルを渡すと、メタデータだけが入り、load_skill の後の呼び出しに本文が入る": {
			withSkills: true, wantCalls: 2, wantMetadata: true, wantBodyAfter: true,
		},
		"スキルを渡さないと、メタデータも本文も入らず 1 回で終わる": {
			withSkills: false, wantCalls: 1, wantMetadata: false, wantBodyAfter: false,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var skills tool.Toolset
			if tc.withSkills {
				var err error
				skills, err = NewSkillToolset(t.Context(), skillsDir)
				if err != nil {
					t.Fatal(err)
				}
			}
			m := &skillModel{}
			p, err := NewPlanner(m, skills)
			if err != nil {
				t.Fatal(err)
			}
			var plan *travel.Plan
			req := &travel.PlanRequest{UserID: "u", Message: "京都に1泊2日、予算3万円"}
			for resp, err := range p.Plan(t.Context(), req) {
				if err != nil {
					t.Fatalf("Plan: %v", err)
				}
				if resp.Plan != nil {
					plan = resp.Plan
				}
			}
			if plan == nil || plan.Budget.TotalYen != 40000 {
				t.Fatalf("予算が返らない: %+v", plan)
			}

			if len(m.calls) != tc.wantCalls {
				t.Fatalf("予算の担当の呼び出し = %d, 期待 %d", len(m.calls), tc.wantCalls)
			}
			first, last := m.calls[0], m.calls[len(m.calls)-1]
			if got := strings.Contains(first.system, budgetSkill); got != tc.wantMetadata {
				t.Errorf("最初の呼び出しのスキルのメタデータ = %v, 期待 %v: %s", got, tc.wantMetadata, first.system)
			}
			if got := slices.Contains(first.tools, "load_skill"); got != tc.wantMetadata {
				t.Errorf("load_skill の有無 = %v, 期待 %v: %v", got, tc.wantMetadata, first.tools)
			}
			if strings.Contains(first.system+first.contents, budgetSkillBody) {
				t.Error("load_skill の前に SKILL.md の本文が入った")
			}
			if got := strings.Contains(last.contents, budgetSkillBody); got != tc.wantBodyAfter {
				t.Errorf("最後の呼び出しの SKILL.md の本文 = %v, 期待 %v", got, tc.wantBodyAfter)
			}
		})
	}
}
