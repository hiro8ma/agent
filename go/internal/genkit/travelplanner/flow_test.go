package travelplanner

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/travel"
)

const (
	roleSpots       = "観光スポットの専門家"
	roleRestaurants = "グルメの専門家"
	roleTransport   = "交通手段の専門家"
	roleSchedule    = "旅行スケジュールの作成担当"
	roleBudget      = "旅行予算の計算担当"

	researcherCount = 3
	barrierWait     = 500 * time.Millisecond
	budgetJSON      = `{"currency":"JPY","transportation_yen":28000,"food_yen":9000,"activities_yen":3000,"total_yen":40000,"assumptions":["新幹線のぞみで往復"]}`
)

var (
	marks = map[string]string{
		roleSpots:       "MARK_SPOTS",
		roleRestaurants: "MARK_REST",
		roleTransport:   "MARK_TRANS",
		roleSchedule:    "MARK_SCHEDULE",
	}
	scheduleChunks = []string{"MARK_", "SCHEDULE"}
	toolInputs     = map[string]map[string]any{
		"search_spots":       {"destination": "京都", "preferences": "歴史"},
		"search_restaurants": {"destination": "京都", "cuisine": "和食"},
		"search_transport":   {"origin": "東京", "destination": "京都"},
	}
)

// fakeModel は system prompt で役を見分け、ツールを 1 回呼んでから目印と検索結果を返す。
// 調査の初回呼び出しでは 3 つが揃うまで待ち、同時に実行中だった数を数える。
type fakeModel struct {
	emptyRole string

	mu        sync.Mutex
	active    int
	maxActive int
	prompts   map[string]string
	outputs   map[string]*ai.ModelOutputConfig
	allIn     chan struct{}
	allInOnce sync.Once
}

func newFakeModel() *fakeModel {
	return &fakeModel{
		prompts: map[string]string{},
		outputs: map[string]*ai.ModelOutputConfig{},
		allIn:   make(chan struct{}),
	}
}

func (f *fakeModel) generate(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	var system, prompt strings.Builder
	for _, m := range req.Messages {
		if m.Role == ai.RoleSystem {
			system.WriteString(m.Text())
			continue
		}
		prompt.WriteString(m.Text())
	}
	role := ""
	for _, r := range []string{roleSpots, roleRestaurants, roleTransport, roleSchedule, roleBudget} {
		if strings.Contains(system.String(), r) {
			role = r
		}
	}

	f.mu.Lock()
	f.prompts[role] = prompt.String()
	f.outputs[role] = req.Output
	f.mu.Unlock()

	last := req.Messages[len(req.Messages)-1]
	switch {
	case role == roleBudget:
		return textResponse(budgetJSON), nil
	case len(req.Tools) == 0:
		if cb != nil {
			for _, chunk := range scheduleChunks {
				if err := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(chunk)}}); err != nil {
					return nil, err
				}
			}
		}
		return textResponse(marks[role]), nil
	case last.Role == ai.RoleTool:
		if role == f.emptyRole {
			return textResponse(""), nil
		}
		out, err := json.Marshal(last.Content[0].ToolResponse.Output)
		if err != nil {
			return nil, err
		}
		return textResponse(marks[role] + " " + string(out)), nil
	default:
		f.waitForOtherResearchers(ctx)
		name := req.Tools[0].Name
		return &ai.ModelResponse{
			Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{Name: name, Input: toolInputs[name]}),
			}},
			FinishReason: ai.FinishReasonStop,
		}, nil
	}
}

func (f *fakeModel) waitForOtherResearchers(ctx context.Context) {
	f.mu.Lock()
	f.active++
	f.maxActive = max(f.maxActive, f.active)
	if f.active == researcherCount {
		f.allInOnce.Do(func() { close(f.allIn) })
	}
	f.mu.Unlock()

	select {
	case <-f.allIn:
	case <-time.After(barrierWait):
	case <-ctx.Done():
	}

	f.mu.Lock()
	f.active--
	f.mu.Unlock()
}

func textResponse(text string) *ai.ModelResponse {
	return &ai.ModelResponse{Message: ai.NewModelTextMessage(text), FinishReason: ai.FinishReasonStop}
}

func newFlow(t *testing.T, f *fakeModel) *core.Flow[TripRequest, TripPlan, string] {
	t.Helper()
	g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/fake"))
	genkit.DefineModel(g, "test/fake", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true, Constrained: ai.ConstrainedSupportAll},
	}, f.generate)
	return DefineFlow(g, "")
}

func runFlow(t *testing.T, f *fakeModel) (TripPlan, error) {
	t.Helper()
	return newFlow(t, f).Run(t.Context(), TripRequest{Request: "東京から京都に2泊3日で行きたい"})
}

func TestResearchRunsInParallel(t *testing.T) {
	t.Parallel()
	f := newFakeModel()
	if _, err := runFlow(t, f); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if f.maxActive != researcherCount {
		t.Errorf("同時に実行中だった調査 = %d, 期待 %d", f.maxActive, researcherCount)
	}
}

func TestUpstreamResultsReachPlannerPrompts(t *testing.T) {
	t.Parallel()
	f := newFakeModel()
	if _, err := runFlow(t, f); err != nil {
		t.Fatalf("Run: %v", err)
	}

	testCases := map[string]struct {
		role string
		want []string
	}{
		"日程の呼び出しに依頼と 3 つの調査結果と検索結果が届く": {
			role: roleSchedule,
			want: []string{"2泊3日", "MARK_SPOTS", "MARK_REST", "MARK_TRANS", "金閣寺", "祇園おかる", "新幹線のぞみ"},
		},
		"予算の呼び出しに日程表と 3 つの調査結果が届く": {
			role: roleBudget,
			want: []string{"MARK_SCHEDULE", "MARK_SPOTS", "MARK_REST", "MARK_TRANS"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			for _, w := range tc.want {
				if !strings.Contains(f.prompts[tc.role], w) {
					t.Errorf("%s のプロンプトに %s が無い", tc.role, w)
				}
			}
		})
	}
}

func TestBudgetIsStructuredOutput(t *testing.T) {
	t.Parallel()
	f := newFakeModel()
	plan, err := runFlow(t, f)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := travel.Budget{
		Currency: "JPY", TransportationYen: 28000, FoodYen: 9000, ActivitiesYen: 3000, TotalYen: 40000,
		Assumptions: []string{"新幹線のぞみで往復"},
	}
	if !reflect.DeepEqual(plan.Budget, want) {
		t.Errorf("Budget = %+v, 期待 %+v", plan.Budget, want)
	}
	out := f.outputs[roleBudget]
	if out == nil || out.Format != "json" || !out.Constrained {
		t.Fatalf("予算の呼び出しがスキーマで縛った JSON を求めていない: %+v", out)
	}
	props, _ := out.Schema["properties"].(map[string]any)
	if _, ok := props["total_yen"]; !ok {
		t.Errorf("出力スキーマに total_yen が無い: %v", out.Schema)
	}
}

func TestEmptyResearchStopsFlow(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		emptyRole string
		wantTool  string
	}{
		"観光の調査結果が空": {emptyRole: roleSpots, wantTool: "search_spots"},
		"飲食の調査結果が空": {emptyRole: roleRestaurants, wantTool: "search_restaurants"},
		"交通の調査結果が空": {emptyRole: roleTransport, wantTool: "search_transport"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			f := newFakeModel()
			f.emptyRole = tc.emptyRole
			_, err := runFlow(t, f)
			if err == nil || !strings.Contains(err.Error(), tc.wantTool) {
				t.Fatalf("err = %v, %s の空の結果で止まっていない", err, tc.wantTool)
			}
			if _, ok := f.prompts[roleSchedule]; ok {
				t.Error("調査結果が空なのに日程の呼び出しまで進んだ")
			}
		})
	}
}
