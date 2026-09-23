package travelplanner

import (
	"context"
	"iter"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/travel"
)

const budgetJSON = `{"currency":"JPY","transportation_yen":28000,"food_yen":9000,"activities_yen":3000,"total_yen":40000,"assumptions":["新幹線のぞみで往復"]}`

var roleReplies = map[string][]string{
	"観光スポットの専門家":    {"MARK_", "SPOTS"},
	"グルメの専門家":       {"MARK_", "REST"},
	"交通手段の専門家":      {"MARK_", "TRANS"},
	"旅行スケジュールの作成担当": {"1日目 ", "午前 金閣寺"},
	"旅行予算の計算担当":     {budgetJSON},
}

// streamingModel は system instruction で役を見分け、stream のときは返答を分けて Partial で流してから全文を返す。
type streamingModel struct{}

func (streamingModel) Name() string { return "streaming" }

func (streamingModel) GenerateContent(_ context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	var sys strings.Builder
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, p := range req.Config.SystemInstruction.Parts {
			sys.WriteString(p.Text)
		}
	}
	chunks := []string{"?"}
	for role, reply := range roleReplies {
		if strings.Contains(sys.String(), role) {
			chunks = reply
			break
		}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream {
			for _, c := range chunks {
				if !yield(&model.LLMResponse{Content: genai.NewContentFromText(c, genai.RoleModel), Partial: true}, nil) {
					return
				}
			}
		}
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(strings.Join(chunks, ""), genai.RoleModel),
			TurnComplete: true,
		}, nil)
	}
}

func TestPlannerStreamsScheduleThenReturnsPlanOnce(t *testing.T) {
	t.Parallel()
	var modelCalls atomic.Int32
	counter, err := plugin.New(plugin.Config{
		Name: "counter",
		BeforeModelCallback: func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
			modelCalls.Add(1)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	impl, err := NewPlanner(streamingModel{}, counter)
	if err != nil {
		t.Fatal(err)
	}
	var p travel.Planner = impl

	var deltas []string
	var plans []*travel.Plan
	req := &travel.PlanRequest{UserID: "u", SessionID: "s", Message: "東京から京都に2泊3日で行きたい"}
	for resp, err := range p.Plan(t.Context(), req) {
		if err != nil {
			t.Fatalf("Plan: %v", err)
		}
		if len(plans) > 0 {
			t.Fatalf("Plan の後に応答が届いた: %+v", resp)
		}
		if resp.Plan != nil {
			plans = append(plans, resp.Plan)
			continue
		}
		deltas = append(deltas, resp.Delta)
	}

	schedule := roleReplies["旅行スケジュールの作成担当"]
	if !reflect.DeepEqual(deltas, schedule) {
		t.Errorf("Delta = %q, 期待 %q。日程表の差分だけが流れていない", deltas, schedule)
	}
	if len(plans) != 1 {
		t.Fatalf("Plan の数 = %d, 期待 1", len(plans))
	}
	want := travel.Plan{
		Schedule: strings.Join(schedule, ""),
		Budget: travel.Budget{
			Currency: "JPY", TransportationYen: 28000, FoodYen: 9000, ActivitiesYen: 3000, TotalYen: 40000,
			Assumptions: []string{"新幹線のぞみで往復"},
		},
	}
	if !reflect.DeepEqual(*plans[0], want) {
		t.Errorf("Plan = %+v, 期待 %+v", *plans[0], want)
	}
	if modelCalls.Load() == 0 {
		t.Error("NewPlanner に渡したプラグインがランナーに届いていない")
	}
}
