package travelplanner

import (
	"reflect"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/travel"
)

func TestPlannerStreamsScheduleThenReturnsPlanOnce(t *testing.T) {
	t.Parallel()
	var p travel.Planner = NewPlanner(newFlow(t, newFakeModel()))

	var deltas []string
	var plans []*travel.Plan
	for resp, err := range p.Plan(t.Context(), &travel.PlanRequest{SessionID: "s", Message: "東京から京都に2泊3日で行きたい"}) {
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

	if !reflect.DeepEqual(deltas, scheduleChunks) {
		t.Errorf("Delta = %q, 期待 %q", deltas, scheduleChunks)
	}
	if len(plans) != 1 {
		t.Fatalf("Plan の数 = %d, 期待 1", len(plans))
	}
	if plans[0].Schedule != strings.Join(scheduleChunks, "") {
		t.Errorf("Schedule = %q", plans[0].Schedule)
	}
	if plans[0].Budget.TotalYen != 40000 {
		t.Errorf("Budget = %+v", plans[0].Budget)
	}
}
