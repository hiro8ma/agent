package travelplanner

import (
	"context"
	"iter"

	"github.com/firebase/genkit/go/core"

	"github.com/hiro8ma/agent/go/internal/travel"
)

var _ travel.Planner = (*Planner)(nil)

// Planner はフローを包んで travel.Planner を満たす。Genkit のフローはセッションを持たないので SessionID は使わない。
type Planner struct {
	flow *core.Flow[TripRequest, TripPlan, string]
}

// NewPlanner は DefineFlow で登録したフローを受け取る。
func NewPlanner(flow *core.Flow[TripRequest, TripPlan, string]) *Planner {
	return &Planner{flow: flow}
}

// Plan はフローのストリームを日程表の差分として流し、最後に日程と予算を返す。
func (p *Planner) Plan(ctx context.Context, req *travel.PlanRequest) iter.Seq2[*travel.PlanResponse, error] {
	return func(yield func(*travel.PlanResponse, error) bool) {
		for v, err := range p.flow.Stream(ctx, TripRequest{Request: req.Message}) {
			if err != nil {
				yield(nil, err)
				return
			}
			if v.Done {
				yield(&travel.PlanResponse{Plan: &travel.Plan{Schedule: v.Output.Schedule, Budget: v.Output.Budget}}, nil)
				return
			}
			if v.Stream == "" {
				continue
			}
			if !yield(&travel.PlanResponse{Delta: v.Stream}, nil) {
				return
			}
		}
	}
}
