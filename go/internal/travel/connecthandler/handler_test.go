package connecthandler_test

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"connectrpc.com/connect"

	travelv1 "github.com/hiro8ma/agent/go/gen/travel/v1"
	"github.com/hiro8ma/agent/go/gen/travel/v1/travelv1connect"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/travel"
	"github.com/hiro8ma/agent/go/internal/travel/connecthandler"
)

// fakePlanner はモデルを呼ばずに差分を 2 つ返してから日程と予算を返す。受け取った依頼を覚える。
type fakePlanner struct {
	got chan *travel.PlanRequest
}

func (f fakePlanner) Plan(_ context.Context, req *travel.PlanRequest) iter.Seq2[*travel.PlanResponse, error] {
	f.got <- req
	return func(yield func(*travel.PlanResponse, error) bool) {
		for _, d := range []string{"1日目 ", "金閣寺"} {
			if !yield(&travel.PlanResponse{Delta: d}, nil) {
				return
			}
		}
		yield(&travel.PlanResponse{Plan: &travel.Plan{
			Schedule: "1日目 金閣寺",
			Budget:   travel.Budget{Currency: "JPY", TransportationYen: 28000, FoodYen: 5000, ActivitiesYen: 1000, TotalYen: 34000, Assumptions: []string{"大人 1 名"}},
		}}, nil)
	}
}

func newClient(t *testing.T, p travel.Planner) travelv1connect.TravelPlannerServiceClient {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle(connecthandler.Route(connecthandler.New(p), libconnect.HeaderAuthenticator))
	srv := httptest.NewTestServer(t, mux)
	return travelv1connect.NewTravelPlannerServiceClient(srv.Client(), srv.URL)
}

func plan(ctx context.Context, c travelv1connect.TravelPlannerServiceClient, msg *travelv1.PlanRequest) ([]*travelv1.PlanResponse, error) {
	req := connect.NewRequest(msg)
	req.Header().Set(libconnect.UserHeader, "alice")
	stream, err := c.Plan(ctx, req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = stream.Close() }()
	var out []*travelv1.PlanResponse
	for stream.Receive() {
		out = append(out, stream.Msg())
	}
	return out, stream.Err()
}

func TestPlanStreamsDeltasThenPlan(t *testing.T) {
	t.Parallel()
	p := fakePlanner{got: make(chan *travel.PlanRequest, 1)}
	c := newClient(t, p)

	got, err := plan(t.Context(), c, &travelv1.PlanRequest{SessionId: "s1", Message: "京都に行きたい"})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("応答の数 = %d, want 3（delta 2 通と plan 1 通）", len(got))
	}

	var deltas []string
	for _, res := range got[:len(got)-1] {
		d, ok := res.GetEvent().(*travelv1.PlanResponse_Delta)
		if !ok {
			t.Fatalf("途中の応答が delta ではない: %v", res)
		}
		deltas = append(deltas, d.Delta)
	}
	if want := []string{"1日目 ", "金閣寺"}; !slices.Equal(deltas, want) {
		t.Errorf("deltas = %q, want %q", deltas, want)
	}
	last := got[len(got)-1].GetPlan()
	if last == nil {
		t.Fatalf("最後の応答が plan ではない: %v", got[len(got)-1])
	}
	if last.GetSchedule() != "1日目 金閣寺" || last.GetBudget().GetTotalYen() != 34000 || last.GetBudget().GetTransportationYen() != 28000 {
		t.Errorf("plan = %v", last)
	}
	req := <-p.got
	if req.UserID != "alice" || req.SessionID != "s1" || req.Message != "京都に行きたい" {
		t.Errorf("planner が受けた依頼 = %+v", req)
	}
}

func TestPlanRejectsEmptyMessage(t *testing.T) {
	t.Parallel()
	c := newClient(t, fakePlanner{got: make(chan *travel.PlanRequest, 1)})

	_, err := plan(t.Context(), c, &travelv1.PlanRequest{SessionId: "s1"})
	if ce, ok := errors.AsType[*connect.Error](err); !ok || ce.Code() != connect.CodeInvalidArgument {
		t.Errorf("Plan() error = %v, want InvalidArgument", err)
	}
}
