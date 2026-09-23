// Package connecthandler は travel.Planner を TravelPlannerService として ConnectRPC（server streaming）で公開する。
package connecthandler

import (
	"context"
	"errors"
	"net/http"

	"connectrpc.com/connect"

	travelv1 "github.com/hiro8ma/agent/go/gen/travel/v1"
	"github.com/hiro8ma/agent/go/gen/travel/v1/travelv1connect"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/travel"
)

type Handler struct {
	planner travel.Planner
}

var _ travelv1connect.TravelPlannerServiceHandler = (*Handler)(nil)

func New(planner travel.Planner) *Handler {
	return &Handler{planner: planner}
}

// Route は利用者の特定まで含めて公開する。
func Route(h *Handler, auth libconnect.Authenticator) (string, http.Handler) {
	return travelv1connect.NewTravelPlannerServiceHandler(h, connect.WithInterceptors(
		// span を認証の外側に置き、利用者を特定できずに拒否した呼び出しも記録する。
		libconnect.EdgeTelemetry(),
		libconnect.ServerIdentity(auth),
	))
}

func (h *Handler) Plan(ctx context.Context, req *connect.Request[travelv1.PlanRequest], stream *connect.ServerStream[travelv1.PlanResponse]) error {
	if req.Msg.GetMessage() == "" {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("message is required"))
	}
	userID, err := identity.From(ctx)
	if err != nil {
		return libconnect.Error(err)
	}
	in := &travel.PlanRequest{UserID: string(userID), SessionID: req.Msg.GetSessionId(), Message: req.Msg.GetMessage()}
	for res, err := range h.planner.Plan(ctx, in) {
		if err != nil {
			return libconnect.Error(err)
		}
		if err := stream.Send(toPlanResponse(res)); err != nil {
			return err
		}
	}
	return nil
}

func toPlanResponse(res *travel.PlanResponse) *travelv1.PlanResponse {
	if res.Plan == nil {
		return &travelv1.PlanResponse{Event: &travelv1.PlanResponse_Delta{Delta: res.Delta}}
	}
	b := res.Plan.Budget
	return &travelv1.PlanResponse{Event: &travelv1.PlanResponse_Plan{Plan: &travelv1.Plan{
		Schedule: res.Plan.Schedule,
		Budget: &travelv1.Budget{
			Currency:          b.Currency,
			TransportationYen: int64(b.TransportationYen),
			FoodYen:           int64(b.FoodYen),
			ActivitiesYen:     int64(b.ActivitiesYen),
			TotalYen:          int64(b.TotalYen),
			Assumptions:       b.Assumptions,
		},
	}}}
}
