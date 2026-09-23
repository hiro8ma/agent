package travelplanner

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"

	"github.com/hiro8ma/agent/go/internal/travel"
)

const appName = "travel_planner"

var _ travel.Planner = (*Planner)(nil)

// Planner は旅行プランナーの木をランナーで動かし、travel.Planner を満たす。
type Planner struct {
	runner   *runner.Runner
	sessions session.Service
}

// NewPlanner は m で木を組み、plugins を差したランナーを用意する。
func NewPlanner(m model.LLM, plugins ...*plugin.Plugin) (*Planner, error) {
	a, err := NewWithModel(m)
	if err != nil {
		return nil, err
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             a,
		SessionService:    sessions,
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		return nil, fmt.Errorf("create runner: %w", err)
	}
	return &Planner{runner: r, sessions: sessions}, nil
}

// Plan は日程表の生成中の差分を流し、最後に State から日程と予算を読んで返す。
//
// 調査の 3 つは並列に走り、差分が混ざるので流さない。
func (p *Planner) Plan(ctx context.Context, req *travel.PlanRequest) iter.Seq2[*travel.PlanResponse, error] {
	return func(yield func(*travel.PlanResponse, error) bool) {
		sessionID, err := p.sessionID(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}
		msg := genai.NewContentFromText(req.Message, genai.RoleUser)
		cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}
		for ev, err := range p.runner.Run(ctx, req.UserID, sessionID, msg, cfg) {
			if err != nil {
				yield(nil, err)
				return
			}
			if !ev.Partial || ev.Author != scheduleAgentName || ev.Content == nil {
				continue
			}
			for _, part := range ev.Content.Parts {
				if part.Text == "" || part.Thought {
					continue
				}
				if !yield(&travel.PlanResponse{Delta: part.Text}, nil) {
					return
				}
			}
		}
		plan, err := p.readPlan(ctx, req.UserID, sessionID)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(&travel.PlanResponse{Plan: plan}, nil)
	}
}

// sessionID は、SessionID が空なら新しいセッションを作る。ランナーが空の ID で作ったセッションは、後から State を読めないため。
func (p *Planner) sessionID(ctx context.Context, req *travel.PlanRequest) (string, error) {
	if req.SessionID != "" {
		return req.SessionID, nil
	}
	resp, err := p.sessions.Create(ctx, &session.CreateRequest{AppName: appName, UserID: req.UserID})
	if err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}
	return resp.Session.ID(), nil
}

func (p *Planner) readPlan(ctx context.Context, userID, sessionID string) (*travel.Plan, error) {
	resp, err := p.sessions.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return nil, fmt.Errorf("get session: %w", err)
	}
	state := resp.Session.State()
	schedule, err := stateString(state, keySchedule)
	if err != nil {
		return nil, err
	}
	rawBudget, err := stateString(state, keyBudget)
	if err != nil {
		return nil, err
	}
	var budget travel.Budget
	if err := json.Unmarshal([]byte(rawBudget), &budget); err != nil {
		return nil, fmt.Errorf("parse %s: %w", keyBudget, err)
	}
	return &travel.Plan{Schedule: schedule, Budget: budget}, nil
}

func stateString(state session.State, key string) (string, error) {
	v, err := state.Get(key)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", key, err)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("read %s: 文字列ではない（%T）", key, v)
	}
	return s, nil
}
