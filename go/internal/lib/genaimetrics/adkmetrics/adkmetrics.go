// Package adkmetrics は genaimetrics の計器に、ADK Go のランナーのプラグインとして値を流す。
package adkmetrics

import (
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics"
)

// Option はプラグインの設定。
type Option func(*options)

type options struct {
	denied    func(result map[string]any) bool
	escalated func(toolName string, result map[string]any) bool
	now       func() time.Time
}

// WithDenied は、ツールの結果が権限の拒否かを決める。既定は status が "denied"。
func WithDenied(f func(result map[string]any) bool) Option {
	return func(o *options) { o.denied = f }
}

// WithEscalation は、ツールの結果が人への引き継ぎ（承認待ちなど）かを決める。既定は常に false。
func WithEscalation(f func(toolName string, result map[string]any) bool) Option {
	return func(o *options) { o.escalated = f }
}

// WithClock は時刻の取り方を差し替える。
func WithClock(now func() time.Time) Option {
	return func(o *options) { o.now = now }
}

type modelCall struct {
	model string
	start time.Time
}

type invocation struct {
	agent     string
	failed    bool
	escalated bool
}

type state struct {
	mu          sync.Mutex
	models      map[string]modelCall
	tools       map[string]time.Time
	invocations map[string]*invocation
}

// Plugin は r に記録するプラグインを返す。
func Plugin(r *genaimetrics.Recorder, opts ...Option) (*plugin.Plugin, error) {
	o := options{
		denied:    func(result map[string]any) bool { return result["status"] == "denied" },
		escalated: func(string, map[string]any) bool { return false },
		now:       time.Now,
	}
	for _, opt := range opts {
		opt(&o)
	}
	s := &state{models: map[string]modelCall{}, tools: map[string]time.Time{}, invocations: map[string]*invocation{}}

	modelKey := func(ctx agent.Context) string { return ctx.InvocationID() + "/" + ctx.AgentName() }
	toolKey := func(ctx agent.Context, t tool.Tool) string { return modelKey(ctx) + "/" + t.Name() }
	finishModel := func(ctx agent.Context, resp *model.LLMResponse, err error) {
		s.mu.Lock()
		call, ok := s.models[modelKey(ctx)]
		delete(s.models, modelKey(ctx))
		if inv := s.invocations[ctx.InvocationID()]; inv != nil && err != nil {
			inv.failed = true
		}
		s.mu.Unlock()
		if !ok {
			return
		}
		var u genaimetrics.Usage
		if resp != nil {
			u = genaimetrics.UsageFrom(resp.UsageMetadata)
		}
		r.Chat(ctx, ctx.AgentName(), call.model, o.now().Sub(call.start), u, err)
	}
	finishTool := func(ctx agent.Context, t tool.Tool, errorType string, escalated bool) {
		s.mu.Lock()
		start, ok := s.tools[toolKey(ctx, t)]
		delete(s.tools, toolKey(ctx, t))
		if inv := s.invocations[ctx.InvocationID()]; inv != nil && escalated {
			inv.escalated = true
		}
		s.mu.Unlock()
		if !ok {
			start = o.now()
		}
		r.Tool(ctx, ctx.AgentName(), t.Name(), o.now().Sub(start), errorType)
	}

	return plugin.New(plugin.Config{
		Name: "genaimetrics",
		BeforeRunCallback: func(ictx agent.InvocationContext) (*genai.Content, error) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.invocations[ictx.InvocationID()] = &invocation{agent: ictx.Agent().Name()}
			return nil, nil
		},
		AfterRunCallback: func(ictx agent.InvocationContext) {
			s.mu.Lock()
			inv := s.invocations[ictx.InvocationID()]
			delete(s.invocations, ictx.InvocationID())
			s.mu.Unlock()
			if inv == nil {
				return
			}
			outcome := genaimetrics.OutcomeCompleted
			switch {
			case inv.escalated:
				outcome = genaimetrics.OutcomeEscalated
			case inv.failed:
				outcome = genaimetrics.OutcomeFailed
			}
			r.Invocation(ictx, inv.agent, outcome)
		},
		BeforeModelCallback: func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.models[modelKey(ctx)] = modelCall{model: req.Model, start: o.now()}
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
			if err == nil && resp != nil && resp.Partial {
				return nil, nil
			}
			finishModel(ctx, resp, err)
			return nil, nil
		},
		OnModelErrorCallback: func(ctx agent.Context, _ *model.LLMRequest, err error) (*model.LLMResponse, error) {
			finishModel(ctx, nil, err)
			return nil, nil
		},
		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, _ map[string]any) (map[string]any, error) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.tools[toolKey(ctx, t)] = o.now()
			return nil, nil
		},
		AfterToolCallback: func(ctx agent.Context, t tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
			errorType := ""
			switch {
			case o.denied(result):
				errorType = genaimetrics.ErrorTypeDenied
			case err != nil || result["error"] != nil || result["status"] == "error":
				errorType = genaimetrics.ErrorTypeToolError
			}
			finishTool(ctx, t, errorType, err == nil && o.escalated(t.Name(), result))
			return nil, nil
		},
		OnToolErrorCallback: func(ctx agent.Context, t tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
			finishTool(ctx, t, genaimetrics.ErrorTypeToolError, false)
			return nil, nil
		},
	})
}
