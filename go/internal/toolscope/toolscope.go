// Package toolscope は、A2A で呼ばれたエージェントのツールを、認証した主体のスコープで許すかを決める。
//
// 判断に使うのは context の主体（libauth.Principal）だけで、Session の State は使わない。
// State は呼び出し元が書き換えられる（ADK の REST の state_delta など）ので、権限の根拠にならない。
package toolscope

import (
	"context"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"

	"github.com/hiro8ma/agent/go/internal/lib/libauth"
)

// Policy はツールの名前ごとに必要なスコープ。載っていないツールは使えない。
type Policy map[string][]string

var (
	// ErrUnlisted は Policy に無いツール。
	ErrUnlisted = errors.New("toolscope: tool is not allowed over A2A")
	// ErrNoPrincipal は認証した主体が context に無い。
	ErrNoPrincipal = errors.New("toolscope: no authenticated caller")
	// ErrScope はスコープが足りない。
	ErrScope = errors.New("toolscope: insufficient scope")
)

// Check はツールを呼んでよいかを返す。
func (p Policy) Check(ctx context.Context, toolName string) error {
	required, listed := p[toolName]
	if !listed {
		return fmt.Errorf("%w: %s", ErrUnlisted, toolName)
	}
	caller, ok := libauth.FromContext(ctx)
	if !ok {
		return ErrNoPrincipal
	}
	if !caller.HasScopes(required...) {
		return fmt.Errorf("%w: %s requires %v", ErrScope, toolName, required)
	}
	return nil
}

func denied(err error) map[string]any {
	return map[string]any{"status": "denied", "error": err.Error()}
}

// ADKPlugin は ADK のランナーのプラグイン。A2A の口のランナーにだけ付ける。
func ADKPlugin(p Policy) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "toolscope",
		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, _ map[string]any) (map[string]any, error) {
			if err := p.Check(ctx, t.Name()); err != nil {
				return denied(err), nil
			}
			return nil, nil
		},
	})
}

type enforcedKey struct{}

// Enforce は、この context で Genkit のミドルウェアに検査をさせる印を付ける。A2A の実行器が付ける。
func Enforce(ctx context.Context) context.Context {
	return context.WithValue(ctx, enforcedKey{}, true)
}

// GenkitMiddleware は Genkit のツールの呼び出しを検査する。Enforce の印が無い呼び出し（A2A 以外の口）は素通しにする。
func GenkitMiddleware(p Policy) ai.MiddlewareFunc {
	return func(context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				if enforced, _ := ctx.Value(enforcedKey{}).(bool); enforced && params.Request != nil {
					if err := p.Check(ctx, params.Request.Name); err != nil {
						return &ai.MultipartToolResponse{Output: denied(err)}, nil
					}
				}
				return next(ctx, params)
			},
		}, nil
	}
}
