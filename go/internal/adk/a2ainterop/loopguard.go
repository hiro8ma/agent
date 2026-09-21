package a2ainterop

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

// TraceParam は呼び出しの経路（通ってきたエージェントの名前をカンマで並べたもの）を運ぶサービスのパラメータ。
// A2A のプロトコルには循環を防ぐ仕組みが無いので、アプリケーションの層で運ぶ。
const TraceParam = "x-a2a-trace"

type traceKey struct{}

// TraceFrom は受けた呼び出しの経路に、自分を足したものを返す。外へ呼ぶときにこれを運ぶ。
func TraceFrom(ctx context.Context) []string {
	t, _ := ctx.Value(traceKey{}).([]string)
	return t
}

// LoopGuard は、自分が既に経路にいる呼び出しと、経路が maxDepth に達した呼び出しを拒否する。
type LoopGuard struct {
	Self     string
	MaxDepth int
}

var _ a2asrv.CallInterceptor = LoopGuard{}

func (g LoopGuard) Before(ctx context.Context, callCtx *a2asrv.CallContext, _ *a2asrv.Request) (context.Context, any, error) {
	var trace []string
	if values, ok := callCtx.ServiceParams().Get(TraceParam); ok && len(values) > 0 && values[0] != "" {
		trace = strings.Split(values[0], ",")
	}
	if slices.Contains(trace, g.Self) {
		return ctx, nil, fmt.Errorf("%w: 循環した呼び出し %s -> %s", a2a.ErrInvalidRequest, strings.Join(trace, " -> "), g.Self)
	}
	if g.MaxDepth > 0 && len(trace) >= g.MaxDepth {
		return ctx, nil, fmt.Errorf("%w: 呼び出しの深さが上限 %d を超える（%s）", a2a.ErrInvalidRequest, g.MaxDepth, strings.Join(trace, " -> "))
	}
	return context.WithValue(ctx, traceKey{}, append(slices.Clone(trace), g.Self)), nil, nil
}

func (LoopGuard) After(context.Context, *a2asrv.CallContext, *a2asrv.Response) error { return nil }

// TracePropagator は外へ呼ぶときに、context の経路をサービスのパラメータに載せる。
type TracePropagator struct{}

var _ a2aclient.CallInterceptor = TracePropagator{}

func (TracePropagator) Before(ctx context.Context, req *a2aclient.Request) (context.Context, any, error) {
	if trace := TraceFrom(ctx); len(trace) > 0 {
		if req.ServiceParams == nil {
			req.ServiceParams = a2aclient.ServiceParams{}
		}
		req.ServiceParams[TraceParam] = []string{strings.Join(trace, ",")}
	}
	return ctx, nil, nil
}

func (TracePropagator) After(context.Context, *a2aclient.Response) error { return nil }
