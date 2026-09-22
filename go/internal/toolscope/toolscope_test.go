package toolscope_test

import (
	"context"
	"errors"
	"testing"

	"github.com/hiro8ma/agent/go/internal/lib/libauth"
	"github.com/hiro8ma/agent/go/internal/toolscope"
)

func TestCheck(t *testing.T) {
	t.Parallel()
	policy := toolscope.Policy{"get_order": {"orders.read"}, "ping": {}}
	with := func(scopes ...string) context.Context {
		return libauth.WithPrincipal(context.Background(), libauth.Principal{Subject: "caller", Scopes: scopes})
	}
	testCases := map[string]struct {
		ctx     context.Context
		tool    string
		wantErr error
	}{
		"スコープを持てば通る":       {ctx: with("orders.read"), tool: "get_order"},
		"スコープが足りなければ拒否":    {ctx: with("a2a.invoke"), tool: "get_order", wantErr: toolscope.ErrScope},
		"主体が無ければ拒否":        {ctx: context.Background(), tool: "get_order", wantErr: toolscope.ErrNoPrincipal},
		"スコープ不要のツールも主体は要る": {ctx: context.Background(), tool: "ping", wantErr: toolscope.ErrNoPrincipal},
		"一覧に無いツールは拒否":      {ctx: with("orders.read", "orders.write"), tool: "delete_order", wantErr: toolscope.ErrUnlisted},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if err := policy.Check(tc.ctx, tc.tool); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}
