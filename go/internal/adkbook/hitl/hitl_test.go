package hitl

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// fakeCtx は確認要求の記録だけを持つ。
type fakeCtx struct {
	agent.Context
	confirmation *toolconfirmation.ToolConfirmation
	requested    []string
}

func (c *fakeCtx) ToolConfirmation() *toolconfirmation.ToolConfirmation { return c.confirmation }
func (c *fakeCtx) RequestConfirmation(hint string, _ any) error {
	c.requested = append(c.requested, hint)
	return nil
}

var refundPolicy = Any(
	Threshold("amount", 10_000, 100_000, "円"),
	MatchArg("account", "corporate", "payroll"),
)

// 3 つの判断が上限で分かれることを見る。
func TestThresholdSplitsThreeWays(t *testing.T) {
	cases := []struct {
		amount float64
		want   Decision
	}{
		{5_000, Allow},
		{10_000, Allow},
		{10_001, Ask},
		{99_999, Ask},
		{100_001, Deny},
	}
	for _, c := range cases {
		got, reason := refundPolicy(map[string]any{"amount": c.amount})
		if got != c.want {
			t.Errorf("%g 円: %v。%v を期待", c.amount, got, c.want)
		}
		if got != Allow && reason == "" {
			t.Errorf("%g 円: 理由が無い", c.amount)
		}
	}
}

// 上限が 1 つだと「大きすぎる」と「確認が要る」が同じ扱いになる。
func TestAskAndDenyAreDifferent(t *testing.T) {
	ask, _ := refundPolicy(map[string]any{"amount": 50_000.0})
	deny, _ := refundPolicy(map[string]any{"amount": 200_000.0})
	if ask == deny {
		t.Fatal("承認を求める額と拒む額が同じ判断になった")
	}
}

// 未承認なら確認を要求し、ツールを実行しない。
func TestGateRequestsConfirmation(t *testing.T) {
	ctx := &fakeCtx{}
	proceed, res, err := Gate(ctx, map[string]any{"amount": 50_000.0}, refundPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if proceed {
		t.Fatal("承認前に実行した")
	}
	if res["status"] != "pending" {
		t.Errorf("status が %v", res["status"])
	}
	if len(ctx.requested) != 1 {
		t.Fatalf("確認要求が %d 件", len(ctx.requested))
	}
	t.Logf("確認要求: %s", ctx.requested[0])
}

// 承認済みなら実行する。同じ引数で判断が変わる。
func TestGateProceedsAfterApproval(t *testing.T) {
	args := map[string]any{"amount": 50_000.0}

	approved := &fakeCtx{confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}}
	if proceed, _, _ := Gate(approved, args, refundPolicy); !proceed {
		t.Error("承認済みなのに実行しなかった")
	}

	rejected := &fakeCtx{confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false}}
	proceed, res, _ := Gate(rejected, args, refundPolicy)
	if proceed {
		t.Error("拒否されたのに実行した")
	}
	if res["status"] != "rejected" {
		t.Errorf("status が %v", res["status"])
	}
	if len(rejected.requested) != 0 {
		t.Error("応答済みなのに再度確認を求めた")
	}
}

// 拒む判断では確認を求めない。人に聞いても答えは変わらない。
func TestDenyDoesNotAskHuman(t *testing.T) {
	ctx := &fakeCtx{}
	proceed, res, _ := Gate(ctx, map[string]any{"amount": 500_000.0}, refundPolicy)
	if proceed {
		t.Fatal("上限を超えたのに実行した")
	}
	if res["status"] != "denied" {
		t.Errorf("status が %v", res["status"])
	}
	if len(ctx.requested) != 0 {
		t.Error("拒む判断で確認を求めた")
	}
}

// 引数の種類でも承認を求める。金額だけが判断材料ではない。
func TestMatchArgTriggersAsk(t *testing.T) {
	got, reason := refundPolicy(map[string]any{"amount": 100.0, "account": "payroll"})
	if got != Ask {
		t.Fatalf("%v。承認を求めることを期待", got)
	}
	t.Logf("理由: %s", reason)
}

// 判断材料が無ければ通す。既定で止めると通常の呼び出しが全部止まる。
func TestMissingArgAllows(t *testing.T) {
	if got, _ := refundPolicy(map[string]any{}); got != Allow {
		t.Errorf("%v。引数が無ければ通すことを期待", got)
	}
}
