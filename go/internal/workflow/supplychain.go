package workflow

import (
	"context"
	"sync/atomic"

	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// 供給チェーン例の Activity 名。inventory -> transportation -> supplier の順に流す。
const (
	ActivityInventory      = "inventory"
	ActivityTransportation = "transportation"
	ActivitySupplier       = "supplier"
)

// inventoryActivity は在庫引き当てを行う fake Activity（決定論・LLM 不要）。
type inventoryActivity struct{}

func (inventoryActivity) Name() string { return ActivityInventory }

func (inventoryActivity) Execute(_ context.Context, in ActivityInput) (ActivityResult, error) {
	return ActivityResult{Output: map[string]any{
		"reserved":  true,
		"operation": in.Operation,
		"sku":       "SKU-1001",
		"qty":       12,
	}}, nil
}

// transportationActivity は輸送手配を行う fake Activity。
type transportationActivity struct{}

func (transportationActivity) Name() string { return ActivityTransportation }

func (transportationActivity) Execute(_ context.Context, in ActivityInput) (ActivityResult, error) {
	return ActivityResult{Output: map[string]any{
		"carrier":   "FleetA",
		"operation": in.Operation,
		"eta_days":  3,
	}}, nil
}

// supplierActivity は仕入先発注を行う fake Activity。
type supplierActivity struct{}

func (supplierActivity) Name() string { return ActivitySupplier }

func (supplierActivity) Execute(_ context.Context, in ActivityInput) (ActivityResult, error) {
	return ActivityResult{Output: map[string]any{
		"po_number": "PO-7788",
		"operation": in.Operation,
		"supplier":  "Acme Parts",
	}}, nil
}

// NewInventoryActivity は在庫引き当ての fake Activity を返す。
func NewInventoryActivity() Activity { return inventoryActivity{} }

// NewTransportationActivity は輸送手配の fake Activity を返す。
func NewTransportationActivity() Activity { return transportationActivity{} }

// NewSupplierActivity は仕入先発注の fake Activity を返す。
func NewSupplierActivity() Activity { return supplierActivity{} }

// SupplyChainWorkflow は 3 フェーズの供給チェーン Workflow を組み立てる。
func SupplyChainWorkflow(retry RetryPolicy) Workflow {
	return Workflow{
		Name: "supply-chain",
		Steps: []Step{
			{Activity: inventoryActivity{}, Retry: retry, Input: ActivityInput{
				Operation: "reserve", Messages: []string{"reserve stock for order"}}},
			{Activity: transportationActivity{}, Retry: retry, Input: ActivityInput{
				Operation: "schedule", Messages: []string{"book carrier"}}},
			{Activity: supplierActivity{}, Retry: retry, Input: ActivityInput{
				Operation: "purchase", Messages: []string{"place purchase order"}}},
		},
	}
}

// FlakyActivity は最初の failBeforeSuccess 回だけ失敗し、その後成功する Activity。
// WHY: ステップ単位リトライ（一時障害が自己回復するケース）の検証に使う。
type FlakyActivity struct {
	Inner            Activity
	FailBeforeSucces int
	calls            int64
}

// Name は内側 Activity 名を引き継ぐ。
func (f *FlakyActivity) Name() string { return f.Inner.Name() }

// Execute は呼び出し回数が閾値以下のうちは error を返す。
func (f *FlakyActivity) Execute(ctx context.Context, in ActivityInput) (ActivityResult, error) {
	n := atomic.AddInt64(&f.calls, 1)
	if int(n) <= f.FailBeforeSucces {
		return ActivityResult{}, liberrors.Newf(liberrors.CodeUnavailable,
			"transient failure on %q (call %d)", f.Inner.Name(), n)
	}
	return f.Inner.Execute(ctx, in)
}

// CrashActivity は実行されたら常に error を返す Activity。
// WHY: プロセスクラッシュ（このステップに到達した時点で落ちる）を模す。
// このステップ以降はジャーナルに記録されないため、再実行で未完了分から再開できる。
type CrashActivity struct {
	Inner Activity
}

// Name は内側 Activity 名を引き継ぐ。
func (c CrashActivity) Name() string { return c.Inner.Name() }

// Execute は常に error を返す（このプロセスでは完了させない）。
func (c CrashActivity) Execute(_ context.Context, _ ActivityInput) (ActivityResult, error) {
	return ActivityResult{}, liberrors.Newf(liberrors.CodeUnavailable,
		"simulated crash at %q before checkpoint", c.Inner.Name())
}
