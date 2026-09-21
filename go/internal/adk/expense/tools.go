package expense

import (
	"fmt"
	"slices"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"github.com/hiro8ma/agent/go/internal/approval"
)

const (
	// ApprovalThreshold 以上の申請は承認を待つ。
	ApprovalThreshold = 500_000
	// MaxAmount を超える申請は受け付けない。
	MaxAmount = 1_000_000_000

	ToolSubmit  = "submit_expense"
	ToolQuery   = "query_expenses"
	ToolApprove = "approve_expense"
)

// Rules は申請のリスク。金額の検証は申請のツールで行い、ここでは承認の要否だけを決める。
func Rules() map[string]approval.Rule {
	return map[string]approval.Rule{
		ToolSubmit: func(args map[string]any) approval.Risk {
			amount, _ := args["amount"].(int)
			if amount >= ApprovalThreshold {
				return approval.High
			}
			return approval.Low
		},
	}
}

// Deps はツールが使うもの。
type Deps struct {
	Store     *Store
	Approvals *approval.Service
	// Approvers は承認と他人の経費の照会ができる利用者。
	Approvers []string
}

func (d Deps) isApprover(user string) bool {
	return slices.Contains(d.Approvers, user)
}

type submitInput struct {
	Date        string `json:"date"`
	Category    string `json:"category"`
	Amount      int    `json:"amount"`
	Description string `json:"description"`
}

type queryInput struct {
	UserID string `json:"userId,omitempty"`
	Month  string `json:"month,omitempty"`
}

type approveInput struct {
	ExpenseID string `json:"expenseId"`
	Approve   bool   `json:"approve"`
}

func refused(msg string) map[string]any {
	return map[string]any{"status": "rejected", "message": msg}
}

// Submit は経費を申請する。金額は入力の文ではなく引数で確かめる。
// 文の正規表現では「マイナス5000円」や全角の「−5,000円」のような書き方を取りこぼす。
func (d Deps) Submit(ctx agent.Context, in submitInput) (map[string]any, error) {
	switch {
	case in.Amount <= 0:
		return refused("金額は 1 円以上"), nil
	case in.Amount > MaxAmount:
		return refused("金額が上限（10 億円）を超えている"), nil
	case !ValidDate(in.Date):
		return refused("日付は 2025-07-10 の形"), nil
	case in.Category == "":
		return refused("費目が要る"), nil
	}
	e := Expense{UserID: ctx.UserID(), Date: in.Date, Category: in.Category, Amount: in.Amount, Description: in.Description, Status: Submitted}
	args := map[string]any{"amount": in.Amount}
	if d.Approvals.Assess(ToolSubmit, args) == approval.Low {
		saved := d.Store.Add(e)
		return map[string]any{"status": "submitted", "expense": saved}, nil
	}
	e.Status = PendingApproval
	saved := d.Store.Add(e)
	// 承認は申請の ID と金額に結び付ける。承認者は依頼を見て、何を承認するかが分かる。
	r, err := d.Approvals.Open(ctx.UserID(), ToolSubmit, map[string]any{"expenseId": saved.ID, "amount": in.Amount}, approval.High)
	if err != nil {
		return nil, err
	}
	d.Store.setRequest(saved.ID, r.ID)
	return map[string]any{
		"status":     "pending_approval",
		"expense_id": saved.ID,
		"request_id": r.ID,
		"message":    fmt.Sprintf("%d 円以上のため承認者の承認を待つ。期限は 24 時間", ApprovalThreshold),
	}, nil
}

// Query は経費を照会する。他人の経費は承認者だけが見られる。
func (d Deps) Query(ctx agent.Context, in queryInput) (map[string]any, error) {
	target := in.UserID
	if target == "" {
		target = ctx.UserID()
	}
	if target != ctx.UserID() && !d.isApprover(ctx.UserID()) {
		return refused("他の利用者の経費は照会できない"), nil
	}
	return map[string]any{"status": "ok", "expenses": d.Store.List(target, in.Month)}, nil
}

// Approve は承認待ちの経費を承認か却下する。承認者の一覧にいる、申請者以外の利用者だけができる。
func (d Deps) Approve(ctx agent.Context, in approveInput) (map[string]any, error) {
	e, ok := d.Store.Get(in.ExpenseID)
	if !ok {
		return refused("経費が見つからない"), nil
	}
	if e.Status != PendingApproval {
		return refused("承認待ちの経費ではない"), nil
	}
	if err := d.Approvals.Decide(ctx.UserID(), e.RequestID, in.Approve); err != nil {
		return refused(err.Error()), nil
	}
	st := Rejected
	if in.Approve {
		st = Approved
	}
	d.Store.SetStatus(e.ID, st)
	return map[string]any{"status": string(st), "expense_id": e.ID}, nil
}

// Tools は 3 つのツールを返す。
func (d Deps) Tools() ([]tool.Tool, error) {
	submit, err := functiontool.New(functiontool.Config{
		Name:        ToolSubmit,
		Description: "経費を申請する。category は費目名（交通費など）、description は業務目的や補足だけを入れる。50 万円以上は承認を待つ",
	}, d.Submit)
	if err != nil {
		return nil, err
	}
	query, err := functiontool.New(functiontool.Config{
		Name:        ToolQuery,
		Description: "経費を照会する。userId を省くと自分の経費。month は 2025-07 の形",
	}, d.Query)
	if err != nil {
		return nil, err
	}
	approve, err := functiontool.New(functiontool.Config{
		Name:        ToolApprove,
		Description: "承認待ちの経費を承認または却下する。承認者だけが使える",
	}, d.Approve)
	if err != nil {
		return nil, err
	}
	return []tool.Tool{submit, query, approve}, nil
}
