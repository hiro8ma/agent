package model

// ToolUpdatePaymentMethod は承認の窓口がリスクを決めるときの鍵になるので、窓口の設定と同じ名前にする。
const ToolUpdatePaymentMethod = "update_order_payment_method"

const ActionStatusPending = "pending_approval"

type ActionOutcome string

const (
	ActionAllow         ActionOutcome = "allow"
	ActionNeedsApproval ActionOutcome = "pending"
	ActionDeny          ActionOutcome = "forbidden"
)

type ActionDecision struct {
	Outcome   ActionOutcome
	Risk      string
	RequestID string
}

type ApprovedAction struct {
	ID   string
	Tool string
	Args map[string]any
}
