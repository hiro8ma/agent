// Package domain は ADK のサンプルが使うポートを定義する。
//
// 実装は internal/adk/repository に置く。
// 依存の向きは repository から domain への一方向にする。
package domain

import (
	"context"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

// MemoryStore は Session を越える記憶の読み書き。
//
// ADK の memory.Service と同じ形にしてある。
// それでも自前で宣言するのは、テストで差し替える先を自分の側に持つため。
type MemoryStore interface {
	// AddSessionToMemory は Session の内容を記憶へ取り込む。
	AddSessionToMemory(ctx context.Context, s session.Session) error
	// SearchMemory は問い合わせに関係する記憶を返す。
	SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error)
}

// BankLocation は Memory Bank の接続先。
//
// 3 つそろわないと接続できない。
// そろわない場合に黙って既定値へ落ちると、意図せず課金される経路ができるので、
// 判定は値オブジェクト側に持たせて呼び出し側から隠さない。
type BankLocation struct {
	ProjectID       string
	Location        string
	ReasoningEngine string
}

// Complete は接続に必要な値がそろっているかを返す。
func (b BankLocation) Complete() bool {
	return b.ProjectID != "" && b.Location != "" && b.ReasoningEngine != ""
}

// Missing は欠けている項目の名前を返す。設定漏れをそのまま利用者へ見せる用途に使う。
func (b BankLocation) Missing() []string {
	var missing []string
	if b.ProjectID == "" {
		missing = append(missing, "ProjectID")
	}
	if b.Location == "" {
		missing = append(missing, "Location")
	}
	if b.ReasoningEngine == "" {
		missing = append(missing, "ReasoningEngine")
	}
	return missing
}
