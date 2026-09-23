package adkagent

import (
	"errors"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/agentcore/actionexec"
)

type getOrderInput struct {
	OrderID string `json:"orderId"`
}

type updatePaymentMethodInput struct {
	OrderID       string `json:"orderId"`
	PaymentMethod string `json:"paymentMethod"`
}

type resolveAreaNamesInput struct {
	AreaIDs []string `json:"areaIds"`
}

type searchKnowledgeInput struct {
	Query string `json:"query"`
}

// toolResult は genkit 版と同じ規約。エラーを返さず {"error": ...} に畳み込み、
// エラーで会話全体を落とさずモデルに続きを判断させる。
func toolResult(data map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	return data, nil
}

// ResearchTools は技術調査エージェントのツール群。
func ResearchTools(knowledge agentcore.KnowledgeSearcher) ([]tool.Tool, error) {
	searchKnowledge, err := functiontool.New(functiontool.Config{
		Name:        "search_knowledge",
		Description: "社内ナレッジ（データストア）をキーワード検索して関連ドキュメントを取得する",
	}, func(ctx agent.Context, in searchKnowledgeInput) (map[string]any, error) {
		if in.Query == "" {
			return toolResult(nil, errors.New("search_knowledge: query is required"))
		}
		docs, err := knowledge.Search(ctx, in.Query, 3)
		if err != nil {
			return toolResult(nil, err)
		}
		return toolResult(map[string]any{"documents": docs}, nil)
	})
	if err != nil {
		return nil, err
	}
	return []tool.Tool{searchKnowledge}, nil
}

// OperationsTools は申請処理エージェントのツール群。
// 書き込み系は action の窓口が実行してよいかを決め、承認待ちなら依頼の ID を返すだけで実行しない。
func OperationsTools(orders agentcore.OrderService, geo agentcore.GeoService, gate action.Gate) ([]tool.Tool, error) {
	getOrder, err := functiontool.New(functiontool.Config{
		Name:        "get_order",
		Description: "指定された ID の注文情報を取得する",
	}, func(ctx agent.Context, in getOrderInput) (map[string]any, error) {
		if in.OrderID == "" {
			return toolResult(nil, errors.New("get_order: orderId is required"))
		}
		order, err := orders.GetOrder(ctx, in.OrderID)
		if err != nil {
			return toolResult(nil, err)
		}
		return toolResult(map[string]any{"order": order}, nil)
	})
	if err != nil {
		return nil, err
	}

	resolveAreaNames, err := functiontool.New(functiontool.Config{
		Name:        "resolve_area_names",
		Description: "エリア ID（areaIds）を地名に変換する",
	}, func(ctx agent.Context, in resolveAreaNamesInput) (map[string]any, error) {
		if len(in.AreaIDs) == 0 {
			return toolResult(nil, errors.New("resolve_area_names: areaIds is required"))
		}
		names, err := geo.ResolveAreaNames(ctx, in.AreaIDs)
		if err != nil {
			return toolResult(nil, err)
		}
		return toolResult(map[string]any{"areaNames": names}, nil)
	})
	if err != nil {
		return nil, err
	}

	updatePaymentMethod, err := functiontool.New(functiontool.Config{
		Name:        action.ToolUpdatePaymentMethod,
		Description: "注文の支払い方法を変更する。承認が要る場合は承認の依頼だけを登録し、利用者に承認待ちであることを伝える",
	}, func(ctx agent.Context, in updatePaymentMethodInput) (map[string]any, error) {
		out, err := actionexec.RequestPaymentChange(ctx, gate, orders, in.OrderID, in.PaymentMethod)
		if err != nil {
			return toolResult(nil, err)
		}
		return out, nil
	})
	if err != nil {
		return nil, err
	}

	return []tool.Tool{getOrder, resolveAreaNames, updatePaymentMethod}, nil
}
