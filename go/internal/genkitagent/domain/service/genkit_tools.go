package service

import (
	"context"
	"errors"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

type getOrderInput struct {
	OrderID string `json:"orderId" jsonschema_description:"注文 ID"`
}

type resolveAreaNamesInput struct {
	AreaIDs []string `json:"areaIds" jsonschema_description:"エリア ID のリスト"`
}

type searchKnowledgeInput struct {
	Query string `json:"query" jsonschema_description:"検索クエリ"`
}

type updateOrderPaymentMethodInput struct {
	OrderID       string `json:"orderId"       jsonschema_description:"注文 ID"`
	PaymentMethod string `json:"paymentMethod" jsonschema_description:"変更後の支払い方法"`
}

// DefineResearchTools は技術調査エージェントのツール群。
func DefineResearchTools(g *genkit.Genkit, knowledge externalservice.KnowledgeSearcher) []ai.ToolRef {
	return defineKnowledgeTools(g, knowledge)
}

// DefineOperationsTools は申請処理エージェントのツール群。
func DefineOperationsTools(g *genkit.Genkit, orders externalservice.OrderService, geo externalservice.GeoService, gate externalservice.ActionGate) []ai.ToolRef {
	tools := defineOrderTools(g, orders)
	tools = append(tools, defineGeoTools(g, geo)...)
	return append(tools, defineWriteTools(g, gate, orders)...)
}

// ツールはエラーを返さず {"error": ...} を結果に含める。エラーで会話全体を落とさず、モデルに続きを判断させるため。
func toolResult(data map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	return data, nil
}

func defineOrderTools(g *genkit.Genkit, orders externalservice.OrderService) []ai.ToolRef {
	getOrder := genkit.DefineTool(g, "get_order",
		"指定された ID の注文情報を取得する",
		func(ctx *ai.ToolContext, in getOrderInput) (map[string]any, error) {
			if in.OrderID == "" {
				return toolResult(nil, errors.New("get_order: orderId is required"))
			}
			order, err := orders.GetOrder(ctx, in.OrderID)
			if err != nil {
				return toolResult(nil, err)
			}
			return toolResult(map[string]any{"order": order}, nil)
		},
	)
	return []ai.ToolRef{getOrder}
}

func defineGeoTools(g *genkit.Genkit, geo externalservice.GeoService) []ai.ToolRef {
	resolveAreaNames := genkit.DefineTool(g, "resolve_area_names",
		"エリア ID（areaIds）を地名に変換する",
		func(ctx *ai.ToolContext, in resolveAreaNamesInput) (map[string]any, error) {
			if len(in.AreaIDs) == 0 {
				return toolResult(nil, errors.New("resolve_area_names: areaIds is required"))
			}
			names, err := geo.ResolveAreaNames(ctx, in.AreaIDs)
			if err != nil {
				return toolResult(nil, err)
			}
			return toolResult(map[string]any{"areaNames": names}, nil)
		},
	)
	return []ai.ToolRef{resolveAreaNames}
}

func defineKnowledgeTools(g *genkit.Genkit, knowledge externalservice.KnowledgeSearcher) []ai.ToolRef {
	searchKnowledge := genkit.DefineTool(g, "search_knowledge",
		"社内ナレッジ（データストア）をキーワード検索して関連ドキュメントを取得する",
		func(ctx *ai.ToolContext, in searchKnowledgeInput) (map[string]any, error) {
			if in.Query == "" {
				return toolResult(nil, errors.New("search_knowledge: query is required"))
			}
			docs, err := knowledge.Search(ctx, in.Query, 3)
			if err != nil {
				return toolResult(nil, err)
			}
			return toolResult(map[string]any{"documents": docs}, nil)
		},
	)
	return []ai.ToolRef{searchKnowledge}
}

// defineWriteTools は書き込み系ツールを定義する。
// 実行してよいかは承認の窓口が決める。承認待ちなら依頼の ID を返すだけで実行しない。
func defineWriteTools(g *genkit.Genkit, gate externalservice.ActionGate, orders externalservice.OrderService) []ai.ToolRef {
	updatePaymentMethod := genkit.DefineTool(g, model.ToolUpdatePaymentMethod,
		"注文の支払い方法を変更する。承認が要る場合は承認の依頼だけを登録し、利用者に承認待ちであることを伝える",
		func(ctx *ai.ToolContext, in updateOrderPaymentMethodInput) (map[string]any, error) {
			out, err := requestPaymentChange(ctx, gate, orders, in.OrderID, in.PaymentMethod)
			if err != nil {
				return toolResult(nil, err)
			}
			return out, nil
		},
	)
	return []ai.ToolRef{updatePaymentMethod}
}

// requestPaymentChange の結果はモデルが読むので、承認待ちなら依頼の ID と、何を待っているかを入れる。
func requestPaymentChange(ctx context.Context, gate externalservice.ActionGate, orders externalservice.OrderService, orderID, method string) (map[string]any, error) {
	if orderID == "" || method == "" {
		return map[string]any{"status": "error", "message": "orderId と paymentMethod が要る"}, nil
	}
	args := map[string]any{"orderId": orderID, "paymentMethod": method}
	d, err := gate.Authorize(ctx, model.ToolUpdatePaymentMethod, args)
	if err != nil {
		return nil, err
	}
	switch d.Outcome {
	case model.ActionAllow:
		order, err := orders.UpdatePaymentMethod(ctx, orderID, method)
		if err != nil {
			return nil, err
		}
		return map[string]any{"status": "done", "order": order}, nil
	case model.ActionDeny:
		return map[string]any{"status": "forbidden", "message": "この操作は承認があっても実行できない"}, nil
	case model.ActionNeedsApproval:
	}
	return map[string]any{
		"status":     model.ActionStatusPending,
		"request_id": d.RequestID,
		"risk":       d.Risk,
		"input":      args,
		"message":    "承認者の承認を待っている。承認されたら依頼者が実行する",
	}, nil
}

func pendingFromResult(tool string, out map[string]any) (model.PendingToolCall, bool) {
	if out["status"] != model.ActionStatusPending {
		return model.PendingToolCall{}, false
	}
	id, _ := out["request_id"].(string)
	input, _ := out["input"].(map[string]any)
	return model.PendingToolCall{ID: id, Name: tool, Input: input}, id != ""
}
