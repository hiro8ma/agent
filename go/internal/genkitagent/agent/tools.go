package agent

import (
	"context"
	"errors"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// OrderService は注文マイクロサービスへの接続。実運用では gRPC クライアントが実装する。
type OrderService interface {
	GetOrder(ctx context.Context, orderID string) (*Order, error)
	UpdatePaymentMethod(ctx context.Context, orderID, paymentMethod string) (*Order, error)
}

// GeoService はエリアマイクロサービスへの接続。
type GeoService interface {
	ResolveAreaNames(ctx context.Context, areaIDs []string) (map[string]string, error)
}

// KnowledgeSearcher はデータストア（社内ナレッジ検索）への接続。
// 実運用では Vertex AI Search / RAG Engine / pgvector が実装する。
type KnowledgeSearcher interface {
	Search(ctx context.Context, query string, limit int) ([]KnowledgeDoc, error)
}

// KnowledgeDoc は検索結果の 1 ドキュメント。
type KnowledgeDoc struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// Order は注文情報。
type Order struct {
	ID            string `json:"id"`
	CustomerName  string `json:"customerName"`
	PaymentMethod string `json:"paymentMethod"`
	AmountJPY     int    `json:"amountJpy"`
}

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

const updateOrderPaymentMethodTool = "update_order_payment_method"

// DefineResearchTools は技術調査エージェントのツール群。
func DefineResearchTools(g *genkit.Genkit, knowledge KnowledgeSearcher) []ai.ToolRef {
	return defineKnowledgeTools(g, knowledge)
}

// DefineOperationsTools は申請処理エージェントのツール群。
func DefineOperationsTools(g *genkit.Genkit, orders OrderService, geo GeoService, pending PendingStore) []ai.ToolRef {
	tools := defineOrderTools(g, orders)
	tools = append(tools, defineGeoTools(g, geo)...)
	return append(tools, defineWriteTools(g, pending)...)
}

// ツールはエラーを返さず {"error": ...} を結果に含める。エラーで会話全体を落とさず、モデルに続きを判断させるため。
func toolResult(data map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	return data, nil
}

func defineOrderTools(g *genkit.Genkit, orders OrderService) []ai.ToolRef {
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

func defineGeoTools(g *genkit.Genkit, geo GeoService) []ai.ToolRef {
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

func defineKnowledgeTools(g *genkit.Genkit, knowledge KnowledgeSearcher) []ai.ToolRef {
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
// 直接実行せず承認待ちとして登録し、ExecuteConfirmedToolCall による人間の承認後に実行する。
func defineWriteTools(g *genkit.Genkit, pending PendingStore) []ai.ToolRef {
	updatePaymentMethod := genkit.DefineTool(g, updateOrderPaymentMethodTool,
		"注文の支払い方法を変更する。実行には人間の承認が必要で、このツールは承認依頼の登録だけを行う",
		func(ctx *ai.ToolContext, in updateOrderPaymentMethodInput) (map[string]any, error) {
			if in.OrderID == "" || in.PaymentMethod == "" {
				return toolResult(nil, errors.New(updateOrderPaymentMethodTool+": orderId and paymentMethod are required"))
			}
			p := PendingToolCall{
				ID:   newToolCallID(),
				Name: updateOrderPaymentMethodTool,
				Input: map[string]any{
					"orderId":       in.OrderID,
					"paymentMethod": in.PaymentMethod,
				},
			}
			if err := pending.Save(ctx, p); err != nil {
				return toolResult(nil, err)
			}
			return map[string]any{
				"status":     "confirmation_required",
				"toolCallId": p.ID,
				"input":      p.Input,
				"message":    "この操作は人間の承認後に実行されます。ユーザーには承認が必要である旨を伝えてください。",
			}, nil
		},
	)
	return []ai.ToolRef{updatePaymentMethod}
}
