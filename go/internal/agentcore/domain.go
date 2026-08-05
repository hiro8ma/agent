package agentcore

import "context"

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
