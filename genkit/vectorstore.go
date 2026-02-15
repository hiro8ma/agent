package genkit

import (
	"math"
	"sort"
	"sync"
)

// VectorDocument はベクトル化されたドキュメント
type VectorDocument struct {
	ID        string
	Content   string
	Embedding []float32
	Metadata  map[string]string
}

// VectorStore はインメモリのベクトルストア
type VectorStore struct {
	mu        sync.RWMutex
	documents []VectorDocument
}

// NewVectorStore は新しいベクトルストアを作成
func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents: make([]VectorDocument, 0),
	}
}

// Add はドキュメントをストアに追加
func (vs *VectorStore) Add(doc VectorDocument) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.documents = append(vs.documents, doc)
}

// Search はコサイン類似度で類似ドキュメントを検索
func (vs *VectorStore) Search(queryEmbedding []float32, topK int) []VectorDocument {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	type scored struct {
		doc   VectorDocument
		score float64
	}

	var results []scored
	for _, doc := range vs.documents {
		score := cosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, scored{doc: doc, score: score})
	}

	// スコア降順でソート
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// 上位K件を返す
	var topDocs []VectorDocument
	for i := 0; i < topK && i < len(results); i++ {
		topDocs = append(topDocs, results[i].doc)
	}
	return topDocs
}

// Len はドキュメント数を返す
func (vs *VectorStore) Len() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.documents)
}

// cosineSimilarity はコサイン類似度を計算
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
