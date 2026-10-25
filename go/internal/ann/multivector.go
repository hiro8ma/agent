package ann

import (
	"fmt"
	"math"
	"slices"
)

// MultiVector は全文書のトークンを 1 つの行列に続けて詰め、文書 d のトークンは Tokens の行 Start[d] から Start[d+1]-1、DocOf[t] はトークン t の文書。
type MultiVector struct {
	Tokens Matrix
	Start  []int32
	DocOf  []int32
}

func NewMultiVector(docs []Matrix) (*MultiVector, error) {
	mv := &MultiVector{Start: make([]int32, 1, len(docs)+1)}
	if len(docs) == 0 {
		return mv, nil
	}
	dim := docs[0].Dim
	var data []float32
	for d, m := range docs {
		if m.Dim != dim {
			return nil, fmt.Errorf("ann: document %d has %d dimensions, want %d", d, m.Dim, dim)
		}
		data = append(data, m.Data[:m.N*m.Dim]...)
		for range m.N {
			mv.DocOf = append(mv.DocOf, int32(d))
		}
		mv.Start = append(mv.Start, int32(len(mv.DocOf)))
	}
	mv.Tokens = Matrix{N: len(mv.DocOf), Dim: dim, Data: data}
	return mv, nil
}

func (mv *MultiVector) Docs() int { return len(mv.Start) - 1 }

func (mv *MultiVector) Doc(d int) Matrix {
	lo, hi, dim := int(mv.Start[d]), int(mv.Start[d+1]), mv.Tokens.Dim
	return Matrix{N: hi - lo, Dim: dim, Data: mv.Tokens.Data[lo*dim : hi*dim : hi*dim]}
}

// MaxSim はクエリの各トークンについて文書のトークンとの内積の最大を取り、クエリのトークンについて足す。長さ 1 のベクトルでは内積が cos になる。
func MaxSim(q, doc Matrix) float32 {
	var s float32
	for i := range q.N {
		qi := q.Row(i)
		best := float32(math.Inf(-1))
		for j := range doc.N {
			best = max(best, dot(qi, doc.Row(j)))
		}
		s += best
	}
	return s
}

// SearchMaxSim は全文書の MaxSim を計算し、高い順に k 件を返す。Dist には -MaxSim を入れ、kBest と Recall をそのまま使う。
func (mv *MultiVector) SearchMaxSim(q Matrix, k int) []Neighbor {
	t := newKBest(min(k, mv.Docs()))
	for d := range mv.Docs() {
		t.push(Neighbor{ID: int32(d), Dist: -MaxSim(q, mv.Doc(d))})
	}
	return t.sorted()
}

// TokenSearcher はトークンの近い k 件を返す索引。Exact と HNSW が満たす。
type TokenSearcher interface {
	Search(q []float32, k int) []Neighbor
}

// SearchCandidates はクエリの各トークンで ix から近い kTok 件のトークンを取り、それを含む文書だけを MaxSim で並べ直す。2 つ目の値は候補の文書数。
func (mv *MultiVector) SearchCandidates(q Matrix, k, kTok int, ix TokenSearcher) ([]Neighbor, int) {
	var docs []int32
	for i := range q.N {
		for _, n := range ix.Search(q.Row(i), kTok) {
			docs = append(docs, mv.DocOf[n.ID])
		}
	}
	slices.Sort(docs)
	docs = slices.Compact(docs)
	t := newKBest(min(k, len(docs)))
	for _, d := range docs {
		t.push(Neighbor{ID: d, Dist: -MaxSim(q, mv.Doc(int(d)))})
	}
	return t.sorted(), len(docs)
}

// TopicConfig は TopicMultiVector の設定。Tokens と Topics は 1 文書あたりの個数の範囲（両端を含む）。
type TopicConfig struct {
	Docs, Dim, NumTopics int
	MinTokens, MaxTokens int
	MinTopics, MaxTopics int
	Noise                float64
	Seed, SampleSeed     uint64
}

// TopicMultiVector は各トークンを文書の話題の 1 つの周りに散らし、散らばりを文書ごとに Noise の 0.5〜1.5 倍で変える。Seed を共有して SampleSeed を変えると同じ話題からクエリを作れる。
func TopicMultiVector(cfg TopicConfig) *MultiVector {
	cents := randomCenters(cfg.NumTopics, cfg.Dim, cfg.Seed)
	scale := 1 / math.Sqrt(float64(cfg.Dim))
	s := newRand(cfg.SampleSeed)
	mv := &MultiVector{Start: make([]int32, 1, cfg.Docs+1)}
	var data []float32
	row := make([]float64, cfg.Dim)
	for d := range cfg.Docs {
		topics := s.Perm(cfg.NumTopics)[:cfg.MinTopics+s.IntN(cfg.MaxTopics-cfg.MinTopics+1)]
		noise := cfg.Noise * (0.5 + s.Float64())
		ntok := cfg.MinTokens + s.IntN(cfg.MaxTokens-cfg.MinTokens+1)
		for range ntok {
			off := len(data)
			data = append(data, make([]float32, cfg.Dim)...)
			scatter(data[off:], cents[topics[s.IntN(len(topics))]*cfg.Dim:], noise, scale, s, row)
			mv.DocOf = append(mv.DocOf, int32(d))
		}
		mv.Start = append(mv.Start, int32(len(mv.DocOf)))
	}
	mv.Tokens = Matrix{N: len(mv.DocOf), Dim: cfg.Dim, Data: data}
	return mv
}
