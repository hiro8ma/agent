package search

import (
	"cmp"
	"context"
	"hash/fnv"
	"slices"
	"time"
)

// SearchType は Elasticsearch の search_type に対応し、BM25 の文書数 / 文書頻度 / 平均文書長をどこで数えるかを決める。
type SearchType int

const (
	// QueryThenFetch は各シャードが自分の中の統計で採点する。
	QueryThenFetch SearchType = iota
	// DFSQueryThenFetch は先に全シャードの統計を集め、全体の値で採点する。
	DFSQueryThenFetch
)

// Router は文書を置くシャードの番号を返す。範囲の外の値はシャード数で割った余りにする。
type Router func(d Doc) int

// HashRouter は文書 ID の FNV-1a ハッシュで振り分ける。ID が同じなら常に同じシャードに置く。
func HashRouter(shards int) Router {
	return func(d Doc) int {
		h := fnv.New32a()
		_, _ = h.Write([]byte(d.ID))
		return int(h.Sum32() % uint32(max(shards, 1)))
	}
}

// Sharded は文書をシャードに分け、シャードごとに Index を持つ。Type の既定は Elasticsearch と同じ QueryThenFetch。
type Sharded struct {
	Type   SearchType
	shards []*Index
	global [][]int
}

// NewSharded は router で文書を shards 個に分けて索引する。router が nil なら HashRouter を使う。opts はすべてのシャードに同じものを渡す。
func NewSharded(docs []Doc, shards int, router Router, opts ...Option) *Sharded {
	shards = max(shards, 1)
	if router == nil {
		router = HashRouter(shards)
	}
	parts := make([][]Doc, shards)
	global := make([][]int, shards)
	for i, d := range docs {
		s := ((router(d) % shards) + shards) % shards
		parts[s] = append(parts[s], d)
		global[s] = append(global[s], i)
	}
	sh := &Sharded{shards: make([]*Index, shards), global: global}
	for s, p := range parts {
		sh.shards[s] = New(p, opts...)
	}
	return sh
}

// ShardDocs はシャードごとの文書数を返す。
func (sh *Sharded) ShardDocs() []int {
	n := make([]int, len(sh.shards))
	for i, ix := range sh.shards {
		n[i] = len(ix.docs)
	}
	return n
}

// ShardedResult の MatchTime / RankTime は全シャードの和。StatsTime は DFSQueryThenFetch で統計を集めた時間、MergeTime は上位を並べ直した時間。
type ShardedResult struct {
	Result
	StatsTime time.Duration
	MergeTime time.Duration
}

func (sh *Sharded) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	res := sh.Rank(query, limit)
	docs := make([]Doc, len(res.Hits))
	for i, h := range res.Hits {
		docs[i] = h.Doc
	}
	return docs, nil
}

func (sh *Sharded) Rank(query string, limit int) ShardedResult {
	return sh.RankQuery(Query{Text: query}, limit)
}

// RankQuery は各シャードで上位 limit 件を取り、点数の高い順に並べ直す。同点は元の文書の並びで決め、1 台の Index と同じ順にする。
func (sh *Sharded) RankQuery(q Query, limit int) ShardedResult {
	var res ShardedResult
	var c *corpus
	if sh.Type == DFSQueryThenFetch {
		start := time.Now()
		c = sh.collect(q)
		res.StatsTime = time.Since(start)
	}
	type ranked struct {
		hit Hit
		pos int
	}
	var all []ranked
	for s, ix := range sh.shards {
		r, ids := ix.rankWith(q, limit, c)
		res.Scored += r.Scored
		res.MatchTime += r.MatchTime
		res.RankTime += r.RankTime
		for i, h := range r.Hits {
			all = append(all, ranked{hit: h, pos: sh.global[s][ids[i]]})
		}
	}
	start := time.Now()
	slices.SortFunc(all, func(a, b ranked) int {
		return cmp.Or(cmp.Compare(b.hit.Score, a.hit.Score), cmp.Compare(a.pos, b.pos))
	})
	if limit >= 0 && len(all) > limit {
		all = all[:limit]
	}
	res.Hits = make([]Hit, len(all))
	for i, r := range all {
		res.Hits[i] = r.hit
	}
	res.MergeTime = time.Since(start)
	return res
}

func (sh *Sharded) collect(q Query) *corpus {
	q = q.scoringText()
	c := &corpus{df: make(map[string]int)}
	var total [numFields]int
	for _, ix := range sh.shards {
		c.docs += len(ix.docs)
		for f := range numFields {
			total[f] += ix.totalLen[f]
		}
		pq := ix.parse(q)
		for i, n := range ix.docFreq(pq.terms, selectFields(q.Fields)) {
			c.df[pq.words[i]] += n
		}
	}
	c.avgLen = averageLength(total, c.docs)
	return c
}
