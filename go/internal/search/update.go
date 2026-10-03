package search

import (
	"cmp"
	"maps"
	"slices"
)

// Updatable は主の索引と、後から足した文書だけを持つ小さな補助の索引の 2 つで検索する。補助の文書の番号は主の文書の後ろに続く。
// BM25 は 2 つの索引の文書数、項目ごとの長さの和、語ごとの文書頻度の和で採点するので、すべての文書で作り直した索引と同じ点になる。
// TF-IDF のコサインの文書ベクトルの長さは索引ごとの統計のままなので、Merge するまで作り直した索引と一致しない。
// Add / Merge と検索を同時に呼んではならない。
type Updatable struct {
	opts  []Option
	main  *Index
	aux   *Index
	added []Doc
}

func NewUpdatable(docs []Doc, opts ...Option) *Updatable {
	return &Updatable{opts: opts, main: New(docs, opts...)}
}

// Add は足した文書すべてから補助の索引を作り直す。主の索引には触れないので、かかる時間は補助の文書の数で決まる。
func (u *Updatable) Add(docs ...Doc) {
	u.added = append(u.added, docs...)
	u.aux = New(slices.Clone(u.added), u.opts...)
}

// Sizes は主と補助の索引の文書の数を返す。
func (u *Updatable) Sizes() (main, aux int) {
	return len(u.main.docs), len(u.added)
}

// Merge は補助の索引を主の索引に畳み込み、補助を空にする。
func (u *Updatable) Merge() {
	if u.aux == nil {
		return
	}
	u.main = mergeIndex(u.main, u.aux, u.opts)
	u.aux, u.added = nil, nil
}

// mergeIndex は aux の文書番号を main の文書数だけずらして postings を語ごとに後ろにつなぐ。文書を解析し直さない。
// 語の番号は main の語の後ろに aux で初めて出た語を aux の順に足すので、すべての文書を順に New に渡したときと同じになる。
// 文書拡張と doc2query は語と文書の組ごとの値を持つので、文書から作り直す。
func mergeIndex(main, aux *Index, opts []Option) *Index {
	if main.docExpansion != nil || main.d2q != nil {
		return New(slices.Concat(main.docs, aux.docs), opts...)
	}
	ix := *main
	off := int32(len(main.docs))
	ix.docs = slices.Concat(main.docs, aux.docs)
	ix.lengths = slices.Concat(main.lengths, aux.lengths)
	for f := range numFields {
		ix.totalLen[f] = main.totalLen[f] + aux.totalLen[f]
	}
	ix.avgLen = averageLength(ix.totalLen, len(ix.docs))
	ix.vocab = maps.Clone(main.vocab)
	ix.words = slices.Clone(main.words)
	ix.postings = slices.Clone(main.postings)
	for t, w := range aux.words {
		shifted := make([]posting, len(aux.postings[t]))
		for i, p := range aux.postings[t] {
			shifted[i] = posting{doc: p.doc + off, split: p.split, pos: p.pos}
		}
		id := ix.termID(w)
		ix.postings[id] = slices.Concat(ix.postings[id], shifted)
	}
	ix.norms = ix.docNorms(len(ix.docs), nil)
	ix.trie = newVocabTrie(ix.words)
	return &ix
}

func (u *Updatable) Rank(query string, limit int) Result {
	return u.RankQuery(Query{Text: query}, limit)
}

// RankQuery は 2 つの索引をそれぞれ全体の統計で採点し、点数の高い順に並べ直す。同点は文書の番号の順で、1 つの索引と同じ順にする。
func (u *Updatable) RankQuery(q Query, limit int) Result {
	if u.aux == nil {
		return u.main.RankQuery(q, limit)
	}
	ixs := []*Index{u.main, u.aux}
	c := collectCorpus(ixs, q)
	type ranked struct {
		hit Hit
		pos int
	}
	var (
		all []ranked
		res Result
	)
	for i, ix := range ixs {
		r, ids := ix.rankWith(q, limit, c)
		res.Scored += r.Scored
		res.Expanded, res.ExpansionDropped = r.Expanded, r.ExpansionDropped
		res.MatchTime += r.MatchTime
		res.RankTime += r.RankTime
		off := 0
		if i == 1 {
			off = len(u.main.docs)
		}
		for j, h := range r.Hits {
			all = append(all, ranked{hit: h, pos: ids[j] + off})
		}
	}
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
	return res
}
