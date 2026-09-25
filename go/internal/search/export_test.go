package search

import "strings"

// WithGallopRatio は galloping に切り替える長さの比を変える。0 なら常に 2 つのポインタで共通部分を取る。
func WithGallopRatio(ratio int) Option {
	return func(ix *Index) { ix.gallopRatio = ratio }
}

// WithWhitespaceTokenizer は空白で区切った語をそのまま索引語にする。
func WithWhitespaceTokenizer() AnalyzerOption {
	return func(c *analyzerConfig) { c.tokenize = strings.Fields }
}

type TestPosting struct {
	Doc       int
	TF        int
	Positions []int32
}

// PostingsOf は索引語の postings を文書番号の昇順で返す。位置は索引の中と同じく 0 から数え、タイトルと本文をつないだもの。
func PostingsOf(ix *Index, term string) []TestPosting {
	ps := ix.postingsOf(ix.lookupTerm(term))
	out := make([]TestPosting, len(ps))
	for i, p := range ps {
		out[i] = TestPosting{Doc: int(p.doc), TF: len(p.pos), Positions: p.pos}
	}
	return out
}

const DefaultGallopRatio = defaultGallopRatio

// WithGallopRatioOf は索引を共有したまま、galloping に切り替える比だけを変えた写しを返す。
func WithGallopRatioOf(ix *Index, ratio int) *Index {
	cp := *ix
	cp.gallopRatio = ratio
	return &cp
}
