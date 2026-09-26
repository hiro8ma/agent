package search

import (
	"strings"
	"unsafe"
)

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

// MatchAnd はクエリの語すべてを含む文書の番号を、採点せずに昇順で返す。
func MatchAnd(ix *Index, text string) []int {
	ids, _ := ix.matchAll(ix.parse(Query{Text: text, Operator: OperatorAnd}), selectFields(nil), nil)
	return ids
}

// QueryTerms はクエリを索引と同じ方法で分けた索引語を返す。
func QueryTerms(ix *Index, text string) []string {
	return ix.parse(Query{Text: text}).words
}

// PostingsBytes はメモリ上の postings の大きさを、posting の構造体の分と位置の配列の分に分けて返す。
func PostingsBytes(ix *Index) (structs, positions int64) {
	for _, ps := range ix.postings {
		structs += int64(len(ps)) * int64(unsafe.Sizeof(posting{}))
		for _, p := range ps {
			positions += int64(len(p.pos)) * int64(unsafe.Sizeof(int32(0)))
		}
	}
	return structs, positions
}
