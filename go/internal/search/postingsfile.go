package search

import (
	"bufio"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
)

// FilePosting はファイルに書いた出現記録の 1 件。位置は書かず、TF はタイトルと本文を合わせた出現回数。
type FilePosting struct {
	Doc int
	TF  int
}

type postingsSpan struct {
	off, size int64
	count     int
}

// PostingsFile は索引の postings を語ごとに（文書番号の差分, 出現回数）の varint の列で 1 つのファイルに並べたもの。語ごとの開始位置と長さはメモリ上の表に持つ。
type PostingsFile struct {
	f     *os.File
	size  int64
	table map[string]postingsSpan
}

// WritePostingsFile は ix の postings を path に書き出し、読むために開いた PostingsFile を返す。使い終えたら Close する。
func WritePostingsFile(ix *Index, path string) (*PostingsFile, error) {
	path = filepath.Clean(path)
	out, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriter(out)
	table := make(map[string]postingsSpan, len(ix.words))
	var (
		off int64
		buf []byte
	)
	for id, word := range ix.words {
		buf = encodePostings(buf[:0], ix.postings[id])
		if _, err := w.Write(buf); err != nil {
			return nil, errors.Join(err, out.Close())
		}
		table[word] = postingsSpan{off: off, size: int64(len(buf)), count: len(ix.postings[id])}
		off += int64(len(buf))
	}
	if err := w.Flush(); err != nil {
		return nil, errors.Join(err, out.Close())
	}
	if err := out.Close(); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &PostingsFile{f: f, size: off, table: table}, nil
}

func encodePostings(buf []byte, ps []posting) []byte {
	var prev int32
	for _, p := range ps {
		buf = binary.AppendUvarint(buf, uint64(p.doc-prev))
		buf = binary.AppendUvarint(buf, uint64(len(p.pos)))
		prev = p.doc
	}
	return buf
}

func (pf *PostingsFile) Close() error { return pf.f.Close() }

// Size はファイルの大きさ（バイト）を返す。
func (pf *PostingsFile) Size() int64 { return pf.size }

// Postings は語の出現記録をファイルから読み戻す。語彙に無い語なら空を返す。
func (pf *PostingsFile) Postings(term string) ([]FilePosting, error) {
	s, ok := pf.table[term]
	if !ok {
		return []FilePosting{}, nil
	}
	c := pf.cursor(s)
	out := make([]FilePosting, 0, s.count)
	for {
		ok, err := c.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			return out, nil
		}
		out = append(out, FilePosting{Doc: int(c.doc), TF: c.tf})
	}
}

// postingsCursor は 1 語の出現記録をファイルの先頭から順に 1 件ずつ読む。
type postingsCursor struct {
	r    *bufio.Reader
	left int
	doc  int32
	tf   int
}

func (pf *PostingsFile) cursor(s postingsSpan) *postingsCursor {
	return &postingsCursor{r: bufio.NewReader(io.NewSectionReader(pf.f, s.off, s.size)), left: s.count}
}

func (c *postingsCursor) next() (bool, error) {
	if c.left == 0 {
		return false, nil
	}
	delta, err := binary.ReadUvarint(c.r)
	if err != nil {
		return false, fmt.Errorf("read doc: %w", err)
	}
	tf, err := binary.ReadUvarint(c.r)
	if err != nil {
		return false, fmt.Errorf("read tf: %w", err)
	}
	c.doc += int32(delta)
	c.tf = int(tf)
	c.left--
	return true, nil
}

// seek は文書番号が doc 以上の出現記録まで読み進める。読み切ったら false。
func (c *postingsCursor) seek(doc int32) (bool, error) {
	for c.doc < doc {
		ok, err := c.next()
		if err != nil || !ok {
			return false, err
		}
	}
	return true, nil
}

// And はすべての語を含む文書の番号を昇順で返す。短い語の出現記録を 1 件ずつ読み、そのたびに他の語をその文書番号まで読み進めるので、出現記録をまとめてメモリに載せない。
func (pf *PostingsFile) And(terms ...string) ([]int, error) {
	spans, ok := pf.spans(terms)
	if !ok {
		return []int{}, nil
	}
	cs := make([]*postingsCursor, len(spans))
	for i, s := range spans {
		cs[i] = pf.cursor(s)
	}
	for _, c := range cs[1:] {
		if ok, err := c.next(); err != nil || !ok {
			return []int{}, err
		}
	}
	out := []int{}
	for {
		ok, err := cs[0].next()
		if err != nil {
			return nil, err
		}
		if !ok {
			return out, nil
		}
		doc, all := cs[0].doc, true
		for _, c := range cs[1:] {
			ok, err := c.seek(doc)
			if err != nil {
				return nil, err
			}
			if !ok {
				return out, nil
			}
			all = all && c.doc == doc
		}
		if all {
			out = append(out, int(doc))
		}
	}
}

// AndLoaded は各語の出現記録をファイルから全部読んでメモリに展開してから、メモリ上の検索と同じ方法で共通部分を取る。
func (pf *PostingsFile) AndLoaded(terms ...string) ([]int, error) {
	spans, ok := pf.spans(terms)
	if !ok {
		return []int{}, nil
	}
	lists := make([][]posting, len(spans))
	for i, s := range spans {
		raw := make([]byte, s.size)
		if _, err := pf.f.ReadAt(raw, s.off); err != nil {
			return nil, err
		}
		ps, err := decodePostings(raw, s.count)
		if err != nil {
			return nil, err
		}
		lists[i] = ps
	}
	all := selectFields(nil)
	ids := docsOf(lists[0], all, true)
	for _, ps := range lists[1:] {
		ids = intersect(ids, ps, all, true, defaultGallopRatio)
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out, nil
}

func decodePostings(raw []byte, count int) ([]posting, error) {
	ps := make([]posting, count)
	var doc int32
	for i := range ps {
		delta, n := binary.Uvarint(raw)
		if n <= 0 {
			return nil, errors.New("decode doc")
		}
		raw = raw[n:]
		if _, n = binary.Uvarint(raw); n <= 0 {
			return nil, errors.New("decode tf")
		}
		raw = raw[n:]
		doc += int32(delta)
		ps[i] = posting{doc: doc}
	}
	return ps, nil
}

// spans は語の表を出現記録の短い順に並べて返す。語が無いか語彙に無い語が 1 つでもあれば false。
func (pf *PostingsFile) spans(terms []string) ([]postingsSpan, bool) {
	if len(terms) == 0 {
		return nil, false
	}
	spans := make([]postingsSpan, len(terms))
	for i, t := range terms {
		s, ok := pf.table[t]
		if !ok || s.count == 0 {
			return nil, false
		}
		spans[i] = s
	}
	slices.SortFunc(spans, func(a, b postingsSpan) int { return cmp.Compare(a.count, b.count) })
	return spans, true
}
