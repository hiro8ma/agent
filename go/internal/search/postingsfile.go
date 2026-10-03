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
	"unsafe"
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
	// skips は語の開始位置ごとの飛び先。BuildSkipIndex を呼ぶまで nil。
	skips map[int64]*skipList
}

// WritePostingsFile は ix の postings を path に書き出し、読むために開いた PostingsFile を返す。使い終えたら Close する。
func WritePostingsFile(ix *Index, path string) (*PostingsFile, error) {
	return writePostings(ix.words, ix.postings, path)
}

func writePostings(words []string, postings [][]posting, path string) (*PostingsFile, error) {
	path = filepath.Clean(path)
	out, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriter(out)
	table := make(map[string]postingsSpan, len(words))
	var (
		off int64
		buf []byte
	)
	for id, word := range words {
		buf = encodePostings(buf[:0], postings[id])
		if _, err := w.Write(buf); err != nil {
			return nil, errors.Join(err, out.Close())
		}
		table[word] = postingsSpan{off: off, size: int64(len(buf)), count: len(postings[id])}
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

// postingsCursor は 1 語の出現記録をファイルの先頭から順に 1 件ずつ読む。skips があれば seek で飛び先に移ってから順に読む。
// 読んだ件数とバイト数は 1 件ごとには数えず、飛ぶときと stats を呼んだときに、区間の始まりからの位置の差で足す。
type postingsCursor struct {
	pf    *PostingsFile
	span  postingsSpan
	r     *bufio.Reader
	src   countingReader
	left  int
	doc   int32
	tf    int
	skips *skipList
	sk    int
	// base は今の区間を読み始めた語の先頭からのバイト位置、baseIdx はその位置の出現記録の番号。
	base    int64
	baseIdx int
	done    cursorStats
}

// cursorStats は読んだ出現記録の件数とバイト数、ファイルから読んだバイト数、飛び先の文書番号を比べた回数。
type cursorStats struct {
	entries, decoded, fileBytes int64
	skipCompares                int64
}

func (st *cursorStats) add(o cursorStats) {
	st.entries += o.entries
	st.decoded += o.decoded
	st.fileBytes += o.fileBytes
	st.skipCompares += o.skipCompares
}

// countingReader はファイルから読んだバイト数を、今の区間の分（n）と全体の分（total）で数える。
type countingReader struct {
	r        io.Reader
	n, total int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	c.total += int64(n)
	return n, err
}

func (pf *PostingsFile) cursor(s postingsSpan) *postingsCursor {
	return pf.cursorWith(s, nil)
}

func (pf *PostingsFile) cursorWith(s postingsSpan, skips *skipList) *postingsCursor {
	c := &postingsCursor{pf: pf, span: s, left: s.count, skips: skips}
	c.src.r = io.NewSectionReader(pf.f, s.off, s.size)
	c.r = bufio.NewReader(&c.src)
	return c
}

// pos は語の先頭から読み終えたバイト位置。
func (c *postingsCursor) pos() int64 { return c.base + c.src.n - int64(c.r.Buffered()) }

func (c *postingsCursor) read() int { return c.span.count - c.left }

// closeSpan は今の区間で読んだ件数とバイト数を足す。
func (c *postingsCursor) closeSpan() {
	c.done.entries += int64(c.read() - c.baseIdx)
	c.done.decoded += c.pos() - c.base
}

func (c *postingsCursor) stats() cursorStats {
	st := c.done
	st.entries += int64(c.read() - c.baseIdx)
	st.decoded += c.pos() - c.base
	st.fileBytes = c.src.total
	return st
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
	if c.doc < doc && c.skips != nil {
		c.jump(doc)
	}
	for c.doc < doc {
		ok, err := c.next()
		if err != nil || !ok {
			return false, err
		}
	}
	return true, nil
}

// jump は文書番号が doc より小さい最後の飛び先が今の位置より先にあれば、そこへ移って読み直す。飛び先の文書番号から次の差分を足すので、飛んだ後も順に読める。
func (c *postingsCursor) jump(doc int32) {
	sl := c.skips
	k := sl.find(c.sk, doc, &c.done.skipCompares)
	c.sk = max(c.sk, k)
	if k < 0 || int(sl.entries[k].idx) <= c.read() {
		return
	}
	e := sl.entries[k]
	c.closeSpan()
	// 飛び先が読み込み済みの範囲にあれば、ファイルを読み直さずに読み捨てる。
	if gap := e.off - c.pos(); gap <= int64(c.r.Buffered()) {
		_, _ = c.r.Discard(int(gap))
		c.src.n = int64(c.r.Buffered())
	} else {
		c.src.r = io.NewSectionReader(c.pf.f, c.span.off+e.off, c.span.size-e.off)
		c.src.n = 0
		c.r.Reset(&c.src)
	}
	c.base, c.baseIdx = e.off, int(e.idx)
	c.doc = e.doc
	c.left = c.span.count - int(e.idx)
}

// skipEntry は interval 件目ごとの飛び先。idx 件目の出現記録が語の先頭から off バイト目に始まり、doc はその 1 つ前の出現記録の文書番号。
type skipEntry struct {
	doc int32
	idx int32
	off int64
}

// skipList は 1 語の飛び先。long が正なら long 個おきの飛び先を上の段に持ち、上の段を先にたどってから下の段をたどる。
// linear なら飛び先を今の位置から 1 つずつたどる。そうでなければ二分探索で引く。
type skipList struct {
	entries []skipEntry
	long    int
	linear  bool
}

// find は start 番目以降で、文書番号が doc より小さい最後の飛び先の番号を返す。無ければ start-1 以下を返す。比べた回数を n に足す。
func (sl *skipList) find(start int, doc int32, n *int64) int {
	es := sl.entries
	start = max(start, 0)
	if !sl.linear && sl.long <= 0 {
		k, _ := slices.BinarySearchFunc(es[start:], doc, func(e skipEntry, d int32) int {
			*n++
			return cmp.Compare(e.doc, d)
		})
		return start + k - 1
	}
	k := start
	if sl.long > 0 {
		for k+sl.long < len(es) {
			*n++
			if es[k+sl.long].doc >= doc {
				break
			}
			k += sl.long
		}
	}
	for k < len(es) {
		*n++
		if es[k].doc >= doc {
			break
		}
		k++
	}
	return k - 1
}

// SkipConfig は出現記録のファイルに持たせる飛び先の作り方。Interval は飛び先の間隔の件数で、0 以下なら語ごとに √n。
// Long が正なら Long 個おきの飛び先を上の段に持つ 2 段にし、Linear なら飛び先を 1 つずつたどる。どちらも無ければ飛び先を二分探索で引く。
type SkipConfig struct {
	Interval int
	Long     int
	Linear   bool
}

// BuildSkipIndex は語ごとに、Interval 件おきの文書番号とファイルの中の位置をメモリ上の表に作る。以後の And は飛び先を使って読み進める。
// 間隔が 2 未満になる短い語には作らない。作った表の大きさ（バイト）を返す。
func (pf *PostingsFile) BuildSkipIndex(cfg SkipConfig) (int64, error) {
	pf.skips = make(map[int64]*skipList)
	var size int64
	for _, s := range pf.table {
		iv := cfg.Interval
		if iv <= 0 {
			iv = sqrtInterval(s.count)
		}
		if iv < 2 || s.count <= iv {
			continue
		}
		raw := make([]byte, s.size)
		if _, err := pf.f.ReadAt(raw, s.off); err != nil {
			return 0, err
		}
		sl := &skipList{long: cfg.Long, linear: cfg.Linear}
		var (
			doc int32
			off int64
		)
		for i := range s.count {
			if i > 0 && i%iv == 0 {
				sl.entries = append(sl.entries, skipEntry{doc: doc, idx: int32(i), off: off})
			}
			delta, n := binary.Uvarint(raw[off:])
			if n <= 0 {
				return 0, errors.New("decode doc")
			}
			_, m := binary.Uvarint(raw[off+int64(n):])
			if m <= 0 {
				return 0, errors.New("decode tf")
			}
			off += int64(n + m)
			doc += int32(delta)
		}
		pf.skips[s.off] = sl
		size += int64(len(sl.entries)) * int64(unsafe.Sizeof(skipEntry{}))
	}
	return size, nil
}

// And はすべての語を含む文書の番号を昇順で返す。短い語の出現記録を 1 件ずつ読み、そのたびに他の語をその文書番号まで読み進めるので、出現記録をまとめてメモリに載せない。
func (pf *PostingsFile) And(terms ...string) ([]int, error) {
	return pf.and(terms, &cursorStats{})
}

func (pf *PostingsFile) and(terms []string, stats *cursorStats) ([]int, error) {
	spans, ok := pf.spans(terms)
	if !ok {
		return []int{}, nil
	}
	cs := make([]*postingsCursor, len(spans))
	for i, s := range spans {
		cs[i] = pf.cursorWith(s, pf.skips[s.off])
	}
	defer func() {
		for _, c := range cs {
			stats.add(c.stats())
		}
	}()
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
