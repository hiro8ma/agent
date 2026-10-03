package search

import (
	"bufio"
	"cmp"
	"container/heap"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
)

// ExternalStats の Runs は書き出した一時的な並びの数、MaxBuffered は 1 度にメモリに持った出現記録の最大件数、RunBytes は一時ファイルの大きさ。
type ExternalStats struct {
	Runs        int
	MaxBuffered int
	RunBytes    int64
}

type runEntry struct {
	term, doc, tf int32
}

type runSpan struct {
	off, size int64
}

// BuildPostingsFileExternal は、メモリに載らない文書の集まりを想定して、外部ソートで WritePostingsFile と同じ形のファイルを path に作る。
// 出現記録を budget 件までメモリにため、語の番号の順に並べて dir の一時ファイルに 1 つの並び（run）として書き出し、最後にすべての並びをヒープで併合する。
// 語の番号は語が初めて現れた順で、語彙の表（語と番号）だけはメモリに持つ。a が nil なら NewAnalyzer を使う。文書拡張や doc2query の語は足さない。
func BuildPostingsFileExternal(docs []Doc, a *Analyzer, budget int, dir, path string) (*PostingsFile, ExternalStats, error) {
	if a == nil {
		a = NewAnalyzer()
	}
	budget = max(budget, 1)
	var stats ExternalStats
	tmp, err := os.CreateTemp(dir, "runs-*.bin")
	if err != nil {
		return nil, stats, err
	}
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
	}()
	w := bufio.NewWriter(tmp)
	var (
		runs  []runSpan
		off   int64
		words []string
		buf   []runEntry
		enc   []byte
	)
	vocab := make(map[string]int32)
	counts := make(map[int32]int32)
	flush := func() error {
		slices.SortFunc(buf, func(x, y runEntry) int { return cmp.Or(cmp.Compare(x.term, y.term), cmp.Compare(x.doc, y.doc)) })
		enc = encodeRun(enc[:0], buf)
		if _, err := w.Write(enc); err != nil {
			return err
		}
		runs = append(runs, runSpan{off: off, size: int64(len(enc))})
		off += int64(len(enc))
		buf = buf[:0]
		return nil
	}
	for id, d := range docs {
		clear(counts)
		for f := range numFields {
			for _, t := range a.analyze(d.field(f)) {
				tid, ok := vocab[t.term]
				if !ok {
					tid = int32(len(words))
					vocab[t.term] = tid
					words = append(words, t.term)
				}
				counts[tid]++
			}
		}
		if len(buf) > 0 && len(buf)+len(counts) > budget {
			if err := flush(); err != nil {
				return nil, stats, err
			}
		}
		for tid, n := range counts {
			buf = append(buf, runEntry{term: tid, doc: int32(id), tf: n})
		}
		stats.MaxBuffered = max(stats.MaxBuffered, len(buf))
	}
	if len(buf) > 0 {
		if err := flush(); err != nil {
			return nil, stats, err
		}
	}
	if err := w.Flush(); err != nil {
		return nil, stats, err
	}
	stats.Runs, stats.RunBytes = len(runs), off
	pf, err := mergeRuns(tmp, runs, words, path)
	return pf, stats, err
}

// encodeRun は語ごとに（語の番号, 件数）に続けて（文書番号の差分, 出現回数）を varint で並べる。差分は語ごとに 0 から数える。
func encodeRun(enc []byte, entries []runEntry) []byte {
	for i := 0; i < len(entries); {
		j := i
		for j < len(entries) && entries[j].term == entries[i].term {
			j++
		}
		enc = binary.AppendUvarint(enc, uint64(entries[i].term))
		enc = binary.AppendUvarint(enc, uint64(j-i))
		var prev int32
		for _, e := range entries[i:j] {
			enc = binary.AppendUvarint(enc, uint64(e.doc-prev))
			enc = binary.AppendUvarint(enc, uint64(e.tf))
			prev = e.doc
		}
		i = j
	}
	return enc
}

// runReader は 1 つの並びを先頭から順に読む。term と left は次に読む語の番号と、その語の残りの件数。
type runReader struct {
	r    *bufio.Reader
	idx  int
	term int32
	left int
	doc  int32
}

func (rr *runReader) header() (bool, error) {
	term, err := binary.ReadUvarint(rr.r)
	if errors.Is(err, io.EOF) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read run term: %w", err)
	}
	n, err := binary.ReadUvarint(rr.r)
	if err != nil {
		return false, fmt.Errorf("read run count: %w", err)
	}
	rr.term, rr.left, rr.doc = int32(term), int(n), 0
	return true, nil
}

func (rr *runReader) next() (doc, tf int32, err error) {
	delta, err := binary.ReadUvarint(rr.r)
	if err != nil {
		return 0, 0, fmt.Errorf("read run doc: %w", err)
	}
	t, err := binary.ReadUvarint(rr.r)
	if err != nil {
		return 0, 0, fmt.Errorf("read run tf: %w", err)
	}
	rr.doc += int32(delta)
	rr.left--
	return rr.doc, int32(t), nil
}

// runHeap は語の番号が小さい順に並び、同じ語なら並びの番号が小さい順に取り出す。
// 文書は番号の順に読んで並びに分けたので、同じ語を並びの番号の順につなげば文書番号の昇順になる。
type runHeap []*runReader

func (h runHeap) Len() int { return len(h) }
func (h runHeap) Less(i, j int) bool {
	return cmp.Or(cmp.Compare(h[i].term, h[j].term), cmp.Compare(h[i].idx, h[j].idx)) < 0
}
func (h runHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *runHeap) Push(x any)   { *h = append(*h, x.(*runReader)) }
func (h *runHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// mergeRuns はすべての並びを先頭から順に読み、語ごとに文書番号の差分を付け直して path に書く。どの並びも読み戻さない。
func mergeRuns(tmp *os.File, runs []runSpan, words []string, path string) (*PostingsFile, error) {
	h := make(runHeap, 0, len(runs))
	for i, s := range runs {
		rr := &runReader{r: bufio.NewReaderSize(io.NewSectionReader(tmp, s.off, s.size), 4096), idx: i}
		ok, err := rr.header()
		if err != nil {
			return nil, err
		}
		if ok {
			h = append(h, rr)
		}
	}
	heap.Init(&h)
	path = filepath.Clean(path)
	out, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriter(out)
	table := make(map[string]postingsSpan, len(words))
	var (
		off     int64
		scratch [2 * binary.MaxVarintLen64]byte
	)
	for h.Len() > 0 {
		term, start, count := h[0].term, off, 0
		var prev int32
		for h.Len() > 0 && h[0].term == term {
			rr := heap.Pop(&h).(*runReader)
			for rr.left > 0 {
				doc, tf, err := rr.next()
				if err != nil {
					return nil, errors.Join(err, out.Close())
				}
				b := binary.AppendUvarint(scratch[:0], uint64(doc-prev))
				b = binary.AppendUvarint(b, uint64(tf))
				if _, err := w.Write(b); err != nil {
					return nil, errors.Join(err, out.Close())
				}
				off += int64(len(b))
				prev = doc
				count++
			}
			ok, err := rr.header()
			if err != nil {
				return nil, errors.Join(err, out.Close())
			}
			if ok {
				heap.Push(&h, rr)
			}
		}
		table[words[term]] = postingsSpan{off: start, size: off - start, count: count}
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
