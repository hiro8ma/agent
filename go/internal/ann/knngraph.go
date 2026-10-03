package ann

// NewKNNGraph は各点を正確な k 近傍だけと結んだ 1 層のグラフを作る。近い点同士の短い辺しか無いので、
// 遠くへ移るには短い辺を何度もたどる必要があり、塊の間がつながらないこともある。比較用で、全点の全探索を伴う。
func NewKNNGraph(data Matrix, k int) *HNSW {
	h := &HNSW{
		data: data, m: k, m0: k, efC: k, efSearch: k, simple: true,
		levels: make([]uint8, data.N), upper: make([][]int32, data.N),
		layer0: make([]int32, data.N*(k+1)),
	}
	h.pool.New = func() any { return &visited{marks: make([]uint32, data.N)} }
	e := NewExact(data)
	const batch = 64
	for lo := 0; lo < data.N; lo += batch {
		hi := min(lo+batch, data.N)
		qs := make([][]float32, 0, hi-lo)
		for i := lo; i < hi; i++ {
			qs = append(qs, data.Row(i))
		}
		res := e.SearchBatchParallel(qs, k+1)
		for qi, ns := range res {
			id := int32(lo + qi)
			out := make([]Neighbor, 0, k)
			for _, n := range ns {
				if n.ID != id && len(out) < k {
					out = append(out, n)
				}
			}
			h.setNeighbors(id, 0, out)
		}
	}
	return h
}
