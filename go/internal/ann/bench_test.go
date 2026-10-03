package ann

import (
	"sync"
	"testing"
)

var (
	benchOnce sync.Once
	benchPQ   *PQIndex
	benchQ    []float32
	sinkNs    []Neighbor
)

func pqSetup(b *testing.B) {
	b.Helper()
	benchOnce.Do(func() {
		data := RandomNormal(100_000, 128, 1)
		pq, err := TrainPQ(data, PQConfig{M: 8, K: 256, Seed: 1})
		if err != nil {
			b.Fatal(err)
		}
		benchPQ = NewPQIndex(pq, data)
		benchQ = RandomNormal(1, 128, 2).Row(0)
	})
}

// 距離表を引く ADC と、符号ごとに中心との距離を計算し直す版を比べる。表は 1 クエリで M×K 回の距離計算だけで済む。
func BenchmarkPQSearch(b *testing.B) {
	b.Run("adc-table", func(b *testing.B) {
		pqSetup(b)
		for b.Loop() {
			sinkNs = benchPQ.Search(benchQ, 20)
		}
	})
	b.Run("no-table", func(b *testing.B) {
		pqSetup(b)
		for b.Loop() {
			sinkNs = pqSearchDirect(benchPQ, benchQ, 20)
		}
	})
}
