package dedup

import "math"

// LSH は署名の先頭 Bands × Rows 個を、Rows 個ずつの帯に分ける。どれか1本の帯の値がすべて一致する2つの文書を候補の組にする。
// Rows を1にすると、署名の次元ごとに値から文書を引く転置インデックスで、どれか1次元で一致すれば候補にする方式と同じになる。
type LSH struct {
	Bands int
	Rows  int
}

// CandidateProbability はジャッカード係数 j の2つの文書が候補になる確率 1 − (1 − j^r)^b を返す。
func (l LSH) CandidateProbability(j float64) float64 {
	return -math.Expm1(float64(l.Bands) * math.Log1p(-math.Pow(j, float64(l.Rows))))
}

// Match は2つの署名がどれか1本の帯で一致するかを返す。
func (l LSH) Match(a, b []uint64) bool {
	for band := range l.Bands {
		same := true
		for i := band * l.Rows; i < (band+1)*l.Rows; i++ {
			if a[i] != b[i] {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

// Candidates は、帯ごとに値の組から文書を引く表を作り、どれかの表で同じ所に入った組 (i, j)（i < j）を1回ずつ fn に渡す。
func (l LSH) Candidates(sigs [][]uint64, fn func(i, j int)) {
	keys := make([][]uint64, len(sigs))
	tables := make([]map[uint64][]int, l.Bands)
	for band := range tables {
		tables[band] = map[uint64][]int{}
	}
	for d, sig := range sigs {
		keys[d] = make([]uint64, l.Bands)
		for band := range l.Bands {
			k := l.bandKey(sig, band)
			keys[d][band] = k
			tables[band][k] = append(tables[band][k], d)
		}
	}
	seen := make([]int, len(sigs))
	for i := range sigs {
		for band := range l.Bands {
			for _, j := range tables[band][keys[i][band]] {
				if j > i && seen[j] != i+1 {
					seen[j] = i + 1
					fn(i, j)
				}
			}
		}
	}
}

// bandKey は帯の Rows 個の値を1つの64ビットの値に畳む。異なる値の組が同じ値になる確率は 2^−64 ほどで、候補の数に表れない。
func (l LSH) bandKey(sig []uint64, band int) uint64 {
	if l.Rows == 1 {
		return sig[band]
	}
	var k uint64
	for i := band * l.Rows; i < (band+1)*l.Rows; i++ {
		k = mix64(k ^ sig[i])
	}
	return k
}
