package dedup

import (
	"hash/fnv"
	"math"
)

// Bloom はビットの長さ N、ハッシュの数 k のブルームフィルタ。入れた要素は必ず「おそらく含まれる」になり、入れていない要素も偽陽性率の割合で同じ答えになる。
type Bloom struct {
	words []uint64
	bits  uint64
	k     int
}

func NewBloom(bits uint64, k int) *Bloom {
	bits = max(bits, 1)
	return &Bloom{words: make([]uint64, (bits+63)/64), bits: bits, k: max(k, 1)}
}

// BloomParams は要素数 m と目標の偽陽性率 p から、N = ⌈−m ln p / (ln 2)²⌉ と、その N で偽陽性率を最小にする k を返す。
func BloomParams(m int, p float64) (bits uint64, k int) {
	bits = uint64(math.Ceil(-float64(m) * math.Log(p) / (math.Ln2 * math.Ln2)))
	return bits, OptimalK(bits, m)
}

// OptimalK は (N/m) ln 2 を四捨五入した k（1以上）を返す。
func OptimalK(bits uint64, m int) int {
	return max(1, int(math.Round(float64(bits)/float64(m)*math.Ln2)))
}

// FalsePositiveRate は m 個を入れたときの偽陽性率の近似 (1 − e^(−km/N))^k を返す。
func FalsePositiveRate(bits uint64, k, m int) float64 {
	return math.Pow(-math.Expm1(-float64(k)*float64(m)/float64(bits)), float64(k))
}

// Key はバイト列を Add と Contains に渡す64ビットの値にする。
func Key(b []byte) uint64 {
	h := fnv.New64a()
	h.Write(b)
	return h.Sum64()
}

// Add は x を入れる。
func (b *Bloom) Add(x uint64) {
	h1, h2 := b.hashes(x)
	for i := range uint64(b.k) {
		p := mix64(h1+i*h2) % b.bits
		b.words[p/64] |= 1 << (p % 64)
	}
}

// Contains は x のビットがすべて立っているか（おそらく含まれるか）を返す。
func (b *Bloom) Contains(x uint64) bool {
	h1, h2 := b.hashes(x)
	for i := range uint64(b.k) {
		p := mix64(h1+i*h2) % b.bits
		if b.words[p/64]&(1<<(p%64)) == 0 {
			return false
		}
	}
	return true
}

// hashes は k 個の位置を h1 + i × h2 を混ぜ直して作るための2つの値を返す。混ぜ直さないと位置が N を法にほぼ等間隔に並び、N=719、k=10、50件で偽陽性率が式の1.4倍になった。
func (b *Bloom) hashes(x uint64) (uint64, uint64) {
	return mix64(x), mix64(x+0x9e3779b97f4a7c15) | 1
}

// Bytes はビットの列に使うバイト数を返す。
func (b *Bloom) Bytes() int { return len(b.words) * 8 }

// mix64 は splitmix64 の仕上げの混ぜ方で、入力の1ビットの違いを全ビットに広げる。
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}
