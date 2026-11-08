package dedup

import (
	"math"
	"strings"
)

// CharShingles は text の連続する k 文字（rune）を前から並べる。日本語は単語の区切りが無いので、こちらを使う。k 文字に満たない空でない text は、それ自体を1つの断片にする。
func CharShingles(text string, k int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	if len(runes) < k {
		return []string{text}
	}
	out := make([]string, 0, len(runes)-k+1)
	for i := 0; i+k <= len(runes); i++ {
		out = append(out, string(runes[i:i+k]))
	}
	return out
}

// WordShingles は空白で区切った単語の連続する k 個を、空白でつないで前から並べる。
func WordShingles(text string, k int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	if len(words) < k {
		return []string{strings.Join(words, " ")}
	}
	out := make([]string, 0, len(words)-k+1)
	for i := 0; i+k <= len(words); i++ {
		out = append(out, strings.Join(words[i:i+k], " "))
	}
	return out
}

// Set は断片の集合。
type Set map[string]struct{}

func NewSet(items []string) Set {
	s := make(Set, len(items))
	for _, it := range items {
		s[it] = struct{}{}
	}
	return s
}

// Jaccard は |A ∩ B| / |A ∪ B| を返す。両方が空なら1。
func Jaccard(a, b Set) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	if len(a) > len(b) {
		a, b = b, a
	}
	inter := 0
	for x := range a {
		if _, ok := b[x]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

// MinHasher は K 個のハッシュの関数で、集合ごとに各関数の最小値を並べた署名を作る。2つの署名で同じ位置の値が一致する確率はジャッカード係数に等しい。
type MinHasher struct {
	seeds []uint64
}

func NewMinHasher(k int, seed uint64) *MinHasher {
	seeds := make([]uint64, k)
	s := seed
	for i := range seeds {
		s += 0x9e3779b97f4a7c15
		seeds[i] = mix64(s)
	}
	return &MinHasher{seeds: seeds}
}

// Signature は集合の署名を返す。空の集合はすべての位置が最大値になる。
func (m *MinHasher) Signature(set Set) []uint64 {
	sig := make([]uint64, len(m.seeds))
	for i := range sig {
		sig[i] = math.MaxUint64
	}
	for s := range set {
		x := Key([]byte(s))
		for i, seed := range m.seeds {
			if v := mix64(x ^ seed); v < sig[i] {
				sig[i] = v
			}
		}
	}
	return sig
}

// EstimateJaccard は2つの署名で値が一致する位置の割合を返す。
func EstimateJaccard(a, b []uint64) float64 {
	same := 0
	for i := range a {
		if a[i] == b[i] {
			same++
		}
	}
	return float64(same) / float64(len(a))
}
