package gpt

import "sort"

// CharTokenizer は文字レベルのトークナイザ。
// コーパスに出現した文字（rune）を語彙として ID を割り当てる。
type CharTokenizer struct {
	stoi map[rune]int
	itos []rune
}

func NewCharTokenizer(text string) *CharTokenizer {
	seen := map[rune]bool{}
	for _, r := range text {
		seen[r] = true
	}
	itos := make([]rune, 0, len(seen))
	for r := range seen {
		itos = append(itos, r)
	}
	sort.Slice(itos, func(i, j int) bool { return itos[i] < itos[j] })
	stoi := make(map[rune]int, len(itos))
	for i, r := range itos {
		stoi[r] = i
	}
	return &CharTokenizer{stoi: stoi, itos: itos}
}

func (t *CharTokenizer) VocabSize() int { return len(t.itos) }

// Encode は未知文字をスキップする（生成プロンプトに語彙外文字が来た場合の安全側動作）。
func (t *CharTokenizer) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		if id, ok := t.stoi[r]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func (t *CharTokenizer) Decode(ids []int) string {
	rs := make([]rune, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < len(t.itos) {
			rs = append(rs, t.itos[id])
		}
	}
	return string(rs)
}
