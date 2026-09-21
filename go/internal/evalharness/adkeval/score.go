package adkeval

import (
	"encoding/json"
	"reflect"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// MatchType はツール呼び出しの照合の方法。ADK の ToolTrajectoryCriterion.MatchType と同じ意味。
type MatchType int

const (
	// Exact は呼び出しの数、順番、名前、引数がすべて一致する。ADK の既定。
	Exact MatchType = iota
	// InOrder は期待する呼び出しがこの順で現れる。余分な呼び出しは許す。
	InOrder
	// AnyOrder は期待する呼び出しがすべて現れる。順番と余分な呼び出しは問わない。
	AnyOrder
)

// TrajectoryScore はターンごとに一致なら 1、不一致なら 0 を付け、その平均を返す。
func TrajectoryScore(actual, expected []Invocation, mode MatchType) float64 {
	if len(expected) == 0 {
		return 1
	}
	total := 0.0
	for i, want := range expected {
		var got []ToolUse
		if i < len(actual) {
			got = actual[i].IntermediateData.ToolUses
		}
		if trajectoryMatches(got, want.IntermediateData.ToolUses, mode) {
			total++
		}
	}
	return total / float64(len(expected))
}

func trajectoryMatches(got, want []ToolUse, mode MatchType) bool {
	switch mode {
	case Exact:
		if len(got) != len(want) {
			return false
		}
		for i := range want {
			if !sameCall(got[i], want[i]) {
				return false
			}
		}
		return true
	case InOrder:
		j := 0
		for _, g := range got {
			if j < len(want) && sameCall(g, want[j]) {
				j++
			}
		}
		return j == len(want)
	case AnyOrder:
		used := make([]bool, len(got))
	next:
		for _, w := range want {
			for i, g := range got {
				if !used[i] && sameCall(g, w) {
					used[i] = true
					continue next
				}
			}
			return false
		}
		return true
	}
	return false
}

// sameCall は名前と引数が同じかを見る。引数は JSON を通して比べ、2 と 2.0 を同じに扱う。
func sameCall(a, b ToolUse) bool {
	return a.Name == b.Name && reflect.DeepEqual(normalizeArgs(a.Args), normalizeArgs(b.Args))
}

func normalizeArgs(args map[string]any) any {
	if len(args) == 0 {
		return map[string]any{}
	}
	raw, err := json.Marshal(args)
	if err != nil {
		return args
	}
	var out any
	if err := json.Unmarshal(raw, &out); err != nil {
		return args
	}
	return out
}

// BigramF1 は文字の 2-gram の重なりの F1 を返す。
// NFKC で全角と半角をそろえ、文字と数字以外（Python の \W と _）を捨ててから比べる。
// Python の samples/evaluation/ja_response_match.py と同じ値になる。
func BigramF1(candidate, reference string) float64 {
	got, want := bigrams(candidate), bigrams(reference)
	overlap, gotTotal, wantTotal := 0, 0, 0
	for g, n := range got {
		overlap += min(n, want[g])
		gotTotal += n
	}
	for _, n := range want {
		wantTotal += n
	}
	if overlap == 0 {
		return 0
	}
	precision := float64(overlap) / float64(gotTotal)
	recall := float64(overlap) / float64(wantTotal)
	return 2 * precision * recall / (precision + recall)
}

func bigrams(text string) map[string]int {
	var kept []rune
	for _, r := range strings.ToLower(norm.NFKC.String(text)) {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			kept = append(kept, r)
		}
	}
	out := map[string]int{}
	for i := 0; i+1 < len(kept); i++ {
		out[string(kept[i:i+2])]++
	}
	return out
}

// ResponseScore はターンごとの BigramF1 の平均を返す。期待する応答の無いターンは数えない。
func ResponseScore(actual, expected []Invocation) (float64, bool) {
	total, n := 0.0, 0
	for i, want := range expected {
		if want.FinalResponse == nil || want.FinalResponse.Text() == "" {
			continue
		}
		got := ""
		if i < len(actual) {
			got = actual[i].FinalResponse.Text()
		}
		total += BigramF1(got, want.FinalResponse.Text())
		n++
	}
	if n == 0 {
		return 0, false
	}
	return total / float64(n), true
}
