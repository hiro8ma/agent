package morph_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/morph"
)

// 教材の品詞 名詞 を、結果の表示に合わせて 固有名詞 に置き換えた遷移のコスト。
func textbookTransitions() morph.Transitions {
	const pn, v, p = "固有名詞", "動詞", "助詞"
	return morph.Transitions{
		{From: pn, To: pn}: 1, {From: pn, To: v}: 1, {From: pn, To: p}: 1,
		{From: v, To: pn}: 2, {From: v, To: v}: 5, {From: v, To: p}: 1,
		{From: p, To: pn}: 1, {From: p, To: v}: 1, {From: p, To: p}: 3,
		{From: "名詞", To: "名詞"}: 1, {From: "名詞", To: p}: 1, {From: p, To: "名詞"}: 1,
		{From: morph.BOS, To: pn}: 0, {From: morph.BOS, To: "名詞"}: 0,
		{From: pn, To: morph.EOS}: 0, {From: v, To: morph.EOS}: 0, {From: "名詞", To: morph.EOS}: 0,
	}
}

func textbookDict() morph.Dictionary {
	return morph.Dictionary{
		"とうきょう": {{Label: "固有名詞", Cost: 2}},
		"と":     {{Label: "名詞", Cost: 2}, {Label: "助詞", Cost: 1}},
		"なら":    {{Label: "動詞", Cost: 2}, {Label: "固有名詞", Cost: 1}},
	}
}

func format(nodes []morph.Node) string {
	parts := make([]string, len(nodes))
	for i, n := range nodes {
		parts[i] = n.Surface + "/" + n.Label
	}
	return strings.Join(parts, " ")
}

func TestAnalyze(t *testing.T) {
	t.Parallel()
	withSplits := textbookDict()
	withSplits["とう"] = []morph.Entry{{Label: "名詞", Cost: 3}}
	withSplits["きょう"] = []morph.Entry{{Label: "名詞", Cost: 3}}
	withSplits["うと"] = []morph.Entry{{Label: "名詞", Cost: 3}}
	withSplits["とな"] = []morph.Entry{{Label: "名詞", Cost: 4}}
	withoutTokyo := textbookDict()
	delete(withoutTokyo, "とうきょう")
	withoutTokyo["とう"] = []morph.Entry{{Label: "名詞", Cost: 3}}
	withoutTokyo["きょう"] = []morph.Entry{{Label: "名詞", Cost: 3}}

	testCases := map[string]struct {
		dict morph.Dictionary
		text string
		want string
		cost float64
	}{
		"教材の辞書で区切りと品詞が決まる": {
			dict: textbookDict(), text: "とうきょうとなら",
			want: "とうきょう/固有名詞 と/助詞 なら/固有名詞", cost: 6,
		},
		"とう / きょう / うと / となを足してもコストでとうきょうが選ばれる": {
			dict: withSplits, text: "とうきょうとなら",
			want: "とうきょう/固有名詞 と/助詞 なら/固有名詞", cost: 6,
		},
		"とうきょうが辞書に無ければとう / きょうに分かれる": {
			dict: withoutTokyo, text: "とうきょうとなら",
			want: "とう/名詞 きょう/名詞 と/助詞 なら/固有名詞", cost: 11,
		},
		"辞書にない文字は 1 文字ずつ未知語にする": {
			dict: textbookDict(), text: "とうきょうとぱりとなら",
			want: "とうきょう/固有名詞 と/助詞 ぱ/未知語 り/未知語 と/助詞 なら/固有名詞", cost: 42,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			a := morph.Analyzer{Dict: tc.dict, Trans: textbookTransitions(), DefaultTrans: 5, UnknownLabel: "未知語", UnknownCost: 10}
			nodes, cost := a.Analyze(tc.text)
			if got := format(nodes); got != tc.want || cost != tc.cost {
				t.Fatalf("got %q (%v), want %q (%v)", got, cost, tc.want, tc.cost)
			}
			var sb strings.Builder
			for i, n := range nodes {
				if i > 0 && nodes[i-1].End != n.Begin {
					t.Fatalf("nodes %d and %d are not adjacent", i-1, i)
				}
				sb.WriteString(n.Surface)
			}
			if sb.String() != tc.text {
				t.Fatalf("surfaces = %q, want %q", sb.String(), tc.text)
			}
		})
	}
}

func TestLatticeNodes(t *testing.T) {
	t.Parallel()
	a := morph.Analyzer{Dict: textbookDict(), UnknownLabel: "未知語", UnknownCost: 10}
	l := a.Lattice("とうきょうとなら")
	var got []string
	for _, n := range l.Nodes {
		got = append(got, n.Surface+"/"+n.Label)
	}
	slices.Sort(got)
	want := []string{
		"と/助詞", "と/名詞", "とうきょう/固有名詞", "う/未知語", "き/未知語", "ょ/未知語", "う/未知語",
		"と/助詞", "と/名詞", "なら/動詞", "なら/固有名詞", "ら/未知語",
	}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("nodes = %v, want %v", got, want)
	}
}
