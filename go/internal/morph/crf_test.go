package morph_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/morph"
)

func sentence(s string) morph.Sentence {
	var out morph.Sentence
	for f := range strings.FieldsSeq(s) {
		tok, label, _ := strings.Cut(f, "/")
		out.Tokens = append(out.Tokens, tok)
		out.Labels = append(out.Labels, label)
	}
	return out
}

var trainData = []morph.Sentence{
	sentence("とうきょう/固有名詞 と/助詞 なら/固有名詞"),
	sentence("これ/代名詞 は/助詞 ペン/名詞 です/助動詞"),
	sentence("カレーライス/名詞 は/助詞 おいしい/形容詞"),
	sentence("ねこ/名詞 と/助詞 いぬ/名詞"),
	sentence("と/名詞 は/助詞 ひらがな/名詞 です/助動詞"),
	sentence("おおさか/固有名詞 と/助詞 きょうと/固有名詞"),
	sentence("これ/代名詞 は/助詞 ねこ/名詞 です/助動詞"),
	sentence("カレーライス/名詞 は/助詞 からい/形容詞"),
}

var heldOut = []morph.Sentence{
	sentence("ペン/名詞 と/助詞 カレーライス/名詞"),
	sentence("これ/代名詞 は/助詞 ほん/名詞 です/助動詞"),
	sentence("と/名詞 は/助詞 もじ/名詞 です/助動詞"),
	sentence("なら/固有名詞 と/助詞 おおさか/固有名詞"),
}

func bruteLogZ(p morph.Problem) float64 {
	var costs []float64
	morph.EachPath(len(p.Emit), len(p.Trans), func(path []int) {
		costs = append(costs, -morph.PathCost(p, path))
	})
	m := slices.Max(costs)
	s := 0.0
	for _, c := range costs {
		s += math.Exp(c - m)
	}
	return m + math.Log(s)
}

func TestLogPartitionMatchesBruteForce(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		n, k     int
		withEnds bool
	}{
		"1 語 3 品詞":         {n: 1, k: 3},
		"3 語 3 品詞":         {n: 3, k: 3},
		"6 語 4 品詞で文頭と文末つき": {n: 6, k: 4, withEnds: true},
		"8 語 2 品詞で文頭と文末つき": {n: 8, k: 2, withEnds: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := rand.New(rand.NewPCG(uint64(tc.n), 7))
			for range 20 {
				p := randomProblem(r, tc.n, tc.k, tc.withEnds)
				got, want := morph.LogPartition(p), bruteLogZ(p)
				if math.Abs(got-want) > 1e-9 {
					t.Fatalf("forward = %v, brute force = %v", got, want)
				}
				node, edge, _ := morph.Marginals(p)
				for pos := range node {
					if s := floatsSum(node[pos]); math.Abs(s-1) > 1e-9 {
						t.Fatalf("node marginals at %d sum to %v", pos, s)
					}
					if pos == 0 {
						continue
					}
					for j := range tc.k {
						col := 0.0
						for i := range tc.k {
							col += edge[pos][i][j]
						}
						if math.Abs(col-node[pos][j]) > 1e-9 {
							t.Fatalf("edge marginals into (%d,%d) = %v, node = %v", pos, j, col, node[pos][j])
						}
					}
				}
			}
		})
	}
}

func TestLogPartitionTextbook(t *testing.T) {
	t.Parallel()
	p := textbookProblem()
	if got, want := morph.LogPartition(p), bruteLogZ(p); math.Abs(got-want) > 1e-12 {
		t.Fatalf("forward = %v, brute force = %v", got, want)
	}
}

func floatsSum(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}

func TestNLLGradCheck(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		l2   float64
		seed uint64
	}{
		"正則化なしでランダムなコスト": {l2: 0, seed: 1},
		"L2 つきでランダムなコスト": {l2: 0.1, seed: 2},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			m := morph.NewCRF(trainData)
			r := rand.New(rand.NewPCG(tc.seed, 3))
			for i := range m.Params {
				m.Params[i] = r.NormFloat64()
			}
			_, grad := m.NLL(trainData, tc.l2)
			const h = 1e-5
			for i := range m.Params {
				orig := m.Params[i]
				m.Params[i] = orig + h
				plus, _ := m.NLL(trainData, tc.l2)
				m.Params[i] = orig - h
				minus, _ := m.NLL(trainData, tc.l2)
				m.Params[i] = orig
				numeric := (plus - minus) / (2 * h)
				if math.Abs(numeric-grad[i]) > 1e-6*math.Max(1, math.Abs(numeric)) {
					t.Fatalf("param %d: analytic = %.10f, numeric = %.10f", i, grad[i], numeric)
				}
			}
		})
	}
}

func accuracy(m *morph.CRF, data []morph.Sentence) (tokens, sentences float64) {
	var hit, total, sentHit int
	for _, s := range data {
		got := m.Tag(s.Tokens)
		for i := range got {
			if got[i] == s.Labels[i] {
				hit++
			}
		}
		total += len(got)
		if slices.Equal(got, s.Labels) {
			sentHit++
		}
	}
	return float64(hit) / float64(total), float64(sentHit) / float64(len(data))
}

func TestCRFTrain(t *testing.T) {
	t.Parallel()
	m := morph.NewCRF(trainData)
	beforeTok, beforeSent := accuracy(m, trainData)
	beforeHeldTok, beforeHeldSent := accuracy(m, heldOut)
	losses := m.Train(trainData, morph.TrainConfig{Steps: 300, LR: 0.1, L2: 0.01})
	afterTok, afterSent := accuracy(m, trainData)
	afterHeldTok, afterHeldSent := accuracy(m, heldOut)

	if losses[len(losses)-1] > losses[0]*0.2 {
		t.Fatalf("loss did not decrease: %v -> %v", losses[0], losses[len(losses)-1])
	}
	if afterSent != 1 {
		t.Fatalf("training sentence accuracy = %v, want 1", afterSent)
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "\nloss %.3f -> %.3f\n", losses[0], losses[len(losses)-1])
	fmt.Fprintf(&sb, "| データ | 学習前 語 | 学習前 文 | 学習後 語 | 学習後 文 |\n")
	fmt.Fprintf(&sb, "| 学習 | %.2f | %.2f | %.2f | %.2f |\n", beforeTok, beforeSent, afterTok, afterSent)
	fmt.Fprintf(&sb, "| 未知 | %.2f | %.2f | %.2f | %.2f |\n", beforeHeldTok, beforeHeldSent, afterHeldTok, afterHeldSent)
	for _, s := range heldOut {
		fmt.Fprintf(&sb, "%v -> %v (want %v)\n", s.Tokens, m.Tag(s.Tokens), s.Labels)
	}
	fmt.Fprintf(&sb, "| 割り当て | コスト |\n")
	for _, e := range [][2]string{{"と", "助詞"}, {"と", "名詞"}, {"と", "固有名詞"}, {"なら", "固有名詞"}, {"とうきょう", "固有名詞"}, {"は", "助詞"}} {
		fmt.Fprintf(&sb, "| %s/%s | %.3f |\n", e[0], e[1], m.EmitCost(e[0], e[1]))
	}
	fmt.Fprintf(&sb, "| 遷移 | コスト |\n")
	for _, e := range [][2]string{
		{"固有名詞", "助詞"},
		{"助詞", "固有名詞"},
		{"名詞", "助詞"},
		{"助詞", "名詞"},
		{"固有名詞", "固有名詞"},
		{"助詞", "助詞"},
		{morph.BOS, "名詞"},
		{morph.BOS, "助詞"},
		{"助詞", morph.EOS},
	} {
		fmt.Fprintf(&sb, "| %s→%s | %.3f |\n", e[0], e[1], m.TransCost(e[0], e[1]))
	}
	t.Log(sb.String())
}
