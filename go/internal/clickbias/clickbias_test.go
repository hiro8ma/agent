package clickbias_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/clickbias"
)

const (
	items = 50
	slots = 10
)

// truth は商品の本当の良さ。0.05〜0.6 に散らす。
func truth(seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 1))
	rel := make([]float64, items)
	for i := range rel {
		rel[i] = 0.05 + 0.55*r.Float64()
	}
	return rel
}

// productionPolicy は本番の推薦にあたる。本当の良さに毎回ノイズを足して上位 slots 件を並べるので、良い商品ほど上に出やすいが、順番は揺れる。
func productionPolicy(rel []float64, noise float64) func(*rand.Rand) []int {
	return func(r *rand.Rand) []int {
		score := make([]float64, len(rel))
		for i := range score {
			score[i] = rel[i] + noise*r.NormFloat64()
		}
		return clickbias.Ranking(score)[:slots]
	}
}

// boostedPolicy は、本番の推薦が良さの低い商品を boosted だけ上に押し上げる場合（広告や在庫の都合）にあたる。
func boostedPolicy(rel []float64, noise float64, boosted []int) func(*rand.Rand) []int {
	return func(r *rand.Rand) []int {
		score := make([]float64, len(rel))
		for i := range score {
			score[i] = rel[i] + noise*r.NormFloat64()
		}
		for _, id := range boosted {
			score[id] += 1
		}
		return clickbias.Ranking(score)[:slots]
	}
}

// middleItems は本当の良さが中くらい（上位 20〜22 位あたり）の商品を n 件返す。上位には入らないが、上に出せばクリックは集まる。
func middleItems(rel []float64, n int) []int {
	return clickbias.Ranking(rel)[20 : 20+n]
}

// randomPolicy は上位候補を毎回ランダムに並べ替える。位置の見られやすさを推定するための少量の実験の枠にあたる。
func randomPolicy(rel []float64, noise float64) func(*rand.Rand) []int {
	prod := productionPolicy(rel, noise)
	return func(r *rand.Rand) []int {
		slate := prod(r)
		r.Shuffle(len(slate), func(i, j int) { slate[i], slate[j] = slate[j], slate[i] })
		return slate
	}
}

// onlineClicks は、並び slate を実際に出したときの 1 回あたりの期待クリック数 Σ θ(k)·r(k 位の商品)。
func onlineClicks(slate []int, theta, rel []float64) float64 {
	sum := 0.0
	for k, id := range slate {
		sum += theta[k] * rel[id]
	}
	return sum
}

func TestNaiveCTRIsBiasedAndIPSRecoversRelevance(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	log := clickbias.NewSimulator(rel, theta, 11).Log(200_000, productionPolicy(rel, 0.15))

	naive := clickbias.NaiveCTR(log, items)
	ips := clickbias.IPS(log, items, theta, 0)

	naiveErr := clickbias.MeanAbsError(naive, rel)
	ipsErr := clickbias.MeanAbsError(ips, rel)
	naiveP := clickbias.PrecisionAtK(naive, rel, slots)
	ipsP := clickbias.PrecisionAtK(ips, rel, slots)
	t.Logf("平均絶対誤差 素の CTR %.3f / IPS %.3f、上位 10 件の一致 素の CTR %.2f / IPS %.2f", naiveErr, ipsErr, naiveP, ipsP)

	if ipsErr >= naiveErr/3 {
		t.Errorf("IPS の誤差 %.3f が素の CTR の誤差 %.3f の 1/3 以上", ipsErr, naiveErr)
	}
	if ipsP < naiveP {
		t.Errorf("IPS の上位 10 件の一致 %.2f が素の CTR %.2f より低い", ipsP, naiveP)
	}
}

func TestEstimateExaminationFromRandomizedTraffic(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	randomized := clickbias.NewSimulator(rel, theta, 13).Log(100_000, randomPolicy(rel, 0.15))
	got := clickbias.EstimateExamination(randomized, slots)
	for k := range slots {
		t.Logf("位置 %2d 本当の θ %.3f 推定 %.3f", k+1, theta[k], got[k])
		if math.Abs(got[k]-theta[k]) > 0.03 {
			t.Errorf("位置 %d の推定 %.3f が本当の値 %.3f から 0.03 以上ずれた", k+1, got[k], theta[k])
		}
	}
}

func TestIPSWithEstimatedExamination(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	sim := clickbias.NewSimulator(rel, theta, 17)
	// 本番の枠の 5% だけをランダムに並べ替え、そこで θ を推定してから、本番のログ全体を IPS で補正する。
	randomized := sim.Log(10_000, randomPolicy(rel, 0.15))
	prod := sim.Log(190_000, productionPolicy(rel, 0.15))
	estimated := clickbias.EstimateExamination(randomized, slots)

	known := clickbias.MeanAbsError(clickbias.IPS(prod, items, theta, 0), rel)
	est := clickbias.MeanAbsError(clickbias.IPS(prod, items, estimated, 0), rel)
	naive := clickbias.MeanAbsError(clickbias.NaiveCTR(prod, items), rel)
	t.Logf("平均絶対誤差 素の CTR %.3f / 推定した θ で IPS %.3f / 本当の θ で IPS %.3f", naive, est, known)
	if est >= naive/2 {
		t.Errorf("推定した θ での IPS の誤差 %.3f が、素の CTR の誤差 %.3f の半分以上", est, naive)
	}
}

func TestClippingTradesVarianceForBias(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	// 深い位置にしか出ない商品の推定が、ログを取り直すたびにどれだけ揺れるかを見る。
	deep := clickbias.Ranking(rel)[slots-2]
	clips := []float64{0, 5, 3}
	values := make([][]float64, len(clips))
	for trial := range 40 {
		log := clickbias.NewSimulator(rel, theta, uint64(100+trial)).Log(5_000, productionPolicy(rel, 0.15))
		for c, clip := range clips {
			values[c] = append(values[c], clickbias.IPS(log, items, theta, clip)[deep])
		}
	}
	stds := make([]float64, len(clips))
	for c, clip := range clips {
		mean, std := meanStd(values[c])
		stds[c] = std
		t.Logf("頭打ち %v: 平均 %.3f（本当は %.3f）、標準偏差 %.3f", clip, mean, rel[deep], std)
	}
	if !(stds[2] < stds[0]) {
		t.Errorf("頭打ち 3 の標準偏差 %.3f が、頭打ちなし %.3f より小さくない", stds[2], stds[0])
	}
}

// TestNaiveRankingSurvivesWhenProductionIsRight は、本番の推薦がもともと良い商品を上に出していれば、素の CTR は値がずれても上位の顔ぶれは崩れないことを示す。
func TestNaiveRankingSurvivesWhenProductionIsRight(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	log := clickbias.NewSimulator(rel, theta, 19).Log(200_000, productionPolicy(rel, 0.15))
	naive := clickbias.NaiveCTR(log, items)
	if p := clickbias.PrecisionAtK(naive, rel, slots); p != 1 {
		t.Errorf("素の CTR の上位 10 件の一致 = %.2f, want 1", p)
	}
	if e := clickbias.MeanAbsError(naive, rel); e < 0.2 {
		t.Errorf("素の CTR の平均絶対誤差 = %.3f, want 0.2 以上（値は大きくずれる）", e)
	}
}

// TestOfflineEvaluationPicksTheBetterPolicy は、本番が中くらいの商品を押し上げていると、素の CTR はその商品を良く見せ、IPS は見抜くことを示す。
func TestOfflineEvaluationPicksTheBetterPolicy(t *testing.T) {
	t.Parallel()
	rel := truth(7)
	theta := clickbias.Examination(slots, 1)
	boosted := middleItems(rel, 3)
	log := clickbias.NewSimulator(rel, theta, 19).Log(200_000, boostedPolicy(rel, 0.15, boosted))
	naive := clickbias.NaiveCTR(log, items)
	ips := clickbias.IPS(log, items, theta, 0)
	for _, id := range boosted {
		t.Logf("押し上げた商品 %d 本当の良さ %.3f / 素の CTR %.3f / IPS %.3f", id, rel[id], naive[id], ips[id])
	}
	t.Logf("上位 10 件の一致 素の CTR %.2f / IPS %.2f", clickbias.PrecisionAtK(naive, rel, slots), clickbias.PrecisionAtK(ips, rel, slots))

	// ログから作った 2 つの新しい並び。素の CTR の順と、IPS で補正した順。
	byNaive := clickbias.Ranking(clickbias.NaiveCTR(log, items))[:slots]
	byIPS := clickbias.Ranking(clickbias.IPS(log, items, theta, 0))[:slots]
	best := clickbias.Ranking(rel)[:slots]

	naiveOnline := onlineClicks(byNaive, theta, rel)
	ipsOnline := onlineClicks(byIPS, theta, rel)
	bestOnline := onlineClicks(best, theta, rel)
	t.Logf("実際に出したときの 1 回あたりの期待クリック 素の CTR の順 %.4f / IPS の順 %.4f / 本当の良さの順 %.4f", naiveOnline, ipsOnline, bestOnline)
	if ipsOnline <= naiveOnline {
		t.Errorf("IPS の順 %.4f が素の CTR の順 %.4f より良くない", ipsOnline, naiveOnline)
	}
}

func TestExaminationShape(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		eta  float64
		want []float64
	}{
		"eta=1 なら 1/k":    {eta: 1, want: []float64{1, 0.5, 1.0 / 3}},
		"eta=0 なら位置によらない": {eta: 0, want: []float64{1, 1, 1}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := clickbias.Examination(3, tc.eta)
			if !slices.EqualFunc(got, tc.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-12 }) {
				t.Errorf("Examination = %v, want %v", got, tc.want)
			}
		})
	}
}

func meanStd(xs []float64) (float64, float64) {
	mean := 0.0
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	v := 0.0
	for _, x := range xs {
		v += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(v / float64(len(xs)))
}
