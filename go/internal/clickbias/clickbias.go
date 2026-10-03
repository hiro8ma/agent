// Package clickbias は、表示位置の偏り（位置バイアス）を含むクリックのログから、商品の良さを取り出す方法を確かめる。
// クリックの確率を「その位置が見られる確率 θ(k)」と「商品が良い確率 r」の積とみなす位置に基づくクリックモデル（PBM）を使う。
package clickbias

import (
	"math"
	"math/rand/v2"
	"slices"
)

// Impression は 1 回の表示で、どの商品をどの位置（0 が先頭）に出し、クリックされたかを表す。
type Impression struct {
	Item     int
	Position int
	Click    bool
}

// Examination は位置 k（0 が先頭）が見られる確率 θ(k) = 1 / (k+1)^eta。eta が大きいほど下の位置が見られにくい。
func Examination(slots int, eta float64) []float64 {
	theta := make([]float64, slots)
	for k := range theta {
		theta[k] = 1 / math.Pow(float64(k+1), eta)
	}
	return theta
}

// Simulator は商品の本当の良さ relevance と、位置の見られやすさ theta から、クリックのログを作る。
type Simulator struct {
	relevance []float64
	theta     []float64
	rng       *rand.Rand
}

func NewSimulator(relevance, theta []float64, seed uint64) *Simulator {
	return &Simulator{relevance: relevance, theta: theta, rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
}

// Show は slate の順に商品を並べて 1 回見せ、位置ごとのクリックを返す。見られて、かつ商品が良かったときだけクリックされる。
func (s *Simulator) Show(slate []int) []Impression {
	out := make([]Impression, len(slate))
	for k, item := range slate {
		examined := s.rng.Float64() < s.theta[k]
		relevant := s.rng.Float64() < s.relevance[item]
		out[k] = Impression{Item: item, Position: k, Click: examined && relevant}
	}
	return out
}

// Log は policy が選んだ並びを n 回見せたログを返す。
func (s *Simulator) Log(n int, policy func(rng *rand.Rand) []int) []Impression {
	var log []Impression
	for range n {
		log = append(log, s.Show(policy(s.rng))...)
	}
	return log
}

// NaiveCTR は商品ごとにクリック数 ÷ 表示回数を返す。位置の偏りを補正しない。表示されなかった商品は NaN。
func NaiveCTR(log []Impression, items int) []float64 {
	clicks := make([]float64, items)
	shows := make([]float64, items)
	for _, im := range log {
		shows[im.Item]++
		if im.Click {
			clicks[im.Item]++
		}
	}
	out := make([]float64, items)
	for i := range out {
		out[i] = clicks[i] / shows[i]
	}
	return out
}

// IPS は各クリックをその位置の見られやすさ θ(k) で割り戻してから、表示回数で割る。PBM の下では商品の良さ r の偏りのない推定になる。
// clip が 0 より大きければ、重み 1/θ(k) を clip で頭打ちにする（分散を抑える代わりに、深い位置の商品を小さく見積もる）。
func IPS(log []Impression, items int, theta []float64, clip float64) []float64 {
	sum := make([]float64, items)
	shows := make([]float64, items)
	for _, im := range log {
		shows[im.Item]++
		if im.Click {
			w := 1 / theta[im.Position]
			if clip > 0 {
				w = min(w, clip)
			}
			sum[im.Item] += w
		}
	}
	out := make([]float64, items)
	for i := range out {
		out[i] = sum[i] / shows[i]
	}
	return out
}

// EstimateExamination は、並びをランダムに入れ替えたログから θ(k) を推定し、先頭を 1 にそろえて返す。
// 商品がどの位置にも同じ確率で出るので、位置ごとのクリック率の違いは θ(k) の違いだけから来る。
func EstimateExamination(randomized []Impression, slots int) []float64 {
	clicks := make([]float64, slots)
	shows := make([]float64, slots)
	for _, im := range randomized {
		shows[im.Position]++
		if im.Click {
			clicks[im.Position]++
		}
	}
	theta := make([]float64, slots)
	for k := range theta {
		theta[k] = clicks[k] / shows[k]
	}
	top := theta[0]
	for k := range theta {
		theta[k] /= top
	}
	return theta
}

// Ranking は推定値の高い順に商品の番号を並べる。NaN（表示されなかった商品）は最後に回す。
func Ranking(score []float64) []int {
	ids := make([]int, len(score))
	for i := range ids {
		ids[i] = i
	}
	slices.SortStableFunc(ids, func(a, b int) int {
		sa, sb := score[a], score[b]
		switch {
		case math.IsNaN(sa) && math.IsNaN(sb):
			return 0
		case math.IsNaN(sa):
			return 1
		case math.IsNaN(sb):
			return -1
		case sa > sb:
			return -1
		case sa < sb:
			return 1
		}
		return 0
	})
	return ids
}

// PrecisionAtK は、推定の上位 k 件のうち、本当の良さの上位 k 件に入っている割合を返す。
func PrecisionAtK(estimate, truth []float64, k int) float64 {
	want := make(map[int]bool, k)
	for _, id := range Ranking(truth)[:k] {
		want[id] = true
	}
	hit := 0
	for _, id := range Ranking(estimate)[:k] {
		if want[id] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

// MeanAbsError は、表示された商品について推定値と本当の良さの差の絶対値の平均を返す。
func MeanAbsError(estimate, truth []float64) float64 {
	sum, n := 0.0, 0
	for i := range estimate {
		if math.IsNaN(estimate[i]) {
			continue
		}
		sum += math.Abs(estimate[i] - truth[i])
		n++
	}
	return sum / float64(n)
}
