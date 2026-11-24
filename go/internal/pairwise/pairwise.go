package pairwise

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"
)

const (
	// Level は区間の信頼水準。
	Level = 0.95
	// Alpha と Power は、要る件数を出すときの有意水準（両側）と検出力。
	Alpha = 0.05
	Power = 0.8
)

// Counts は1つの比較の、前側のモデルから見た勝ち / 負け / 引き分けの数。
type Counts struct {
	Name   string `json:"name"`
	Wins   int    `json:"wins"`
	Losses int    `json:"losses"`
	Ties   int    `json:"ties"`
}

func (c Counts) Validate() error {
	if c.Wins < 0 || c.Losses < 0 || c.Ties < 0 {
		return fmt.Errorf("%s: 勝ち %d / 負け %d / 引き分け %d に負の数がある", c.Name, c.Wins, c.Losses, c.Ties)
	}
	return nil
}

// TieMode は勝率を出すときの引き分けの扱い。
type TieMode int

const (
	ExcludeTies TieMode = iota
	HalfWin
)

var TieModes = [...]TieMode{ExcludeTies, HalfWin}

func (m TieMode) String() string {
	return [...]string{"引き分けを除く", "引き分けを0.5勝"}[m]
}

// Verdict は勝率の区間が0.5をまたぐかでの判定。
type Verdict int

const (
	Indistinguishable Verdict = iota
	Better
	Worse
)

func (v Verdict) String() string {
	return [...]string{"区別できない", "50%と区別できる（勝ち）", "50%と区別できる（負け）"}[v]
}

// Interval は1つの引き分けの扱いでの勝率と Wilson の区間。N は勝率の分母にした件数。
type Interval struct {
	Ties         TieMode
	N            int
	Rate         float64
	Lower, Upper float64
	Verdict      Verdict
}

// Interval は引き分けの扱い m で勝率と Wilson の区間を出す。件数が0なら勝率0.5、区間 [0, 1] とする。
func (c Counts) Interval(m TieMode) Interval {
	x, n := float64(c.Wins), c.Wins+c.Losses
	if m == HalfWin {
		x, n = float64(c.Wins)+0.5*float64(c.Ties), n+c.Ties
	}
	iv := Interval{Ties: m, N: n, Rate: 0.5}
	if n > 0 {
		iv.Rate = x / float64(n)
	}
	iv.Lower, iv.Upper = Wilson(x, n)
	switch {
	case iv.Lower > 0.5:
		iv.Verdict = Better
	case iv.Upper < 0.5:
		iv.Verdict = Worse
	}
	return iv
}

// Wilson は n 件のうち x 件が成功したときの、勝率の Level の Wilson の区間。x は端数を許す。n が0なら [0, 1]。
func Wilson(x float64, n int) (lower, upper float64) {
	if n <= 0 {
		return 0, 1
	}
	z := normalQuantile(1 - (1-Level)/2)
	fn := float64(n)
	p := x / fn
	d := 1 + z*z/fn
	center := (p + z*z/(2*fn)) / d
	half := z / d * math.Sqrt(p*(1-p)/fn+z*z/(4*fn*fn))
	return max(0, center-half), min(1, center+half)
}

func (c Counts) SignTest() float64 {
	return SignTest(c.Wins, c.Losses)
}

// SignTest は wins 勝 losses 敗の、勝率0.5の正確な二項検定（両側）の p 値を返す。勝ち負けが0件なら1。
func SignTest(wins, losses int) float64 {
	n := wins + losses
	if n <= 0 {
		return 1
	}
	lgN, _ := math.Lgamma(float64(n + 1))
	tail := 0.0
	for i := range min(wins, losses) + 1 {
		lgI, _ := math.Lgamma(float64(i + 1))
		lgRest, _ := math.Lgamma(float64(n - i + 1))
		tail += math.Exp(lgN - lgI - lgRest - float64(n)*math.Ln2)
	}
	return min(1, 2*tail)
}

// RequiredN は、真の勝率 p を50%と見分けるのに要る、勝ち負けの付いた件数を正規近似で返す。有意水準は Alpha（両側）、検出力は Power。
func RequiredN(p float64) (int, error) {
	if !(p > 0 && p < 1) || p == 0.5 {
		return 0, errors.New("真の勝率は0より大きく1より小さい、0.5以外の値にする")
	}
	za, zb := normalQuantile(1-Alpha/2), normalQuantile(Power)
	num, d := za*0.5+zb*math.Sqrt(p*(1-p)), p-0.5
	return int(math.Ceil(num * num / (d * d))), nil
}

// Holm は p 値の並びを Holm の方法で補正し、元の順で返す。
func Holm(ps []float64) []float64 {
	order := make([]int, len(ps))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(ps[a], ps[b]) })
	out := make([]float64, len(ps))
	running := 0.0
	for rank, i := range order {
		running = max(running, min(1, float64(len(ps)-rank)*ps[i]))
		out[i] = running
	}
	return out
}

// Result は1つの比較の結果。Holm は同時に渡した比較の全部を1つの族として補正した p 値。
type Result struct {
	Counts
	Intervals [len(TieModes)]Interval
	P         float64
	Holm      float64
}

// Analyze は比較ごとに区間と符号検定の p 値を出し、全部の p 値に Holm の補正をかける。
func Analyze(cs []Counts) []Result {
	out := make([]Result, len(cs))
	ps := make([]float64, len(cs))
	for i, c := range cs {
		out[i].Counts = c
		for j, m := range TieModes {
			out[i].Intervals[j] = c.Interval(m)
		}
		out[i].P = c.SignTest()
		ps[i] = out[i].P
	}
	for i, h := range Holm(ps) {
		out[i].Holm = h
	}
	return out
}

func normalQuantile(q float64) float64 {
	return math.Sqrt2 * math.Erfinv(2*q-1)
}
