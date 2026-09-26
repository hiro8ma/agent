package morph

import "math"

// Problem はトークン列が決まったときの品詞の割り当ての問題。Trans[i][j] は品詞 i から j への遷移のコスト。
// Start と End は文頭と文末の遷移のコストで、nil なら 0 とみなす。
type Problem struct {
	Emit  [][]float64
	Trans [][]float64
	Start []float64
	End   []float64
}

// Result はビタビの結果。Min[t][j] は位置 t で品詞 j に至る経路の最小のコスト（文末の遷移は含まない）。
type Result struct {
	Path []int
	Cost float64
	Min  [][]float64
	Back [][]int
}

func (p Problem) states() int {
	if len(p.Emit) == 0 {
		return len(p.Trans)
	}
	return len(p.Emit[0])
}

func (p Problem) start(j int) float64 {
	if p.Start == nil {
		return 0
	}
	return p.Start[j]
}

func (p Problem) end(j int) float64 {
	if p.End == nil {
		return 0
	}
	return p.End[j]
}

// Viterbi は割り当てと遷移のコストの合計が最小の品詞列を O(n k²) で求める。
func Viterbi(p Problem) Result {
	n, k := len(p.Emit), p.states()
	if n == 0 || k == 0 {
		return Result{}
	}
	minCost := make([][]float64, n)
	back := make([][]int, n)
	for t := range n {
		minCost[t] = make([]float64, k)
		back[t] = make([]int, k)
	}
	for j := range k {
		minCost[0][j] = p.start(j) + p.Emit[0][j]
		back[0][j] = -1
	}
	for t := 1; t < n; t++ {
		for j := range k {
			best, arg := math.Inf(1), 0
			for i := range k {
				if c := minCost[t-1][i] + p.Trans[i][j]; c < best {
					best, arg = c, i
				}
			}
			minCost[t][j] = best + p.Emit[t][j]
			back[t][j] = arg
		}
	}
	best, last := math.Inf(1), 0
	for j := range k {
		if c := minCost[n-1][j] + p.end(j); c < best {
			best, last = c, j
		}
	}
	path := make([]int, n)
	path[n-1] = last
	for t := n - 1; t > 0; t-- {
		path[t-1] = back[t][path[t]]
	}
	return Result{Path: path, Cost: best, Min: minCost, Back: back}
}

// PathCost は品詞列 path のコストの合計を返す。
func PathCost(p Problem, path []int) float64 {
	if len(path) == 0 {
		return 0
	}
	c := p.start(path[0]) + p.end(path[len(path)-1])
	for t, j := range path {
		c += p.Emit[t][j]
		if t > 0 {
			c += p.Trans[path[t-1]][j]
		}
	}
	return c
}

// EachPath は k^n 通りの品詞列を順に fn へ渡す。fn に渡すスライスは使い回す。
func EachPath(n, k int, fn func(path []int)) {
	if n == 0 || k == 0 {
		return
	}
	path := make([]int, n)
	for {
		fn(path)
		t := n - 1
		for t >= 0 && path[t] == k-1 {
			path[t] = 0
			t--
		}
		if t < 0 {
			return
		}
		path[t]++
	}
}

// BruteForce は k^n 通りをすべて数えて最小の品詞列を求める。
func BruteForce(p Problem) ([]int, float64) {
	n, k := len(p.Emit), p.states()
	var best []int
	bestCost := math.Inf(1)
	EachPath(n, k, func(path []int) {
		if c := PathCost(p, path); c < bestCost {
			bestCost = c
			best = append(best[:0], path...)
		}
	})
	return best, bestCost
}
