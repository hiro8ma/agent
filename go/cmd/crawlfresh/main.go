// crawlfresh は、先方への負荷を抑える規則で取り込みがどれだけ遅くなるか、取り直しの方針で鮮度がどう変わるか、掲載を終えた記事をどれだけ出し続けるかを、模擬の時計と乱数で測る。
// ネットワークには出ない。
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

func main() {
	var (
		hostPages = flag.Int("host-pages", 10_000, "表 A で1ホストから取る件数")
		pages     = flag.Int("pages", 10_000, "表 B のページ数")
		budgets   = flag.String("budgets", "0.25,1,4", "表 B の1日の取り直しの予算を、1ページあたりの回数で並べる")
		intervals = flag.Int("intervals", 200, "表 B の模擬で1ページを取り直す回数")
		seed      = flag.Uint64("seed", 1, "乱数の種")
	)
	flag.Parse()
	perPage, err := parseFloats(*budgets)
	if err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	w := os.Stdout
	politeness(w, *hostPages, *seed)
	freshness(w, *pages, perPage, *intervals, *seed)
	delisting(w, *seed)
	fmt.Fprintf(w, "\n所要時間は%.2f秒\n", time.Since(start).Seconds())
}

func parseFloats(s string) ([]float64, error) {
	var out []float64
	for f := range strings.SplitSeq(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

type latency struct {
	name   string
	sample func(*rand.Rand) time.Duration
}

// politeness は1ホストの n 件を crawl.Frontier に入れ、取得の時間を引いて模擬の時計で取り切る。
func politeness(w io.Writer, n int, seed uint64) {
	const median = 200 * time.Millisecond
	const sigma = 1.0
	latencies := []latency{
		{name: "固定", sample: func(*rand.Rand) time.Duration { return median }},
		{name: "対数正規", sample: func(r *rand.Rand) time.Duration {
			return time.Duration(float64(median) * math.Exp(sigma*r.NormFloat64()))
		}},
	}
	fmt.Fprintf(w, "## 表 A 配慮の費用\n\n")
	fmt.Fprintf(w, "- 1ホストから%d件を取る。同じホストへは同時に1本で、次の取得は、前の取得が終わってから k × その取得の時間だけ待つ\n", n)
	fmt.Fprintf(w, "- 取得の時間は、固定の%vと、中央値%vで対数の標準偏差%.1fの対数正規（種%d）\n\n", median, median, sigma, seed)
	fmt.Fprintln(w, "| 取得時間の分布 | k | 平均の取得時間 | 99パーセンタイル | 全件の所要 | 1時間あたりの件数 | 接続している割合 | 1/(1+k) | 最も長い待ち |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|")
	for _, lat := range latencies {
		for _, k := range []float64{0, 1, crawl.DefaultPoliteness} {
			r := rand.New(rand.NewPCG(seed, 1))
			f := crawl.NewFrontier(crawl.FrontierConfig{Politeness: k})
			for i := range n {
				f.Add(fmt.Sprintf("https://example.com/articles/%d", i), 1)
			}
			t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
			now := t0
			var busy, longestWait time.Duration
			var prevEnd time.Time
			durations := make([]time.Duration, 0, n)
			for {
				it, ok := f.Next(now)
				if !ok {
					wake, ok := f.Wake()
					if !ok {
						break
					}
					now = wake
					continue
				}
				if !prevEnd.IsZero() {
					longestWait = max(longestWait, now.Sub(prevEnd))
				}
				d := lat.sample(r)
				durations = append(durations, d)
				busy += d
				end := now.Add(d)
				f.Done(it.Host, now, end)
				now, prevEnd = end, end
			}
			total := now.Sub(t0)
			slices.Sort(durations)
			fmt.Fprintf(w, "| %s | %g | %s | %s | %s | %.0f | %.4f | %.4f | %s |\n",
				lat.name, k, seconds(busy/time.Duration(n)), seconds(durations[len(durations)*99/100]),
				span(total), float64(n)/total.Hours(), float64(busy)/float64(total), 1/(1+k), seconds(longestWait))
		}
	}
}

// freshness は変化の頻度を対数正規から引き、3つの方針で取り直しの予算を配って、式と模擬の平均の鮮度を並べる。
func freshness(w io.Writer, n int, perPage []float64, intervals int, seed uint64) {
	const mean, sigma = 1.0, 1.5
	r := rand.New(rand.NewPCG(seed, 2))
	mu := math.Log(mean) - sigma*sigma/2
	lambdas := make([]float64, n)
	for i := range lambdas {
		lambdas[i] = math.Exp(mu + sigma*r.NormFloat64())
	}
	sorted := slices.Sorted(slices.Values(lambdas))
	var total float64
	for _, l := range lambdas {
		total += l
	}
	fmt.Fprintf(w, "\n## 表 B 鮮度\n\n")
	fmt.Fprintf(w, "- %dページ、変化の頻度 λ は平均%.1f回/日、対数の標準偏差%.1fの対数正規（種%d）\n", n, mean, sigma, seed)
	fmt.Fprintf(w, "- 引いた λ は中央値%.3f、平均%.3f、99パーセンタイル%.2f、最大%.1f（回/日）\n", sorted[n/2], total/float64(n), sorted[n*99/100], sorted[n-1])
	fmt.Fprintf(w, "- 式は (1 − e^(−λI)) / (λI)。模擬は、ページごとに変化をポアソン過程で引き、一定の間隔で%d回取り直して、最新だった時間の割合を数える\n\n", intervals)
	fmt.Fprintln(w, "| 1日の予算 B | 方針 | 式の平均の鮮度 | 模擬の平均の鮮度 | 差 | 0回のページ | 最も多いページの回数/日 |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|")
	policies := []struct {
		name     string
		allocate func([]float64, float64) []float64
	}{
		{name: "一様", allocate: crawl.UniformAllocation},
		{name: "比例", allocate: crawl.ProportionalAllocation},
		{name: "最適", allocate: crawl.OptimalAllocation},
	}
	for _, p := range perPage {
		budget := p * float64(n)
		for _, pol := range policies {
			freqs := pol.allocate(lambdas, budget)
			formula := crawl.MeanFreshness(lambdas, freqs)
			sim := crawl.SimulateFreshness(lambdas, freqs, intervals, seed)
			zeros := 0
			for _, f := range freqs {
				if f == 0 {
					zeros++
				}
			}
			fmt.Fprintf(w, "| %.0f | %s | %.4f | %.4f | %+.4f | %d | %.2f |\n",
				budget, pol.name, formula, sim, sim-formula, zeros, slices.Max(freqs))
		}
	}
}

// delisting は記事の出る時刻と掲載の期間を引き、存在の確認と一覧の間隔の組ごとに、終えた記事を出し続けた時間と新しい記事の遅れを並べる。
func delisting(w io.Writer, seed uint64) {
	base := crawl.DelistConfig{Days: 400, Warmup: 100, NewPerDay: 50, EndPerDay: 0.05, CheckCost: 1, Seed: seed}
	const conditionalCost = 0.1
	fmt.Fprintf(w, "\n## 表 C 掲載の終了\n\n")
	fmt.Fprintf(w, "- 1日に新しい記事が平均%.0f件出て、掲載中の記事は1日に%.0f%%の確率で掲載を終える（平均の掲載期間%.1f日、掲載中は平均%.0f件）\n",
		base.NewPerDay, base.EndPerDay*100, 1/-math.Log1p(-base.EndPerDay), base.NewPerDay/-math.Log1p(-base.EndPerDay))
	fmt.Fprintf(w, "- %.0f日を模擬し、最初の%.0f日を除いて数える（種%d）。一覧で見つけた記事はその場で1回取り、その後は確認の間隔ごとに取って、404なら出すのをやめる\n", base.Days, base.Warmup, seed)
	fmt.Fprintf(w, "- 費用は全件の取得を1とする。最後の行は、確認を条件付きの取得（変わっていなければ304）にして1回の費用を%.1fと仮定し、確認の間隔を%.0f%%に縮めた。存在の確認が1日で一覧が1時間の行と、費用がほぼ同じになる\n\n", conditionalCost, conditionalCost*100)
	fmt.Fprintln(w, "| 存在の確認の間隔 | 一覧を見る間隔 | 確認の費用 | (a) 終えた記事をまだ出していた時間の平均 | (b) 新しい記事が出るまでの遅れの平均 | 見つける前に終えた割合 | (c) 1日の取得の回数 | 1日の費用 |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|")
	rows := []struct{ check, list, cost float64 }{
		{check: 0.25, list: 1.0 / 24, cost: 1},
		{check: 0.25, list: 1, cost: 1},
		{check: 1, list: 1.0 / 24, cost: 1},
		{check: 1, list: 1, cost: 1},
		{check: 3, list: 1.0 / 24, cost: 1},
		{check: 3, list: 1, cost: 1},
		{check: conditionalCost, list: 1.0 / 24, cost: conditionalCost},
	}
	for _, row := range rows {
		cfg := base
		cfg.CheckInterval, cfg.ListInterval, cfg.CheckCost = row.check, row.list, row.cost
		res := crawl.SimulateDelist(cfg)
		fmt.Fprintf(w, "| %s | %s | %.1f | %.2f時間 | %.2f時間 | %.2f%% | %.0f | %.0f |\n",
			days(row.check), days(row.list), row.cost, res.StaleDays*24, res.DiscoverDays*24, res.MissedRate*100, res.FetchesPerDay, res.CostPerDay)
	}
}

func seconds(d time.Duration) string { return fmt.Sprintf("%.3f秒", d.Seconds()) }

func span(d time.Duration) string {
	if d >= time.Hour {
		return fmt.Sprintf("%.2f時間", d.Hours())
	}
	return fmt.Sprintf("%.1f分", d.Minutes())
}

func days(d float64) string {
	if d >= 1 {
		return fmt.Sprintf("%g日", d)
	}
	return fmt.Sprintf("%.1f時間", d*24)
}
