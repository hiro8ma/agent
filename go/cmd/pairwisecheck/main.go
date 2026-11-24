// pairwisecheck は、ペア比較の評価の勝ち / 負け / 引き分けの数から、勝率の Wilson の95%区間と判定、符号検定の p 値と Holm の補正、50%と見分けるのに要る件数を出す。
// ローカルのファイルだけを読み、ネットワークには出ない。
//
//	go run ./cmd/pairwisecheck -wins 11 -losses 9 -ties 0
//	go run ./cmd/pairwisecheck internal/pairwise/testdata/comparisons.json
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/hiro8ma/agent/go/internal/pairwise"
)

func main() {
	var (
		name   = flag.String("name", "比較", "-wins / -losses / -ties で渡す比較の名前")
		wins   = flag.Int("wins", 0, "前側のモデルの勝ちの数")
		losses = flag.Int("losses", 0, "前側のモデルの負けの数")
		ties   = flag.Int("ties", 0, "引き分けの数")
		rates  = flag.String("p", "0.55,0.6,0.65,0.7", "要る件数を出す真の勝率（カンマ区切り）")
	)
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "使い方: pairwisecheck [-name 名前] [-wins 数 -losses 数 -ties 数] [-p 0.55,0.6] [比較の .json か .csv...]")
		flag.PrintDefaults()
	}
	flag.Parse()
	var cs []pairwise.Counts
	flag.Visit(func(f *flag.Flag) {
		if (f.Name == "wins" || f.Name == "losses" || f.Name == "ties") && len(cs) == 0 {
			cs = append(cs, pairwise.Counts{Name: *name, Wins: *wins, Losses: *losses, Ties: *ties})
		}
	})
	for _, path := range flag.Args() {
		loaded, err := pairwise.Load(path)
		if err != nil {
			fail(err)
		}
		cs = append(cs, loaded...)
	}
	if len(cs) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	for _, c := range cs {
		if err := c.Validate(); err != nil {
			fail(err)
		}
	}
	ps, err := parseRates(*rates)
	if err != nil {
		fail(err)
	}
	results := pairwise.Analyze(cs)
	w := os.Stdout
	intervals(w, results)
	signTests(w, results)
	required(w, ps)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(2)
}

func parseRates(s string) ([]float64, error) {
	var out []float64
	for f := range strings.SplitSeq(s, ",") {
		p, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
		if err != nil {
			return nil, fmt.Errorf("-p の %q: %w", f, err)
		}
		if _, err := pairwise.RequiredN(p); err != nil {
			return nil, fmt.Errorf("-p の %q: %w", f, err)
		}
		out = append(out, p)
	}
	return out, nil
}

func intervals(w io.Writer, results []pairwise.Result) {
	fmt.Fprintf(w, "## 勝率の区間と判定\n\n")
	fmt.Fprintf(w, "勝率は名前の前側のモデルから見た値で、区間は Wilson の%g%%区間\n\n", pairwise.Level*100)
	fmt.Fprintln(w, "| 比較 | 勝ち | 負け | 引き分け | 引き分けの扱い | 件数 | 勝率 | 区間 | 判定 |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|")
	for _, r := range results {
		for _, iv := range r.Intervals {
			fmt.Fprintf(w, "| %s | %d | %d | %d | %s | %d | %.3f | [%.3f, %.3f] | %s |\n",
				r.Name, r.Wins, r.Losses, r.Ties, iv.Ties, iv.N, iv.Rate, iv.Lower, iv.Upper, iv.Verdict)
		}
	}
}

func signTests(w io.Writer, results []pairwise.Result) {
	fmt.Fprintf(w, "\n## 符号検定\n\n")
	fmt.Fprintln(w, "引き分けを除いた勝ち負けで、勝率0.5の正確な二項検定（両側）")
	fmt.Fprintf(w, "Holm の補正は%d件の比較を1つの族として行う\n\n", len(results))
	fmt.Fprintln(w, "| 比較 | 勝ち負けの付いた件数 | p 値 | Holm の補正後の p 値 |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range results {
		fmt.Fprintf(w, "| %s | %d | %.4g | %.4g |\n", r.Name, r.Wins+r.Losses, r.P, r.Holm)
	}
}

func required(w io.Writer, ps []float64) {
	fmt.Fprintf(w, "\n## 50%%と見分けるのに要る件数\n\n")
	fmt.Fprintf(w, "有意水準%g%%（両側）、検出力%g%%の正規近似で出した、勝ち負けの付いた件数（引き分けを除く）\n\n", pairwise.Alpha*100, pairwise.Power*100)
	fmt.Fprintln(w, "| 真の勝率 | 件数 |")
	fmt.Fprintln(w, "|---|---|")
	for _, p := range ps {
		n, _ := pairwise.RequiredN(p)
		fmt.Fprintf(w, "| %g | %d |\n", p, n)
	}
}
