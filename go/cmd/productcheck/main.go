// productcheck は、商品ページの HTML の JSON-LD に、エージェントが比較に使う項目が揃っているかと、予算 / 届くまでの日数 / 在庫の条件で判定できる商品の数を出す。
// ローカルのファイルだけを読み、ネットワークには出ない。
//
//	go run ./cmd/productcheck -max-price 5000 -max-days 3 -in-stock internal/productld/testdata
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/hiro8ma/agent/go/internal/productld"
)

type named struct {
	label string
	cond  productld.Condition
}

func main() {
	var (
		maxPrice = flag.Float64("max-price", -1, "予算の上限。負なら見ない")
		currency = flag.String("currency", "JPY", "予算の通貨")
		maxDays  = flag.Float64("max-days", -1, "届くまでの日数の上限。負なら見ない")
		inStock  = flag.Bool("in-stock", false, "在庫ありだけにする")
	)
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "使い方: productcheck [-max-price 円] [-currency JPY] [-max-days 日] [-in-stock] HTML のファイルかディレクトリ...")
		flag.PrintDefaults()
	}
	flag.Parse()
	conds := conditions(*maxPrice, *currency, *maxDays, *inStock)
	if flag.NArg() == 0 || len(conds) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	var results []productld.Result
	for _, path := range flag.Args() {
		r, err := productld.Load(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		results = append(results, r...)
	}
	w := os.Stdout
	pages(w, results)
	fields(w, results)
	counts(w, results, conds)
	verdicts(w, results, conds[len(conds)-1])
}

func conditions(maxPrice float64, currency string, maxDays float64, inStock bool) []named {
	var out []named
	var all productld.Condition
	if maxPrice >= 0 {
		all.MaxPrice, all.Currency = new(maxPrice), currency
		out = append(out, named{fmt.Sprintf("予算%g %s 以下", maxPrice, currency), productld.Condition{MaxPrice: all.MaxPrice, Currency: currency}})
	}
	if maxDays >= 0 {
		all.MaxDays = new(maxDays)
		out = append(out, named{fmt.Sprintf("届くまで%g日以内", maxDays), productld.Condition{MaxDays: all.MaxDays}})
	}
	if inStock {
		all.InStock = true
		out = append(out, named{"在庫あり", productld.Condition{InStock: true}})
	}
	if len(out) > 1 {
		out = append(out, named{"すべて", all})
	}
	return out
}

func label(p productld.Product) string {
	if p.Name == "" {
		return "（name 無し）"
	}
	return p.Name
}

func pages(w io.Writer, results []productld.Result) {
	fmt.Fprintf(w, "## 読んだページ\n\n")
	fmt.Fprintln(w, "| ページ | JSON-LD の script | 壊れた script | Product |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range results {
		fmt.Fprintf(w, "| %s | %d | %d | %d |\n", r.File, r.Scripts, r.Broken, len(r.Products))
	}
}

func fields(w io.Writer, results []productld.Result) {
	names := make([]string, len(productld.Fields))
	for i, f := range productld.Fields {
		names[i] = f.String()
	}
	fmt.Fprintf(w, "\n## 項目の有無\n\n")
	fmt.Fprintf(w, "| ページ | 商品 | %s |\n", strings.Join(names, " | "))
	fmt.Fprintf(w, "|---|---|%s\n", strings.Repeat("---|", len(names)))
	have := make([]int, len(productld.Fields))
	total := 0
	for _, r := range results {
		for _, p := range r.Products {
			total++
			cells := make([]string, len(productld.Fields))
			for i, f := range productld.Fields {
				cells[i] = "無"
				if p.Has(f) {
					cells[i] = "有"
					have[i]++
				}
			}
			fmt.Fprintf(w, "| %s | %s | %s |\n", r.File, label(p), strings.Join(cells, " | "))
		}
	}
	cells := make([]string, len(have))
	for i, n := range have {
		cells[i] = fmt.Sprint(n)
	}
	fmt.Fprintf(w, "| 有る商品の数 | %d件中 | %s |\n", total, strings.Join(cells, " | "))
}

func counts(w io.Writer, results []productld.Result, conds []named) {
	var products []productld.Product
	for _, r := range results {
		products = append(products, r.Products...)
	}
	fmt.Fprintf(w, "\n## 条件ごとの件数\n\n")
	fmt.Fprintln(w, "| 条件 | 満たす | 満たさない | 判定できない |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, c := range conds {
		n := productld.Count(products, c.cond)
		fmt.Fprintf(w, "| %s | %d | %d | %d |\n", c.label, n[productld.Satisfied], n[productld.Failed], n[productld.Unknown])
	}
}

func verdicts(w io.Writer, results []productld.Result, c named) {
	fmt.Fprintf(w, "\n## 商品ごとの判定（%s）\n\n", c.label)
	fmt.Fprintln(w, "| ページ | 商品 | 判定 | 判定できない理由 |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range results {
		for _, p := range r.Products {
			v := productld.Judge(p, c.cond)
			fmt.Fprintf(w, "| %s | %s | %s | %s |\n", r.File, label(p), v.Status, strings.Join(v.Missing, ", "))
		}
	}
}
