// wikilint は LLM Wiki のページの出典のハッシュを確かめ、古いページが 1 件でもあれば終了コード 1 で終わる。push の前に流す。
//
//	go run ./cmd/wikilint -root path/to/wiki
//	go run ./cmd/wikilint -root path/to/wiki -stamp billing/refund.md   # 読み直して直したページのハッシュを書き込む
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hiro8ma/agent/go/internal/wikilint"
)

func main() {
	root := flag.String("root", ".", "wiki の根のディレクトリ")
	stamp := flag.String("stamp", "", "読み直して直したページ（root からの相対パス）。出典のハッシュを今の値に書き換える")
	flag.Parse()

	pages, err := wikilint.Load(*root)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if *stamp != "" {
		text, err := wikilint.Stamp(*root, pages, *stamp)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		if err := os.WriteFile(filepath.Join(*root, *stamp), []byte(text), 0o600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		for _, p := range wikilint.Downstream(pages, *stamp) {
			fmt.Printf("次に読み直す: %s\n", p)
		}
		return
	}

	problems := wikilint.Lint(*root, pages)
	for _, p := range problems {
		fmt.Println(p)
	}
	if len(problems) > 0 {
		fmt.Fprintf(os.Stderr, "%d 件の問題。古いページを読み直して直し、-stamp でハッシュを書き込む\n", len(problems))
		os.Exit(1)
	}
	fmt.Printf("%d ページ、問題なし\n", len(pages))
}
