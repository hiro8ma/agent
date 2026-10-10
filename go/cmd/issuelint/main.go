// issuelint は Issue の本文が定義になっているかを確かめ、問題が 1 件でもあれば終了コード 1 で終わる。エージェントに Issue を渡す前に流す。
//
//	gh issue view 123 --json body -q .body | go run ./cmd/issuelint
//	go run ./cmd/issuelint -file issue-context/body.md
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/hiro8ma/agent/go/internal/issuelint"
)

func main() {
	file := flag.String("file", "", "Issue の本文のファイル。空か - なら標準入力から読む")
	flag.Parse()

	body, err := read(*file)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	r := issuelint.Check(string(body))
	if !r.OK() {
		fmt.Print(r)
		fmt.Fprintln(os.Stderr, "Issue の本文を直してから、ラベルを付け直す")
		os.Exit(1)
	}
	fmt.Println("Issue の本文に問題なし")
}

func read(file string) ([]byte, error) {
	if file == "" || file == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(filepath.Clean(file))
}
