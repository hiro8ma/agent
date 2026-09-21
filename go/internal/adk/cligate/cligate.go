// Package cligate は CLI をエージェントのツールとして包む。
//
// 引数をシェルに通さないだけでは足りない。シェルを通らなくても、モデルは任意のサブコマンドとフラグを渡せる
// （gcloud なら auth print-access-token でトークンを表示させ、--impersonate-service-account で別の権限を使う）。
// ここでは、許可するコマンドの経路と、その経路で使ってよいフラグの両方を決め、それ以外は実行しない。
package cligate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"time"
)

// Spec は許可する 1 つのコマンド。
type Spec struct {
	// Path はサブコマンドの経路。例 compute instances list
	Path []string
	// Flags は使ってよいフラグの名前。--name=value の形だけを受ける。
	Flags []string
}

// Gate は 1 つの CLI を包む。
type Gate struct {
	Binary string
	Specs  []Spec
	// Fixed は最後に必ず付けるフラグ。モデルは同じ名前のフラグを渡せない。
	Fixed []string
	// Env は子プロセスに渡す環境変数のすべて。親の環境変数は引き継がない。
	Env     []string
	Timeout time.Duration
	// MaxOutput は返す標準出力の上限（バイト）。
	MaxOutput int
}

// ErrNotAllowed は許可していないコマンドやフラグを示す。
var ErrNotAllowed = errors.New("cligate: 許可していない")

// Build は argv を検証し、実行する引数の並びを返す。
func (g Gate) Build(argv []string) ([]string, error) {
	spec, ok := g.match(argv)
	if !ok {
		return nil, fmt.Errorf("%w: コマンド %q", ErrNotAllowed, strings.Join(argv, " "))
	}
	rest := argv[len(spec.Path):]
	for _, a := range rest {
		name, _, ok := strings.Cut(a, "=")
		if !ok || !strings.HasPrefix(name, "--") {
			return nil, fmt.Errorf("%w: 位置引数や値の無いフラグ %q", ErrNotAllowed, a)
		}
		if !slices.Contains(spec.Flags, name) || g.fixed(name) {
			return nil, fmt.Errorf("%w: フラグ %q", ErrNotAllowed, name)
		}
	}
	out := slices.Concat(spec.Path, rest, g.Fixed)
	return out, nil
}

func (g Gate) match(argv []string) (Spec, bool) {
	for _, s := range g.Specs {
		if len(argv) >= len(s.Path) && slices.Equal(argv[:len(s.Path)], s.Path) {
			// 経路の後ろにさらにサブコマンドが続く形（list の後ろに delete など）は、フラグの検査で落ちる。
			return s, true
		}
	}
	return Spec{}, false
}

func (g Gate) fixed(name string) bool {
	for _, f := range g.Fixed {
		if n, _, _ := strings.Cut(f, "="); n == name {
			return true
		}
	}
	return false
}

// Result は実行の結果。
type Result struct {
	Stdout    string `json:"stdout"`
	Stderr    string `json:"stderr"`
	ExitCode  int    `json:"exitCode"`
	Truncated bool   `json:"truncated,omitempty"`
}

// Run は検証してから実行する。
func (g Gate) Run(ctx context.Context, argv []string) (Result, error) {
	args, err := g.Build(argv)
	if err != nil {
		return Result{}, err
	}
	if g.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, g.Timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, g.Binary, args...) //nolint:gosec // 引数は Build で許可した経路とフラグだけ
	cmd.Env = g.Env
	if cmd.Env == nil {
		cmd.Env = []string{}
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err = cmd.Run()
	if ctx.Err() != nil {
		return Result{}, fmt.Errorf("cligate: 時間切れ: %w", ctx.Err())
	}
	r := Result{Stdout: stdout.String(), Stderr: stderr.String()}
	if g.MaxOutput > 0 && len(r.Stdout) > g.MaxOutput {
		r.Stdout, r.Truncated = r.Stdout[:g.MaxOutput], true
	}
	if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
		r.ExitCode = exitErr.ExitCode()
		return r, nil
	}
	return r, err
}
