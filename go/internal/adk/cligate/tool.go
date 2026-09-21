package cligate

import (
	"errors"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type toolInput struct {
	// Args は 1 つずつ分けた引数。文字列 1 つで受けて分割しないので、分割の食い違いが起きない。
	Args []string `json:"args"`
}

// Tool は Gate を ADK のツールにする。許可していない呼び出しは、理由を結果としてモデルに返す。
func (g Gate) Tool(name, description string) (tool.Tool, error) {
	return functiontool.New(functiontool.Config{Name: name, Description: description},
		func(ctx agent.Context, in toolInput) (map[string]any, error) {
			r, err := g.Run(ctx, in.Args)
			if errors.Is(err, ErrNotAllowed) {
				return map[string]any{"status": "rejected", "message": err.Error()}, nil
			}
			if err != nil {
				return nil, err
			}
			return map[string]any{"status": "ok", "stdout": r.Stdout, "stderr": r.Stderr, "exitCode": r.ExitCode, "truncated": r.Truncated}, nil
		})
}
