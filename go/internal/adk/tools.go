// Package adk は usecase.Ask を Google ADK で実装する。
// ADK は Runner が内部ループを持つため、ここで usecase.Ask を直接実装する。
package adk

import (
	"context"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"

	sharedtool "github.com/hiro8ma/agent/go/internal/tool"
)

// ADK の functiontool.New は typed struct を要求するため、
// 共有 Tool Handler（map[string]any ベース）を struct 型にラップする。

// --- calculator ---

type calcArgs struct {
	Op string  `json:"op" description:"演算子 (add / sub / mul / div)"`
	A  float64 `json:"a"  description:"左オペランド"`
	B  float64 `json:"b"  description:"右オペランド"`
}

type calcResult struct {
	Result float64 `json:"result"`
}

func calculator(ctx tool.Context, args calcArgs) (calcResult, error) {
	out, err := sharedtool.Calculator(context.Background(), map[string]any{
		"op": args.Op,
		"a":  args.A,
		"b":  args.B,
	})
	if err != nil {
		return calcResult{}, err
	}
	r, _ := out["result"].(float64)
	return calcResult{Result: r}, nil
}

// --- search_knowledge ---

type searchArgs struct {
	Query string `json:"query" description:"検索クエリ"`
}

type searchResult struct {
	Result string `json:"result"`
}

func searchKnowledge(ctx tool.Context, args searchArgs) (searchResult, error) {
	out, err := sharedtool.SearchKnowledge(context.Background(), map[string]any{
		"query": args.Query,
	})
	if err != nil {
		return searchResult{}, err
	}
	r, _ := out["result"].(string)
	return searchResult{Result: r}, nil
}

// buildTools は共有 Tool 定義を ADK 形式に変換する。
func buildTools() ([]tool.Tool, error) {
	calc, err := functiontool.New(functiontool.Config{
		Name:        sharedtool.CalculatorSchema.Name,
		Description: sharedtool.CalculatorSchema.Description,
	}, calculator)
	if err != nil {
		return nil, err
	}

	search, err := functiontool.New(functiontool.Config{
		Name:        sharedtool.SearchKnowledgeSchema.Name,
		Description: sharedtool.SearchKnowledgeSchema.Description,
	}, searchKnowledge)
	if err != nil {
		return nil, err
	}

	return []tool.Tool{calc, search}, nil
}
