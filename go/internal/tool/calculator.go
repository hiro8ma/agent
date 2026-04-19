package tool

import (
	"context"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// CalculatorSchema は calculator Tool 宣言。
var CalculatorSchema = model.ToolSchema{
	Name:        "calculator",
	Description: "四則演算を行う計算ツール。op には add / sub / mul / div を指定する",
	Parameters: model.ParameterSchema{
		Type: "object",
		Properties: map[string]model.PropertySchema{
			"op": {Type: "string", Description: "演算子 (add / sub / mul / div)"},
			"a":  {Type: "number", Description: "左オペランド"},
			"b":  {Type: "number", Description: "右オペランド"},
		},
		Required: []string{"op", "a", "b"},
	},
}

// Calculator は CalculatorSchema の Handler。
func Calculator(ctx context.Context, args map[string]any) (map[string]any, error) {
	op, _ := args["op"].(string)
	a, _ := args["a"].(float64)
	b, _ := args["b"].(float64)

	var result float64
	switch op {
	case "add":
		result = a + b
	case "sub":
		result = a - b
	case "mul":
		result = a * b
	case "div":
		if b == 0 {
			return nil, fmt.Errorf("0 で割ることはできない")
		}
		result = a / b
	default:
		return nil, fmt.Errorf("不明な演算子 %q", op)
	}
	fmt.Printf("  [Tool calculator] %v %s %v = %v\n", a, op, b, result)
	return map[string]any{"result": result}, nil
}
