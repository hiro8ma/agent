package model

// ToolSchema は Tool の宣言。LLM に公開する際の入力スキーマを表す。
type ToolSchema struct {
	Name        string
	Description string
	Parameters  ParameterSchema
}

// ParameterSchema は JSON Schema 相当の簡易表現。
type ParameterSchema struct {
	Type       string
	Properties map[string]PropertySchema
	Required   []string
}

// PropertySchema は個別プロパティの型情報。
type PropertySchema struct {
	Type        string
	Description string
}
