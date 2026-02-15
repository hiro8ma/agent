package genkit

// CalcInput は計算ツールの入力スキーマ
type CalcInput struct {
	A  float64 `json:"a" jsonschema:"description=最初の数値"`
	B  float64 `json:"b" jsonschema:"description=2番目の数値"`
	Op string  `json:"op" jsonschema:"description=演算子: add, sub, mul, div"`
}

// CalcOutput は計算ツールの出力スキーマ
type CalcOutput struct {
	Result float64 `json:"result"`
}

// JokeInput はジョークFlowの入力
type JokeInput struct {
	Topic string `json:"topic"`
}

// JokeOutput はジョークFlowの出力（構造化出力）
type JokeOutput struct {
	Setup     string `json:"setup" jsonschema:"description=ジョークの前振り"`
	Punchline string `json:"punchline" jsonschema:"description=ジョークのオチ"`
	Rating    int    `json:"rating" jsonschema:"description=面白さ評価 1-10"`
}

// SentimentOutput は感情分析の構造化出力
type SentimentOutput struct {
	Sentiment  string   `json:"sentiment" jsonschema:"description=感情: positive, negative, neutral"`
	Confidence float64  `json:"confidence" jsonschema:"description=信頼度 0.0-1.0"`
	Keywords   []string `json:"keywords" jsonschema:"description=重要なキーワード"`
}
