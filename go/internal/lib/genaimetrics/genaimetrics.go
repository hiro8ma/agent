// Package genaimetrics は、生成 AI の呼び出しとエージェントの振る舞いを OpenTelemetry のメトリクスで出す。
//
// 名前と属性は OTel の GenAI の規約に寄せる。規約に無いものは gen_ai. の下に足し、下の定数に並べる。
// 属性に利用者の ID、本文、検索語は入れない。
//
// 環境変数（ConfigFromEnv）
//
//	GENAI_PRICES            モデルごとの 100 万トークンあたりの単価（USD）。"model=入力/キャッシュ入力/出力;model2=..."
//	GENAI_DAILY_BUDGET_USD  1 日の予算（USD）。0 なら予算のメトリクスを出さない
package genaimetrics

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	"google.golang.org/genai"
)

const (
	// TokenUsage は規約の gen_ai.client.token.usage。種類は input / output に加えて、その内訳の cached_input / reasoning を出す。
	TokenUsage = "gen_ai.client.token.usage"
	// OperationDuration は規約の gen_ai.client.operation.duration。chat と execute_tool を出す。
	OperationDuration = "gen_ai.client.operation.duration"
	// Cost は単価の表から計算した費用（USD）。
	Cost = "gen_ai.client.cost"
	// CostBudget は 1 日の予算（USD）。
	CostBudget = "gen_ai.client.cost.budget"
	// Retries はモデルの呼び出しの再試行の回数。
	Retries = "gen_ai.client.retries"
	// Invocations はエージェントの 1 回の実行の結果（completed / failed / escalated）ごとの回数。
	Invocations = "gen_ai.agent.invocations"
	// GuardrailBlocks はガードレールが止めた回数。
	GuardrailBlocks = "gen_ai.guardrail.blocks"
)

const (
	// TokenTypeCachedInput は input のうち、キャッシュから読んだ分。
	TokenTypeCachedInput = "cached_input"
	// TokenTypeReasoning は output のうち、思考の分。
	TokenTypeReasoning = "reasoning"

	OutcomeCompleted = "completed"
	OutcomeFailed    = "failed"
	OutcomeEscalated = "escalated"

	ErrorTypeDenied    = "denied"
	ErrorTypeToolError = "tool_error"

	outcomeKey = attribute.Key("gen_ai.agent.outcome")
	stageKey   = attribute.Key("gen_ai.guardrail.stage")
	ruleKey    = attribute.Key("gen_ai.guardrail.rule")
)

// Price は 100 万トークンあたりの単価（USD）。思考のトークンは出力の単価で数える。
type Price struct {
	Input       float64
	CachedInput float64
	Output      float64
}

// Config は単価の表と 1 日の予算。
type Config struct {
	Prices      map[string]Price
	DailyBudget float64
}

// ConfigFromEnv は GENAI_PRICES と GENAI_DAILY_BUDGET_USD を読む。
func ConfigFromEnv() (Config, error) {
	c := Config{Prices: map[string]Price{}}
	for entry := range strings.SplitSeq(os.Getenv("GENAI_PRICES"), ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, values, ok := strings.Cut(entry, "=")
		parts := strings.Split(values, "/")
		if !ok || len(parts) != 3 {
			return Config{}, fmt.Errorf("genaimetrics: GENAI_PRICES の %q は model=入力/キャッシュ入力/出力 の形にする", entry)
		}
		var nums [3]float64
		for i, p := range parts {
			v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil {
				return Config{}, fmt.Errorf("genaimetrics: GENAI_PRICES の %q: %w", entry, err)
			}
			nums[i] = v
		}
		c.Prices[strings.TrimSpace(name)] = Price{Input: nums[0], CachedInput: nums[1], Output: nums[2]}
	}
	if s := os.Getenv("GENAI_DAILY_BUDGET_USD"); s != "" {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return Config{}, fmt.Errorf("genaimetrics: GENAI_DAILY_BUDGET_USD: %w", err)
		}
		c.DailyBudget = v
	}
	return c, nil
}

// Usage は 1 回のモデルの呼び出しのトークン数。Input はキャッシュ分を含み、Output は思考を含む。
type Usage struct {
	Input       int64
	CachedInput int64
	Output      int64
	Reasoning   int64
}

// UsageFrom は genai の使用量を Usage にする。ツールの結果（ToolUsePrompt）は入力に数える。
func UsageFrom(u *genai.GenerateContentResponseUsageMetadata) Usage {
	if u == nil {
		return Usage{}
	}
	return Usage{
		Input:       int64(u.PromptTokenCount) + int64(u.ToolUsePromptTokenCount),
		CachedInput: int64(u.CachedContentTokenCount),
		Output:      int64(u.CandidatesTokenCount) + int64(u.ThoughtsTokenCount),
		Reasoning:   int64(u.ThoughtsTokenCount),
	}
}

// Cost は u を p の単価で計算する。
func (u Usage) Cost(p Price) float64 {
	return (float64(u.Input-u.CachedInput)*p.Input + float64(u.CachedInput)*p.CachedInput + float64(u.Output)*p.Output) / 1e6
}

// Recorder はメトリクスの計器を持つ。
type Recorder struct {
	cfg         Config
	tokens      metric.Int64Histogram
	duration    metric.Float64Histogram
	cost        metric.Float64Counter
	retries     metric.Int64Counter
	invocations metric.Int64Counter
	guardrail   metric.Int64Counter
}

// New は mp の Meter で計器を作る。
func New(mp metric.MeterProvider, cfg Config) (*Recorder, error) {
	m := mp.Meter("genaimetrics")
	r := &Recorder{cfg: cfg}
	var errs []error
	var err error
	r.tokens, err = m.Int64Histogram(TokenUsage, metric.WithUnit("{token}"), metric.WithDescription("モデルの呼び出しごとのトークン数"),
		metric.WithExplicitBucketBoundaries(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576))
	errs = append(errs, err)
	r.duration, err = m.Float64Histogram(OperationDuration, metric.WithUnit("s"), metric.WithDescription("モデルの呼び出しとツールの実行の所要時間"),
		metric.WithExplicitBucketBoundaries(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92))
	errs = append(errs, err)
	r.cost, err = m.Float64Counter(Cost, metric.WithUnit("{USD}"), metric.WithDescription("単価の表から計算した費用"))
	errs = append(errs, err)
	r.retries, err = m.Int64Counter(Retries, metric.WithUnit("{retry}"), metric.WithDescription("モデルの呼び出しの再試行"))
	errs = append(errs, err)
	r.invocations, err = m.Int64Counter(Invocations, metric.WithUnit("{invocation}"), metric.WithDescription("エージェントの実行の結果"))
	errs = append(errs, err)
	r.guardrail, err = m.Int64Counter(GuardrailBlocks, metric.WithUnit("{block}"), metric.WithDescription("ガードレールが止めた回数"))
	errs = append(errs, err)
	if cfg.DailyBudget > 0 {
		_, err = m.Float64ObservableGauge(CostBudget, metric.WithUnit("{USD}"), metric.WithDescription("1 日の予算"),
			metric.WithFloat64Callback(func(_ context.Context, o metric.Float64Observer) error {
				o.Observe(cfg.DailyBudget)
				return nil
			}))
		errs = append(errs, err)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return r, nil
}

// Chat はモデルの呼び出し 1 回を記録する。err が nil でなければ error.type を付け、トークンと費用は数えない。
func (r *Recorder) Chat(ctx context.Context, agentName, model string, d time.Duration, u Usage, err error) {
	base := []attribute.KeyValue{semconv.GenAIOperationNameChat, semconv.GenAIRequestModelKey.String(model), semconv.GenAIAgentNameKey.String(agentName)}
	durAttrs := base
	if err != nil {
		durAttrs = append(durAttrs, semconv.ErrorTypeKey.String(ErrorType(err)))
	}
	r.duration.Record(ctx, d.Seconds(), metric.WithAttributes(durAttrs...))
	if err != nil {
		return
	}
	for typ, n := range map[string]int64{"input": u.Input, "output": u.Output, TokenTypeCachedInput: u.CachedInput, TokenTypeReasoning: u.Reasoning} {
		if n > 0 || typ == "input" || typ == "output" {
			r.tokens.Record(ctx, n, metric.WithAttributes(append(base, semconv.GenAITokenTypeKey.String(typ))...))
		}
	}
	if p, ok := r.cfg.Prices[model]; ok {
		r.cost.Add(ctx, u.Cost(p), metric.WithAttributes(semconv.GenAIRequestModelKey.String(model), semconv.GenAIAgentNameKey.String(agentName)))
	}
}

// Tool はツールの実行 1 回を記録する。errorType は成功なら空、失敗なら ErrorTypeToolError か ErrorTypeDenied。
func (r *Recorder) Tool(ctx context.Context, agentName, tool string, d time.Duration, errorType string) {
	attrs := []attribute.KeyValue{semconv.GenAIOperationNameExecuteTool, semconv.GenAIAgentNameKey.String(agentName), semconv.GenAIToolNameKey.String(tool)}
	if errorType != "" {
		attrs = append(attrs, semconv.ErrorTypeKey.String(errorType))
	}
	r.duration.Record(ctx, d.Seconds(), metric.WithAttributes(attrs...))
}

// Retry はモデルの呼び出しの再試行 1 回を記録する。
func (r *Recorder) Retry(ctx context.Context, model string, err error) {
	r.retries.Add(ctx, 1, metric.WithAttributes(semconv.GenAIRequestModelKey.String(model), semconv.ErrorTypeKey.String(ErrorType(err))))
}

// Invocation はエージェントの実行 1 回の結果を記録する。
func (r *Recorder) Invocation(ctx context.Context, agentName, outcome string) {
	r.invocations.Add(ctx, 1, metric.WithAttributes(semconv.GenAIAgentNameKey.String(agentName), outcomeKey.String(outcome)))
}

// GuardrailBlocked はガードレールが止めた 1 回を記録する。rule は固定の規則名にし、入力の文を入れない。
func (r *Recorder) GuardrailBlocked(ctx context.Context, stage, rule string) {
	r.guardrail.Add(ctx, 1, metric.WithAttributes(stageKey.String(stage), ruleKey.String(rule)))
}

// ErrorType はエラーを error.type の値にする。genai の API のエラーは HTTP のステータス、それ以外は _OTHER。
func ErrorType(err error) string {
	if apiErr, ok := errors.AsType[genai.APIError](err); ok {
		return strconv.Itoa(apiErr.Code)
	}
	if apiErr, ok := errors.AsType[*genai.APIError](err); ok {
		return strconv.Itoa(apiErr.Code)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout"
	}
	return "_OTHER"
}
