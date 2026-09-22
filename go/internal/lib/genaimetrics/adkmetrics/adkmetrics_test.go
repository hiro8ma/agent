package adkmetrics_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/toolgov"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics/adkmetrics"
)

const modelName = "scripted-model"

// scripted は、利用者の文にツール名があればそのツールを 1 回呼び、結果を受けたら答える。
// 「落ちる」を含めば 503 を返す。どの応答にも使用量を付ける。
type scripted struct{}

func (scripted) Name() string { return modelName }

func (scripted) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		last := req.Contents[len(req.Contents)-1]
		usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 1000, CachedContentTokenCount: 400, CandidatesTokenCount: 50, ThoughtsTokenCount: 30}
		if last.Parts[0].FunctionResponse != nil {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("答え", genai.RoleModel), UsageMetadata: usage}, nil)
			return
		}
		text := last.Parts[0].Text
		if strings.Contains(text, "落ちる") {
			yield(nil, genai.APIError{Code: 503, Status: "UNAVAILABLE"})
			return
		}
		for _, name := range []string{"lookup", "broken", "forbidden", "approve"} {
			if strings.Contains(text, name) {
				yield(&model.LLMResponse{
					Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: name, Args: map[string]any{}}}}},
					UsageMetadata: usage,
				}, nil)
				return
			}
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("答え", genai.RoleModel), UsageMetadata: usage}, nil)
	}
}

type empty struct{}

func newTool(t *testing.T, name string, result map[string]any) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name}, func(agent.Context, empty) (map[string]any, error) {
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return tl
}

type harness struct {
	reader *sdkmetric.ManualReader
	runner *runner.Runner
}

func newHarness(t *testing.T, pluginsFirst bool) harness {
	t.Helper()
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	rec, err := genaimetrics.New(mp, genaimetrics.Config{
		Prices:      map[string]genaimetrics.Price{modelName: {Input: 1, CachedInput: 0.1, Output: 10}},
		DailyBudget: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	metrics, err := adkmetrics.Plugin(rec, adkmetrics.WithEscalation(func(_ string, r map[string]any) bool {
		return r["status"] == "pending_approval"
	}))
	if err != nil {
		t.Fatal(err)
	}
	gov, err := toolgov.New(toolgov.Policy{"assistant": {"lookup", "broken", "approve"}}, &toolgov.Memory{})
	if err != nil {
		t.Fatal(err)
	}
	plugins := []*plugin.Plugin{gov, metrics}
	if pluginsFirst {
		plugins = []*plugin.Plugin{metrics, gov}
	}
	ag, err := llmagent.New(llmagent.Config{
		Name: "assistant", Model: scripted{}, Instruction: "x",
		Tools: []tool.Tool{
			newTool(t, "lookup", map[string]any{"hit": 1}),
			newTool(t, "broken", map[string]any{"error": "backend down"}),
			newTool(t, "forbidden", map[string]any{"hit": 1}),
			newTool(t, "approve", map[string]any{"status": "pending_approval"}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: ag, SessionService: session.InMemoryService(), AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatal(err)
	}
	return harness{reader: reader, runner: r}
}

func (h harness) ask(t *testing.T, text string) {
	t.Helper()
	for _, err := range h.runner.Run(t.Context(), "u", "s-"+text, genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
		_ = err
	}
}

func (h harness) collect(t *testing.T) metricdata.ResourceMetrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := h.reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	return rm
}

func find(rm metricdata.ResourceMetrics, name string) metricdata.Aggregation {
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			if m.Name == name {
				return m.Data
			}
		}
	}
	return nil
}

func attr(set attribute.Set, key string) string {
	v, _ := set.Value(attribute.Key(key))
	return v.String()
}

// histCounts は属性の組ごとの記録の回数と合計を返す。
func histCounts[N int64 | float64](data metricdata.Aggregation, keys ...string) map[string][2]float64 {
	out := map[string][2]float64{}
	h, ok := data.(metricdata.Histogram[N])
	if !ok {
		return out
	}
	for _, dp := range h.DataPoints {
		var parts []string
		for _, k := range keys {
			parts = append(parts, attr(dp.Attributes, k))
		}
		k := strings.Join(parts, "|")
		cur := out[k]
		out[k] = [2]float64{cur[0] + float64(dp.Count), cur[1] + float64(dp.Sum)}
	}
	return out
}

func sums[N int64 | float64](data metricdata.Aggregation, keys ...string) map[string]float64 {
	out := map[string]float64{}
	s, ok := data.(metricdata.Sum[N])
	if !ok {
		return out
	}
	for _, dp := range s.DataPoints {
		var parts []string
		for _, k := range keys {
			parts = append(parts, attr(dp.Attributes, k))
		}
		out[strings.Join(parts, "|")] += float64(dp.Value)
	}
	return out
}

func TestTokensAndCost(t *testing.T) {
	t.Parallel()
	h := newHarness(t, false)
	h.ask(t, "こんにちは")
	rm := h.collect(t)

	tokens := histCounts[int64](find(rm, genaimetrics.TokenUsage), "gen_ai.token.type")
	want := map[string]float64{"input": 1000, "output": 80, "cached_input": 400, "reasoning": 30}
	for typ, n := range want {
		if tokens[typ][1] != n {
			t.Errorf("%s = %v, want %v", typ, tokens[typ][1], n)
		}
	}
	cost := sums[float64](find(rm, genaimetrics.Cost), "gen_ai.request.model", "gen_ai.agent.name")
	// (1000-400)*1 + 400*0.1 + 80*10 = 1440 を 100 万で割る。
	if got := cost[modelName+"|assistant"]; got < 0.00143999 || got > 0.00144001 {
		t.Errorf("cost = %v", cost)
	}
	if g, ok := find(rm, genaimetrics.CostBudget).(metricdata.Gauge[float64]); !ok || g.DataPoints[0].Value != 5 {
		t.Errorf("budget = %v", find(rm, genaimetrics.CostBudget))
	}
}

func TestToolOutcomes(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		pluginsFirst bool
	}{
		"toolgov の後に置く": {pluginsFirst: false},
		"toolgov の前に置く": {pluginsFirst: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			h := newHarness(t, tc.pluginsFirst)
			for _, text := range []string{"lookup", "broken", "forbidden", "approve"} {
				h.ask(t, text)
			}
			rm := h.collect(t)
			tools := histCounts[float64](find(rm, genaimetrics.OperationDuration), "gen_ai.operation.name", "gen_ai.tool.name", "error.type")
			want := map[string]float64{
				"execute_tool|lookup|":           1,
				"execute_tool|broken|tool_error": 1,
				"execute_tool|forbidden|denied":  1,
				"execute_tool|approve|":          1,
			}
			for k, n := range want {
				if tools[k][0] != n {
					t.Errorf("%s = %v, want %v（全体 %v）", k, tools[k][0], n, tools)
				}
			}
			inv := sums[int64](find(rm, genaimetrics.Invocations), "gen_ai.agent.name", "gen_ai.agent.outcome")
			if inv["assistant|escalated"] != 1 || inv["assistant|completed"] != 3 {
				t.Errorf("invocations = %v", inv)
			}
		})
	}
}

func TestModelErrorIsCountedOnce(t *testing.T) {
	t.Parallel()
	h := newHarness(t, false)
	h.ask(t, "落ちる")
	rm := h.collect(t)
	chats := histCounts[float64](find(rm, genaimetrics.OperationDuration), "gen_ai.operation.name", "error.type")
	if chats["chat|503"][0] != 1 {
		t.Errorf("chat の失敗 = %v", chats)
	}
	if n := histCounts[int64](find(rm, genaimetrics.TokenUsage), "gen_ai.token.type")["input"][0]; n != 0 {
		t.Errorf("失敗した呼び出しのトークンを数えた: %v", n)
	}
	inv := sums[int64](find(rm, genaimetrics.Invocations), "gen_ai.agent.outcome")
	if inv["failed"] != 1 {
		t.Errorf("invocations = %v", inv)
	}
}

func TestErrorType(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		err  error
		want string
	}{
		"genai の値のエラーはステータス":    {err: genai.APIError{Code: 429}, want: "429"},
		"genai のポインタのエラーもステータス": {err: &genai.APIError{Code: 500}, want: "500"},
		"包まれていても読む":             {err: errors.Join(errors.New("x"), genai.APIError{Code: 503}), want: "503"},
		"時間切れは timeout":         {err: context.DeadlineExceeded, want: "timeout"},
		"それ以外は _OTHER":          {err: errors.New("x"), want: "_OTHER"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := genaimetrics.ErrorType(tc.err); got != tc.want {
				t.Errorf("ErrorType = %s, want %s", got, tc.want)
			}
		})
	}
}
