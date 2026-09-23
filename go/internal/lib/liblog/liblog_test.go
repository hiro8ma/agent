package liblog_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/trace"

	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
)

var sc = trace.NewSpanContext(trace.SpanContextConfig{
	TraceID:    trace.TraceID{0x4b, 0xf9, 0x2f, 0x35, 0x77, 0xb3, 0x4d, 0xa6, 0xa3, 0xce, 0x92, 0x9d, 0x0e, 0x0e, 0x47, 0x36},
	SpanID:     trace.SpanID{0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7},
	TraceFlags: trace.FlagsSampled,
})

const (
	traceHex = "4bf92f3577b34da6a3ce929d0e0e4736"
	spanHex  = "00f067aa0ba902b7"
)

func decode(t *testing.T, buf *bytes.Buffer) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("JSON ではない: %v: %s", err, buf)
	}
	return m
}

func TestFormats(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		format  liblog.Format
		want    map[string]any
		missing []string
	}{
		"Cloud Logging は severity と message と trace の特別なキーで書く": {
			format: liblog.FormatCloudLogging,
			want: map[string]any{
				"severity":                             "WARNING",
				"message":                              "遅い応答",
				"logging.googleapis.com/trace":         "projects/demo-project/traces/" + traceHex,
				"logging.googleapis.com/spanId":        spanHex,
				"logging.googleapis.com/trace_sampled": true,
				"service":                              "conversation",
				"version":                              "v1.2.3",
			},
			missing: []string{"level", "msg", "trace_id", "source"},
		},
		"Datadog は message と 16 進の trace_id で書く": {
			format:  liblog.FormatDatadog,
			want:    map[string]any{"level": "WARN", "message": "遅い応答", "trace_id": traceHex, "span_id": spanHex},
			missing: []string{"msg", "severity", "logging.googleapis.com/trace"},
		},
		"Plain は slog のキーのまま": {
			format:  liblog.FormatPlain,
			want:    map[string]any{"level": "WARN", "msg": "遅い応答", "trace_id": traceHex},
			missing: []string{"message", "severity"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var buf bytes.Buffer
			l := liblog.New(liblog.WithWriter(&buf), liblog.WithFormat(tc.format),
				liblog.WithProjectID("demo-project"), liblog.WithService("conversation"), liblog.WithVersion("v1.2.3"))
			l.WarnContext(trace.ContextWithSpanContext(t.Context(), sc), "遅い応答")
			got := decode(t, &buf)
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("%s = %v, want %v", k, got[k], v)
				}
			}
			for _, k := range tc.missing {
				if _, ok := got[k]; ok {
					t.Errorf("%s があってはならない: %v", k, got)
				}
			}
		})
	}
}

func TestSeverity(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		level slog.Level
		want  string
	}{
		"DEBUG はそのまま":            {level: slog.LevelDebug, want: "DEBUG"},
		"INFO はそのまま":             {level: slog.LevelInfo, want: "INFO"},
		"WARN は WARNING になる":     {level: slog.LevelWarn, want: "WARNING"},
		"ERROR はそのまま":            {level: slog.LevelError, want: "ERROR"},
		"ERROR より 4 上は CRITICAL": {level: slog.LevelError + 4, want: "CRITICAL"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var buf bytes.Buffer
			l := liblog.New(liblog.WithWriter(&buf), liblog.WithFormat(liblog.FormatCloudLogging), liblog.WithLevel(slog.LevelDebug))
			l.Log(t.Context(), tc.level, "x")
			if got := decode(t, &buf)["severity"]; got != tc.want {
				t.Errorf("severity = %v, want %s", got, tc.want)
			}
		})
	}
}

func TestTraceStaysTopLevelInsideGroup(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	l := liblog.New(liblog.WithWriter(&buf), liblog.WithFormat(liblog.FormatCloudLogging)).
		WithGroup("rpc").With("method", "Chat")
	l.InfoContext(trace.ContextWithSpanContext(t.Context(), sc), "受けた", "status", "ok")
	got := decode(t, &buf)
	if got["logging.googleapis.com/spanId"] != spanHex {
		t.Errorf("trace のキーが最上位に無い: %v", got)
	}
	rpc, _ := got["rpc"].(map[string]any)
	if rpc["method"] != "Chat" || rpc["status"] != "ok" {
		t.Errorf("rpc = %v", got["rpc"])
	}
}

func TestNoTraceWithoutSpan(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	liblog.New(liblog.WithWriter(&buf), liblog.WithFormat(liblog.FormatDatadog)).InfoContext(t.Context(), "x")
	if _, ok := decode(t, &buf)["trace_id"]; ok {
		t.Error("span が無いのに trace_id がある")
	}
}

func TestParseFormat(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		in      string
		want    liblog.Format
		wantErr bool
	}{
		"空は Plain":            {in: "", want: liblog.FormatPlain},
		"gcp は Cloud Logging": {in: "gcp", want: liblog.FormatCloudLogging},
		"大文字の Datadog も読む":    {in: "Datadog", want: liblog.FormatDatadog},
		"知らない名前はエラー":          {in: "splunk", wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := liblog.ParseFormat(tc.in)
			if (err != nil) != tc.wantErr || got != tc.want {
				t.Errorf("ParseFormat(%q) = %v, %v", tc.in, got, err)
			}
		})
	}
}

// 以下はプロセスのロガー（Init）を書き換えるので並列にしない。

func TestPackageFunctionsUseInitLogger(t *testing.T) {
	var buf bytes.Buffer
	liblog.Init(liblog.WithWriter(&buf), liblog.WithFormat(liblog.FormatCloudLogging))
	t.Cleanup(func() { liblog.Init(liblog.WithWriter(&bytes.Buffer{})) })

	ctx := trace.ContextWithSpanContext(context.Background(), sc)
	err := liberrors.Wrap(liberrors.Code("unavailable"), errors.New("dial"), "knowledge に接続できない")
	liblog.Error(ctx, "検索に失敗", err, liblog.NeedAction())

	got := decode(t, &buf)
	if got["severity"] != "ERROR" || got["error_code"] != "unavailable" || got["need_action"] != true {
		t.Errorf("got = %v", got)
	}
	if !strings.Contains(got["error"].(string), "dial") {
		t.Errorf("error = %v", got["error"])
	}
	src, _ := got["logging.googleapis.com/sourceLocation"].(map[string]any)
	if file, _ := src["file"].(string); !strings.HasSuffix(file, "liblog_test.go") {
		t.Errorf("source が呼び出し元を指していない: %v", src)
	}
	if liblog.Logger() != slog.Default() {
		t.Error("Init が slog.Default を設定していない")
	}
}

func TestLevelFiltersPackageFunctions(t *testing.T) {
	var buf bytes.Buffer
	var level slog.LevelVar
	level.Set(slog.LevelWarn)
	liblog.Init(liblog.WithWriter(&buf), liblog.WithLevel(&level))
	t.Cleanup(func() { liblog.Init(liblog.WithWriter(&bytes.Buffer{})) })

	liblog.Info(context.Background(), "書かない")
	if buf.Len() != 0 {
		t.Fatalf("WARN 未満を書いた: %s", buf.String())
	}
	level.Set(slog.LevelDebug)
	liblog.Debug(context.Background(), "書く")
	if !strings.Contains(buf.String(), "書く") {
		t.Errorf("動かしながら下げた重大度が効かない: %s", buf.String())
	}
}

func TestConcurrentInitAndLog(t *testing.T) {
	t.Cleanup(func() { liblog.Init(liblog.WithWriter(&bytes.Buffer{})) })
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				liblog.Init(liblog.WithWriter(&safeDiscard{}))
				liblog.Info(context.Background(), "x")
			}
		})
	}
	wg.Wait()
}

type safeDiscard struct{}

func (safeDiscard) Write(p []byte) (int, error) { return len(p), nil }

func TestInitFromEnv(t *testing.T) {
	testCases := map[string]struct {
		format, level string
		wantErr       bool
	}{
		"Cloud Logging と debug を読む": {format: "cloudlogging", level: "debug"},
		"知らない出力先はエラー":               {format: "splunk", wantErr: true},
		"知らない重大度はエラー":               {format: "plain", level: "verbose", wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Setenv("LOG_FORMAT", tc.format)
			t.Setenv("LOG_LEVEL", tc.level)
			t.Setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
			t.Cleanup(func() { liblog.Init(liblog.WithWriter(&bytes.Buffer{})) })
			l, err := liblog.InitFromEnv("knowledge-server")
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v", err)
			}
			if tc.wantErr {
				return
			}
			if !l.Enabled(t.Context(), slog.LevelDebug) || liblog.Logger() != l {
				t.Error("LOG_LEVEL=debug が効いていない、またはプロセスのロガーになっていない")
			}
		})
	}
}
