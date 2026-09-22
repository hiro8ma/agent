package libotel_test

import (
	"slices"
	"testing"

	"go.opentelemetry.io/otel"

	"github.com/hiro8ma/agent/go/internal/lib/libotel"
)

func TestSetup(t *testing.T) {
	testCases := map[string]struct {
		traces, metrics string
		wantErr         bool
	}{
		"既定は送らないが provider と propagator は設定する": {},
		"console は標準出力へ":                       {traces: "console", metrics: "console"},
		"知らない exporter はエラー":                   {traces: "zipkin", wantErr: true},
		"メトリクスの知らない exporter もエラー":             {metrics: "prometheus", wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Setenv("OTEL_TRACES_EXPORTER", tc.traces)
			t.Setenv("OTEL_METRICS_EXPORTER", tc.metrics)
			shutdown, err := libotel.Setup(t.Context(), "test-service")
			if tc.wantErr {
				if err == nil {
					t.Fatal("Setup() error = nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Setup() error = %v", err)
			}
			t.Cleanup(func() { _ = shutdown(t.Context()) })
			fields := otel.GetTextMapPropagator().Fields()
			for _, want := range []string{"traceparent", "baggage"} {
				if !slices.Contains(fields, want) {
					t.Errorf("propagator の fields = %v, %s が無い", fields, want)
				}
			}
		})
	}
}
