// Package libotel は OpenTelemetry の TracerProvider / MeterProvider / propagator を global に設定する。
//
// 環境変数
//
//	OTEL_TRACES_EXPORTER   otlp / console / none（既定 none）
//	OTEL_METRICS_EXPORTER  otlp / console / none（既定 none）
//	OTEL_EXPORTER_OTLP_ENDPOINT などの OTLP の設定は、各 exporter が標準の名前で読む
//	OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES は Setup に渡した名前より優先する
//	OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT  "1" か "true" のときだけ、プロンプトやツールの入出力を span に残す
package libotel

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime/debug"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutmetric"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
)

// Shutdown は溜めた span とメトリクスを送り切ってから止める。
type Shutdown func(context.Context) error

// Setup は service の名前で global の provider と propagator を設定する。
//
// exporter が none でも SDK の TracerProvider と propagator は設定する。
// 受けた trace を下流へ引き継ぎ、Genkit が自前の provider を作らないようにするため。
func Setup(ctx context.Context, service string) (Shutdown, error) {
	res, err := resource.New(ctx,
		resource.WithAttributes(semconv.ServiceName(service), semconv.ServiceVersion(version())),
		resource.WithFromEnv(),
		resource.WithTelemetrySDK(),
	)
	if err != nil {
		return nil, fmt.Errorf("libotel: resource: %w", err)
	}

	spanExporter, err := newSpanExporter(ctx, os.Getenv("OTEL_TRACES_EXPORTER"))
	if err != nil {
		return nil, err
	}
	traceOpts := []sdktrace.TracerProviderOption{sdktrace.WithResource(res)}
	if spanExporter != nil {
		if !captureContent() {
			spanExporter = Redact(spanExporter)
		}
		traceOpts = append(traceOpts, sdktrace.WithBatcher(spanExporter))
	}
	tp := sdktrace.NewTracerProvider(traceOpts...)

	reader, err := newMetricReader(ctx, os.Getenv("OTEL_METRICS_EXPORTER"))
	if err != nil {
		return nil, errors.Join(err, tp.Shutdown(ctx))
	}
	meterOpts := []sdkmetric.Option{sdkmetric.WithResource(res)}
	if reader != nil {
		meterOpts = append(meterOpts, sdkmetric.WithReader(reader))
	}
	mp := sdkmetric.NewMeterProvider(meterOpts...)

	otel.SetTracerProvider(tp)
	otel.SetMeterProvider(mp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	return func(ctx context.Context) error {
		return errors.Join(tp.Shutdown(ctx), mp.Shutdown(ctx))
	}, nil
}

func newSpanExporter(ctx context.Context, kind string) (sdktrace.SpanExporter, error) {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "", "none":
		return nil, nil
	case "otlp":
		return otlptracehttp.New(ctx)
	case "console":
		return stdouttrace.New()
	default:
		return nil, fmt.Errorf("libotel: OTEL_TRACES_EXPORTER=%q は otlp / console / none のどれか", kind)
	}
}

func newMetricReader(ctx context.Context, kind string) (sdkmetric.Reader, error) {
	var exp sdkmetric.Exporter
	var err error
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "", "none":
		return nil, nil
	case "otlp":
		exp, err = otlpmetrichttp.New(ctx)
	case "console":
		exp, err = stdoutmetric.New()
	default:
		return nil, fmt.Errorf("libotel: OTEL_METRICS_EXPORTER=%q は otlp / console / none のどれか", kind)
	}
	if err != nil {
		return nil, err
	}
	return sdkmetric.NewPeriodicReader(exp), nil
}

func captureContent() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")))
	return v == "1" || v == "true"
}

func version() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" {
		return info.Main.Version
	}
	return "unknown"
}
