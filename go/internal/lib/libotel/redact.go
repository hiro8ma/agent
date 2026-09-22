package libotel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// ContentKeys は、ADK と Genkit が本文（プロンプト、応答、ツールの引数と結果）を載せる span の属性。
// どちらも取り込みの設定にかかわらず常に載せるので、送る前に落とす。
var ContentKeys = map[attribute.Key]bool{
	"genkit:input":                    true,
	"genkit:output":                   true,
	"gcp.vertex.agent.tool_call_args": true,
	"gcp.vertex.agent.tool_response":  true,
}

// Redact は ContentKeys の属性を落としてから next に渡す exporter を返す。
func Redact(next sdktrace.SpanExporter) sdktrace.SpanExporter {
	return redactor{next: next}
}

type redactor struct {
	next sdktrace.SpanExporter
}

func (r redactor) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	out := make([]sdktrace.ReadOnlySpan, len(spans))
	for i, s := range spans {
		out[i] = redactedSpan{ReadOnlySpan: s}
	}
	return r.next.ExportSpans(ctx, out)
}

func (r redactor) Shutdown(ctx context.Context) error {
	return r.next.Shutdown(ctx)
}

type redactedSpan struct {
	sdktrace.ReadOnlySpan
}

func (s redactedSpan) Attributes() []attribute.KeyValue {
	var kept []attribute.KeyValue
	for _, kv := range s.ReadOnlySpan.Attributes() {
		if !ContentKeys[kv.Key] {
			kept = append(kept, kv)
		}
	}
	return kept
}
