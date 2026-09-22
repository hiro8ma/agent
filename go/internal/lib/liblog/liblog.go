// Package liblog は、プロセスで 1 つの slog のロガーを持ち、context の trace を付けて JSON で書く。
//
// 出力先（Cloud Logging / Datadog）に合わせてキーの名前を変える。trace の ID は OTel の 16 進のまま出す。
package liblog

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel/trace"

	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// Format はログの受け手の種類。
type Format int

const (
	// FormatPlain は slog の JSON のキーのまま書く。
	FormatPlain Format = iota
	// FormatCloudLogging は Cloud Logging の構造化ログの特別なキーで書く。
	FormatCloudLogging
	// FormatDatadog は Datadog が属性として読むキーで書く。
	FormatDatadog
)

// ParseFormat は環境変数などの文字列から Format を返す。空なら FormatPlain。
func ParseFormat(s string) (Format, error) {
	switch strings.ToLower(s) {
	case "", "plain":
		return FormatPlain, nil
	case "cloudlogging", "gcp":
		return FormatCloudLogging, nil
	case "datadog":
		return FormatDatadog, nil
	}
	return FormatPlain, errors.New("liblog: unknown format " + s)
}

type config struct {
	level     slog.Leveler
	format    Format
	projectID string
	service   string
	version   string
	w         io.Writer
}

// Option はロガーの設定。
type Option func(*config)

// WithLevel は書く最低の重大度。slog.LevelVar を渡すと動かしながら変えられる。
func WithLevel(l slog.Leveler) Option { return func(c *config) { c.level = l } }

// WithFormat は受け手の種類。
func WithFormat(f Format) Option { return func(c *config) { c.format = f } }

// WithProjectID は Cloud Logging の trace のキーに付ける GCP のプロジェクト ID。
func WithProjectID(id string) Option { return func(c *config) { c.projectID = id } }

// WithService は全てのログに付けるサービス名。
func WithService(name string) Option { return func(c *config) { c.service = name } }

// WithVersion は全てのログに付けるデプロイの版。
func WithVersion(v string) Option { return func(c *config) { c.version = v } }

// WithWriter は書き込み先。既定は標準出力。
func WithWriter(w io.Writer) Option { return func(c *config) { c.w = w } }

var current atomic.Pointer[slog.Logger]

// Init はプロセスのロガーを作り直し、slog.Default にも設定する。起動時に 1 度呼ぶ。
func Init(opts ...Option) *slog.Logger {
	l := New(opts...)
	current.Store(l)
	slog.SetDefault(l)
	return l
}

// InitFromEnv は環境変数から設定を読んで Init する。
//
//	LOG_FORMAT            plain / cloudlogging / datadog（既定 plain）
//	LOG_LEVEL             debug / info / warn / error（既定 info）
//	GOOGLE_CLOUD_PROJECT  Cloud Logging の trace のキーに付けるプロジェクト ID
func InitFromEnv(service string) (*slog.Logger, error) {
	format, err := ParseFormat(os.Getenv("LOG_FORMAT"))
	if err != nil {
		return nil, err
	}
	level := slog.LevelInfo
	if s := os.Getenv("LOG_LEVEL"); s != "" {
		if err := level.UnmarshalText([]byte(s)); err != nil {
			return nil, err
		}
	}
	return Init(
		WithFormat(format),
		WithLevel(level),
		WithProjectID(os.Getenv("GOOGLE_CLOUD_PROJECT")),
		WithService(service),
		WithVersion(buildVersion()),
	), nil
}

func buildVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		return info.Main.Version
	}
	return ""
}

// Logger はプロセスのロガーを返す。Init の前は slog.Default を返す。
func Logger() *slog.Logger {
	if l := current.Load(); l != nil {
		return l
	}
	return slog.Default()
}

// New は Init と同じ設定のロガーを作るが、プロセスのロガーは変えない。
func New(opts ...Option) *slog.Logger {
	c := config{level: slog.LevelInfo, w: os.Stdout}
	for _, opt := range opts {
		opt(&c)
	}
	base := slog.NewJSONHandler(c.w, &slog.HandlerOptions{
		AddSource:   true,
		Level:       c.level,
		ReplaceAttr: replacer(c.format),
	})
	var attrs []slog.Attr
	if c.service != "" {
		attrs = append(attrs, slog.String("service", c.service))
	}
	if c.version != "" {
		attrs = append(attrs, slog.String("version", c.version))
	}
	return slog.New(&traceHandler{base: base.WithAttrs(attrs), format: c.format, projectID: c.projectID})
}

// Debug は context の trace を付けて DEBUG で書く。
func Debug(ctx context.Context, msg string, args ...any) { write(ctx, slog.LevelDebug, msg, args) }

// Info は context の trace を付けて INFO で書く。
func Info(ctx context.Context, msg string, args ...any) { write(ctx, slog.LevelInfo, msg, args) }

// Warn は context の trace を付けて WARN で書く。
func Warn(ctx context.Context, msg string, args ...any) { write(ctx, slog.LevelWarn, msg, args) }

// Error は err の内容と、liberrors のコードがあればそれを付けて ERROR で書く。
func Error(ctx context.Context, msg string, err error, args ...any) {
	if err != nil {
		args = append(args, slog.String("error", err.Error()))
		if le, ok := errors.AsType[*liberrors.Error](err); ok {
			args = append(args, slog.String("error_code", string(le.Code)))
		}
	}
	write(ctx, slog.LevelError, msg, args)
}

// NeedAction は、人が対応しなければならないログに付ける印。アラートの条件に使う。
func NeedAction() slog.Attr { return slog.Bool("need_action", true) }

func write(ctx context.Context, level slog.Level, msg string, args []any) {
	l := Logger()
	if !l.Enabled(ctx, level) {
		return
	}
	var pcs [1]uintptr
	// write と、それを呼んだ Info などの 2 段を飛ばし、呼び出し元の位置を source にする。
	runtime.Callers(3, pcs[:])
	r := slog.NewRecord(time.Now(), level, msg, pcs[0])
	r.Add(args...)
	_ = l.Handler().Handle(ctx, r)
}

// traceHandler は context の span から trace のキーを最上位に足す。
//
// WithGroup の後に足すと group の中に入ってしまうので、WithAttrs と WithGroup は記録しておき、
// trace のキーを足した後で適用し直す。
type traceHandler struct {
	base      slog.Handler
	ops       []func(slog.Handler) slog.Handler
	format    Format
	projectID string
}

func (h *traceHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.base.Enabled(ctx, level)
}

func (h *traceHandler) Handle(ctx context.Context, r slog.Record) error {
	next := h.base
	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		next = next.WithAttrs(h.traceAttrs(sc))
	}
	for _, op := range h.ops {
		next = op(next)
	}
	return next.Handle(ctx, r)
}

func (h *traceHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return h.with(func(n slog.Handler) slog.Handler { return n.WithAttrs(attrs) })
}

func (h *traceHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	return h.with(func(n slog.Handler) slog.Handler { return n.WithGroup(name) })
}

func (h *traceHandler) with(op func(slog.Handler) slog.Handler) *traceHandler {
	c := *h
	c.ops = append(append([]func(slog.Handler) slog.Handler(nil), h.ops...), op)
	return &c
}

func (h *traceHandler) traceAttrs(sc trace.SpanContext) []slog.Attr {
	traceID, spanID := sc.TraceID().String(), sc.SpanID().String()
	if h.format != FormatCloudLogging {
		return []slog.Attr{slog.String("trace_id", traceID), slog.String("span_id", spanID)}
	}
	if h.projectID != "" {
		traceID = "projects/" + h.projectID + "/traces/" + traceID
	}
	return []slog.Attr{
		slog.String("logging.googleapis.com/trace", traceID),
		slog.String("logging.googleapis.com/spanId", spanID),
		slog.Bool("logging.googleapis.com/trace_sampled", sc.IsSampled()),
	}
}

func replacer(f Format) func([]string, slog.Attr) slog.Attr {
	return func(groups []string, a slog.Attr) slog.Attr {
		if len(groups) > 0 || f == FormatPlain {
			return a
		}
		switch a.Key {
		case slog.MessageKey:
			a.Key = "message"
		case slog.LevelKey:
			if f == FormatCloudLogging {
				return slog.String("severity", severity(a.Value.Any().(slog.Level)))
			}
		case slog.SourceKey:
			if f == FormatCloudLogging {
				a.Key = "logging.googleapis.com/sourceLocation"
			}
		}
		return a
	}
}

// severity は slog の重大度を Cloud Logging の LogSeverity の名前にする。WARN は WARNING になる。
func severity(l slog.Level) string {
	switch {
	case l >= slog.LevelError+4:
		return "CRITICAL"
	case l >= slog.LevelError:
		return "ERROR"
	case l >= slog.LevelWarn:
		return "WARNING"
	case l >= slog.LevelInfo:
		return "INFO"
	default:
		return "DEBUG"
	}
}
