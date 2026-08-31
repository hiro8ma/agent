package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"
)

// config をそのままログに渡してもキーが平文で出ないことを確かめる。
// %+v や logger.Info("config", "cfg", cfg) を書いた瞬間に漏れる事故を防ぐ。
func TestConfigDoesNotLeakAPIKey(t *testing.T) {
	const secret = "EXAMPLE-KEY-MUST-NEVER-APPEAR-IN-LOGS"
	cfg := &config{
		port:         "19910",
		geminiAPIKey: secret,
		defaultModel: "googleai/gemini-3.5-flash",
	}

	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	logger.Info("starting", "cfg", cfg)

	out := buf.String()
	if strings.Contains(out, secret) {
		t.Fatalf("API キーがログに出ている: %s", out)
	}
	if !strings.Contains(out, "EXAM****") {
		t.Fatalf("マスク後の値が見当たらない: %s", out)
	}

	// ログが JSON として壊れていないことも確かめる
	var v map[string]any
	if err := json.Unmarshal(buf.Bytes(), &v); err != nil {
		t.Fatalf("ログが JSON として不正: %v", err)
	}
}

// fmt 経由（%v / %+v / %s）でも漏れないことを確かめる。
func TestConfigStringDoesNotLeakAPIKey(t *testing.T) {
	const secret = "EXAMPLE-KEY-MUST-NEVER-APPEAR-IN-LOGS"
	cfg := &config{geminiAPIKey: secret}

	for _, format := range []string{"%v", "%+v", "%s"} {
		got := sprintf(format, cfg)
		if strings.Contains(got, secret) {
			t.Fatalf("%s で API キーが漏れている: %s", format, got)
		}
	}
}

func sprintf(format string, a ...any) string {
	return fmt.Sprintf(format, a...)
}
