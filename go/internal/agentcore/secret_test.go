package agentcore_test

import (
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/agentcore"
)

func TestMaskSecret(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"空", "", "(未設定)"},
		{"短い", "abc", "****"},
		{"境界", "abcd", "****"},
		{"通常", "fc-EXAMPLE0000000000000000000000000", "fc-E****"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := agentcore.MaskSecret(tt.input); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// 伏せ字から元の長さが復元できないことを確かめる。
func TestMaskSecretHidesLength(t *testing.T) {
	short := agentcore.MaskSecret(strings.Repeat("x", 20))
	long := agentcore.MaskSecret(strings.Repeat("x", 200))
	if short != long {
		t.Fatalf("長さが漏れている: %q vs %q", short, long)
	}
}
