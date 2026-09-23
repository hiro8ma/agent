package liblog_test

import (
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/lib/liblog"
)

func TestMaskSecret(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		input string
		want  string
	}{
		"空は未設定と出す":          {input: "", want: "(未設定)"},
		"4 文字未満は全部伏せる":      {input: "abc", want: "****"},
		"ちょうど 4 文字も全部伏せる":   {input: "abcd", want: "****"},
		"5 文字以上は先頭 4 文字を残す": {input: "fc-EXAMPLE0000000000000000000000000", want: "fc-E****"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := liblog.MaskSecret(tc.input); got != tc.want {
				t.Fatalf("MaskSecret(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestMaskSecretHidesLength(t *testing.T) {
	t.Parallel()
	short := liblog.MaskSecret(strings.Repeat("x", 20))
	long := liblog.MaskSecret(strings.Repeat("x", 200))
	if short != long {
		t.Fatalf("長さが漏れている: %q vs %q", short, long)
	}
}
