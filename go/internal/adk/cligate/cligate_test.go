package cligate_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/adk/cligate"
)

// fakeGcloud は受け取った引数と環境変数を JSON で出し、呼ばれた印のファイルを作る偽の gcloud を置く。
func fakeGcloud(t *testing.T, body string) (bin, marker string) {
	t.Helper()
	dir := t.TempDir()
	marker = filepath.Join(dir, "called")
	bin = filepath.Join(dir, "gcloud")
	if body == "" {
		body = `printf '%s\n' "$@" > /dev/null; touch "` + marker + `"; printf '{"args":['; first=1; for a in "$@"; do if [ $first = 0 ]; then printf ','; fi; printf '"%s"' "$a"; first=0; done; printf '],"secret":"%s"}' "$SECRET_TOKEN"`
	}
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"+body+"\n"), 0o700); err != nil { //nolint:gosec // テスト用の偽の実行ファイル
		t.Fatal(err)
	}
	return bin, marker
}

func gate(bin string) cligate.Gate {
	return cligate.Gate{
		Binary: bin,
		Specs: []cligate.Spec{
			{Path: []string{"compute", "instances", "list"}, Flags: []string{"--project", "--zones", "--filter", "--limit"}},
			{Path: []string{"storage", "buckets", "list"}, Flags: []string{"--project"}},
		},
		Fixed:     []string{"--format=json"},
		Env:       []string{"PATH=/usr/bin:/bin"},
		Timeout:   5 * time.Second,
		MaxOutput: 1 << 16,
	}
}

func TestAllowedCommandRunsWithFixedFormat(t *testing.T) {
	t.Setenv("SECRET_TOKEN", "should-not-leak")
	bin, _ := fakeGcloud(t, "")
	r, err := gate(bin).Run(t.Context(), []string{"compute", "instances", "list", "--project=p", "--zones=asia-northeast1-a"})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Args   []string `json:"args"`
		Secret string   `json:"secret"`
	}
	if err := json.Unmarshal([]byte(r.Stdout), &got); err != nil {
		t.Fatalf("出力 %q: %v", r.Stdout, err)
	}
	want := []string{"compute", "instances", "list", "--project=p", "--zones=asia-northeast1-a", "--format=json"}
	if !slices.Equal(got.Args, want) {
		t.Errorf("渡った引数 = %v, want %v", got.Args, want)
	}
	if got.Secret != "" {
		t.Errorf("親の環境変数が子に渡った: %q", got.Secret)
	}
}

func TestRejectedCommandsNeverRun(t *testing.T) {
	t.Parallel()
	testCases := map[string][]string{
		"アクセストークンの表示":      {"auth", "print-access-token"},
		"インスタンスの削除":        {"compute", "instances", "delete", "web-1", "--quiet"},
		"list の後ろに位置引数":    {"compute", "instances", "list", "; rm -rf ~"},
		"許可していないフラグでなりすます": {"compute", "instances", "list", "--impersonate-service-account=a@b"},
		"値の無いフラグ":          {"compute", "instances", "list", "--quiet"},
		"固定した出力の形式を上書きする":  {"compute", "instances", "list", "--format=text"},
		"別の経路のフラグを持ち込む":    {"storage", "buckets", "list", "--zones=x"},
		"gcloud の設定を書き換える": {"config", "set", "project", "another"},
		"空":                {},
	}
	for tn, argv := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			bin, marker := fakeGcloud(t, "")
			_, err := gate(bin).Run(t.Context(), argv)
			if !errors.Is(err, cligate.ErrNotAllowed) {
				t.Errorf("err = %v, want ErrNotAllowed", err)
			}
			if _, statErr := os.Stat(marker); statErr == nil {
				t.Error("許可していないコマンドが実行された")
			}
		})
	}
}

func TestTimeoutStopsLongCommands(t *testing.T) {
	t.Parallel()
	bin, _ := fakeGcloud(t, "exec sleep 10")
	g := gate(bin)
	g.Timeout = 200 * time.Millisecond
	start := time.Now()
	_, err := g.Run(t.Context(), []string{"storage", "buckets", "list"})
	if err == nil || time.Since(start) > 5*time.Second {
		t.Errorf("err = %v, 経過 %v", err, time.Since(start))
	}
}

func TestExitCodeIsReturnedNotAsError(t *testing.T) {
	t.Parallel()
	bin, _ := fakeGcloud(t, `echo "permission denied" >&2; exit 1`)
	r, err := gate(bin).Run(t.Context(), []string{"storage", "buckets", "list", "--project=p"})
	if err != nil || r.ExitCode != 1 || r.Stderr == "" {
		t.Errorf("r = %+v, err = %v", r, err)
	}
}
