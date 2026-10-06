package wikilint_test

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/wikilint"
)

// newWiki は root の下に files を書き、wiki のページを読み込む。
func newWiki(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for name, body := range files {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func load(t *testing.T, root string) map[string]*wikilint.Page {
	t.Helper()
	pages, err := wikilint.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	return pages
}

func write(t *testing.T, root, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

// stamp は path の出典のハッシュを今の値に書き込む。
func stamp(t *testing.T, root, path string) {
	t.Helper()
	text, err := wikilint.Stamp(root, load(t, root), path)
	if err != nil {
		t.Fatal(err)
	}
	write(t, root, path, text)
}

func kinds(problems []wikilint.Problem) []string {
	var out []string
	for _, p := range problems {
		out = append(out, p.Page+" "+p.Kind)
	}
	slices.Sort(out)
	return out
}

// 返金のページ A は、コードとヘルプ記事を根拠にし、請求書のページ B は A を根拠にする。
func refundWiki(t *testing.T) string {
	t.Helper()
	root := newWiki(t, map[string]string{
		"src/refund.go":   "func Refund() { /* 30 日以内 */ }\n",
		"help/refund.md":  "返金は 30 日以内に申し込めます\n",
		"wiki/refund.md":  "---\ntitle: 返金\nsources:\n  - file: ../src/refund.go\n  - file: ../help/refund.md\n---\n返金は購入から 30 日以内\n",
		"wiki/invoice.md": "---\nsources:\n  - page: wiki/refund.md\n---\n返金したら請求書は取り消される（返金の条件は返金のページ）\n",
	})
	stamp(t, root, "wiki/refund.md")
	stamp(t, root, "wiki/invoice.md")
	return root
}

func TestFreshWikiPasses(t *testing.T) {
	t.Parallel()
	root := refundWiki(t)
	if got := wikilint.Lint(root, load(t, root)); len(got) != 0 {
		t.Fatalf("問題 = %v, want なし", got)
	}
}

func TestStampKeepsOtherFrontmatter(t *testing.T) {
	t.Parallel()
	root := refundWiki(t)
	raw, err := os.ReadFile(filepath.Join(root, "wiki/refund.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "title: 返金\n") {
		t.Errorf("stamp で sources 以外の frontmatter が消えた:\n%s", raw)
	}
}

// TestSourceChangePropagatesOneStepAtATime は、出典の変更が A を古くし、A を直すと今度は B が古くなることを確かめる。
func TestSourceChangePropagatesOneStepAtATime(t *testing.T) {
	t.Parallel()
	root := refundWiki(t)

	// コードが 14 日以内に変わった。LLM が「答えは変わらない」と判断しても、ハッシュは必ず気づく。
	write(t, root, "src/refund.go", "func Refund() { /* 14 日以内 */ }\n")
	if got, want := kinds(wikilint.Lint(root, load(t, root))), []string{"wiki/refund.md stale"}; !slices.Equal(got, want) {
		t.Fatalf("1 段目 = %v, want %v", got, want)
	}
	if got, want := wikilint.Downstream(load(t, root), "wiki/refund.md"), []string{"wiki/invoice.md"}; !slices.Equal(got, want) {
		t.Fatalf("下流 = %v, want %v", got, want)
	}

	// A を読み直して本文を直し、ハッシュを書き込む。A の本文が変わったので、今度は B が古くなる。
	pages := load(t, root)
	write(t, root, "wiki/refund.md", strings.Replace(must(wikilint.Stamp(root, pages, "wiki/refund.md")), "30 日", "14 日", 1))
	stamp(t, root, "wiki/refund.md")
	if got, want := kinds(wikilint.Lint(root, load(t, root))), []string{"wiki/invoice.md stale"}; !slices.Equal(got, want) {
		t.Fatalf("2 段目 = %v, want %v", got, want)
	}

	// B も読み直してハッシュを書き込めば、古いページはなくなる。
	stamp(t, root, "wiki/invoice.md")
	if got := wikilint.Lint(root, load(t, root)); len(got) != 0 {
		t.Fatalf("3 段目 = %v, want なし", got)
	}
}

// TestRestampingUpstreamDoesNotStaleDownstream は、出典が変わっても答えが変わらず、出典の欄のハッシュだけを書き直した場合、下流が古くならないことを確かめる。本文のハッシュで結んでいるため。
func TestRestampingUpstreamDoesNotStaleDownstream(t *testing.T) {
	t.Parallel()
	root := refundWiki(t)
	// コードのコメントだけが変わった。A を読み直して答えは変わらないと確かめ、ハッシュだけを書き直す。
	write(t, root, "src/refund.go", "func Refund() { /* 30 日以内（整理） */ }\n")
	stamp(t, root, "wiki/refund.md")
	if got := wikilint.Lint(root, load(t, root)); len(got) != 0 {
		t.Fatalf("問題 = %v, want なし", got)
	}
}

func TestLintProblems(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		page string
		want []string
	}{
		"一時置き場の回答を根拠にするとエラー": {
			page: "---\nsources:\n  - file: ../inbox/answer-001.md\n    sha256: x\n---\n本文\n",
			want: []string{"wiki/p.md inbox"},
		},
		"出典が無いとエラー": {
			page: "---\nsources:\n  - file: ../src/gone.go\n    sha256: x\n---\n本文\n",
			want: []string{"wiki/p.md missing"},
		},
		"ハッシュを書いていないとエラー": {
			page: "---\nsources:\n  - file: ../src/refund.go\n---\n本文\n",
			want: []string{"wiki/p.md unhashed"},
		},
		"出典なしのページは問題なし": {
			page: "本文だけ\n",
			want: nil,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			root := newWiki(t, map[string]string{
				"src/refund.go":       "code\n",
				"inbox/answer-001.md": "確認前の回答\n",
				"wiki/p.md":           tc.page,
			})
			if got := kinds(wikilint.Lint(root, load(t, root))); !slices.Equal(got, tc.want) {
				t.Errorf("問題 = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestInboxPagesAreNotLoaded(t *testing.T) {
	t.Parallel()
	root := newWiki(t, map[string]string{
		"inbox/answer-001.md": "---\nsources:\n  - file: ../nowhere.go\n---\n確認前\n",
		"wiki/p.md":           "本文\n",
	})
	pages := load(t, root)
	if _, ok := pages["inbox/answer-001.md"]; ok {
		t.Error("一時置き場のファイルをページとして読んだ")
	}
	if got := wikilint.Lint(root, pages); len(got) != 0 {
		t.Errorf("問題 = %v, want なし（一時置き場は lint の対象外）", got)
	}
}

func must(s string, err error) string {
	if err != nil {
		panic(err)
	}
	return s
}
