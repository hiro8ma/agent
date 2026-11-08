package crawl_test

import (
	"testing"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

func TestNormalize(t *testing.T) {
	t.Parallel()
	n := crawl.Normalizer{DropParams: []string{"utm_*", "fbclid"}}
	testCases := map[string]struct {
		raw  string
		want string
	}{
		"scheme と host を小文字にする":      {raw: "HTTP://Example.COM/Path", want: "http://example.com/Path"},
		"http の80を外す":                {raw: "http://example.com:80/a", want: "http://example.com/a"},
		"https の443を外す":              {raw: "https://example.com:443/a", want: "https://example.com/a"},
		"既定でないポートは残す":                {raw: "http://example.com:8080/a", want: "http://example.com:8080/a"},
		"https の80は既定でないので残す":        {raw: "https://example.com:80/a", want: "https://example.com:80/a"},
		"# 以降を外す":                    {raw: "http://example.com/a.html#comments", want: "http://example.com/a.html"},
		". と .. の区間を解く":              {raw: "http://example.com/a/./b/../c", want: "http://example.com/a/c"},
		"末尾の .. は / で終える":            {raw: "http://example.com/a/b/..", want: "http://example.com/a/"},
		"根より上の .. は捨てる":              {raw: "http://example.com/../a", want: "http://example.com/a"},
		"空のパスを / にする":                {raw: "http://example.com", want: "http://example.com/"},
		"計測用のパラメータを外す":               {raw: "http://example.com/a?utm_source=news&id=3&utm_medium=mail&fbclid=x", want: "http://example.com/a?id=3"},
		"パラメータがすべて外れたら ? も外す":        {raw: "http://example.com/a?utm_source=news", want: "http://example.com/a"},
		"前方一致でない名前は残す":               {raw: "http://example.com/a?utmost=1", want: "http://example.com/a?utmost=1"},
		"残りのパラメータを名前の順に並べる":          {raw: "http://example.com/a?b=2&a=1&c=3", want: "http://example.com/a?a=1&b=2&c=3"},
		"同じ名前のパラメータは元の順を保つ":          {raw: "http://example.com/a?b=2&a=9&a=1", want: "http://example.com/a?a=9&a=1&b=2"},
		"%XX の表記はそのまま残す":             {raw: "http://example.com/%E3%83%84?q=a%20b", want: "http://example.com/%E3%83%84?q=a%20b"},
		"IPv6 の host は角括弧を保ってポートを外す": {raw: "http://[::1]:80/a", want: "http://[::1]/a"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := n.Normalize(tc.raw)
			if err != nil {
				t.Fatalf("Normalize(%q): %v", tc.raw, err)
			}
			if got != tc.want {
				t.Errorf("Normalize(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestNormalizeRejectsRelative(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		raw string
	}{
		"相対 URL":       {raw: "/articles/1.html"},
		"host の無い URL": {raw: "mailto:info@example.com"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got, err := (crawl.Normalizer{}).Normalize(tc.raw); err == nil {
				t.Errorf("Normalize(%q) = %q, want error", tc.raw, got)
			}
		})
	}
}

func TestNormalizeMergesDuplicates(t *testing.T) {
	t.Parallel()
	n := crawl.Normalizer{DropParams: []string{"utm_*"}}
	variants := []string{
		"http://example.com/articles/1.html",
		"HTTP://EXAMPLE.com:80/articles/1.html",
		"http://example.com/articles/1.html#top",
		"http://example.com/articles/./x/../1.html",
		"http://example.com/articles/1.html?utm_source=news&utm_campaign=fall",
	}
	want, err := n.Normalize(variants[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range variants[1:] {
		got, err := n.Normalize(v)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("Normalize(%q) = %q, want %q", v, got, want)
		}
	}
}
