package crawl_test

import (
	"testing"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

// rfcExample は RFC 9309 の 5.1 の例。
const rfcExample = `User-Agent: *
Disallow: *.gif$
Disallow: /example/
Allow: /publications/

User-Agent: foobot
Disallow:/
Allow:/example/page.html
Allow:/example/allowed.gif

User-Agent: barbot
User-Agent: bazbot
Disallow: /example/page.html

User-Agent: quxbot

EOF
`

func TestRobotsAllowed(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		robots string
		agent  string
		path   string
		want   bool
	}{
		"該当の無い名前は * のグループで /example/ を禁止":           {robots: rfcExample, agent: "otherbot", path: "/example/a.html", want: false},
		"該当の無い名前は * のグループで /publications/ を許可":      {robots: rfcExample, agent: "otherbot", path: "/publications/a.html", want: true},
		"* と $ で .gif で終わるパスを禁止":                    {robots: rfcExample, agent: "otherbot", path: "/images/a.gif", want: false},
		"$ はパスの終わりにだけ一致し、クエリが続けば許可":                 {robots: rfcExample, agent: "otherbot", path: "/images/a.gif?size=2", want: true},
		"foobot は自分のグループの Disallow: / に従う":          {robots: rfcExample, agent: "foobot", path: "/index.html", want: false},
		"foobot は長い Allow で /example/page.html を許可": {robots: rfcExample, agent: "foobot", path: "/example/page.html", want: true},
		"foobot には * のグループの .gif の禁止を当てない":          {robots: rfcExample, agent: "foobot", path: "/example/allowed.gif", want: true},
		"名前は大文字小文字を区別しない":                           {robots: rfcExample, agent: "FooBot", path: "/example/page.html", want: true},
		"2つの User-agent の行は同じグループ":                  {robots: rfcExample, agent: "bazbot", path: "/example/page.html", want: false},
		"barbot には * のグループの /example/ の禁止を当てない":     {robots: rfcExample, agent: "barbot", path: "/example/other.html", want: true},
		"規則の無いグループに合う名前は * に戻らず全部許可":                {robots: rfcExample, agent: "quxbot", path: "/example/a.html", want: true},
		"最も長い一致の Allow が勝つ": {
			robots: "User-Agent: foobot\nAllow: /example/page/\nDisallow: /example/page/disallowed.gif\n",
			agent:  "foobot", path: "/example/page/", want: true,
		},
		"最も長い一致の Disallow が勝つ": {
			robots: "User-Agent: foobot\nAllow: /example/page/\nDisallow: /example/page/disallowed.gif\n",
			agent:  "foobot", path: "/example/page/disallowed.gif", want: false,
		},
		"短い Disallow が先に書かれていても長い Allow が勝つ": {
			robots: "User-agent: *\nDisallow: /a\nAllow: /a/b\n",
			agent:  "testbot", path: "/a/b/c", want: true,
		},
		"同じ長さなら Disallow が後でも Allow が勝つ": {
			robots: "User-agent: *\nAllow: /page\nDisallow: /page\n",
			agent:  "testbot", path: "/page.html", want: true,
		},
		"同じ長さなら Disallow が先でも Allow が勝つ": {
			robots: "User-agent: *\nDisallow: /page\nAllow: /page\n",
			agent:  "testbot", path: "/page.html", want: true,
		},
		"途中の * は任意の文字列に一致する": {
			robots: "User-agent: *\nDisallow: /private*/secret\n",
			agent:  "testbot", path: "/private-2026/archive/secret.html", want: false,
		},
		"途中の * の後ろが合わなければ許可": {
			robots: "User-agent: *\nDisallow: /private*/secret\n",
			agent:  "testbot", path: "/private-2026/public.html", want: true,
		},
		"$ の前と同じ終わりでなければ許可": {
			robots: "User-agent: *\nDisallow: /*.php$\n",
			agent:  "testbot", path: "/index.php5", want: true,
		},
		"空の Disallow は全部許可": {
			robots: "User-agent: *\nDisallow:\n",
			agent:  "testbot", path: "/anything", want: true,
		},
		"同じ名前の2つのグループは合わせる": {
			robots: "User-agent: testbot\nDisallow: /x\n\nUser-agent: other\nDisallow: /y\n\nUser-agent: testbot\nDisallow: /z\n",
			agent:  "testbot", path: "/z/1.html", want: false,
		},
		"合わせても別の名前のグループの規則は入らない": {
			robots: "User-agent: testbot\nDisallow: /x\n\nUser-agent: other\nDisallow: /y\n\nUser-agent: testbot\nDisallow: /z\n",
			agent:  "testbot", path: "/y/1.html", want: true,
		},
		"User-agent の版の表記は名前の部分で比べる": {
			robots: "User-agent: TestBot/1.2 (+https://example.com/bot)\nDisallow: /\n",
			agent:  "testbot", path: "/a", want: false,
		},
		"行の # 以降は読まない": {
			robots: "User-agent: * # all\nDisallow: /tmp # temp files\n",
			agent:  "testbot", path: "/tmp/a", want: false,
		},
		"最初の User-agent より前の規則は読まない": {
			robots: "Disallow: /\nUser-agent: *\nDisallow: /b\n",
			agent:  "testbot", path: "/a", want: true,
		},
		"ASCII でない規則は %XX にして比べる": {
			robots: "User-agent: *\nDisallow: /foo/bar/ツ\n",
			agent:  "testbot", path: "/foo/bar/%e3%83%84", want: false,
		},
		"/robots.txt は全部禁止でも許可": {
			robots: "User-agent: *\nDisallow: /\n",
			agent:  "testbot", path: "/robots.txt", want: true,
		},
		"グループが無ければ全部許可": {
			robots: "# empty\n",
			agent:  "testbot", path: "/a", want: true,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := crawl.ParseRobots(tc.robots).Allowed(tc.agent, tc.path); got != tc.want {
				t.Errorf("Allowed(%q, %q) = %v, want %v", tc.agent, tc.path, got, tc.want)
			}
		})
	}
}

func TestRobotsFromFetch(t *testing.T) {
	t.Parallel()
	const disallowAll = "User-agent: *\nDisallow: /\n"
	const disallowPrivate = "User-agent: *\nDisallow: /private/\n"
	testCases := map[string]struct {
		status int
		body   string
		path   string
		want   bool
	}{
		"200は本文の規則で禁止":               {status: 200, body: disallowPrivate, path: "/private/a", want: false},
		"200は本文の規則で許可":               {status: 200, body: disallowPrivate, path: "/public/a", want: true},
		"404は本文に禁止があっても全部許可":         {status: 404, body: disallowAll, path: "/private/a", want: true},
		"410は全部許可":                   {status: 410, body: "", path: "/a", want: true},
		"リダイレクトを辿りきれない3xx は全部許可":     {status: 301, body: "", path: "/a", want: true},
		"500は全部禁止":                   {status: 500, body: "", path: "/a", want: false},
		"503は本文があっても全部禁止":            {status: 503, body: "User-agent: *\nAllow: /\n", path: "/a", want: false},
		"到達できないときは全部禁止":              {status: crawl.StatusUnreachable, body: "", path: "/a", want: false},
		"5xx でも robots.txt 自体は取ってよい": {status: 503, body: "", path: "/robots.txt", want: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := crawl.RobotsFromFetch(tc.status, []byte(tc.body)).Allowed("testbot", tc.path); got != tc.want {
				t.Errorf("status %d: Allowed(%q) = %v, want %v", tc.status, tc.path, got, tc.want)
			}
		})
	}
}
