package crawl

import (
	"fmt"
	"net/url"
	"slices"
	"strings"
)

var defaultPorts = map[string]string{"http": "80", "https": "443"}

// Normalizer は重複の判定に使う URL の正規形を作る。
// DropParams は外す計測用のパラメータの名前で、末尾の * は前方一致にする（utm_* など）。
type Normalizer struct {
	DropParams []string
}

// Normalize は scheme と host を小文字にし、既定のポートと # 以降を外し、. と .. の区間を解き、DropParams を外して残りのパラメータを名前の順に並べる。
func (n Normalizer) Normalize(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if !u.IsAbs() || u.Host == "" {
		return "", fmt.Errorf("crawl: 絶対 URL ではない: %q", raw)
	}
	scheme := strings.ToLower(u.Scheme)
	host := strings.ToLower(u.Hostname())
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	if port := u.Port(); port != "" && port != defaultPorts[scheme] {
		host += ":" + port
	}
	path := removeDotSegments(u.EscapedPath())
	if path == "" {
		path = "/"
	}
	var b strings.Builder
	b.WriteString(scheme + "://")
	if u.User != nil {
		b.WriteString(u.User.String() + "@")
	}
	b.WriteString(host + path)
	if q := n.query(u.RawQuery); q != "" {
		b.WriteString("?" + q)
	}
	return b.String(), nil
}

// query は同じ名前のパラメータの順を保ったまま、名前の順に並べる。値の表記はそのまま残す。
func (n Normalizer) query(raw string) string {
	type param struct{ key, raw string }
	var params []param
	for piece := range strings.SplitSeq(raw, "&") {
		if piece == "" {
			continue
		}
		key, _, _ := strings.Cut(piece, "=")
		if k, err := url.QueryUnescape(key); err == nil {
			key = k
		}
		if n.drop(key) {
			continue
		}
		params = append(params, param{key: key, raw: piece})
	}
	slices.SortStableFunc(params, func(a, b param) int { return strings.Compare(a.key, b.key) })
	out := make([]string, len(params))
	for i, p := range params {
		out[i] = p.raw
	}
	return strings.Join(out, "&")
}

func (n Normalizer) drop(key string) bool {
	for _, p := range n.DropParams {
		if prefix, ok := strings.CutSuffix(p, "*"); ok {
			if strings.HasPrefix(key, prefix) {
				return true
			}
		} else if key == p {
			return true
		}
	}
	return false
}

// removeDotSegments は RFC 3986 の 5.2.4 の手順で . と .. の区間を解く。path.Clean と違い、末尾の / と空の区間は残す。
func removeDotSegments(in string) string {
	var out strings.Builder
	dropLast := func() {
		s := out.String()
		out.Reset()
		if i := strings.LastIndexByte(s, '/'); i >= 0 {
			out.WriteString(s[:i])
		}
	}
	for in != "" {
		switch {
		case strings.HasPrefix(in, "../"):
			in = in[3:]
		case strings.HasPrefix(in, "./"):
			in = in[2:]
		case strings.HasPrefix(in, "/./"):
			in = in[2:]
		case in == "/.":
			in = "/"
		case strings.HasPrefix(in, "/../"):
			in = in[3:]
			dropLast()
		case in == "/..":
			in = "/"
			dropLast()
		case in == "." || in == "..":
			in = ""
		default:
			i := strings.IndexByte(in[1:], '/')
			if i < 0 {
				out.WriteString(in)
				in = ""
			} else {
				out.WriteString(in[:i+1])
				in = in[i+1:]
			}
		}
	}
	return out.String()
}
