package mcpguard

import (
	"fmt"
	"path/filepath"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/tool"
)

// PathArgs は場所を受ける引数の名前。公式のファイルシステムの MCP サーバーの引数に合わせる。
var PathArgs = []string{"path", "paths", "source", "destination"}

// Within は p が roots のどれかの中を指すかを返す。相対パスは roots[0] から解決する。
//
// 文字列に ".." を含むかではなく、正規化した結果で判定する。
// 「notes..txt」のような名前は止めず、許可したディレクトリの絶対パスは通す。
// シンボリックリンクの先はクライアントからは分からないので、最後の判定はサーバーに任せる。
func Within(roots []string, p string) bool {
	if len(roots) == 0 || p == "" {
		return false
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(roots[0], p)
	}
	p = filepath.Clean(p)
	for _, root := range roots {
		rel, err := filepath.Rel(filepath.Clean(root), p)
		if err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// PathGuard は場所を受ける引数（配列の中も）が roots の外を指していれば、ツールを実行せずに止める。
func PathGuard(roots []string) llmagent.BeforeToolCallback {
	return func(_ agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
		for _, key := range PathArgs {
			for _, p := range stringsOf(args[key]) {
				if !Within(roots, p) {
					return map[string]any{"error": fmt.Sprintf("%s の %s は許可したディレクトリの外を指している", t.Name(), key)}, nil
				}
			}
		}
		return nil, nil
	}
}

func stringsOf(v any) []string {
	switch x := v.(type) {
	case string:
		return []string{x}
	case []any:
		var out []string
		for _, e := range x {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return x
	}
	return nil
}
