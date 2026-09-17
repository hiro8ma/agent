// 鍵の保存範囲を宣言し、実際の State と突き合わせる。
//
// 複数のエージェントが同じ State を読み書きするため、
// 誰が何をどの範囲で書くかが散らばる。意図を 1 か所に置く。
package blackboard

import (
	"fmt"
	"sort"
	"strings"

	"google.golang.org/adk/v2/session"
)

// Scope は値をどこまで残すか。
type Scope int

const (
	// Session はそのセッションだけ。接頭辞なし。
	Session Scope = iota
	// Invocation はその 1 回の呼び出しだけ。temp:
	Invocation
	// User は同じ利用者の別セッションでも残る。user:
	User
	// App は全利用者で共有する。app:
	App
)

func (s Scope) prefix() string {
	switch s {
	case Invocation:
		return session.KeyPrefixTemp
	case User:
		return session.KeyPrefixUser
	case App:
		return session.KeyPrefixApp
	default:
		return ""
	}
}

func (s Scope) String() string {
	switch s {
	case Invocation:
		return "呼び出し内"
	case User:
		return "利用者"
	case App:
		return "アプリ全体"
	default:
		return "セッション内"
	}
}

// Schema は、どの名前をどの範囲で持つかの宣言。
//
// 鍵は接頭辞を含まない名前で書く。接頭辞は範囲から導く。
type Schema map[string]Scope

// Key は宣言された範囲に対応する完全な鍵を返す。
// 宣言に無い名前は空文字を返す。既定値へ倒すと誤りが通る。
func (s Schema) Key(name string) string {
	sc, ok := s[name]
	if !ok {
		return ""
	}
	return sc.prefix() + name
}

// MustKey は宣言に無い名前で panic する。
// 宣言漏れを実行時の空データではなく起動時の停止にする。
func (s Schema) MustKey(name string) string {
	k := s.Key(name)
	if k == "" {
		panic(fmt.Sprintf("blackboard: %q が Schema に宣言されていない", name))
	}
	return k
}

// Issue は 1 件の食い違い。
type Issue struct {
	// Key は実際に State にあった鍵。
	Key string
	// Want は宣言された範囲。
	Want Scope
	// Got は実際の鍵から読み取れる範囲。
	Got Scope
	// Reason は何が問題か。
	Reason string
}

func (i Issue) String() string {
	if i.Reason != "" {
		return fmt.Sprintf("%s: %s", i.Key, i.Reason)
	}
	return fmt.Sprintf("%s: 宣言は %s だが実際は %s", i.Key, i.Want, i.Got)
}

// Verify は State の鍵を宣言と突き合わせ、3 種類の食い違いを返す。
//
//	範囲のずれ    付け忘れ（消える）と付けすぎ（漏れる）
//	宣言に無い鍵  宣言せずに書かれた
//	未書き込み    宣言されているが書かれていない
func Verify(st session.State, schema Schema) []Issue {
	actual := map[string]any{}
	for k, v := range st.All() {
		actual[k] = v
	}

	var issues []Issue
	seen := map[string]bool{}

	for key := range actual {
		name, got := split(key)
		want, declared := schema[name]
		if !declared {
			issues = append(issues, Issue{Key: key, Got: got,
				Reason: fmt.Sprintf("宣言に無い（%s で書かれている）", got)})
			continue
		}
		seen[name] = true
		if want != got {
			reason := ""
			switch {
			case want == User && got == Session:
				reason = "user: の付け忘れ。次のセッションで消える"
			case want == App && got == Session:
				reason = "app: の付け忘れ。別のセッションから読めない"
			case want == Session && got == App:
				reason = "app: の付けすぎ。他の利用者から読める"
			case want == User && got == App:
				reason = "app: の付けすぎ。他の利用者から読める"
			}
			issues = append(issues, Issue{Key: key, Want: want, Got: got, Reason: reason})
		}
	}

	for name := range schema {
		if !seen[name] {
			issues = append(issues, Issue{Key: schema.Key(name),
				Reason: "宣言されているが書かれていない"})
		}
	}

	sort.Slice(issues, func(a, b int) bool { return issues[a].Key < issues[b].Key })
	return issues
}

// split は鍵を接頭辞のない名前と範囲へ分ける。
func split(key string) (string, Scope) {
	switch {
	case strings.HasPrefix(key, session.KeyPrefixTemp):
		return strings.TrimPrefix(key, session.KeyPrefixTemp), Invocation
	case strings.HasPrefix(key, session.KeyPrefixUser):
		return strings.TrimPrefix(key, session.KeyPrefixUser), User
	case strings.HasPrefix(key, session.KeyPrefixApp):
		return strings.TrimPrefix(key, session.KeyPrefixApp), App
	default:
		return key, Session
	}
}
