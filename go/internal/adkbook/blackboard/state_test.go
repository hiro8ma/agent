// Package blackboard は ADK の State をブラックボードとして使う際の
// 保存範囲を扱う。
//
// State の鍵の接頭辞が保存範囲を決める。
//
//	接頭辞なし  そのセッションだけ
//	temp:       その 1 回の呼び出しだけ
//	user:       同じ利用者の別セッションでも残る
//	app:        全利用者・全セッションで共有
//
// 付け忘れると次のセッションで消え、付けすぎると他の利用者へ漏れる。
// どちらも書き込みは成功し、例外は出ない。
package blackboard

import (
	"context"
	"testing"

	"google.golang.org/adk/v2/session"
)

const (
	app  = "blackboard_demo"
	user = "u1"
)

func newSession(t *testing.T, svc session.Service, id string, state map[string]any) session.Session {
	t.Helper()
	res, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: app, UserID: user, SessionID: id, State: state,
	})
	if err != nil {
		t.Fatal(err)
	}
	return res.Session
}

func get(t *testing.T, svc session.Service, id string) map[string]any {
	t.Helper()
	res, err := svc.Get(context.Background(), &session.GetRequest{
		AppName: app, UserID: user, SessionID: id,
	})
	if err != nil {
		t.Fatal(err)
	}
	return flatten(res.Session.State())
}

// flatten は State を素の map へ写す。
func flatten(st session.State) map[string]any {
	out := map[string]any{}
	for k, v := range st.All() {
		out[k] = v
	}
	return out
}

// TestPrefixDecidesScope は接頭辞が保存範囲を決めることを確かめる。
//
// 同じ利用者の 2 つのセッションを作り、
// 1 つ目で 4 種類の鍵を書いて 2 つ目から読む。
func TestPrefixDecidesScope(t *testing.T) {
	svc := session.InMemoryService()

	newSession(t, svc, "s1", map[string]any{
		"finding":      "セッション内だけ",
		"temp:draft":   "この呼び出しだけ",
		"user:profile": "この利用者のどのセッションでも",
		"app:policy":   "全利用者で共有",
	})
	second := newSession(t, svc, "s2", nil)

	st := flatten(second.State())
	t.Logf("2 つ目のセッションから見える鍵:")
	for _, k := range []string{"finding", "temp:draft", "user:profile", "app:policy"} {
		v, ok := st[k]
		t.Logf("  %-14s 見える=%v 値=%v", k, ok, v)
	}

	// 接頭辞なしは越えない。これが落とし穴になる。
	if _, ok := st["finding"]; ok {
		t.Error("接頭辞なしの鍵が別セッションから見えた")
	}
	// user: と app: は越える。
	if v, ok := st["user:profile"]; !ok || v != "この利用者のどのセッションでも" {
		t.Errorf("user: が越えていない: %v", st)
	}
	if v, ok := st["app:policy"]; !ok || v != "全利用者で共有" {
		t.Errorf("app: が越えていない: %v", st)
	}
}

// TestForgettingPrefixFailsSilently は、接頭辞の付け忘れが
// 例外を出さずに保存先だけ変えることを確かめる。
func TestForgettingPrefixFailsSilently(t *testing.T) {
	svc := session.InMemoryService()

	// 意図: 利用者の設定を保存し、次のセッションでも使いたい。
	// 実際: 接頭辞を付け忘れている。
	newSession(t, svc, "write", map[string]any{
		"preference": "日本語で回答",  // user: を付け忘れた
		"user:tier":  "premium", // こちらは正しい
	})

	// 書いた側から読むと、両方見える。ここでは気づけない。
	written := get(t, svc, "write")
	if _, ok := written["preference"]; !ok {
		t.Fatal("書いた側からも見えない。前提が違う")
	}
	t.Log("書いた側からは両方見える。ここでは異常に見えない")

	// 次のセッションで読むと、片方だけ消えている。
	next := flatten(newSession(t, svc, "read", nil).State())
	if _, ok := next["preference"]; ok {
		t.Error("接頭辞なしが越えた。この試験の前提が崩れている")
	}
	if _, ok := next["user:tier"]; !ok {
		t.Error("user: が越えていない")
	}
	t.Logf("次のセッションでは preference が消え、user:tier だけ残る")

}

// TestAppScopeLeaksAcrossUsers は app: が利用者をまたぐことを確かめる。
func TestAppScopeLeaksAcrossUsers(t *testing.T) {
	svc := session.InMemoryService()
	ctx := context.Background()

	// 利用者 A が、本来 user: であるべき値を app: で書いた。
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "userA", SessionID: "a1",
		State: map[string]any{"app:last_query": "A の検索履歴"},
	}); err != nil {
		t.Fatal(err)
	}

	// 別の利用者 B から見える。
	resB, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "userB", SessionID: "b1",
	})
	if err != nil {
		t.Fatal(err)
	}
	v, ok := flatten(resB.Session.State())["app:last_query"]
	if !ok {
		t.Fatal("app: が利用者をまたがなかった。前提が違う")
	}
	t.Logf("利用者 B から利用者 A の値が見える: %v", v)

}
