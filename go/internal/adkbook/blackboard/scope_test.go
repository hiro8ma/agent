package blackboard

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
)

// バグ調査のブラックボード。教材の例に合わせる。
var debugSchema = Schema{
	"hypothesis":  Session, // この調査の中だけ
	"test_result": Session,
	"draft":       Invocation, // 1 回の呼び出しで捨てる
	"reporter":    User,       // 報告した利用者。別セッションでも残す
	"known_bugs":  App,        // 既知の不具合。全利用者で共有
}

// TestSchemaBuildsKeys は宣言から鍵を組み立てられることを見る。
func TestSchemaBuildsKeys(t *testing.T) {
	cases := map[string]string{
		"hypothesis": "hypothesis",
		"draft":      "temp:draft",
		"reporter":   "user:reporter",
		"known_bugs": "app:known_bugs",
	}
	for name, want := range cases {
		if got := debugSchema.MustKey(name); got != want {
			t.Errorf("%s: %q。%q を期待", name, got, want)
		}
	}

	// 宣言に無い名前は空。既定値へ倒すと誤りが通る。
	if got := debugSchema.Key("undeclared"); got != "" {
		t.Errorf("宣言に無い名前が %q を返した", got)
	}

	defer func() {
		if recover() == nil {
			t.Error("MustKey が宣言漏れで止まらなかった")
		}
	}()
	debugSchema.MustKey("undeclared")
}

func stateWith(t *testing.T, kv map[string]any) session.State {
	t.Helper()
	svc := session.InMemoryService()
	res, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "debug", UserID: "u1", SessionID: "s1", State: kv,
	})
	if err != nil {
		t.Fatal(err)
	}
	return res.Session.State()
}

// TestVerifyCatchesForgottenPrefix は付け忘れを捕まえることを見る。
func TestVerifyCatchesForgottenPrefix(t *testing.T) {
	st := stateWith(t, map[string]any{
		"hypothesis":  "N+1 クエリ",
		"test_result": "再現した",
		"temp:draft":  "書きかけ",
		"reporter":    "tanaka", // user: の付け忘れ
		"known_bugs":  "既知 3 件", // app: の付け忘れ
	})

	issues := Verify(st, debugSchema)
	var msgs []string
	for _, i := range issues {
		msgs = append(msgs, i.String())
		t.Log(" ", i)
	}
	joined := strings.Join(msgs, "\n")

	if !strings.Contains(joined, "user: の付け忘れ") {
		t.Error("user: の付け忘れを検出していない")
	}
	if !strings.Contains(joined, "app: の付け忘れ") {
		t.Error("app: の付け忘れを検出していない")
	}
}

// TestVerifyCatchesOverscopedKey は付けすぎを捕まえることを見る。
func TestVerifyCatchesOverscopedKey(t *testing.T) {
	st := stateWith(t, map[string]any{
		"app:hypothesis": "N+1 クエリ", // 調査ごとの値を全体で共有してしまった
		"test_result":    "再現した",
		"temp:draft":     "書きかけ",
		"app:reporter":   "tanaka", // 報告者が他の利用者から見える
		"app:known_bugs": "既知 3 件",
	})

	var found int
	for _, i := range Verify(st, debugSchema) {
		t.Log(" ", i)
		if strings.Contains(i.Reason, "他の利用者から読める") {
			found++
		}
	}
	if found != 2 {
		t.Errorf("漏れる誤りを %d 件検出。2 件を期待", found)
	}
}

// TestVerifyReportsMissingAndUndeclared は
// 書かれていない鍵と、宣言に無い鍵を報告することを見る。
func TestVerifyReportsMissingAndUndeclared(t *testing.T) {
	st := stateWith(t, map[string]any{
		"hypothesis": "N+1 クエリ",
		"scratch":    "誰かが宣言せずに書いた",
	})

	var missing, undeclared int
	for _, i := range Verify(st, debugSchema) {
		t.Log(" ", i)
		switch {
		case strings.Contains(i.Reason, "書かれていない"):
			missing++
		case strings.Contains(i.Reason, "宣言に無い"):
			undeclared++
		}
	}
	// test_result / draft / reporter / known_bugs の 4 件が未書き込み。
	if missing != 4 {
		t.Errorf("未書き込みを %d 件検出。4 件を期待", missing)
	}
	if undeclared != 1 {
		t.Errorf("宣言外を %d 件検出。1 件を期待", undeclared)
	}
}

// TestVerifyPassesOnCorrectState は正しい State で 0 件になることを見る。
func TestVerifyPassesOnCorrectState(t *testing.T) {
	st := stateWith(t, map[string]any{
		debugSchema.MustKey("hypothesis"):  "N+1 クエリ",
		debugSchema.MustKey("test_result"): "再現した",
		debugSchema.MustKey("draft"):       "書きかけ",
		debugSchema.MustKey("reporter"):    "tanaka",
		debugSchema.MustKey("known_bugs"):  "既知 3 件",
	})
	if issues := Verify(st, debugSchema); len(issues) != 0 {
		for _, i := range issues {
			t.Error(i)
		}
	}
}
