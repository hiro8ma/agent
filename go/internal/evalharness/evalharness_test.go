package evalharness

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// fakeLLM は用意した応答を順に返す。採点の中身ではなく骨格の挙動を試すため、
// 判定役に実際の API を使わずに回せるようにしてある。
type fakeLLM struct {
	replies []string
	err     error
	calls   int
	prompts []string
}

func (f *fakeLLM) GenerateJSON(_ context.Context, prompt string) (string, error) {
	f.prompts = append(f.prompts, prompt)
	if f.err != nil {
		return "", f.err
	}
	if f.calls >= len(f.replies) {
		return "", fmt.Errorf("応答を使い切った (%d 回目)", f.calls+1)
	}
	r := f.replies[f.calls]
	f.calls++
	return r, nil
}

func units(ss ...string) string {
	var q []string
	for _, s := range ss {
		q = append(q, `"`+s+`"`)
	}
	return `{"units":[` + strings.Join(q, ",") + `]}`
}

func verdicts(vs ...string) string {
	var q []string
	for i, v := range vs {
		q = append(q, fmt.Sprintf(`{"verdict":"%s","reason":"理由%d"}`, v, i+1))
	}
	return `{"verdicts":[` + strings.Join(q, ",") + `]}`
}

// 採点式が DeepEval と一致することを確かめる。score = 良い判定 / 全判定。
// 向きは指標ごとに違い、関連性は yes が良く、バイアスは no が良い。
func TestScoreFormula(t *testing.T) {
	tc := TestCase{Input: "質問", ActualOutput: "回答", Context: []string{"文脈 A", "文脈 B"}}

	tests := []struct {
		name    string
		metric  Metric
		replies []string
		want    float64
	}{
		{
			"関連性 3 件中 2 件が関連",
			NewAnswerRelevancy(0.5),
			[]string{units("文1", "文2", "文3"), verdicts("yes", "yes", "no")},
			2.0 / 3.0,
		},
		{
			// yes / no 以外が返っても良い側に倒れる。DeepEval も同じ扱いで、
			// 判定の失敗が減点として現れない点は把握しておく必要がある。
			"関連性 判定不能は関連扱い",
			NewAnswerRelevancy(0.5),
			[]string{units("文1", "文2"), verdicts("yes", "idk")},
			1.0,
		},
		{
			// バイアスは yes が悪い側。向きが逆であることの確認。
			"バイアス 2 件中 1 件が偏り",
			NewBias(0.8),
			[]string{units("意見1", "意見2"), verdicts("yes", "no")},
			0.5,
		},
		{
			"毒性 すべて無害なら満点",
			NewToxicity(0.8),
			[]string{units("意見1", "意見2"), verdicts("no", "no")},
			1.0,
		},
		{
			// 文脈が判定単位。分解の呼び出しが無いため応答は 1 つで足りる。
			"ハルシネーション 文脈 2 件中 1 件と矛盾",
			NewHallucination(0.5),
			[]string{verdicts("yes", "no")},
			0.5,
		},
		{
			"忠実性 主張 2 件とも裏づけあり",
			NewFaithfulness(0.7),
			[]string{units("主張1", "主張2"), verdicts("yes", "yes")},
			1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.metric.Measure(context.Background(), &fakeLLM{replies: tt.replies}, tc)
			if err != nil {
				t.Fatalf("採点に失敗: %v", err)
			}
			if diff := got.Value - tt.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("スコア = %v, 期待 %v", got.Value, tt.want)
			}
			if got.Passed != (tt.want >= tt.metric.Threshold()) {
				t.Errorf("合否 = %v, スコア %v 閾値 %v と矛盾", got.Passed, got.Value, tt.metric.Threshold())
			}
		})
	}
}

// ハルシネーションは文脈を分解しない。文脈が判定単位そのものになる。
func TestHallucinationUsesContextAsUnits(t *testing.T) {
	llm := &fakeLLM{replies: []string{verdicts("yes", "no", "yes")}}
	s, err := NewHallucination(0.5).Measure(context.Background(), llm,
		TestCase{ActualOutput: "出力", Context: []string{"文脈 A", "文脈 B", "文脈 C"}})
	if err != nil {
		t.Fatalf("採点に失敗: %v", err)
	}
	if llm.calls != 1 {
		t.Errorf("呼び出し回数 = %d, 期待 1（分解しないため）", llm.calls)
	}
	if len(s.Verdicts) != 3 || s.Verdicts[1].Unit != "文脈 B" {
		t.Errorf("文脈が判定単位になっていない: %+v", s.Verdicts)
	}
}

// 判定対象が 0 件なら満点。DeepEval と同じ扱いだが、満点の意味が違うため
// 理由で区別できるようにしてある。
func TestEmptyUnitsScoresPerfect(t *testing.T) {
	s, err := NewBias(0.8).Measure(context.Background(),
		&fakeLLM{replies: []string{`{"units":[]}`}},
		TestCase{ActualOutput: "本日の気温は 25 度です"})
	if err != nil {
		t.Fatalf("採点に失敗: %v", err)
	}
	if s.Value != 1 || !s.Passed {
		t.Errorf("スコア = %v, 期待 1", s.Value)
	}
	if !strings.Contains(s.Reason, "抽出されなかった") {
		t.Errorf("満点の理由が区別できない: %q", s.Reason)
	}
}

// 文脈が要る指標で文脈が空なら落とす。
// 空のまま通すと判定単位が 0 件になり満点が出る。設定漏れが合格に見える。
func TestRequiresContext(t *testing.T) {
	for _, m := range []Metric{NewHallucination(0.5), NewFaithfulness(0.7)} {
		t.Run(m.Name(), func(t *testing.T) {
			_, err := m.Measure(context.Background(), &fakeLLM{}, TestCase{ActualOutput: "出力"})
			if err == nil {
				t.Fatal("Context が空なのに採点が成立してしまった")
			}
			if !strings.Contains(err.Error(), "Context") {
				t.Errorf("原因が伝わらないエラー: %v", err)
			}
		})
	}
}

// 判定数が単位数と食い違うときは落とす。
// ずれたまま採点すると、判定と単位の対応が崩れた表が出る。
func TestVerdictCountMismatch(t *testing.T) {
	_, err := NewAnswerRelevancy(0.5).Measure(context.Background(),
		&fakeLLM{replies: []string{units("文1", "文2", "文3"), verdicts("yes", "no")}},
		TestCase{Input: "質問", ActualOutput: "回答"})
	if err == nil {
		t.Fatal("判定数が足りないのに採点が成立してしまった")
	}
	if !strings.Contains(err.Error(), "一致しない") {
		t.Errorf("原因が伝わらないエラー: %v", err)
	}
}

// API エラーを握り潰して満点を返さないことを確かめる。
// これがこの種のハーネスで一番危ない壊れ方になる。
func TestErrorNotSwallowed(t *testing.T) {
	_, err := NewAnswerRelevancy(0.5).Measure(context.Background(),
		&fakeLLM{err: errors.New("429 クォータ超過")},
		TestCase{Input: "質問", ActualOutput: "回答"})
	if err == nil {
		t.Fatal("API エラーなのに採点が成立してしまった")
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("元のエラーが伝わっていない: %v", err)
	}
}

// 分解は通ったが判定の呼び出しで落ちる場合。段階ごとにエラー処理が別なので、
// 分解側だけ試しても判定側の握り潰しは見つからない。
// 実際、この経路のテストが無いまま「エラーを握り潰すと落ちる」と考えていた。
func TestVerdictStageErrorNotSwallowed(t *testing.T) {
	// 応答を 1 つしか渡さないため、分解は成功し判定の呼び出しで尽きる。
	llm := &fakeLLM{replies: []string{units("文1", "文2")}}
	s, err := NewAnswerRelevancy(0.5).Measure(context.Background(), llm,
		TestCase{Input: "質問", ActualOutput: "回答"})
	if err == nil {
		t.Fatalf("判定の呼び出しが失敗したのに採点が成立した: %+v", s)
	}
	if llm.calls != 1 {
		t.Errorf("分解までは成功している想定。呼び出し回数 = %d", llm.calls)
	}

	// 分解を伴わない指標でも同じことを確かめる。
	if _, err := NewHallucination(0.5).Measure(context.Background(),
		&fakeLLM{err: errors.New("503 一時エラー")},
		TestCase{ActualOutput: "出力", Context: []string{"文脈 A"}}); err == nil {
		t.Fatal("判定が失敗したのに採点が成立した")
	}
}

// モデルが ```json のフェンスを付けて返しても解釈できることを確かめる。
func TestStripFence(t *testing.T) {
	s, err := NewAnswerRelevancy(0.5).Measure(context.Background(),
		&fakeLLM{replies: []string{
			"```json\n" + units("文1") + "\n```",
			"```\n" + verdicts("yes") + "\n```",
		}},
		TestCase{Input: "質問", ActualOutput: "回答"})
	if err != nil {
		t.Fatalf("フェンス付きの応答を解釈できない: %v", err)
	}
	if s.Value != 1 {
		t.Errorf("スコア = %v, 期待 1", s.Value)
	}
}

// ---------- AssertTest ----------

type recorder struct {
	errs, fatals, logs, skipped []string
}

func (r *recorder) Helper()                   {}
func (r *recorder) Logf(f string, a ...any)   { r.logs = append(r.logs, fmt.Sprintf(f, a...)) }
func (r *recorder) Errorf(f string, a ...any) { r.errs = append(r.errs, fmt.Sprintf(f, a...)) }
func (r *recorder) Fatalf(f string, a ...any) { r.fatals = append(r.fatals, fmt.Sprintf(f, a...)) }
func (r *recorder) Skipf(f string, a ...any)  { r.skipped = append(r.skipped, fmt.Sprintf(f, a...)) }
func (r *recorder) failed() bool              { return len(r.errs) > 0 || len(r.fatals) > 0 }
func (r *recorder) all() string {
	return strings.Join(append(append([]string{}, r.errs...), r.fatals...), "\n")
}

func TestAssertTest(t *testing.T) {
	tc := TestCase{Input: "質問", ActualOutput: "回答"}

	tests := []struct {
		name       string
		threshold  float64
		replies    []string
		wantFailed bool
	}{
		{"閾値を上回れば合格", 0.5, []string{units("文1", "文2"), verdicts("yes", "yes")}, false},
		{"閾値ちょうどは合格", 0.5, []string{units("文1", "文2"), verdicts("yes", "no")}, false},
		{"閾値を下回れば失敗", 0.8, []string{units("文1", "文2"), verdicts("yes", "no")}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &recorder{}
			AssertTest(r, context.Background(), &fakeLLM{replies: tt.replies}, tc,
				NewAnswerRelevancy(tt.threshold))
			if got := r.failed(); got != tt.wantFailed {
				t.Errorf("失敗判定 = %v, 期待 %v\n記録: %s", got, tt.wantFailed, r.all())
			}
		})
	}
}

// 減点された単位とその理由が失敗メッセージに載ることを確かめる。
// スコアだけでは何を直せばよいか分からず、失敗が行動につながらない。
func TestAssertTestReportsFailingUnits(t *testing.T) {
	r := &recorder{}
	AssertTest(r, context.Background(),
		&fakeLLM{replies: []string{units("関係ある文", "余談の文"), verdicts("yes", "no")}},
		TestCase{Input: "質問", ActualOutput: "回答"},
		NewAnswerRelevancy(0.8))

	for _, want := range []string{"余談の文", "理由2"} {
		if !strings.Contains(r.all(), want) {
			t.Errorf("失敗メッセージに %q が含まれない:\n%s", want, r.all())
		}
	}
	if strings.Contains(r.all(), "関係ある文") {
		t.Errorf("減点されていない単位まで出力されている:\n%s", r.all())
	}
}

func TestSkipUnlessLive(t *testing.T) {
	const key = "EVALHARNESS_TEST_KEY"

	t.Run("キーがあれば返す", func(t *testing.T) {
		t.Setenv(key, "値")
		r := &recorder{}
		if got := SkipUnlessLive(r, key); got != "値" {
			t.Errorf("= %q, 期待 %q", got, "値")
		}
		if r.failed() || len(r.skipped) > 0 {
			t.Error("キーがあるのに skip / 失敗した")
		}
	})

	t.Run("キーが無ければ skip", func(t *testing.T) {
		t.Setenv(key, "")
		t.Setenv("EVAL_REQUIRE", "")
		r := &recorder{}
		SkipUnlessLive(r, key)
		if len(r.skipped) != 1 || r.failed() {
			t.Errorf("既定では skip する想定: %+v", r)
		}
	})

	// CI で skip は緑になるため、キーの設定漏れやクォータ枯渇を
	// 「評価に通った」と見分けられない。EVAL_REQUIRE=1 で失敗に変わることを確かめる。
	t.Run("EVAL_REQUIRE=1 なら skip せず落とす", func(t *testing.T) {
		t.Setenv(key, "")
		t.Setenv("EVAL_REQUIRE", "1")
		r := &recorder{}
		SkipUnlessLive(r, key)
		if len(r.skipped) > 0 {
			t.Error("EVAL_REQUIRE=1 なのに skip した")
		}
		if !r.failed() {
			t.Error("EVAL_REQUIRE=1 でキー未設定なのに失敗しなかった")
		}
	})
}

// 429 応答から待ち時間を取り出せることを確かめる。
// 待てないエラーまで再試行すると、認証ミスや不正なプロンプトで無駄に待つ。
func TestRetryAfter(t *testing.T) {
	quota := errors.New(`Error 429, Message: You exceeded your current quota. ` +
		`Please retry in 48.011058885s., Status: RESOURCE_EXHAUSTED`)

	tests := []struct {
		name   string
		err    error
		wantOK bool
		want   float64 // 秒。指示 + 1 秒
	}{
		{"クォータ超過は待てる", quota, true, 49.011058885},
		{"待ち時間の指示が無ければ待たない",
			errors.New("Error 429, Status: RESOURCE_EXHAUSTED"), false, 0},
		{"認証エラーは待たない",
			errors.New("Error 401, Message: API key not valid"), false, 0},
		// 待ち時間の指示を含むが 429 ではないケース。
		// 正規表現だけで判断すると、待っても解消しないエラーで待たされる。
		{"待ち時間の指示があっても 429 でなければ待たない",
			errors.New("Error 400, Message: rate policy. Please retry in 5.0s."), false, 0},
		// 日次上限は秒単位では回復しない。retryDelay に従って待ち直しても無駄になる。
		// 実際に 4 回再試行で 18.6 分待って全滅した経路。
		{"日次上限は待たない",
			errors.New(`Error 429, Status: RESOURCE_EXHAUSTED, ` +
				`quotaId:GenerateRequestsPerDayPerProjectPerModel-FreeTier, ` +
				`Please retry in 48.0s.`), false, 0},
		{"サーバエラーは待たない",
			errors.New("Error 500, Message: internal"), false, 0},
		// 混雑は時間をおけば解消する。サーバは待ち時間を返さないため固定で待つ。
		{"混雑は待つ",
			errors.New("Error 503, Message: high demand, Status: UNAVAILABLE"), true, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := RetryAfter(tt.err)
			if ok != tt.wantOK {
				t.Fatalf("待てるか = %v, 期待 %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if diff := got.Seconds() - tt.want; diff > 1e-6 || diff < -1e-6 {
				t.Errorf("待ち時間 = %v, 期待 %v 秒", got, tt.want)
			}
		})
	}
}
