package evalharness

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// TB は testing.TB のうちハーネスが使う部分だけを取り出したもの。
//
// *testing.T を直接受けるとハーネス自身をテストできない。
// 「落ちるべきときに落ちるか」を確かめるには、失敗を記録できる受け口が要る。
// 評価が素通しになっていてもテストは緑になるため、ここは実際に確かめる価値がある。
type TB interface {
	Helper()
	Logf(format string, args ...any)
	Errorf(format string, args ...any)
	Fatalf(format string, args ...any)
	Skipf(format string, args ...any)
}

// AssertTest は DeepEval の assert_test に対応する。
// 不合格の指標があればテストを落とし、判定の内訳をそのまま出す。
//
// 内訳を出すのは、スコアだけでは何を直せばよいか分からないため。
// 0.4 という数字は、どの単位が減点されたかを伝えない。
func AssertTest(t TB, ctx context.Context, llm LLM, tc TestCase, metrics ...Metric) {
	t.Helper()

	for _, m := range metrics {
		s, err := m.Measure(ctx, llm, tc)
		if err != nil {
			t.Fatalf("%s の採点に失敗: %v", m.Name(), err)
			return
		}

		good := 0
		for _, v := range s.Verdicts {
			if v.Good {
				good++
			}
		}
		label := fmt.Sprintf("%s: %.2f (閾値 %.2f, 判定 %d/%d 件が良好)",
			s.Metric, s.Value, s.Threshold, good, len(s.Verdicts))

		if !s.Passed {
			t.Errorf("%s\n%s", label, formatVerdicts(s))
			continue
		}
		t.Logf("%s", label)

		// 合格でも閾値ぎりぎりは知らせる。次のモデル更新で落ちる予備軍になる。
		if d := s.Value - s.Threshold; d < 0.1 {
			t.Logf("  閾値との差が %.2f しかない。次の変更で落ちる可能性がある", d)
		}
	}
}

func formatVerdicts(s Score) string {
	var b strings.Builder
	for _, v := range s.Verdicts {
		if v.Good {
			continue
		}
		fmt.Fprintf(&b, "  減点: %s\n    → %s\n", truncate(v.Unit, 120), v.Reason)
	}
	if b.Len() == 0 && s.Reason != "" {
		fmt.Fprintf(&b, "  理由: %s\n", s.Reason)
	}
	return strings.TrimRight(b.String(), "\n")
}

// SkipUnlessLive は認証情報が無いときテストを飛ばす。
//
// ただし EVAL_REQUIRE=1 なら飛ばさずに落とす。CI で skip は緑になるため、
// 「評価に通った」と「評価が走らなかった」がテスト結果の色で区別できない。
// キーの設定漏れやクォータ枯渇を緑のまま見逃す事故は、この一行で防げる。
//
// ローカルでは既定の skip、CI では EVAL_REQUIRE=1 を立てる運用を想定する。
func SkipUnlessLive(t TB, envKey string) string {
	t.Helper()

	if v := os.Getenv(envKey); v != "" {
		return v
	}
	if os.Getenv("EVAL_REQUIRE") == "1" {
		t.Fatalf("%s が未設定。EVAL_REQUIRE=1 のため skip せず失敗させた", envKey)
		return ""
	}
	t.Skipf("%s が未設定のため skip。CI では EVAL_REQUIRE=1 を立てて skip を失敗に変える", envKey)
	return ""
}
