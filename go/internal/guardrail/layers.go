// 入力と出力とツール実行の 3 層に置くガードレール。
//
// Python 版（samples/callbacks/guardrails.py）と対になる。
// 打ち切りの規則は言語で違う。
//
//	Go      非 nil を返したら止まる
//	Python  真を返したら止まる。空の map は止めないがツールは飛ぶ
//
// Go では nil でない限り止まるので、空の map を返すと「理由の無い遮断」になる。
//
// 判定は純粋関数として公開し、コールバックはその薄い包みにする。
// agent.Context は実装が重く、偽物を作ると検査の方が壊れやすい。
package guardrail

import (
	"fmt"
	"regexp"
	"slices"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// InjectionFlagKey は検出の印を置く鍵。監視と、後段での扱いを変える判断に使う。
const InjectionFlagKey = "injection_detected"

// ConfirmedKey は破壊的操作の確認済みを示す鍵。
const ConfirmedKey = "confirmed_destructive"

// InjectionPatterns は指示の上書きを狙う入力の形。
//
// 既知の形しか拾えない。分類器と権限設計を併せて使う。
var InjectionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions`),
	regexp.MustCompile(`(?i)disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions)`),
	regexp.MustCompile(`(?i)reveal\s+(your\s+)?(system\s+)?prompt`),
	regexp.MustCompile(`(以前|これまで)の指示を(すべて)?無視`),
	regexp.MustCompile(`(システムプロンプト|あなたの指示)を(表示|教えて|出力)`),
}

// piiRules は伏せる形と、その置き換え先。何を伏せたか分かる名前にする。
var piiRules = []struct {
	re   *regexp.Regexp
	mask string
}{
	{regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.-]+`), "[EMAIL_MASKED]"},
	{regexp.MustCompile(`\b(?:\d[ -]?){13,16}\b`), "[CARD_MASKED]"},
	{regexp.MustCompile(`\b0\d{1,4}-\d{1,4}-\d{4}\b`), "[PHONE_MASKED]"},
}

// LastUserText は直近の利用者発話だけを返す。
//
// 全履歴を対象にすると、一度弾いた入力が履歴に残る限り毎回落ちる。
func LastUserText(req *model.LLMRequest) string {
	if req == nil {
		return ""
	}
	for _, c := range slices.Backward(req.Contents) {

		if c == nil || c.Role != "user" {
			continue
		}
		var b strings.Builder
		for _, p := range c.Parts {
			if p != nil {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}

// MatchInjection は入力に指示の上書きを狙う形があれば、その形を返す。
func MatchInjection(text string, patterns []*regexp.Regexp) *regexp.Regexp {
	for _, re := range patterns {
		if re != nil && re.MatchString(text) {
			return re
		}
	}
	return nil
}

// DetectInjection は入力ガードレール。兆候があればモデルを呼ばずに返す。
func DetectInjection(log *Log, patterns []*regexp.Regexp, msg string) llmagent.BeforeModelCallback {
	return func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		hit := MatchInjection(LastUserText(req), patterns)
		if hit == nil {
			log.add(Verdict{Stage: "before_model"})
			return nil, nil
		}
		markState(ctx, InjectionFlagKey, hit.String())
		log.add(Verdict{
			Stage: "before_model", Rule: "インジェクション", Blocked: true,
			Detail: fmt.Sprintf("入力が %s に一致", hit.String()),
		})
		return refuse(msg), nil
	}
}

// MaskText は個人情報の形を伏せる。伏せたかどうかも返す。
func MaskText(text string) (string, bool) {
	masked := text
	for _, rule := range piiRules {
		masked = rule.re.ReplaceAllString(masked, rule.mask)
	}
	return masked, masked != text
}

// MaskPII は出力ガードレール。個人情報の形を伏せてから返す。
//
// 伏せたときだけ書き換えた応答を返す。無ければ nil を返して元の応答を通す。
func MaskPII(log *Log) llmagent.AfterModelCallback {
	return func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err != nil || resp == nil || resp.Content == nil {
			return nil, nil
		}
		changed := false
		for i, part := range resp.Content.Parts {
			if part == nil || part.Text == "" {
				continue
			}
			masked, hit := MaskText(part.Text)
			if hit {
				resp.Content.Parts[i].Text = masked
				changed = true
			}
		}
		if !changed {
			log.add(Verdict{Stage: "after_model"})
			return nil, nil
		}
		log.add(Verdict{
			Stage: "after_model", Rule: "個人情報", Blocked: true,
			Detail: "出力の個人情報を伏せた",
		})
		return resp, nil
	}
}

// NeedsConfirmation は、そのツールが確認を要するかを返す。
//
// state が nil のときは未確認として扱う。確認済みの印が読めない以上、
// 通してしまうより止める方が安全側になる。
func NeedsConfirmation(state session.ReadonlyState, toolName, flagKey string, targets []string) bool {
	found := slices.Contains(targets, toolName)
	if !found {
		return false
	}
	if state == nil {
		return true
	}
	v, err := state.Get(flagKey)
	if err != nil {
		return true
	}
	done, _ := v.(bool)
	return !done
}

// RequireConfirmation は実行ガードレール。壊す操作は確認済みでなければ実行しない。
//
// 返す map は中身のあるものにする。Go では非 nil なら止まるので、
// 空の map でも止まるが、モデルは何を聞き返せばよいか分からない。
func RequireConfirmation(log *Log, flagKey string, targets ...string) llmagent.BeforeToolCallback {
	return func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
		name := ""
		if t != nil {
			name = t.Name()
		}
		if !NeedsConfirmation(readonlyState(ctx), name, flagKey, targets) {
			log.add(Verdict{Stage: "before_tool"})
			return nil, nil
		}
		log.add(Verdict{
			Stage: "before_tool", Rule: "要確認", Blocked: true,
			Detail: fmt.Sprintf("%s は確認が要る", name),
		})
		return map[string]any{
			"status":  "confirmation_required",
			"tool":    name,
			"args":    args,
			"message": fmt.Sprintf("%s は取り消せない操作です。実行してよいか確認してください", name),
		}, nil
	}
}

// readonlyState は Context から読み取り用の State を取り出す。無ければ nil を返す。
func readonlyState(ctx agent.Context) session.ReadonlyState {
	if ctx == nil {
		return nil
	}
	return ctx.ReadonlyState()
}

// markState は印を残す。書けなくても本来の遮断は続ける。
func markState(ctx agent.Context, key string, value any) {
	if ctx == nil {
		return
	}
	state := ctx.State()
	if state == nil {
		return
	}
	_ = state.Set(key, value)
}
