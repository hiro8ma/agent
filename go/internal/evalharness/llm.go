// Package evalharness は DeepEval の評価指標を Go に移植し、go test に載せる。
//
// DeepEval は Python 専用で Go 版が無い。ただし各指標の中身は
//
//	判定単位に分解する → 1 単位ずつ二値で判定する → 良い判定の割合を採点にする
//
// という共通の骨格でできている。単発で「点数をつけて」と聞くより安定するのは、
// 分解によって judge の判断が 1 単位ごとの yes / no に落ちるため。
// この骨格は言語に依存しないので Go でそのまま書ける。
//
// アプリが Go なら評価も Go に置いたほうが、ビルド・依存・CI を一本化できる。
package evalharness

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"
)

// LLM は判定役の言語モデル。JSON を返すことだけを求める。
//
// SDK 型をインターフェースに出さないのは、判定役を Gemini 以外に
// 差し替えられるようにするため。テストではフェイクを入れて API を叩かずに回す。
type LLM interface {
	GenerateJSON(ctx context.Context, prompt string) (string, error)
}

// GeminiLLM は Gemini を判定役に使う LLM 実装。
type GeminiLLM struct {
	client *genai.Client
	model  string

	// MinInterval は呼び出し間隔の下限。無料枠は 1 分あたり 5 リクエスト、
	// 1 日 20 リクエストの上限がある。指標を分解すると 1 ケースで
	// 複数回呼ぶことになるため、間隔を空けないと 429 で落ちる。
	MinInterval time.Duration

	// Calls は実際に呼んだ回数。分解方式は呼び出し回数が読みにくいため、
	// クォータ管理のために数えておく。
	Calls int

	// MaxRetries は 429 を受けたときの再試行回数。0 なら再試行しない。
	//
	// 待つ価値があるのは 1 分あたりの上限に当たった場合だけ。
	// 日次上限（quotaId に PerDay を含む）は数十秒待っても回復しない。
	// retryDelay は日次上限でも数十秒を返すため、指示をそのまま信じると
	// 回復しない待機を繰り返す。実際に 4 回再試行で 18.6 分待って全滅した。
	MaxRetries int

	mu   sync.Mutex
	last time.Time
}

// NewGeminiLLM は API キーから判定役を作る。
func NewGeminiLLM(ctx context.Context, apiKey, model string) (*GeminiLLM, error) {
	c, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("genai クライアント生成: %w", err)
	}
	return &GeminiLLM{client: c, model: model}, nil
}

// GenerateJSON はプロンプトを投げ、JSON 文字列を受け取る。
func (g *GeminiLLM) GenerateJSON(ctx context.Context, prompt string) (string, error) {
	g.throttle()

	// 判定役に温度を与えると同じ入力でも結果が動く。再現性を優先して 0 に固定する。
	var temp float32
	cfg := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		Temperature:      &temp,
	}

	resp, err := g.generateWithRetry(ctx, prompt, cfg)
	if err != nil {
		return "", err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("応答が空。finish reason=%s", finishReason(resp))
	}

	var b strings.Builder
	for _, p := range resp.Candidates[0].Content.Parts {
		b.WriteString(p.Text)
	}
	return strings.TrimSpace(b.String()), nil
}

// generateWithRetry は 429 のときサーバが指示した時間だけ待って再試行する。
func (g *GeminiLLM) generateWithRetry(ctx context.Context, prompt string, cfg *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	for attempt := 0; ; attempt++ {
		resp, err := g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), cfg)
		if err == nil {
			return resp, nil
		}

		wait, ok := RetryAfter(err)
		if !ok || attempt >= g.MaxRetries {
			return nil, fmt.Errorf("生成リクエスト (%d 回目, 再試行 %d): %w", g.Calls, attempt, err)
		}

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("再試行の待機中に中断: %w", ctx.Err())
		case <-time.After(wait):
		}
	}
}

// retryAfter は 429 応答から待つべき時間を取り出す。
// 待てる種類のエラーでなければ ok=false を返し、呼び出し側は即座に諦める。
// RetryAfter は 429 / 503 応答から待つべき時間を取り出す。
// 待てない種類のエラーなら ok=false を返す。
func RetryAfter(err error) (time.Duration, bool) {
	msg := err.Error()

	// 混雑による 503 は時間をおけば解消する。サーバは待ち時間を返さないため、
	// 一定時間おいて試す。実測で I2T の評価中に遭遇し、
	// 再試行が無かったため測定が 1 件落ちた。
	if strings.Contains(msg, "503") || strings.Contains(msg, "UNAVAILABLE") {
		return 10 * time.Second, true
	}

	if !strings.Contains(msg, "RESOURCE_EXHAUSTED") && !strings.Contains(msg, "429") {
		return 0, false
	}

	// 日次上限は秒単位の待機では回復しない。retryDelay は数十秒を返すが、
	// それに従って待ち直しても同じ 429 が返るだけで時間を失う。
	if strings.Contains(msg, "PerDay") {
		return 0, false
	}

	// "Please retry in 48.011058885s." の形で待ち時間が入る。
	m := retryDelayRe.FindStringSubmatch(msg)
	if m == nil {
		return 0, false
	}
	sec, convErr := strconv.ParseFloat(m[1], 64)
	if convErr != nil {
		return 0, false
	}
	// 指示された時間ちょうどだと窓の境界で弾かれることがあるため 1 秒足す。
	return time.Duration(sec*float64(time.Second)) + time.Second, true
}

var retryDelayRe = regexp.MustCompile(`retry in ([0-9.]+)s`)

func (g *GeminiLLM) throttle() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.Calls++
	if g.MinInterval <= 0 {
		return
	}
	if wait := g.MinInterval - time.Since(g.last); wait > 0 && !g.last.IsZero() {
		time.Sleep(wait)
	}
	g.last = time.Now()
}

func finishReason(r *genai.GenerateContentResponse) string {
	if len(r.Candidates) == 0 {
		return "candidates なし"
	}
	return string(r.Candidates[0].FinishReason)
}

// askJSON はプロンプトを投げて JSON を型に流し込む。
//
// モデルが ```json のフェンスを付けて返すことがあるため、剥がしてから解釈する。
// 構造化出力を指定していても起きるので、ここで吸収しておく。
func askJSON(ctx context.Context, llm LLM, prompt string, out any) error {
	raw, err := llm.GenerateJSON(ctx, prompt)
	if err != nil {
		return err
	}
	cleaned := stripFence(raw)
	if err := json.Unmarshal([]byte(cleaned), out); err != nil {
		return fmt.Errorf("JSON の解釈に失敗 (%s): %w", truncate(cleaned, 200), err)
	}
	return nil
}

func stripFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if i := strings.Index(s, "\n"); i >= 0 {
		s = s[i+1:]
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
