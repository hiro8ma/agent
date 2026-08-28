package i2t

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/evalharness"
)

// Describer は画像から説明文を生成する。評価の対象になるアプリケーション本体。
type Describer struct {
	client *genai.Client
	model  string

	// Calls は呼び出し回数。無料枠には上限があるため数えておく。
	Calls int

	// MinInterval は呼び出し間隔の下限。無料枠は 1 分あたり 5 リクエスト。
	// 変換ごとに説明を作ると連続で叩くことになり、間隔を空けないと 429 になる。
	// 実測でロバストネスの測定が 1 件これで落ちた。
	MinInterval time.Duration

	// MaxRetries は 429 / 503 を受けたときの再試行回数。
	// 日次上限は待っても回復しないため、evalharness.RetryAfter が判定する。
	MaxRetries int

	mu   sync.Mutex
	last time.Time
}

// throttle は前回の呼び出しから MinInterval 空ける。
func (d *Describer) throttle() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.MinInterval <= 0 {
		return
	}
	if wait := d.MinInterval - time.Since(d.last); wait > 0 && !d.last.IsZero() {
		time.Sleep(wait)
	}
	d.last = time.Now()
}

// generate は待機と再試行を挟んで 1 回生成する。
func (d *Describer) generate(ctx context.Context, contents []*genai.Content) (string, time.Duration, error) {
	var temp float32 // 再現性を優先して 0 に固定する
	cfg := &genai.GenerateContentConfig{Temperature: &temp}

	var total time.Duration
	for attempt := 0; ; attempt++ {
		d.throttle()
		d.Calls++

		t0 := time.Now()
		resp, err := d.client.Models.GenerateContent(ctx, d.model, contents, cfg)
		total += time.Since(t0)

		if err == nil {
			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
				return "", total, fmt.Errorf("i2t: 応答が空")
			}
			var b strings.Builder
			for _, p := range resp.Candidates[0].Content.Parts {
				b.WriteString(p.Text)
			}
			return strings.TrimSpace(b.String()), total, nil
		}

		wait, ok := evalharness.RetryAfter(err)
		if !ok || attempt >= d.MaxRetries {
			return "", total, fmt.Errorf("i2t: 生成 (%d 回目, 再試行 %d): %w", d.Calls, attempt, err)
		}
		select {
		case <-ctx.Done():
			return "", total, fmt.Errorf("i2t: 再試行の待機中に中断: %w", ctx.Err())
		case <-time.After(wait):
		}
	}
}

// NewDescriber は API キーから生成器を作る。
func NewDescriber(ctx context.Context, apiKey, model string) (*Describer, error) {
	c, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: apiKey, Backend: genai.BackendGeminiAPI})
	if err != nil {
		return nil, fmt.Errorf("i2t: クライアント生成: %w", err)
	}
	return &Describer{client: c, model: model}, nil
}

// 説明の指示。評価の対象になるので、変えたら評価もやり直す。
//
// 位置を必ず書かせるのは、鏡像テストで方向の理解を測るため。
// 指示しなければ位置に触れず、測定対象が消える。
const describePrompt = `この画像に写っているものを日本語で説明してください。

次の点を必ず含めてください。
- 写っている図形の種類（円 / 四角 / 三角）
- それぞれの色
- それぞれの位置（左上・中央・右下 のように）
- 文字が写っていれば、その文字列をそのまま

写っていないものは書かないでください。推測で補わないでください。`

// Describe は画像を説明する。
func (d *Describer) Describe(ctx context.Context, img image.Image) (string, time.Duration, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return "", 0, fmt.Errorf("i2t: PNG への符号化: %w", err)
	}

	return d.generate(ctx, []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
		genai.NewPartFromBytes(buf.Bytes(), "image/png"),
		genai.NewPartFromText(describePrompt),
	}}})
}

// ExtractText は画像内の文字だけを読み取る。OCR 精度の測定に使う。
//
// 説明文から文字を切り出すのではなく、読み取りだけを指示する。
// 説明の一部として書かせると、余計な語が混ざって CER が説明文の量に左右される。
func (d *Describer) ExtractText(ctx context.Context, img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return "", fmt.Errorf("i2t: PNG への符号化: %w", err)
	}

	out, _, err := d.generate(ctx, []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
		genai.NewPartFromBytes(buf.Bytes(), "image/png"),
		genai.NewPartFromText(
			"この画像に書かれている文字だけを、そのまま出力してください。" +
				"説明や前置きは書かないでください。文字が無ければ空で返してください。"),
	}}})
	return out, err
}
