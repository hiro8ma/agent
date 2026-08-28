package i2t

import (
	"fmt"
	"sort"
	"strings"

	"github.com/hiro8ma/agent/go/internal/evalharness"
)

// Score は評価結果。evalharness と同じ形にして、
// テキスト側の指標と 1 つのダッシュボードに並べられるようにする。
type Score = evalharness.AgentScore

// Mention は説明文から抽出した物体の言及。
type Mention struct {
	Shape Shape
	Color string // 空なら色の言及なし
	Raw   string
}

// ExtractMentions は説明文から物体の言及を取り出す。
//
// 形の名前を探し、その直前にある色名を対応づける。「赤い円」「青の四角」のような
// 語順を想定する。日本語の構文解析はせず、語の並びだけで判断する。
//
// 解析を挟むと、解析器の誤りと モデルの誤りが混ざる。
// 評価の対象はモデルなので、抽出は単純に保って取りこぼしを許す。
// 取りこぼしは減点に働かない（言及していないものは幻覚として数えない）ため、
// CHAIR は過大評価ではなく過小評価の側に倒れる。
func ExtractMentions(caption string) []Mention {
	colors := ColorNames()
	var out []Mention

	for _, sh := range ShapeNames() {
		shape := Shape(sh)
		for idx := 0; ; {
			i := strings.Index(caption[idx:], sh)
			if i < 0 {
				break
			}
			at := idx + i
			idx = at + len(sh)

			// 形の直前 12 バイトに色名があれば対応づける。
			// 12 は「オレンジ色の」程度が入る長さ。
			start := at - 12
			if start < 0 {
				start = 0
			}
			window := caption[start:at]

			m := Mention{Shape: shape, Raw: sh}
			best := -1
			for _, c := range colors {
				if j := strings.LastIndex(window, c); j > best {
					best, m.Color = j, c
				}
			}
			out = append(out, m)
		}
	}
	return out
}

// CHAIR は説明文に含まれる、画像に存在しない物体の割合を測る。
//
// Caption Hallucination Assessment with Image Relevance。
// 元の指標は「キャプション内の名詞と実際のアノテーションを照合する」もので、
// ここでは合成画像なのでアノテーションは定義上正しい。
//
// スコアは他の指標と揃えて 1 が良い。1 - 幻覚率 を返す。
// CHAIR 自体は「幻覚の割合」で 0 が良い向きの指標だが、
// 向きの違う値を同じ表に並べると平均で意味が壊れる。取り込む側で揃える。
func CHAIR(s Scene, caption string, threshold float64) Score {
	mentions := ExtractMentions(caption)
	if len(mentions) == 0 {
		return Score{Metric: "CHAIR", Threshold: threshold,
			Reason: "採点できない: 説明文に物体の言及がない"}
	}

	halluc := 0
	var details []string
	for _, m := range mentions {
		switch {
		case !s.HasShape(m.Shape):
			halluc++
			details = append(details, fmt.Sprintf("%s は画像に無い", m.Shape))
		case m.Color != "" && !s.Contains(m.Shape, m.Color):
			halluc++
			if s.HasColor(m.Color) {
				details = append(details, fmt.Sprintf("%sの%s という組み合わせは無い", m.Color, m.Shape))
			} else {
				details = append(details, fmt.Sprintf("%s は画像に無い色", m.Color))
			}
		}
	}
	sort.Strings(details)

	v := 1 - float64(halluc)/float64(len(mentions))
	return Score{
		Metric: "CHAIR", Value: v, Threshold: threshold, Passed: v >= threshold, Scored: true,
		Reason:  fmt.Sprintf("言及 %d 件のうち %d 件が画像に無い", len(mentions), halluc),
		Details: details,
	}
}

// Coverage は画像内の物体をどれだけ説明できたかを測る。
//
// CHAIR は「無いものを言ったか」を測る。取りこぼしは測らない。
// 何も言わなければ幻覚も 0 件になるため、CHAIR だけでは
// 「黙っていれば満点」の実装が最高得点になる。対で測る。
func Coverage(s Scene, caption string, threshold float64) Score {
	if len(s.Objects) == 0 {
		return Score{Metric: "Coverage", Threshold: threshold,
			Reason: "採点できない: 画像に物体がない"}
	}

	mentions := ExtractMentions(caption)
	found := 0
	var missed []string
	for _, o := range s.Objects {
		hit := false
		for _, m := range mentions {
			if m.Shape == o.Shape && (m.Color == "" || m.Color == o.Color) {
				hit = true
				break
			}
		}
		if hit {
			found++
		} else {
			missed = append(missed, fmt.Sprintf("%sの%s に言及なし", o.Color, o.Shape))
		}
	}
	sort.Strings(missed)

	v := float64(found) / float64(len(s.Objects))
	return Score{
		Metric: "Coverage", Value: v, Threshold: threshold, Passed: v >= threshold, Scored: true,
		Reason:  fmt.Sprintf("物体 %d 件のうち %d 件に言及", len(s.Objects), found),
		Details: missed,
	}
}

// MirrorConsistency は左右反転した画像で、方向の記述が入れ替わるかを測る。
//
// 位置の記述が正しいかを直接測るには、説明文から
// 「どの物体がどこにあると言ったか」を取り出す必要があり、日本語の構文解析が要る。
// 解析器の誤りとモデルの誤りが混ざるため、評価としては筋が悪い。
//
// 代わりに変換に対する性質を見る。元画像で「左」と言った回数は、
// 反転した画像では「右」と言った回数に一致するはず。
// 一致しなければ、方向を理解せずに書いているか、反転に追随できていない。
//
// 構文解析を使わずに位置の理解を測れる。入力変換に対する期待関係を
// 性質として定義する形になっている。
//
// ただし語数の一致は弱い代理指標になる。限界は 2 つある。
//
//	追随しなくても 0 にはならない。左 2 右 1 のまま変わらなければ 0.67 になる。
//	追随した 1.00 と閾値で分ける形になり、程度の判断には使えない
//
//	方向語が説明の別の箇所に現れると数がずれる。実測で
//	「右側（中央の右）」という表現により右が 2 回数えられた
//
// 位置理解の有無を大まかに見る煙感知器として使う。
// 程度まで測るなら、物体と方向の対応を取り出す必要がある。
func MirrorConsistency(original, mirrored string, threshold float64) Score {
	lo, ro := strings.Count(original, "左"), strings.Count(original, "右")
	lm, rm := strings.Count(mirrored, "左"), strings.Count(mirrored, "右")

	if lo+ro == 0 && lm+rm == 0 {
		return Score{Metric: "MirrorConsistency", Threshold: threshold,
			Reason: "採点できない: どちらの説明にも方向の記述がない"}
	}

	// 左右が同数だと、反転に追随した場合としなかった場合が同じ数になり区別できない。
	// 満点が出るが何も測っていない。実測で左 1 右 1 の配置を使って踏んだ。
	// 判別できない条件では採点しない。
	if lo == ro && lm == rm {
		return Score{Metric: "MirrorConsistency", Threshold: threshold,
			Reason: fmt.Sprintf(
				"採点できない: 左右の言及数が同数（元 左%d 右%d / 反転後 左%d 右%d）。"+
					"反転に追随した場合としなかった場合を区別できない。"+
					"左右で物体数の異なる配置を使う", lo, ro, lm, rm)}
	}

	// 元の「左」が反転後の「右」に、元の「右」が反転後の「左」に対応する。
	diff := abs(lo-rm) + abs(ro-lm)
	total := lo + ro + lm + rm

	v := 1 - float64(diff)/float64(total)
	if v < 0 {
		v = 0
	}
	return Score{
		Metric: "MirrorConsistency", Value: v, Threshold: threshold,
		Passed: v >= threshold, Scored: true,
		Reason: fmt.Sprintf("元画像 左%d 右%d / 反転後 左%d 右%d", lo, ro, lm, rm),
	}
}

// ---------- OCR ----------

// CER は文字単位の誤り率。編集距離 / 正解の文字数。
//
// 0 が完全一致。1 を超えることもある（余計な文字を大量に出した場合）。
func CER(want, got string) float64 {
	w, g := []rune(want), []rune(got)
	if len(w) == 0 {
		if len(g) == 0 {
			return 0
		}
		return 1
	}
	return float64(levenshtein(w, g)) / float64(len(w))
}

// WER は単語単位の誤り率。空白で区切る。
func WER(want, got string) float64 {
	w, g := strings.Fields(want), strings.Fields(got)
	if len(w) == 0 {
		if len(g) == 0 {
			return 0
		}
		return 1
	}
	return float64(levenshteinStr(w, g)) / float64(len(w))
}

// OCRAccuracy は画像内の文字の読み取りを採点する。
//
// 空白の違いは誤読として数えない。実測で、改行で区切って返された結果が
// CER 0.083 になった。文字はすべて正しく読めており、区切り文字だけが違う。
// これを誤読に数えると、読み取り精度がレイアウトの表現方法に左右される。
//
// レイアウト保持性（複数行・段組・読み順を再現できるか）は別の観点になる。
// 分けて測るべきもので、ここでは扱わない。
//
// CER は 0 が良い向きなので、1 - CER に直して他の指標と揃える。
// 1 を超える CER があるため 0 で下限を切る。
func OCRAccuracy(s Scene, extracted string, threshold float64) Score {
	if len(s.Texts) == 0 {
		return Score{Metric: "OCRAccuracy", Threshold: threshold,
			Reason: "採点できない: 画像に文字がない"}
	}

	var want []string
	for _, t := range s.Texts {
		want = append(want, t.Content)
	}
	joined := strings.Join(want, " ")

	// 空白の種類と連続を 1 個の半角空白に潰してから比べる。
	norm := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	cer, wer := CER(norm(joined), norm(extracted)), WER(norm(joined), norm(extracted))
	v := 1 - cer
	if v < 0 {
		v = 0
	}
	return Score{
		Metric: "OCRAccuracy", Value: v, Threshold: threshold, Passed: v >= threshold, Scored: true,
		Reason: fmt.Sprintf("CER %.3f / WER %.3f", cer, wer),
		Details: []string{
			fmt.Sprintf("正解 %q", joined),
			fmt.Sprintf("読み取り %q", extracted),
		},
	}
}

func levenshtein(a, b []rune) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min3(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

func levenshteinStr(a, b []string) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min3(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
