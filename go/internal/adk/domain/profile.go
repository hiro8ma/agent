package domain

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// ProfileField は利用者の輪郭として保存してよい項目。
//
// モデルに鍵名を決めさせると、role のような権限の鍵まで書き換えられる。
// 書ける項目をここで列挙し、それ以外は受け付けない。
type ProfileField string

const (
	FieldExpertise   ProfileField = "expertise"
	FieldLanguage    ProfileField = "language"
	FieldAnswerStyle ProfileField = "answer_style"
	FieldFocus       ProfileField = "focus"
)

// userScope は State のスコープの接頭辞。ADK に依存させないため domain では文字列で持つ。
const userScope = "user:"

// ProfileFields は書ける項目の一覧。モデルへの説明にも使う。
var ProfileFields = []ProfileField{FieldExpertise, FieldLanguage, FieldAnswerStyle, FieldFocus}

var (
	enumValues = map[ProfileField][]string{
		FieldExpertise:   {"beginner", "intermediate", "advanced"},
		FieldAnswerStyle: {"concise", "detailed"},
	}
	languagePattern = regexp.MustCompile(`^[a-z0-9+#.-]{1,20}$`)
	// 自由記述は 1 行に限る。改行や見出し記号を許すと、利用者が system instruction の続きを書ける。
	focusForbidden = regexp.MustCompile("[\n\r#`<>]")
)

const focusMaxRunes = 40

// ParseProfileField は項目名を検証する。
func ParseProfileField(name string) (ProfileField, error) {
	for _, f := range ProfileFields {
		if string(f) == name {
			return f, nil
		}
	}
	return "", fmt.Errorf("%q は記録できない項目。記録できるのは %v", name, ProfileFields)
}

// StateKey は State に置く鍵を返す。user: なので同じ利用者の別 Session へ引き継がれる。
func (f ProfileField) StateKey() string {
	return userScope + "profile_" + string(f)
}

// Normalize は値を検証し、保存する形に整える。
func (f ProfileField) Normalize(value string) (string, error) {
	v := strings.TrimSpace(value)
	if v == "" {
		return "", fmt.Errorf("%s の値が空", f)
	}
	switch f {
	case FieldExpertise, FieldAnswerStyle:
		v = strings.ToLower(v)
		if slices.Contains(enumValues[f], v) {
			return v, nil
		}
		return "", fmt.Errorf("%s は %v のいずれか", f, enumValues[f])
	case FieldLanguage:
		v = strings.ToLower(v)
		if !languagePattern.MatchString(v) {
			return "", fmt.Errorf("language は英小文字と記号 +#.- で 20 文字まで")
		}
		return v, nil
	case FieldFocus:
		if focusForbidden.MatchString(v) {
			return "", fmt.Errorf("focus に改行と記号 # ` < > は使えない")
		}
		if len([]rune(v)) > focusMaxRunes {
			return "", fmt.Errorf("focus は %d 文字まで", focusMaxRunes)
		}
		return v, nil
	}
	return "", fmt.Errorf("未知の項目 %s", f)
}

// Profile は利用者の輪郭。
type Profile struct {
	Expertise   string
	Language    string
	AnswerStyle string
	Focus       string
}

// IsEmpty は何も分かっていないかを返す。
func (p Profile) IsEmpty() bool {
	return p == Profile{}
}

var expertiseGuide = map[string]string{
	"beginner":     "初学者。用語は初出で説明し、手順を省略しない",
	"intermediate": "中級者。基本は説明せず、判断の理由を添える",
	"advanced":     "上級者。基礎の説明は省き、設計判断とトレードオフを中心に答える",
}

var answerStyleGuide = map[string]string{
	"concise":  "簡潔。結論を先に書き、補足は求められたら出す",
	"detailed": "詳しく。前提と根拠と代替案まで書く",
}

// Instruction は輪郭を system instruction に差し込む文へ描き出す。
//
// 事実を並べるだけでなく、答え方の指示に変換して渡す。
// 「上級者」とだけ書くより「基礎の説明は省く」と書く方が、モデルの答えが変わる。
func (p Profile) Instruction() string {
	if p.IsEmpty() {
		return ""
	}
	lines := []string{"## この利用者について（過去の対話から記録したもの）"}
	if g, ok := expertiseGuide[p.Expertise]; ok {
		lines = append(lines, "- 技術レベルは "+g)
	}
	if p.Language != "" {
		lines = append(lines, fmt.Sprintf("- 主に使う言語は %s。コード例はこの言語で書く", p.Language))
	}
	if g, ok := answerStyleGuide[p.AnswerStyle]; ok {
		lines = append(lines, "- 回答の長さは "+g)
	}
	if p.Focus != "" {
		lines = append(lines, fmt.Sprintf("- 関心のある領域は「%s」。例はこの領域から選ぶ", p.Focus))
	}
	return strings.Join(lines, "\n")
}
