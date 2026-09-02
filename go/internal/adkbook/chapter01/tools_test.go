package chapter01

import (
	"iter"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// mapState は State の最小実装。ADK の ContextMock は nil を返す。
type mapState map[string]any

func (m mapState) Get(k string) (any, error) {
	v, ok := m[k]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (m mapState) Set(k string, v any) error { m[k] = v; return nil }

func (m mapState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range m {
			if !yield(k, v) {
				return
			}
		}
	}
}

type statefulContext struct {
	*agent.ContextMock
	st mapState
}

func (c *statefulContext) State() session.State                 { return c.st }
func (c *statefulContext) ReadonlyState() session.ReadonlyState { return c.st }

func newContext() *statefulContext {
	return &statefulContext{ContextMock: &agent.ContextMock{}, st: mapState{}}
}

// 観光情報が返るかを見る。
func TestGetSightseeingReturnsSpots(t *testing.T) {
	got, err := GetSightseeing(newContext(), SightseeingInput{City: "tokyo"})
	if err != nil {
		t.Fatalf("get sightseeing: %v", err)
	}
	if got.Status != "success" {
		t.Errorf("status = %q（success のはず）", got.Status)
	}
	if len(got.Spots) == 0 {
		t.Error("spots が空")
	}
	if got.RecommendedSeason == "" {
		t.Error("recommendedSeason が空")
	}
}

// 日本語の都市名を両ツールが受けるかを見る。
//
// 英語だけ受けると、表記の変換をモデルに任せることになり失敗点が増える。
func TestToolsAcceptJapaneseCityNames(t *testing.T) {
	for _, city := range []string{"東京", "大阪", "札幌", "福岡", "とうきょう"} {
		w, err := GetWeather(newContext(), WeatherInput{City: city})
		if err != nil || w.Status != "success" {
			t.Errorf("get_weather(%q) status = %q err = %v", city, w.Status, err)
		}
		s, err := GetSightseeing(newContext(), SightseeingInput{City: city})
		if err != nil || s.Status != "success" {
			t.Errorf("get_sightseeing(%q) status = %q err = %v", city, s.Status, err)
		}
	}
}

// 2 つの辞書が同じ都市を持つかを見る。
//
// 片方だけに都市があると、天気は返るのに観光が返らない組み合わせができる。
func TestBothToolsCoverTheSameCities(t *testing.T) {
	for city := range weather {
		if _, ok := sightseeing[city]; !ok {
			t.Errorf("%q は天気にあるが観光に無い", city)
		}
	}
	for city := range sightseeing {
		if _, ok := weather[city]; !ok {
			t.Errorf("%q は観光にあるが天気に無い", city)
		}
	}
}

// 成功した都市が State に残るかを見る。
func TestSuccessRecordsLastCity(t *testing.T) {
	ctx := newContext()
	if _, err := GetWeather(ctx, WeatherInput{City: "大阪"}); err != nil {
		t.Fatal(err)
	}
	v, err := ctx.st.Get(LastCityKey)
	if err != nil {
		t.Fatalf("%s が記録されていない", LastCityKey)
	}
	if v != "osaka" {
		t.Errorf("%s = %v（osaka のはず）", LastCityKey, v)
	}
}

// 失敗した都市が State に残らないかを見る。
//
// 未登録の都市を記録すると「前回の都市」として引き継がれる。
func TestFailureDoesNotRecordLastCity(t *testing.T) {
	ctx := newContext()
	if _, err := GetWeather(ctx, WeatherInput{City: "那覇"}); err != nil {
		t.Fatal(err)
	}
	if _, err := ctx.st.Get(LastCityKey); err == nil {
		t.Errorf("失敗した都市が記録された")
	}
}

// State をまたぐ鍵になっているかを見る。
func TestLastCityKeyIsUserScoped(t *testing.T) {
	if !strings.HasPrefix(LastCityKey, session.KeyPrefixUser) {
		t.Errorf("LastCityKey = %q（%q で始まるはず）", LastCityKey, session.KeyPrefixUser)
	}
}

// 直近の都市が Instruction に差し込まれるかを見る。
func TestBuildInstructionInjectsLastCity(t *testing.T) {
	ctx := newContext()
	if _, err := GetWeather(ctx, WeatherInput{City: "札幌"}); err != nil {
		t.Fatal(err)
	}
	got, err := BuildInstruction(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "sapporo") {
		t.Error("Instruction が直近の都市を差し込んでいない")
	}
}

// 記録が無いときに素の Instruction を返すかを見る。
func TestBuildInstructionWithoutStateIsBase(t *testing.T) {
	got, err := BuildInstruction(newContext())
	if err != nil {
		t.Fatal(err)
	}
	if got != baseInstruction {
		t.Error("記録が無いのに Instruction が変わっている")
	}
}

// 期待する振る舞いを Instruction が指示しているかを見る。
//
// Instruction に無い振る舞いを期待すると、実装ではなく
// 仕様の欠落で落ちる。落ちた側を直しても直らない。
func TestInstructionCoversEveryExpectedBehavior(t *testing.T) {
	for name, phrase := range map[string]string{
		"都市名の未指定":  "聞き返",
		"範囲外の質問":   "無関係",
		"指示の上書き":   "上書き",
		"推測の禁止":    "予報",
		"未登録の都市":   "登録されていない",
		"天気と観光の統合": "組み合わせ",
	} {
		if !strings.Contains(baseInstruction, phrase) {
			t.Errorf("%s を期待しているが Instruction に %q が無い", name, phrase)
		}
	}
}

// Instruction が出力形式を指示しているかを見る。
//
// 目的と例外処理だけ書いて形式を落とすと、応答の長さと体裁が
// モデルの気分で変わる。
func TestInstructionSpecifiesOutputFormat(t *testing.T) {
	for _, phrase := range []string{"3 文以内", "箇条書き"} {
		if !strings.Contains(baseInstruction, phrase) {
			t.Errorf("出力形式の指示に %q が無い", phrase)
		}
	}
}
