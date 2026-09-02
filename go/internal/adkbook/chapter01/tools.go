package chapter01

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// LastCityKey は直近に問い合わせた都市を置く鍵。
//
// user: を付けるとセッションをまたいで残る。
// 接頭辞なしにすると会話が切れた時点で消える。
var LastCityKey = session.KeyPrefixUser + "last_city"

var weather = map[string]struct {
	Condition   string
	Temperature int
	Humidity    int
}{
	"tokyo":   {"晴れ", 28, 55},
	"osaka":   {"曇り", 30, 70},
	"sapporo": {"雨", 21, 85},
	"fukuoka": {"晴れ", 31, 65},
}

var sightseeing = map[string]struct {
	Spots  []string
	Season string
}{
	"tokyo":   {[]string{"浅草寺", "東京タワー", "明治神宮"}, "春（桜の季節）"},
	"osaka":   {[]string{"大阪城", "道頓堀", "通天閣"}, "秋"},
	"sapporo": {[]string{"大通公園", "時計台", "藻岩山"}, "冬（雪まつり）"},
	"fukuoka": {[]string{"太宰府天満宮", "櫛田神社", "福岡城跡"}, "春"},
}

// aliases は日本語表記を鍵へ寄せる。
// 英語だけ受けると、表記の変換をモデルに任せることになり失敗点が増える。
var aliases = map[string]string{
	"東京": "tokyo", "とうきょう": "tokyo",
	"大阪": "osaka", "おおさか": "osaka",
	"札幌": "sapporo", "さっぽろ": "sapporo",
	"福岡": "fukuoka", "ふくおか": "fukuoka",
}

// normalize は都市名を辞書の鍵へ寄せる。
func normalize(city string) string {
	raw := strings.TrimSpace(city)
	if key, ok := aliases[raw]; ok {
		return key
	}
	return strings.ToLower(raw)
}

// remember は問い合わせに成功した都市だけを記録する。
//
// 失敗も記録すると、未登録の都市が「前回の都市」として残る。
func remember(ctx agent.Context, key string) {
	if ctx == nil {
		return
	}
	// ADK の ContextMock は State に nil を返す。
	st := ctx.State()
	if st == nil {
		return
	}
	st.Set(LastCityKey, key)
}

// WeatherInput はツールの入力。
//
// json タグがそのままモデルへ渡るスキーマになる。
// Python 版が docstring から組み立てるところを、Go は型で書く。
type WeatherInput struct {
	// City は都市名。日本語（東京）と英語（tokyo）のどちらでもよい。
	City string `json:"city"`
}

// WeatherOutput はツールの出力。
//
// 失敗も構造化して返す。エラーを返すとモデルが理由を読めず、
// 「登録されていない都市」と「呼び出しに失敗した」を区別できない。
type WeatherOutput struct {
	Status       string `json:"status"`
	City         string `json:"city,omitempty"`
	Report       string `json:"report,omitempty"`
	Condition    string `json:"condition,omitempty"`
	Temperature  int    `json:"temperature,omitempty"`
	Humidity     int    `json:"humidity,omitempty"`
	ErrorMessage string `json:"errorMessage,omitempty"`
}

// GetWeather は指定した都市の現在の天気を返す。
func GetWeather(ctx agent.Context, in WeatherInput) (WeatherOutput, error) {
	key := normalize(in.City)
	d, ok := weather[key]
	if !ok {
		return WeatherOutput{
			Status:       "error",
			ErrorMessage: fmt.Sprintf("%s の天気は登録されていない", in.City),
		}, nil
	}
	remember(ctx, key)
	return WeatherOutput{
		Status:      "success",
		City:        key,
		Report:      fmt.Sprintf("%s、気温 %d 度、湿度 %d%%", d.Condition, d.Temperature, d.Humidity),
		Condition:   d.Condition,
		Temperature: d.Temperature,
		Humidity:    d.Humidity,
	}, nil
}

// SightseeingInput はツールの入力。
type SightseeingInput struct {
	// City は都市名。日本語（東京）と英語（tokyo）のどちらでもよい。
	City string `json:"city"`
}

// SightseeingOutput はツールの出力。
type SightseeingOutput struct {
	Status            string   `json:"status"`
	City              string   `json:"city,omitempty"`
	Spots             []string `json:"spots,omitempty"`
	RecommendedSeason string   `json:"recommendedSeason,omitempty"`
	ErrorMessage      string   `json:"errorMessage,omitempty"`
}

// GetSightseeing は指定した都市の観光情報を返す。
func GetSightseeing(ctx agent.Context, in SightseeingInput) (SightseeingOutput, error) {
	key := normalize(in.City)
	d, ok := sightseeing[key]
	if !ok {
		return SightseeingOutput{
			Status:       "error",
			ErrorMessage: fmt.Sprintf("%s の観光情報は登録されていない", in.City),
		}, nil
	}
	remember(ctx, key)
	return SightseeingOutput{
		Status:            "success",
		City:              key,
		Spots:             d.Spots,
		RecommendedSeason: d.Season,
	}, nil
}
