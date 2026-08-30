package chapter01

import "testing"

// 登録済みの都市で天気が返るかを見る。
func TestGetWeatherReturnsReport(t *testing.T) {
	got, err := GetWeather(nil, WeatherInput{City: "Tokyo"})
	if err != nil {
		t.Fatalf("get weather: %v", err)
	}
	if got.Status != "success" {
		t.Errorf("status = %q（success のはず）", got.Status)
	}
	if got.Report == "" {
		t.Error("report が空")
	}
}

// 大文字と前後の空白を吸収するかを見る。
//
// モデルは都市名を「Tokyo」「 tokyo 」のように揺らして渡す。
// 呼び出し側の表記に依存すると、指示に書いても守られないときに落ちる。
func TestGetWeatherNormalizesCity(t *testing.T) {
	for _, city := range []string{"tokyo", "Tokyo", "TOKYO", "  Tokyo  "} {
		got, err := GetWeather(nil, WeatherInput{City: city})
		if err != nil {
			t.Fatalf("%q: %v", city, err)
		}
		if got.Status != "success" {
			t.Errorf("%q で status = %q", city, got.Status)
		}
	}
}

// 未登録の都市がエラーではなく構造化した結果で返るかを見る。
//
// error を返すとモデルが理由を読めず、
// 「登録されていない」と「呼び出しに失敗した」を区別できない。
func TestGetWeatherReportsUnknownCityAsData(t *testing.T) {
	got, err := GetWeather(nil, WeatherInput{City: "Naha"})
	if err != nil {
		t.Fatalf("未登録の都市で error を返した: %v", err)
	}
	if got.Status != "error" {
		t.Errorf("status = %q（error のはず）", got.Status)
	}
	if got.ErrorMessage == "" {
		t.Error("errorMessage が空。モデルが理由を読めない")
	}
	if got.Report != "" {
		t.Errorf("失敗なのに report が入っている: %q", got.Report)
	}
}

// 既定のモデルに依存していないかを見る。
//
// ADK v2.2.0 の LlmAgent の既定は gemini-3-flash-preview になる。
// 暗黙の既定に任せると、バージョンが上がったときに
// 気づかないままモデルが変わる。
func TestModelIsPinnedExplicitly(t *testing.T) {
	if ModelName != "gemini-3.5-flash" {
		t.Errorf("ModelName = %q（教材のメインモデルのはず）", ModelName)
	}
}
