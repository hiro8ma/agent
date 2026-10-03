package enginecmp

import (
	_ "embed"
	"encoding/json"
)

//go:generate go run ./gen -out kenpou.json

//go:embed kenpou.json
var kenpouJSON []byte

// Kenpou は日本国憲法の第一条から第百三条までを、1 条を 1 つの文字列にして返す。添字に 1 を足すと条の番号になる。
func Kenpou() ([]string, error) {
	var articles []string
	if err := json.Unmarshal(kenpouJSON, &articles); err != nil {
		return nil, err
	}
	return articles, nil
}
