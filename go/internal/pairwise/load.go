package pairwise

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

var csvHeader = []string{"name", "wins", "losses", "ties"}

// Load は比較の一覧を .json（オブジェクトの配列）か .csv（見出しが name,wins,losses,ties）から読む。
func Load(path string) ([]Counts, error) {
	b, err := fs.ReadFile(os.DirFS(filepath.Dir(path)), filepath.Base(path))
	if err != nil {
		return nil, err
	}
	var cs []Counts
	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".json":
		cs, err = parseJSON(b)
	case ".csv":
		cs, err = parseCSV(b)
	default:
		return nil, fmt.Errorf("%s: 拡張子 %q は読まない（.json か .csv にする）", path, ext)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for _, c := range cs {
		if err := c.Validate(); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}
	return cs, nil
}

func parseJSON(b []byte) ([]Counts, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var cs []Counts
	if err := dec.Decode(&cs); err != nil {
		return nil, err
	}
	return cs, nil
}

func parseCSV(b []byte) ([]Counts, error) {
	rows, err := csv.NewReader(bytes.NewReader(b)).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 || !slices.Equal(rows[0], csvHeader) {
		return nil, fmt.Errorf("1行目は %s にする", strings.Join(csvHeader, ","))
	}
	cs := make([]Counts, 0, len(rows)-1)
	for i, row := range rows[1:] {
		var n [3]int
		for j, cell := range row[1:] {
			if n[j], err = strconv.Atoi(strings.TrimSpace(cell)); err != nil {
				return nil, fmt.Errorf("%d行目の %s: %w", i+2, csvHeader[j+1], err)
			}
		}
		cs = append(cs, Counts{Name: row[0], Wins: n[0], Losses: n[1], Ties: n[2]})
	}
	return cs, nil
}
