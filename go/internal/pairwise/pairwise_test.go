package pairwise_test

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/pairwise"
)

const tol = 1e-9

func near(got, want float64) bool {
	return math.Abs(got-want) <= tol*max(1, math.Abs(want))
}

func TestInterval(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		counts       pairwise.Counts
		mode         pairwise.TieMode
		n            int
		rate         float64
		lower, upper float64
		verdict      pairwise.Verdict
	}{
		"11勝9敗は約 [0.34, 0.74] で区別できない": {
			counts: pairwise.Counts{Wins: 11, Losses: 9}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0.55, lower: 0.3420853424503424, upper: 0.7418021417443758, verdict: pairwise.Indistinguishable,
		},
		"15勝5敗は下限が0.5を超え区別できる（勝ち）": {
			counts: pairwise.Counts{Wins: 15, Losses: 5}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0.75, lower: 0.531299122381256, upper: 0.8881382985923343, verdict: pairwise.Better,
		},
		"18勝2敗は区別できる（勝ち）": {
			counts: pairwise.Counts{Wins: 18, Losses: 2}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0.9, lower: 0.6989663547715128, upper: 0.9721335187862319, verdict: pairwise.Better,
		},
		"2勝18敗は上限が0.5を下回り区別できる（負け）": {
			counts: pairwise.Counts{Wins: 2, Losses: 18}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0.1, lower: 0.027866481213768224, upper: 0.3010336452284872, verdict: pairwise.Worse,
		},
		"20勝0敗は上限が1で下限が約0.84": {
			counts: pairwise.Counts{Wins: 20}, mode: pairwise.ExcludeTies,
			n: 20, rate: 1, lower: 0.8388748419471808, upper: 1, verdict: pairwise.Better,
		},
		"0勝20敗は下限が0で上限が約0.16": {
			counts: pairwise.Counts{Losses: 20}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0, lower: 0, upper: 0.16112515805281927, verdict: pairwise.Worse,
		},
		"4勝0敗は件数が少なくても下限が0.5を超える": {
			counts: pairwise.Counts{Wins: 4}, mode: pairwise.ExcludeTies,
			n: 4, rate: 1, lower: 0.5101091635454028, upper: 1, verdict: pairwise.Better,
		},
		"0件は勝率0.5、区間 [0, 1] で区別できない": {
			counts: pairwise.Counts{}, mode: pairwise.ExcludeTies,
			n: 0, rate: 0.5, lower: 0, upper: 1, verdict: pairwise.Indistinguishable,
		},
		"0件は引き分けを0.5勝に数えても区間 [0, 1]": {
			counts: pairwise.Counts{}, mode: pairwise.HalfWin,
			n: 0, rate: 0.5, lower: 0, upper: 1, verdict: pairwise.Indistinguishable,
		},
		"引き分けだけを除くと0件になり区間 [0, 1]": {
			counts: pairwise.Counts{Ties: 10}, mode: pairwise.ExcludeTies,
			n: 0, rate: 0.5, lower: 0, upper: 1, verdict: pairwise.Indistinguishable,
		},
		"引き分けだけを0.5勝に数えると10件の勝率0.5の区間": {
			counts: pairwise.Counts{Ties: 10}, mode: pairwise.HalfWin,
			n: 10, rate: 0.5, lower: 0.23659309051256405, upper: 0.763406909487436, verdict: pairwise.Indistinguishable,
		},
		"引き分けを除くと11勝9敗4分は11勝9敗と同じ区間": {
			counts: pairwise.Counts{Wins: 11, Losses: 9, Ties: 4}, mode: pairwise.ExcludeTies,
			n: 20, rate: 0.55, lower: 0.3420853424503424, upper: 0.7418021417443758, verdict: pairwise.Indistinguishable,
		},
		"引き分けを0.5勝に数えると11勝9敗4分は24件の13勝": {
			counts: pairwise.Counts{Wins: 11, Losses: 9, Ties: 4}, mode: pairwise.HalfWin,
			n: 24, rate: 13.0 / 24, lower: 0.3507486535891949, upper: 0.72108666268789, verdict: pairwise.Indistinguishable,
		},
		"10勝6敗8分を0.5勝に数えると区間の上限が下がる": {
			counts: pairwise.Counts{Wins: 10, Losses: 6, Ties: 8}, mode: pairwise.HalfWin,
			n: 24, rate: 14.0 / 24, lower: 0.38834665865228757, upper: 0.7553239739018823, verdict: pairwise.Indistinguishable,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := tc.counts.Interval(tc.mode)
			if got.Ties != tc.mode || got.N != tc.n || !near(got.Rate, tc.rate) || !near(got.Lower, tc.lower) || !near(got.Upper, tc.upper) || got.Verdict != tc.verdict {
				t.Errorf("Interval = %s / %d件 / 勝率 %.6f / [%.10f, %.10f] / %s, want %s / %d件 / 勝率 %.6f / [%.10f, %.10f] / %s",
					got.Ties, got.N, got.Rate, got.Lower, got.Upper, got.Verdict, tc.mode, tc.n, tc.rate, tc.lower, tc.upper, tc.verdict)
			}
		})
	}
}

func TestSignTest(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		counts pairwise.Counts
		want   float64
	}{
		"11勝9敗は約0.82":            {counts: pairwise.Counts{Wins: 11, Losses: 9}, want: 0.8238029479980469},
		"9勝11敗は11勝9敗と同じ":         {counts: pairwise.Counts{Wins: 9, Losses: 11}, want: 0.8238029479980469},
		"15勝5敗は約0.041":           {counts: pairwise.Counts{Wins: 15, Losses: 5}, want: 0.04138946533203125},
		"18勝2敗は約0.0004":          {counts: pairwise.Counts{Wins: 18, Losses: 2}, want: 0.0004024505615234375},
		"20勝0敗は2の-19乗":           {counts: pairwise.Counts{Wins: 20}, want: 1.9073486328125e-06},
		"4勝0敗は0.125で、区間の判定と食い違う": {counts: pairwise.Counts{Wins: 4}, want: 0.125},
		"10勝10敗は1を超えず1":          {counts: pairwise.Counts{Wins: 10, Losses: 10}, want: 1},
		"0件は1":                   {counts: pairwise.Counts{}, want: 1},
		"引き分けだけは0件と同じ1":          {counts: pairwise.Counts{Ties: 10}, want: 1},
		"引き分けは p 値に入らない":         {counts: pairwise.Counts{Wins: 15, Losses: 5, Ties: 100}, want: 0.04138946533203125},
		"5100勝4900敗は約0.047":      {counts: pairwise.Counts{Wins: 5100, Losses: 4900}, want: 0.04658552770494739},
		"5000勝5000敗は1":           {counts: pairwise.Counts{Wins: 5000, Losses: 5000}, want: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := tc.counts.SignTest(); !near(got, tc.want) {
				t.Errorf("SignTest(%d勝%d敗%d分) = %.12g, want %.12g", tc.counts.Wins, tc.counts.Losses, tc.counts.Ties, got, tc.want)
			}
		})
	}
}

func TestRequiredN(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		p       float64
		want    int
		wantErr bool
	}{
		"真の勝率0.55は783件":           {p: 0.55, want: 783},
		"真の勝率0.60は194件":           {p: 0.6, want: 194},
		"真の勝率0.65は85件":            {p: 0.65, want: 85},
		"真の勝率0.70は47件":            {p: 0.7, want: 47},
		"0.5を挟んで対称で、0.45は0.55と同じ": {p: 0.45, want: 783},
		"0.30は0.70と同じ":            {p: 0.3, want: 47},
		"0.5は見分けられないのでエラー":        {p: 0.5, wantErr: true},
		"0はエラー":                   {p: 0, wantErr: true},
		"1はエラー":                   {p: 1, wantErr: true},
		"負の値はエラー":                 {p: -0.1, wantErr: true},
		"NaN はエラー":                {p: math.NaN(), wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := pairwise.RequiredN(tc.p)
			if (err != nil) != tc.wantErr || got != tc.want {
				t.Errorf("RequiredN(%v) = %d, %v, want %d, エラー %v", tc.p, got, err, tc.want, tc.wantErr)
			}
		})
	}
}

func TestHolm(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		ps, want []float64
	}{
		"小さい順に件数の倍率を減らしてかけ、元の順に戻す": {ps: []float64{0.01, 0.04, 0.03, 0.005}, want: []float64{0.03, 0.06, 0.06, 0.02}},
		"前の補正後の値より小さくしない":          {ps: []float64{0.01, 0.04, 0.03}, want: []float64{0.03, 0.06, 0.06}},
		"1で頭打ちにする":                 {ps: []float64{0.6, 0.7}, want: []float64{1, 1}},
		"同じ値は大きいほうの倍率に揃う":          {ps: []float64{0.02, 0.02}, want: []float64{0.04, 0.04}},
		"1件は補正しない":                 {ps: []float64{0.04}, want: []float64{0.04}},
		"0件は0件":                    {ps: []float64{}, want: []float64{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := pairwise.Holm(tc.ps)
			ok := len(got) == len(tc.want)
			for i := range min(len(got), len(tc.want)) {
				ok = ok && near(got[i], tc.want[i])
			}
			if !ok {
				t.Errorf("Holm(%v) = %v, want %v", tc.ps, got, tc.want)
			}
		})
	}
}

func TestAnalyzeFixtures(t *testing.T) {
	t.Parallel()
	cs, err := pairwise.Load("testdata/comparisons.json")
	if err != nil {
		t.Fatal(err)
	}
	results := pairwise.Analyze(cs)
	byName := map[string]pairwise.Result{}
	for _, r := range results {
		byName[r.Name] = r
	}
	testCases := map[string]struct {
		p, holm          float64
		exclude, halfWin pairwise.Verdict
	}{
		"model-a vs model-b": {p: 0.8238029479980469, holm: 0.90899658203125, exclude: pairwise.Indistinguishable, halfWin: pairwise.Indistinguishable},
		"model-a vs model-c": {p: 0.04138946533203125, holm: 0.165557861328125, exclude: pairwise.Better, halfWin: pairwise.Better},
		"model-a vs model-d": {p: 0.0004024505615234375, holm: 0.002414703369140625, exclude: pairwise.Better, halfWin: pairwise.Better},
		"model-b vs model-c": {p: 0.454498291015625, holm: 0.90899658203125, exclude: pairwise.Indistinguishable, halfWin: pairwise.Indistinguishable},
		"model-b vs model-d": {p: 0.125, holm: 0.375, exclude: pairwise.Better, halfWin: pairwise.Better},
		"model-c vs model-d": {p: 0.0025768280029296875, holm: 0.012884140014648438, exclude: pairwise.Worse, halfWin: pairwise.Worse},
	}
	if len(results) != len(testCases) {
		t.Fatalf("%d件の比較を読んだ, want %d件", len(results), len(testCases))
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r, ok := byName[tn]
			if !ok {
				t.Fatalf("%s を読んでいない", tn)
			}
			ex, hw := r.Intervals[pairwise.ExcludeTies], r.Intervals[pairwise.HalfWin]
			if !near(r.P, tc.p) || !near(r.Holm, tc.holm) || ex.Ties != pairwise.ExcludeTies || ex.Verdict != tc.exclude || hw.Ties != pairwise.HalfWin || hw.Verdict != tc.halfWin {
				t.Errorf("p %.12g / Holm %.12g / %s %s / %s %s, want %.12g / %.12g / %s / %s",
					r.P, r.Holm, ex.Ties, ex.Verdict, hw.Ties, hw.Verdict, tc.p, tc.holm, tc.exclude, tc.halfWin)
			}
		})
	}
}

func TestLoadJSONAndCSVAgree(t *testing.T) {
	t.Parallel()
	fromJSON, err := pairwise.Load("testdata/comparisons.json")
	if err != nil {
		t.Fatal(err)
	}
	fromCSV, err := pairwise.Load("testdata/comparisons.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(fromJSON) == 0 || !reflect.DeepEqual(fromJSON, fromCSV) {
		t.Errorf("JSON %v と CSV %v が一致しない", fromJSON, fromCSV)
	}
}

func TestLoadRejects(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		file, body, wantErr string
	}{
		"負の数":          {file: "c.json", body: `[{"name":"model-a vs model-b","wins":-1,"losses":9,"ties":0}]`, wantErr: "負の数"},
		"JSON の知らない項目": {file: "c.json", body: `[{"name":"model-a vs model-b","win":11,"losses":9,"ties":0}]`, wantErr: "unknown field"},
		"CSV の見出しの違い":  {file: "c.csv", body: "name,losses,wins,ties\nmodel-a vs model-b,9,11,0\n", wantErr: "1行目"},
		"CSV の数でないセル":  {file: "c.csv", body: "name,wins,losses,ties\nmodel-a vs model-b,11,x,0\n", wantErr: "2行目の losses"},
		"CSV の負の数":     {file: "c.csv", body: "name,wins,losses,ties\nmodel-a vs model-b,11,9,-2\n", wantErr: "負の数"},
		"読まない拡張子":      {file: "c.txt", body: "11,9,0", wantErr: "拡張子"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			path := filepath.Join(t.TempDir(), tc.file)
			if err := os.WriteFile(path, []byte(tc.body), 0o600); err != nil {
				t.Fatal(err)
			}
			_, err := pairwise.Load(path)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Load のエラー %v に %q が無い", err, tc.wantErr)
			}
		})
	}
}
