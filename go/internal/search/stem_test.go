package search_test

import (
	"bufio"
	"os"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestPorterStemReferenceVocabulary(t *testing.T) {
	t.Parallel()
	f, err := os.Open("testdata/porter_voc.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	n := 0
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		word, want, ok := strings.Cut(line, "\t")
		if !ok {
			t.Fatalf("malformed line %q", line)
		}
		n++
		if got := search.PorterStem(word); got != want {
			t.Errorf("PorterStem(%q) = %q, want %q", word, got, want)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n < 100 {
		t.Fatalf("read %d pairs, want at least 100", n)
	}
}

func TestPorterStemRules(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		word string
		want string
	}{
		"1a sses は ss にする":          {word: "caresses", want: "caress"},
		"1a ies は i にする":            {word: "ponies", want: "poni"},
		"1a 末尾の s を落とす":             {word: "cats", want: "cat"},
		"1b eed は m>0 のとき ee にする":   {word: "agreed", want: "agre"},
		"1b ing を落とす":               {word: "motoring", want: "motor"},
		"1b 母音の無い語幹からは ing を落とさない":  {word: "sing", want: "sing"},
		"1b 落とした後の at に e を足す":      {word: "conflated", want: "conflat"},
		"1b 落とした後の重なった子音を 1 つにする":   {word: "hopping", want: "hop"},
		"1b 重なった l は残す":             {word: "falling", want: "fall"},
		"1b m=1 の cvc に e を足す":      {word: "filing", want: "file"},
		"1c 母音を含む語幹の y を i にする":     {word: "happy", want: "happi"},
		"1c 母音の無い語幹の y は残す":         {word: "sky", want: "sky"},
		"2 ational を ate にする":       {word: "relational", want: "relat"},
		"2 izer を ize にする":          {word: "digitizer", want: "digit"},
		"2 bli を ble にする（C 版の変更）":   {word: "conformably", want: "conform"},
		"2 logi を log にする（C 版の変更）":  {word: "archaeology", want: "archaeolog"},
		"2 m=0 の語幹には logi の規則をかけない": {word: "theology", want: "theologi"},
		"2 ization を ize にする":       {word: "vietnamization", want: "vietnam"},
		"3 ness を落とす":               {word: "goodness", want: "good"},
		"3 ative を落とす":              {word: "formative", want: "form"},
		"4 ement を m>1 のとき落とす":      {word: "replacement", want: "replac"},
		"4 ion は s か t の後だけ落とす":     {word: "adoption", want: "adopt"},
		"5a m>1 の末尾の e を落とす":        {word: "probate", want: "probat"},
		"5b m>1 の重なった l を 1 つにする":   {word: "controlling", want: "control"},
		"2 文字以下は変えない（C 版の変更）":       {word: "is", want: "is"},
		"複数の段を続けてかける":               {word: "generalizations", want: "gener"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.PorterStem(tc.word); got != tc.want {
				t.Fatalf("PorterStem(%q) = %q, want %q", tc.word, got, tc.want)
			}
		})
	}
}

func TestLemmatize(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		word string
		want string
	}{
		"不規則変化の過去形":          {word: "went", want: "go"},
		"不規則変化の過去分詞":         {word: "been", want: "be"},
		"不規則変化の複数形":          {word: "children", want: "child"},
		"-ies を y に戻す":       {word: "studies", want: "study"},
		"-ches の es を落とす":    {word: "watches", want: "watch"},
		"-xes の es を落とす":     {word: "boxes", want: "box"},
		"-s を落とす":            {word: "pencils", want: "pencil"},
		"-ss は落とさない":         {word: "class", want: "class"},
		"-is は落とさない":         {word: "this", want: "this"},
		"3 文字の語の s は落とさない":   {word: "gas", want: "gas"},
		"-ied を y に戻す":       {word: "studied", want: "study"},
		"-ed を落とす":           {word: "walked", want: "walk"},
		"-eed は -ed として扱わない": {word: "agreed", want: "agreed"},
		"重なった子音を 1 つにする":     {word: "stopped", want: "stop"},
		"at の後に e を足す":       {word: "created", want: "create"},
		"短い cvc の後に e を足す":   {word: "hoped", want: "hope"},
		"-ing を落とす":          {word: "walking", want: "walk"},
		"母音の無い残りなら -ing を残す": {word: "string", want: "string"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.Lemmatize(tc.word); got != tc.want {
				t.Fatalf("Lemmatize(%q) = %q, want %q", tc.word, got, tc.want)
			}
		})
	}
}

func TestStemmingVersusLemmatization(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		words     []string
		wantStem  []string
		wantLemma []string
	}{
		"教材の例 pencil は両方でまとまる": {
			words: []string{"pencil", "pencils"}, wantStem: []string{"pencil", "pencil"}, wantLemma: []string{"pencil", "pencil"},
		},
		"教材の例 walk は両方でまとまる": {
			words: []string{"walk", "walks", "walked"}, wantStem: []string{"walk", "walk", "walk"}, wantLemma: []string{"walk", "walk", "walk"},
		},
		"教材の例 see の saw は語幹化ではまとまらない": {
			words: []string{"see", "sees", "saw"}, wantStem: []string{"see", "see", "saw"}, wantLemma: []string{"see", "see", "see"},
		},
		"語幹化は university と universe をまとめすぎる": {
			words: []string{"university", "universe"}, wantStem: []string{"univers", "univers"}, wantLemma: []string{"university", "universe"},
		},
		"語幹化は organization と organ をまとめすぎる": {
			words: []string{"organization", "organ"}, wantStem: []string{"organ", "organ"}, wantLemma: []string{"organization", "organ"},
		},
		"news と new は両方でまとめすぎる": {
			words: []string{"news", "new"}, wantStem: []string{"new", "new"}, wantLemma: []string{"new", "new"},
		},
		"語幹化は children と child をまとめきれない": {
			words: []string{"children", "child"}, wantStem: []string{"children", "child"}, wantLemma: []string{"child", "child"},
		},
		"見出し語化は品詞を見ないので名詞の saw も see になる": {
			words: []string{"saw"}, wantStem: []string{"saw"}, wantLemma: []string{"see"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			for i, w := range tc.words {
				if got := search.PorterStem(w); got != tc.wantStem[i] {
					t.Errorf("PorterStem(%q) = %q, want %q", w, got, tc.wantStem[i])
				}
				if got := search.Lemmatize(w); got != tc.wantLemma[i] {
					t.Errorf("Lemmatize(%q) = %q, want %q", w, got, tc.wantLemma[i])
				}
			}
		})
	}
}
