package search_test

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// evalDocs は語彙の不一致を意図して作った評価用の文書。適合する文書がクエリの語ではなく同義語や狭い語で書かれている。
func evalDocs() []search.Doc {
	return []search.Doc{
		{ID: "del1", Title: "Delete a file", Content: "Select the file and press delete to delete it from the folder."},
		{ID: "del2", Title: "Remove old files", Content: "You can remove files you no longer need from the trash."},
		{ID: "del3", Title: "Erase documents permanently", Content: "Erase a document so nobody can recover it."},
		{ID: "del4", Title: "Removing attachments", Content: "Remove a file attached to a message."},
		{ID: "del5", Title: "Clean up storage", Content: "Erase large files to free up storage space."},
		{ID: "del6", Title: "Deleting folders", Content: "Deleting a folder also deletes every file inside."},
		{ID: "file1", Title: "Upload a file", Content: "Drag a file into the window to upload the file."},
		{ID: "file2", Title: "Share a file", Content: "Send a link so others can open the file."},
		{ID: "file3", Title: "File formats", Content: "Supported file formats include pdf and png."},
		{ID: "file4", Title: "Rename a file", Content: "Right click the file and choose rename."},
		{ID: "file5", Title: "File size limits", Content: "Each file can be up to two gigabytes."},

		{ID: "sp1", Title: "スマホの充電が遅い", Content: "スマホを急速充電器で充電する方法"},
		{ID: "sp2", Title: "スマートフォンの電池", Content: "スマートフォンを充電できないときの確認"},
		{ID: "sp3", Title: "携帯電話の充電", Content: "携帯電話のバッテリーを長持ちさせる充電のコツ"},
		{ID: "sp4", Title: "スマートフォンの充電器", Content: "純正の充電器を使う"},
		{ID: "sp5", Title: "携帯電話が充電されない", Content: "ケーブルを交換する"},
		{ID: "ev1", Title: "電気自動車の充電", Content: "自宅で電気自動車を充電する"},
		{ID: "cam1", Title: "スマホのカメラ", Content: "スマホで写真を撮るコツ"},
		{ID: "pc1", Title: "ノートパソコンの充電", Content: "ノートパソコンの電池を交換する"},
		{ID: "fee1", Title: "携帯電話の料金", Content: "携帯電話の料金プランを見直す"},
		{ID: "mb1", Title: "モバイルバッテリーの選び方", Content: "容量で選ぶ"},

		{ID: "li1", Title: "ログインできない", Content: "パスワードを忘れてログインできない場合"},
		{ID: "li2", Title: "サインインできない", Content: "サインインに失敗するときの対処"},
		{ID: "li3", Title: "ログオンできないとき", Content: "アカウントがロックされてログオンできない"},
		{ID: "li4", Title: "サインオンのエラー", Content: "シングルサインオンでサインオンできない"},
		{ID: "hist1", Title: "ログインの履歴", Content: "ログインした端末を確認する"},
		{ID: "prt1", Title: "印刷できない", Content: "プリンターで印刷できない場合"},
		{ID: "sav1", Title: "保存できない", Content: "ファイルを保存できないときの対処"},

		{ID: "fruit1", Title: "Apple pie recipe", Content: "Bake an apple pie with fresh fruit, cinnamon and sugar."},
		{ID: "fruit2", Title: "Growing fruit trees", Content: "An apple tree needs sun. Harvest the fruit in the autumn orchard."},
		{ID: "fruit3", Title: "Healthy snacks", Content: "Fruit such as an apple or a banana is a healthy snack."},
		{ID: "fruit4", Title: "Orchard harvest", Content: "Pick ripe fruit in the orchard: apple and pear varieties."},
		{ID: "fruit5", Title: "Cider from the orchard", Content: "Press ripe fruit in autumn to make sweet cider."},
		{ID: "co1", Title: "Apple releases a new iPhone", Content: "Apple announced the iPhone at the Apple event."},
		{ID: "co2", Title: "Apple stock rises", Content: "Shares of Apple rose after strong iPhone and Mac sales."},
		{ID: "co3", Title: "Apple Mac update", Content: "Apple released a macOS update for the Mac."},
		{ID: "co4", Title: "Apple services revenue", Content: "Apple reported record services revenue from the App Store."},
		{ID: "co5", Title: "Apple Watch", Content: "The Apple Watch tracks heart rate with an app."},
		{ID: "co6", Title: "iPhone and Mac sales", Content: "Sales of the iPhone and Mac grew and investors bought the stock."},

		{ID: "ph1", Title: "Phone battery drains fast", Content: "Your phone battery drains when the screen stays bright."},
		{ID: "ph2", Title: "Replace a phone battery", Content: "Most phone batteries can be replaced at a repair shop."},
		{ID: "ph3", Title: "Android phone battery", Content: "Android phone settings show battery usage by app."},
		{ID: "ph4", Title: "Phone battery life", Content: "Lower the phone brightness to extend battery life."},
		{ID: "ph5", Title: "Phone battery swelling", Content: "A swollen phone battery is dangerous."},
		{ID: "px1", Title: "Pixel battery saver", Content: "Turn on battery saver on your Pixel."},
		{ID: "gx1", Title: "Galaxy battery tips", Content: "Galaxy owners can limit charging to protect the battery."},
		{ID: "ip1", Title: "iPhone battery health", Content: "Check iPhone battery health in settings."},
		{ID: "ip2", Title: "iPhone battery replacement", Content: "Apple replaces the iPhone battery for a fee."},
		{ID: "ip3", Title: "iPhone battery drain after update", Content: "Some iPhone users see battery drain after an update."},
		{ID: "ip4", Title: "iPhone optimized battery charging", Content: "Optimized charging slows iPhone battery aging."},
		{ID: "ip5", Title: "iPhone low power mode", Content: "Low power mode saves iPhone battery."},
		{ID: "ip6", Title: "iPhone battery percentage", Content: "Show the battery percentage on iPhone."},
		{ID: "bt1", Title: "Laptop battery", Content: "A laptop battery lasts about three years."},
		{ID: "bt2", Title: "Car battery", Content: "Jump start a car battery with cables."},
		{ID: "bt3", Title: "Battery recycling", Content: "Recycle old batteries at a collection point."},
		{ID: "bt4", Title: "AA battery sizes", Content: "AA and AAA battery sizes compared."},
	}
}

type evalQuery struct {
	text     string
	relevant []string
}

// evalQueries の apple は果物を探す意図とし、会社の文書は適合としない。
func evalQueries() map[string]evalQuery {
	return map[string]evalQuery{
		"delete file": {text: "delete file", relevant: []string{"del1", "del2", "del3", "del4", "del5", "del6"}},
		"スマホ 充電":      {text: "スマホ 充電", relevant: []string{"sp1", "sp2", "sp3", "sp4", "sp5"}},
		"ログイン できない":   {text: "ログイン できない", relevant: []string{"li1", "li2", "li3", "li4"}},
		"apple":       {text: "apple", relevant: []string{"fruit1", "fruit2", "fruit3", "fruit4", "fruit5"}},
		"phone battery": {text: "phone battery", relevant: []string{
			"ph1", "ph2", "ph3", "ph4", "ph5", "px1", "gx1", "ip1", "ip2", "ip3", "ip4", "ip5", "ip6",
		}},
		"iphone battery": {text: "iphone battery", relevant: []string{"ip1", "ip2", "ip3", "ip4", "ip5", "ip6"}},
	}
}

func evalAnalyzer() *search.Analyzer {
	return search.NewAnalyzer(
		search.WithNormalization(),
		search.WithEnglish(),
		search.WithStemming(),
		search.WithStopWords(search.DefaultStopWords()...),
	)
}

// evalThesaurus は教材の SudachiDict の同義語グループの形を借りた辞書。phone から狭い語への関係だけを持ち、iphone から phone は足さない。
func evalThesaurus(opts ...search.ThesaurusOption) *search.Thesaurus {
	return search.NewThesaurus(append([]search.ThesaurusOption{
		search.WithSynonymGroup("delete", "remove", "erase"),
		search.WithSynonymGroup("スマホ", "スマートフォン", "携帯電話"),
		search.WithSynonymGroup("ログイン", "サインイン", "ログオン", "サインオン"),
		search.WithNarrower("phone", "iphone", "pixel", "galaxy"),
	}, opts...)...)
}

type evalMetrics struct {
	recall, ndcg, precision float64
}

const evalK = 10

// measure は上位 10 件の recall / nDCG（適合を 1、不適合を 0 とした 2 値）/ precision を求める。
func measure(hits []search.Hit, relevant []string) evalMetrics {
	rel := make(map[string]bool, len(relevant))
	for _, id := range relevant {
		rel[id] = true
	}
	var m evalMetrics
	var dcg, ideal float64
	found := 0
	for i, h := range hits[:min(len(hits), evalK)] {
		if rel[h.Doc.ID] {
			found++
			dcg += 1 / math.Log2(float64(i+2))
		}
	}
	for i := range min(len(relevant), evalK) {
		ideal += 1 / math.Log2(float64(i+2))
	}
	m.recall = float64(found) / float64(len(relevant))
	m.precision = float64(found) / evalK
	if ideal > 0 {
		m.ndcg = dcg / ideal
	}
	return m
}

// evalSetup は同じ文書と辞書から作った、文書拡張なしと文書拡張ありの索引。
type evalSetup struct {
	plain, docExpanded *search.Index
	th                 *search.Thesaurus
}

func newEvalSetup() evalSetup {
	a := evalAnalyzer()
	th := evalThesaurus()
	ds := evalDocs()
	return evalSetup{
		plain:       search.New(ds, search.WithAnalyzer(a)),
		docExpanded: search.New(ds, search.WithAnalyzer(a), search.WithDocExpansion(th)),
		th:          th,
	}
}

// explicitFeedback は利用者が 1 ページ目の適合文書を選んだとみなす。1 ページ目に無ければ最も上の適合文書を 1 件選んだとみなす。
func explicitFeedback(ix *search.Index, q evalQuery) []string {
	rel := make(map[string]bool)
	for _, id := range q.relevant {
		rel[id] = true
	}
	var out []string
	for i, h := range ix.RankQuery(search.Query{Text: q.text}, -1).Hits {
		if rel[h.Doc.ID] && (i < evalK || len(out) == 0) {
			out = append(out, h.Doc.ID)
		}
		if i >= evalK && len(out) > 0 {
			break
		}
	}
	return out
}

type evalMethod struct {
	name string
	run  func(s evalSetup, q evalQuery) []search.Hit
}

func evalMethods() []evalMethod {
	ms := []evalMethod{
		{name: "no expansion", run: func(s evalSetup, q evalQuery) []search.Hit {
			return s.plain.RankQuery(search.Query{Text: q.text}, evalK).Hits
		}},
		{name: "thesaurus query expansion", run: func(s evalSetup, q evalQuery) []search.Hit {
			return s.plain.RankQuery(search.Query{Text: q.text, Thesaurus: s.th}, evalK).Hits
		}},
		{name: "document expansion", run: func(s evalSetup, q evalQuery) []search.Hit {
			return s.docExpanded.RankQuery(search.Query{Text: q.text}, evalK).Hits
		}},
	}
	for _, k := range []int{3, 10} {
		for _, m := range []int{5, 10, 20} {
			ms = append(ms, evalMethod{name: fmt.Sprintf("PRF K=%d M=%d", k, m), run: func(s evalSetup, q evalQuery) []search.Hit {
				r, _ := s.plain.PseudoRelevanceFeedback(search.Query{Text: q.text}, search.Feedback{Docs: k, Terms: m}, evalK)
				return r.Hits
			}})
		}
	}
	ms = append(ms, evalMethod{name: "explicit relevance feedback", run: func(s evalSetup, q evalQuery) []search.Hit {
		r, _ := s.plain.RelevanceFeedback(search.Query{Text: q.text}, explicitFeedback(s.plain, q), search.Feedback{}, evalK)
		return r.Hits
	}})
	return ms
}

// evaluate は手法ごと、クエリごとの指標と、クエリの平均を返す。
func evaluate(s evalSetup) map[string]map[string]evalMetrics {
	out := make(map[string]map[string]evalMetrics)
	qs := evalQueries()
	for _, m := range evalMethods() {
		out[m.name] = make(map[string]evalMetrics)
		var mean evalMetrics
		for name, q := range qs {
			v := measure(m.run(s, q), q.relevant)
			out[m.name][name] = v
			mean.recall += v.recall / float64(len(qs))
			mean.ndcg += v.ndcg / float64(len(qs))
			mean.precision += v.precision / float64(len(qs))
		}
		out[m.name]["mean"] = mean
	}
	return out
}

func TestExpansionEvalTable(t *testing.T) {
	t.Parallel()
	res := evaluate(newEvalSetup())
	names := slices.Sorted(func(yield func(string) bool) {
		for n := range evalQueries() {
			if !yield(n) {
				return
			}
		}
	})
	var b strings.Builder
	for _, m := range evalMethods() {
		fmt.Fprintf(&b, "%-28s mean R=%.3f nDCG=%.3f P=%.3f |", m.name, res[m.name]["mean"].recall, res[m.name]["mean"].ndcg, res[m.name]["mean"].precision)
		for _, n := range names {
			v := res[m.name][n]
			fmt.Fprintf(&b, " %s R=%.2f N=%.2f P=%.2f |", n, v.recall, v.ndcg, v.precision)
		}
		b.WriteString("\n")
	}
	t.Log("\n" + b.String())
	base := res["no expansion"]["mean"]
	for _, name := range []string{"thesaurus query expansion", "document expansion", "explicit relevance feedback"} {
		if got := res[name]["mean"]; got.recall <= base.recall || got.ndcg <= base.ndcg {
			t.Errorf("%s: mean recall %.3f nDCG %.3f, want above no expansion %.3f %.3f", name, got.recall, got.ndcg, base.recall, base.ndcg)
		}
	}
}
