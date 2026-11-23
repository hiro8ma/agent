package productld_test

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/productld"
)

func ld(scripts ...string) []byte {
	var b strings.Builder
	b.WriteString("<html><head>")
	for _, s := range scripts {
		b.WriteString(`<script type="application/ld+json">` + s + "</script>")
	}
	b.WriteString("</head><body></body></html>")
	return []byte(b.String())
}

func js(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestExtract(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		page []byte
		want productld.Page
	}{
		"単体のオブジェクトの Product を読む": {
			page: ld(`{"@context":"https://schema.org","@type":"Product","name":"見本の皿","sku":"S-1","offers":{"@type":"Offer","price":1200,"priceCurrency":"JPY","availability":"https://schema.org/InStock"}}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{
				{Name: "見本の皿", SKU: "S-1", Offers: []productld.Offer{{Price: new(1200.0), Currency: "JPY", Availability: "InStock"}}},
			}},
		},
		"最上位の配列の Product を全部拾う": {
			page: ld(`[{"@type":"Product","name":"見本の皿"},{"@type":"Product","name":"見本の箸"}]`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{Name: "見本の皿"}, {Name: "見本の箸"}}},
		},
		"@graph の中の Product を拾う": {
			page: ld(`{"@context":"https://schema.org","@graph":[{"@type":"BreadcrumbList"},{"@type":"Product","name":"見本の皿"}]}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{Name: "見本の皿"}}},
		},
		"offers の配列を全部残す": {
			page: ld(`{"@type":"Product","name":"見本の皿","offers":[{"price":1200,"priceCurrency":"JPY"},{"price":900,"priceCurrency":"JPY"}]}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{
				{Name: "見本の皿", Offers: []productld.Offer{{Price: new(1200.0), Currency: "JPY"}, {Price: new(900.0), Currency: "JPY"}}},
			}},
		},
		"ProductGroup の hasVariant を商品ごとに分け、variant に無い項目を受け継ぐ": {
			page: ld(`{"@type":"ProductGroup","name":"見本の椀","brand":"見本窯","aggregateRating":{"ratingValue":4},"hasVariant":[{"@type":"Product","name":"見本の椀 白","sku":"W-1"},{"@type":"Product","sku":"W-2","brand":"別の見本窯"}]}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{
				{Name: "見本の椀 白", Brand: "見本窯", SKU: "W-1", Rating: true},
				{Name: "見本の椀", Brand: "別の見本窯", SKU: "W-2", Rating: true},
			}},
		},
		"Product の中の関連商品は数えない": {
			page: ld(`{"@type":"Product","name":"見本の皿","isRelatedTo":{"@type":"Product","name":"見本の箸"}}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{Name: "見本の皿"}}},
		},
		"壊れた script だけ飛ばして数える": {
			page: ld(`{"@type":"Product",}`, `{"@type":"Product","name":"見本の皿"}`),
			want: productld.Page{Scripts: 2, Broken: 1, Products: []productld.Product{{Name: "見本の皿"}}},
		},
		"空の script を壊れたものとして数える": {
			page: ld(``),
			want: productld.Page{Scripts: 1, Broken: 1},
		},
		"JSON-LD でない script は読まない": {
			page: []byte(`<script type="application/json">{"@type":"Product","name":"見本の皿"}</script><script>var p = {"@type":"Product"};</script>`),
			want: productld.Page{},
		},
		"type の大文字と charset を許す": {
			page: []byte(`<script type="Application/LD+JSON; charset=utf-8">{"@type":"Product","name":"見本の皿"}</script>`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{Name: "見本の皿"}}},
		},
		"@type の配列と URL の形、文字列の price、brand と image のオブジェクト、gtin を読む": {
			page: ld(`{"@type":["https://schema.org/Product","Thing"],"name":"見本の皿","brand":{"@type":"Brand","name":"見本窯"},"image":{"@type":"ImageObject","url":"/a.jpg"},"gtin13":"0000000000017","aggregateRating":{"ratingValue":"4.5"},"hasMerchantReturnPolicy":{"@type":"MerchantReturnPolicy"},"offers":{"price":"1200.5","priceCurrency":"JPY","availability":"http://schema.org/OutOfStock"}}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{
				Name: "見本の皿", Brand: "見本窯", Image: true, GTIN: "0000000000017", Rating: true, ReturnPolicy: true,
				Offers: []productld.Offer{{Price: new(1200.5), Currency: "JPY", Availability: "OutOfStock"}},
			}}},
		},
		"配送の日数は maxValue を読み、日でない単位は読まない": {
			page: ld(`{"@type":"Product","offers":{"shippingDetails":[{"shippingRate":{"value":"0"},"deliveryTime":{"handlingTime":{"minValue":0,"maxValue":1,"unitCode":"DAY"},"transitTime":{"minValue":2,"maxValue":4,"unitCode":"d"}}},{"deliveryTime":{"handlingTime":{"maxValue":12,"unitCode":"HUR"},"transitTime":{"maxValue":3}}}]}}`),
			want: productld.Page{Scripts: 1, Products: []productld.Product{{Offers: []productld.Offer{{Shipping: []productld.Shipping{
				{Rate: new(0.0), Handling: new(1.0), Transit: new(4.0)},
				{Transit: new(3.0)},
			}}}}}},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := productld.Extract(tc.page); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Extract =\n%s\nwant\n%s", js(got), js(tc.want))
			}
		})
	}
}

func missing(p productld.Product) string {
	var out []string
	for _, f := range productld.Fields {
		if !p.Has(f) {
			out = append(out, f.String())
		}
	}
	return p.Name + ": " + strings.Join(out, ", ")
}

func TestLoadFixtures(t *testing.T) {
	t.Parallel()
	results, err := productld.Load("testdata")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		scripts, broken int
		products        []string
	}{
		"broken.html":          {scripts: 2, broken: 1, products: []string{"真鍮のしおり: hasMerchantReturnPolicy, aggregateRating"}},
		"client-rendered.html": {},
		"complete.html":        {scripts: 1, products: []string{"リネンのシャツ: "}},
		"graph-group.html": {scripts: 1, products: []string{
			"ウールのマフラー 紺: ",
			"ウールのマフラー 灰: hasMerchantReturnPolicy",
		}},
		"no-shipping.html": {scripts: 1, products: []string{"ステンレスの水筒: shippingRate, handlingTime, transitTime, hasMerchantReturnPolicy, aggregateRating"}},
		"stock-only.html":  {scripts: 1, products: []string{"木のトレイ: sku / gtin, price, priceCurrency, shippingRate, handlingTime, transitTime, hasMerchantReturnPolicy, aggregateRating"}},
	}
	var files []string
	for _, r := range results {
		files = append(files, r.File)
		w, ok := want[r.File]
		if !ok {
			t.Errorf("想定していないファイル %s", r.File)
			continue
		}
		var got []string
		for _, p := range r.Products {
			got = append(got, missing(p))
		}
		if r.Scripts != w.scripts || r.Broken != w.broken || !slices.Equal(got, w.products) {
			t.Errorf("%s: script %d、壊れた %d、無い項目 %q, want %d、%d、%q", r.File, r.Scripts, r.Broken, got, w.scripts, w.broken, w.products)
		}
	}
	if !slices.IsSorted(files) || len(files) != len(want) {
		t.Errorf("読んだファイル %q は、名前の順の %d件でない", files, len(want))
	}
}

func TestLoadFile(t *testing.T) {
	t.Parallel()
	results, err := productld.Load("testdata/complete.html")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].File != "complete.html" || len(results[0].Products) != 1 {
		t.Errorf("Load(complete.html) = %s", js(results))
	}
}

func TestJudge(t *testing.T) {
	t.Parallel()
	all := productld.Condition{MaxPrice: new(5000.0), Currency: "JPY", MaxDays: new(3.0), InStock: true}
	ship := func(handling, transit float64) []productld.Shipping {
		return []productld.Shipping{{Handling: new(handling), Transit: new(transit)}}
	}
	testCases := map[string]struct {
		offers []productld.Offer
		cond   productld.Condition
		want   productld.Verdict
	}{
		"同じ Offer で全部の条件を満たせば満たす": {
			offers: []productld.Offer{{Price: new(4800.0), Currency: "JPY", Availability: "InStock", Shipping: ship(1, 2)}},
			cond:   all,
			want:   productld.Verdict{Status: productld.Satisfied},
		},
		"届く日数は handlingTime と transitTime の maxValue の和で見る": {
			offers: []productld.Offer{{Shipping: ship(2, 2)}},
			cond:   productld.Condition{MaxDays: new(3.0)},
			want:   productld.Verdict{Status: productld.Failed},
		},
		"届く日数がちょうど上限なら満たす": {
			offers: []productld.Offer{{Shipping: ship(1, 2)}},
			cond:   productld.Condition{MaxDays: new(3.0)},
			want:   productld.Verdict{Status: productld.Satisfied},
		},
		"handlingTime だけでは判定できない": {
			offers: []productld.Offer{{Shipping: []productld.Shipping{{Handling: new(1.0)}}}},
			cond:   productld.Condition{MaxDays: new(3.0)},
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"transitTime の maxValue"}},
		},
		"shippingDetails が無ければ満たさないでなく判定できない": {
			offers: []productld.Offer{{Price: new(4800.0), Currency: "JPY", Availability: "InStock"}},
			cond:   all,
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"shippingDetails"}},
		},
		"price が無ければ予算で判定できない": {
			offers: []productld.Offer{{Currency: "JPY"}},
			cond:   productld.Condition{MaxPrice: new(5000.0), Currency: "JPY"},
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"price"}},
		},
		"priceCurrency が条件の通貨と違えば判定できない": {
			offers: []productld.Offer{{Price: new(30.0), Currency: "USD"}},
			cond:   productld.Condition{MaxPrice: new(5000.0), Currency: "JPY"},
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"priceCurrency が JPY でない"}},
		},
		"予算を超えれば、配送の項目が無くても満たさない": {
			offers: []productld.Offer{{Price: new(6000.0), Currency: "JPY", Availability: "InStock"}},
			cond:   all,
			want:   productld.Verdict{Status: productld.Failed},
		},
		"在庫切れは満たさない": {
			offers: []productld.Offer{{Availability: "OutOfStock"}},
			cond:   productld.Condition{InStock: true},
			want:   productld.Verdict{Status: productld.Failed},
		},
		"列挙に無い availability の値は判定できない": {
			offers: []productld.Offer{{Availability: "在庫あり"}},
			cond:   productld.Condition{InStock: true},
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"availability の値 在庫あり"}},
		},
		"2つ目の Offer だけが満たしても満たす": {
			offers: []productld.Offer{
				{Price: new(6000.0), Currency: "JPY", Availability: "InStock", Shipping: ship(1, 2)},
				{Price: new(4500.0), Currency: "JPY", Availability: "InStock", Shipping: ship(1, 2)},
			},
			cond: all,
			want: productld.Verdict{Status: productld.Satisfied},
		},
		"条件を別々の Offer で満たしても満たさない": {
			offers: []productld.Offer{
				{Price: new(4500.0), Currency: "JPY", Availability: "OutOfStock", Shipping: ship(1, 2)},
				{Price: new(6000.0), Currency: "JPY", Availability: "InStock", Shipping: ship(1, 2)},
			},
			cond: all,
			want: productld.Verdict{Status: productld.Failed},
		},
		"見ない条件の項目は無くてよい": {
			offers: []productld.Offer{{Availability: "InStock"}},
			cond:   productld.Condition{InStock: true},
			want:   productld.Verdict{Status: productld.Satisfied},
		},
		"Offer が無ければ見る条件の項目を理由に返す": {
			cond: all,
			want: productld.Verdict{Status: productld.Unknown, Missing: []string{"price", "priceCurrency", "availability", "shippingDetails"}},
		},
		"配送の詳細が複数なら最も遅いもので見る": {
			offers: []productld.Offer{{Shipping: append(ship(0, 1), ship(1, 4)...)}},
			cond:   productld.Condition{MaxDays: new(3.0)},
			want:   productld.Verdict{Status: productld.Failed},
		},
		"遅い配送で外れれば、ほかの配送に欠けがあっても満たさない": {
			offers: []productld.Offer{{Shipping: append(ship(2, 3), productld.Shipping{Transit: new(1.0)})}},
			cond:   productld.Condition{MaxDays: new(3.0)},
			want:   productld.Verdict{Status: productld.Failed},
		},
		"判定できない Offer の理由を重ねずに並べる": {
			offers: []productld.Offer{{Currency: "JPY", Availability: "InStock"}, {Currency: "JPY"}},
			cond:   productld.Condition{MaxPrice: new(5000.0), Currency: "JPY", InStock: true},
			want:   productld.Verdict{Status: productld.Unknown, Missing: []string{"price", "availability"}},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := productld.Judge(productld.Product{Offers: tc.offers}, tc.cond); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Judge = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestCountFixtures(t *testing.T) {
	t.Parallel()
	results, err := productld.Load("testdata")
	if err != nil {
		t.Fatal(err)
	}
	var products []productld.Product
	for _, r := range results {
		products = append(products, r.Products...)
	}
	testCases := map[string]struct {
		cond productld.Condition
		want [3]int
	}{
		"5000円以下": {cond: productld.Condition{MaxPrice: new(5000.0), Currency: "JPY"}, want: [3]int{5, 0, 1}},
		"3日以内":    {cond: productld.Condition{MaxDays: new(3.0)}, want: [3]int{3, 1, 2}},
		"在庫あり":    {cond: productld.Condition{InStock: true}, want: [3]int{5, 1, 0}},
		"すべて":     {cond: productld.Condition{MaxPrice: new(5000.0), Currency: "JPY", MaxDays: new(3.0), InStock: true}, want: [3]int{2, 2, 2}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := productld.Count(products, tc.cond); got != tc.want {
				t.Errorf("満たす / 満たさない / 判定できない = %v, want %v", got, tc.want)
			}
		})
	}
}
