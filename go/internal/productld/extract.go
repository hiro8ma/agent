// Package productld は、商品ページの HTML から schema.org の Product と Offer の JSON-LD を取り出し、エージェントが比較に使う項目の有無と、予算 / 届くまでの日数 / 在庫の条件で判定できるかを返す。
package productld

import (
	"bytes"
	"encoding/json"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/net/html"
)

// Field はエージェントが商品の比較に使う項目。
type Field int

const (
	Name Field = iota
	Brand
	Image
	Identifier
	Price
	Currency
	Availability
	ShippingRate
	HandlingTime
	TransitTime
	ReturnPolicy
	Rating
)

// Fields は表に並べる順の全部の項目。
var Fields = []Field{Name, Brand, Image, Identifier, Price, Currency, Availability, ShippingRate, HandlingTime, TransitTime, ReturnPolicy, Rating}

func (f Field) String() string {
	return [...]string{
		"name", "brand", "image", "sku / gtin", "price", "priceCurrency", "availability",
		"shippingRate", "handlingTime", "transitTime", "hasMerchantReturnPolicy", "aggregateRating",
	}[f]
}

// Product は JSON-LD の Product 1件。数値の項目は、無いか読めなければ nil。
type Product struct {
	Name         string
	Brand        string
	Image        bool
	SKU          string
	GTIN         string
	Offers       []Offer
	ReturnPolicy bool
	Rating       bool
}

// Offer は Product の offers の1件。Availability は schema.org の URL を外した値（InStock など）。
type Offer struct {
	Price        *float64
	Currency     string
	Availability string
	Shipping     []Shipping
	ReturnPolicy bool
}

// Shipping は OfferShippingDetails の1件。Handling と Transit は日数の maxValue。
type Shipping struct {
	Rate     *float64
	Handling *float64
	Transit  *float64
}

// Has は項目が有るかを返す。Offer の項目は、どれか1つの Offer に有れば有るとする。
func (p Product) Has(f Field) bool {
	switch f {
	case Name:
		return p.Name != ""
	case Brand:
		return p.Brand != ""
	case Image:
		return p.Image
	case Identifier:
		return p.SKU != "" || p.GTIN != ""
	case Rating:
		return p.Rating
	}
	if f == ReturnPolicy && p.ReturnPolicy {
		return true
	}
	return slices.ContainsFunc(p.Offers, func(o Offer) bool { return o.has(f) })
}

func (o Offer) has(f Field) bool {
	switch f {
	case Price:
		return o.Price != nil
	case Currency:
		return o.Currency != ""
	case Availability:
		return o.Availability != ""
	case ReturnPolicy:
		return o.ReturnPolicy
	case ShippingRate:
		return slices.ContainsFunc(o.Shipping, func(s Shipping) bool { return s.Rate != nil })
	case HandlingTime:
		return slices.ContainsFunc(o.Shipping, func(s Shipping) bool { return s.Handling != nil })
	case TransitTime:
		return slices.ContainsFunc(o.Shipping, func(s Shipping) bool { return s.Transit != nil })
	}
	return false
}

// Page は1ページから取り出した結果。Scripts は JSON-LD の script の数、Broken はそのうち JSON として読めなかった数。
type Page struct {
	Products []Product
	Scripts  int
	Broken   int
}

// Extract は HTML の `<script type="application/ld+json">` を全部読み、Product を文書の順に返す。
func Extract(page []byte) Page {
	var p Page
	z := html.NewTokenizer(bytes.NewReader(page))
	inLD := false
	for {
		switch z.Next() {
		case html.ErrorToken:
			return p
		case html.StartTagToken:
			name, hasAttr := z.TagName()
			inLD = string(name) == "script" && hasAttr && isJSONLD(z)
		case html.TextToken:
			if !inLD {
				continue
			}
			inLD = false
			p.Scripts++
			var v any
			if err := json.Unmarshal(z.Text(), &v); err != nil {
				p.Broken++
				continue
			}
			p.Products = collect(v, p.Products)
		case html.EndTagToken:
			if inLD {
				p.Scripts++
				p.Broken++
			}
			inLD = false
		}
	}
}

func isJSONLD(z *html.Tokenizer) bool {
	for {
		key, val, more := z.TagAttr()
		if string(key) == "type" {
			media, _, _ := strings.Cut(string(val), ";")
			return strings.EqualFold(strings.TrimSpace(media), "application/ld+json")
		}
		if !more {
			return false
		}
	}
}

func collect(v any, out []Product) []Product {
	switch t := v.(type) {
	case []any:
		for _, e := range t {
			out = collect(e, out)
		}
	case map[string]any:
		if g, ok := t["@graph"]; ok {
			out = collect(g, out)
		}
		switch {
		case isType(t, "ProductGroup"):
			group := product(t)
			variants := 0
			for _, n := range nodes(t["hasVariant"]) {
				if isType(n, "Product") {
					out = append(out, inherit(product(n), group))
					variants++
				}
			}
			if variants == 0 {
				out = append(out, group)
			}
		case isType(t, "Product"):
			out = append(out, product(t))
		}
	}
	return out
}

func inherit(v, group Product) Product {
	if v.Name == "" {
		v.Name = group.Name
	}
	if v.Brand == "" {
		v.Brand = group.Brand
	}
	v.Image = v.Image || group.Image
	v.ReturnPolicy = v.ReturnPolicy || group.ReturnPolicy
	v.Rating = v.Rating || group.Rating
	return v
}

func product(m map[string]any) Product {
	p := Product{
		Name:         text(m["name"]),
		Brand:        brand(m["brand"]),
		Image:        image(m["image"]),
		SKU:          text(m["sku"]),
		ReturnPolicy: len(nodes(m["hasMerchantReturnPolicy"])) > 0,
	}
	for _, k := range []string{"gtin", "gtin14", "gtin13", "gtin12", "gtin8"} {
		if p.GTIN = text(m[k]); p.GTIN != "" {
			break
		}
	}
	for _, r := range nodes(m["aggregateRating"]) {
		p.Rating = p.Rating || text(r["ratingValue"]) != ""
	}
	for _, o := range nodes(m["offers"]) {
		p.Offers = append(p.Offers, offer(o))
	}
	return p
}

func offer(m map[string]any) Offer {
	o := Offer{
		Price:        number(m["price"]),
		Currency:     text(m["priceCurrency"]),
		Availability: term(text(m["availability"])),
		ReturnPolicy: len(nodes(m["hasMerchantReturnPolicy"])) > 0,
	}
	for _, s := range nodes(m["shippingDetails"]) {
		delivery := first(s["deliveryTime"])
		o.Shipping = append(o.Shipping, Shipping{
			Rate:     number(first(s["shippingRate"])["value"]),
			Handling: days(first(delivery["handlingTime"])),
			Transit:  days(first(delivery["transitTime"])),
		})
	}
	return o
}

func days(q map[string]any) *float64 {
	if unit := text(q["unitCode"]); unit != "" && unit != "DAY" && unit != "d" {
		return nil
	}
	return number(q["maxValue"])
}

func isType(m map[string]any, name string) bool {
	switch t := m["@type"].(type) {
	case string:
		return term(t) == name
	case []any:
		for _, e := range t {
			if s, ok := e.(string); ok && term(s) == name {
				return true
			}
		}
	}
	return false
}

func term(s string) string {
	for _, prefix := range []string{"https://schema.org/", "http://schema.org/", "schema:"} {
		if rest, ok := strings.CutPrefix(s, prefix); ok {
			return rest
		}
	}
	return s
}

func nodes(v any) []map[string]any {
	switch t := v.(type) {
	case map[string]any:
		return []map[string]any{t}
	case []any:
		var out []map[string]any
		for _, e := range t {
			if m, ok := e.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	return nil
}

func first(v any) map[string]any {
	if n := nodes(v); len(n) > 0 {
		return n[0]
	}
	return nil
}

func text(v any) string {
	switch t := v.(type) {
	case string:
		return strings.TrimSpace(t)
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	}
	return ""
}

func number(v any) *float64 {
	switch t := v.(type) {
	case float64:
		return &t
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(t), 64); err == nil {
			return &f
		}
	}
	return nil
}

func brand(v any) string {
	if s := text(v); s != "" {
		return s
	}
	for _, n := range nodes(v) {
		if s := text(n["name"]); s != "" {
			return s
		}
	}
	return ""
}

func image(v any) bool {
	switch t := v.(type) {
	case string:
		return strings.TrimSpace(t) != ""
	case []any:
		return slices.ContainsFunc(t, image)
	case map[string]any:
		return text(t["url"]) != "" || text(t["contentUrl"]) != ""
	}
	return false
}
