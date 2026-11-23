package productld

import (
	"slices"
	"strings"
)

// Condition はエージェントが商品を絞る条件。nil と false の条件は見ない。
type Condition struct {
	MaxPrice *float64
	Currency string
	MaxDays  *float64
	InStock  bool
}

// Status は条件での判定。
type Status int

const (
	Satisfied Status = iota
	Failed
	Unknown
)

func (s Status) String() string {
	return [...]string{"満たす", "満たさない", "判定できない"}[s]
}

// Verdict は商品1件の判定。Missing は判定できないときの理由で、無い項目か読めない値を Offer の順に並べる。
type Verdict struct {
	Status  Status
	Missing []string
}

var (
	inStock    = []string{"InStock", "LimitedAvailability", "OnlineOnly"}
	notInStock = []string{"OutOfStock", "SoldOut", "Discontinued", "BackOrder", "PreOrder", "PreSale", "InStoreOnly", "MadeToOrder", "Reserved"}
)

// Judge は、どれか1つの Offer が条件を全部満たせば満たす、分かっている項目で全部の Offer が外れれば満たさない、それ以外を判定できないとする。
// Offer が無い商品は、項目が全部無い Offer を1つ持つとみなし、見る条件の項目を理由に返す。
func Judge(p Product, c Condition) Verdict {
	offers := p.Offers
	if len(offers) == 0 {
		offers = []Offer{{}}
	}
	v := Verdict{Status: Failed}
	for _, o := range offers {
		failed, missing := judgeOffer(o, c)
		switch {
		case failed:
		case len(missing) == 0:
			return Verdict{Status: Satisfied}
		default:
			v.Status = Unknown
			v.Missing = appendNew(v.Missing, missing...)
		}
	}
	return v
}

// Count は判定ごとの商品の数を、Status の値を添字にして返す。
func Count(products []Product, c Condition) [3]int {
	var n [3]int
	for _, p := range products {
		n[Judge(p, c).Status]++
	}
	return n
}

func judgeOffer(o Offer, c Condition) (failed bool, missing []string) {
	if c.MaxPrice != nil {
		var m []string
		if o.Price == nil {
			m = append(m, Price.String())
		}
		switch {
		case o.Currency == "":
			m = append(m, Currency.String())
		case !strings.EqualFold(o.Currency, c.Currency):
			m = append(m, Currency.String()+" が "+c.Currency+" でない")
		}
		failed = len(m) == 0 && *o.Price > *c.MaxPrice
		missing = append(missing, m...)
	}
	if c.InStock {
		switch {
		case o.Availability == "":
			missing = append(missing, Availability.String())
		case slices.Contains(notInStock, o.Availability):
			failed = true
		case !slices.Contains(inStock, o.Availability):
			missing = append(missing, Availability.String()+" の値 "+o.Availability)
		}
	}
	if c.MaxDays != nil {
		d, m := deliveryDays(o)
		failed = failed || d != nil && *d > *c.MaxDays
		missing = append(missing, m...)
	}
	return failed, missing
}

// deliveryDays は OfferShippingDetails ごとに handlingTime と transitTime の maxValue を足し、その最大を返す。
// 宛先で絞らないので、どの宛先にも届く日数として最も遅いものを使う。
func deliveryDays(o Offer) (*float64, []string) {
	if len(o.Shipping) == 0 {
		return nil, []string{"shippingDetails"}
	}
	var most *float64
	var missing []string
	for _, s := range o.Shipping {
		if s.Handling == nil {
			missing = appendNew(missing, HandlingTime.String()+" の maxValue")
		}
		if s.Transit == nil {
			missing = appendNew(missing, TransitTime.String()+" の maxValue")
		}
		if s.Handling != nil && s.Transit != nil {
			if d := *s.Handling + *s.Transit; most == nil || d > *most {
				most = &d
			}
		}
	}
	return most, missing
}

func appendNew(list []string, items ...string) []string {
	for _, it := range items {
		if !slices.Contains(list, it) {
			list = append(list, it)
		}
	}
	return list
}
