# productld

商品ページの HTML から schema.org の Product / Offer の JSON-LD を取り出し、エージェントが比較に使う項目が揃っているかと、予算 / 届くまでの日数 / 在庫の条件で判定できる商品の数を出す部品

AI のエージェントが商品を比べて選ぶとき、価格に加えて在庫と届くまでの日数を、企業が構造化データで出しているかを確かめるために作った

## 構成

| ファイル | 中身 |
| --- | --- |
| `extract.go` | `Extract`。`<script type="application/ld+json">` を全部読み、単体のオブジェクト / 配列 / `@graph` / ProductGroup の `hasVariant` から Product を取り出す。壊れた JSON はその script だけ飛ばして数える。`Product.Has` で項目ごとの有無を返す |
| `judge.go` | `Judge` / `Count`。条件（予算の上限と通貨 / 届くまでの日数 / 在庫あり）で、満たす / 満たさない / 判定できない（理由つき）に分ける |
| `load.go` | `Load`。HTML のファイル1つか、ディレクトリの直下の .html と .htm を名前の順に読む |

## 動かし方

```bash
go test ./internal/productld   # 1秒ほど
make productcheck              # testdata の合成のページを、予算5000円 / 3日以内 / 在庫ありで見る（ネットワークは使わない）
go run ./cmd/productcheck -max-price 5000 -max-days 3 -in-stock 商品ページのディレクトリ
```

## 判定の決め方

- 条件は、同じ Offer で全部満たしたときだけ満たす。別々の Offer で満たしても満たさない
- 分かっている項目で外れれば、ほかの項目が無くても満たさない。外れていないが項目が無ければ判定できないにして、無い項目を理由に返す
- 届く日数は handlingTime と transitTime の maxValue の和。OfferShippingDetails が複数あれば、宛先で絞らないので最も遅いものを使う
- unitCode が DAY / d / 無し以外の日数は読まない
- priceCurrency が条件の通貨と違えば判定できない
- availability は InStock / LimitedAvailability / OnlineOnly を在庫ありとする。schema.org の列挙に無い値は判定できない
- variant に無い name / brand / image / hasMerchantReturnPolicy / aggregateRating は ProductGroup から受け継ぐ。sku / gtin と offers は受け継がない

## testdata

- ブランドと商品は架空で、URL は相対パス
- `client-rendered.html` は、JSON-LD を JavaScript で後から差し込むので素の HTML には無い、という想定のページ
- `broken.html` は、壊れた JSON の script と読める script を1つずつ持つ

## 読まないもの

- `@id` だけで別のノードを指す参照（ProductGroup の `isVariantOf` や、共有の `shippingDetails`）
- AggregateOffer の lowPrice と priceSpecification の価格。どの Offer の在庫と配送かが分からないため
- shippingDestination での宛先の絞り込み
- URL の取得と、JavaScript を実行した後の HTML
