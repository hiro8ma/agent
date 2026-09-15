---
name: product-inquiry
description: 商品の検索、詳細の取得、在庫の確認を行う手順
license: Apache-2.0
metadata:
  owner: support-domain
---

# 商品の問い合わせ

## 手順

1. カテゴリが分かる場合は `assets/category-list.md` から該当する値を選ぶ
2. `search_products` で候補を絞る。カテゴリが不明なら空文字を渡す
3. 利用者が 1 つに絞ったら `get_product_details` で詳細と在庫を確かめる

## 出力

候補は 3 件までにする。
結果に「全 N 件のうち上位 M 件」と付いている場合は、その旨を利用者へ伝える。
