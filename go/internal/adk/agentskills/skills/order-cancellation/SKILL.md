---
name: order-cancellation
description: 注文のキャンセルを確定し、利用者へ通知を送るときに使う
license: Apache-2.0
metadata:
  owner: adk-agentskills
  disable-model-invocation: "true"
---

# 注文のキャンセルの確定

注文を取り消して通知を送る工程のため、担当者が名前を指定して始める。

## 手順

1. 担当者から注文 ID と対象商品を受け取る。
2. `get_order_status`で発送前であることを確かめる。
3. `cancel_order`でキャンセルを確定する。
4. `send_cancel_notice`で利用者へ通知を送る。

## 停止条件

発送済みの注文はキャンセルせず、担当者へ返品の手続きを案内する。
