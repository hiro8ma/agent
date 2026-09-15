---
name: order-management
description: 注文の状況確認、取り消し、配送の追跡を行う手順
license: Apache-2.0
metadata:
  owner: support-domain
---

# 注文管理

## 手順

1. `get_order_status` で現在の状態と発送済みかどうかを確かめる
2. 取り消しの可否は `references/cancel-policy.md` の条件で判断する
3. 取り消せる場合だけ `cancel_order` を呼ぶ。理由は利用者の言葉をそのまま渡す
4. 取り消せない場合は返品の案内へ切り替える

## 注意

取り消しは元に戻せない。実行の前に利用者へ確認を取る。
金額の変更は扱わない。
