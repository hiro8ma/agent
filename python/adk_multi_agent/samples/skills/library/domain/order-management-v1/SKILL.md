---
name: order-management-v1
description: 注文の照会と取り消しを行う手順（現行版）
license: Apache-2.0
metadata:
  owner: order-domain
  version: v1
  adk_additional_tools:
    - get_order_status
---

# 注文管理 v1

1. `get_order_status` で現在の状態を確かめる
2. 発送前の注文だけ取り消せる。発送後は返品の案内へ切り替える
3. 取り消しの実行前に、利用者へ確認を取る

## 注意

金額の変更は扱わない。
返品は `returns-management` の担当になる。
