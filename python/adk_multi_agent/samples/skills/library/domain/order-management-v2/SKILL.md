---
name: order-management-v2
description: 注文の照会と取り消しを行う手順（取り消し条件を見直した版）
license: Apache-2.0
metadata:
  owner: order-domain
  version: v2
  adk_additional_tools:
    - get_order_status
    - cancel_order
---

# 注文管理 v2

1. `get_order_status` で現在の状態と発送予定を確かめる
2. 発送前に加えて、発送当日の集荷前までは `cancel_order` で取り消せる
3. 取り消しの実行前に、利用者へ確認を取る

## v1 からの変更

取り消せる期限を集荷前まで広げた。
判断に使う値が増えるので、v1 と同時に動かして結果を比べる。
