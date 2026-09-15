---
name: notify-user
description: 利用者への通知文の作り方と、送ってよい条件を決める
license: Apache-2.0
metadata:
  owner: common
---

# 通知の手順

1. 送る前に `get_notification_setting` で受信可否を確かめる
2. 受信が許可されている場合だけ `send_notification` で送る
3. 本文は 3 行以内にし、次に取る行動を 1 つだけ書く

## 送らない場合

- 受信設定が取れないときは送らない。設定が読めないことを応答へ書く
- 同じ用件を 24 時間以内に送っている場合は送らない
