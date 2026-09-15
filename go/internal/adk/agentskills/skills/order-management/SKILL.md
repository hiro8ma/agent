---
name: order-management
description: 注文状況の確認、キャンセル、配送追跡に対応するときに使う
license: Apache-2.0
metadata:
  owner: adk-agentskills
---

# 注文管理

## 手順

1. 利用者から注文 ID を確認する。
2. `get_order_status`で注文情報を取得する。
3. キャンセル依頼では`references/cancel-policy.md`を読む。
4. 回答には`assets/reply-template.md`を使う。

## 停止条件

本人確認が完了していない注文は操作しない。

キャンセル可能か不明な場合は、人間の担当者へ引き継ぐ。
