---
title: "Audience-split ConnectRPC services — チャット基盤を会員種別で物理分離する"
date: "2026-04-21"
tags: [connectrpc, proto, authorization, multi-tenancy, audience-split]
---

# Audience-split ConnectRPC services

1 つのチャット基盤で「管理者」「エンドユーザ」「ビジネスパートナー」など複数の audience を扱うとき、ConnectRPC の service 自体を audience 単位で分割する設計パターン

## よくある失敗

```proto
// 1 つの ChatService に audience を enum で詰める（アンチパターン）
service ChatService {
  rpc Ask(AskRequest) returns (stream AskResponse);
}

message AskRequest {
  Audience audience = 1;  // ADMIN / USER / BUSINESS
  ...
}
```

- 認可ロジックが service 1 本に集中し、`if audience == ADMIN { casbin check... }` 的な分岐が肥大化
- 認可バグが audience を跨いで発生する（管理者権限の漏洩など）
- Firestore / DB のコレクションも 1 本になりがちで IAM 分離が弱い
- 将来 audience を増やすたびに既存 service を触る → レビュー範囲が広がる

## 推奨パターン

```proto
service AdminChatService {
  rpc Ask(AskRequest) returns (stream AskResponse);
  rpc ListMessages(...) returns (...);
}

service UserChatService {
  rpc Ask(AskRequest) returns (stream AskResponse);
  ...
}

service BusinessChatService {
  rpc Ask(AskRequest) returns (stream AskResponse);
  ...
}
```

service を分けると、以下が audience 単位で物理分離される

| 境界 | 分離単位 |
|---|---|
| proto service | `AdminChatService` / `UserChatService` / `BusinessChatService` |
| 認可ミドルウェア | Interceptor を audience ごとにバインド |
| Firestore コレクション | `admin_chats/` / `user_chats/` / `business_chats/` |
| IAM / Workload Identity | SA を audience ごとに分ける選択肢が取れる |
| Istio AuthorizationPolicy | service 単位で principal を絞れる |
| ログ・監査 | service 名で aggregation できる |

## 導入タイミング

audience が 1 つしかない初期段階でも、service 名に audience を含めておく（`AdminChatService` のように）。これだけで後から service を増やす時に既存コードに破壊的変更を入れずに済む

1 つ目の service を実装する時点で、次の audience を見据えた共通 message（`AskRequest` / `AskResponse` / `ToolCall` / `FinishReason`）をパッケージの共通 proto に切り出しておくと、2 つ目以降は service 定義だけコピペで作れる

## 会話履歴ストレージも audience-split にする

Firestore のコレクション構造も audience 単位でトップレベル分離

```
admin_chats/
  {chat_id}/
    messages/{message_id}
user_chats/
  {chat_id}/
    messages/{message_id}
business_chats/
  ...
```

- Firestore セキュリティルール / IAM を audience 単位でかけられる
- TTL policy や BigQuery Export も audience 単位で差をつけられる
- クロス audience での参照事故を構造的に防げる

`chats/` 1 本にして `audience` フィールドで分ける設計は、クエリに `where("audience", "==", ...)` を毎回書く必要があり、書き忘れ 1 つで情報漏洩になる

## 共通化して良いもの / すべきでないもの

| 対象 | 共通化 | 理由 |
|---|---|---|
| `AskRequest` / `AskResponse` proto | ○ | 同じ chat メカニズム |
| Tool 実装 | ○（allowlist で audience ごとに公開範囲を制御） | Tool 自体は audience 非依存 |
| 認可 Interceptor | × | audience ごとに claim / role が異なる |
| Firestore コレクション | × | 構造的に分けたほうが安全 |
| Prompt テンプレート | △（基底は共通、audience ごとに variant） | 応対トーンが audience で変わる |

## proto 規約チェックリスト

audience-split の service を切るとき、proto-lint で以下を守る

- `package` は `aiagent.v1`（アンダースコア禁止、lower_camel_case + `.` 区切り）
- enum 値は型名プレフィックス付き（`PLATFORM_ADMIN_WEBVIEW` / `FINISH_REASON_STOP`）
- インラインコメント禁止、フィールドの上に書く
- field 番号は意味単位でグルーピング、将来予約分は番号帯を空けておくと意図が伝わる
