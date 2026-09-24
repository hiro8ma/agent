# 旅行プランナーの Agent Skills（ADK 版 / Genkit 版）

旅行プランナーの日程と予算の担当に、SKILL.md の手順書を必要なときだけ読み込ませる。
スキルは `internal/travel/skills/` に置き、ADK 版と Genkit 版で同じファイルを使う。

| スキル | 使う場面 |
|---|---|
| `budget-allocation` | 予算額が示された依頼。交通 / 宿 / 食事 / 観光の配分と、足りないときに削る順番 |
| `seasonal-caution` | 時期が示された依頼。桜や紅葉の混雑、夏の暑さ、雨の季節の屋内の代わり |

## 動かし方

`TRAVEL_SKILLS_DIR` を設定したときだけ有効になる。未設定なら今までと同じ入力になる。

```bash
TRAVEL_SKILLS_DIR=internal/travel/skills go run ./cmd/adk-travelplanner
TRAVEL_SKILLS_DIR=internal/travel/skills go run ./cmd/genkit-travelplanner
TRAVEL_SKILLS_DIR=internal/travel/skills go run ./cmd/travelplanner-server -impl adk
```

## ADK 版と Genkit 版の違い

| 要素 | ADK 版 | Genkit 版 |
|---|---|---|
| 部品 | `skilltoolset`（`NewSkillToolset`）を `Toolsets` に渡す | `middleware.Skills` を `ai.WithUse` で差し込む |
| メタデータ | system instruction に `<available_skills>` の XML | system prompt に `<skills>` の一覧 |
| 本文を読むツール | `load_skill`（ほかに `list_skills` / `load_skill_resource`） | `use_skill` だけ |
| 本文の行き先 | セッションの履歴に残り、後段のエージェントにも届く | その生成の中だけ。予算の生成は別の呼び出しなので届かない |
| 予算の構造化出力 | Gemini API ではツールと出力スキーマを併用できず、`set_model_response` の呼び出しに切り替わる | 出力スキーマのまま |

## 検証（2026-09-24、Gemini API `gemini-3.5-flash-lite` 実呼び出し、各 1 回）

| 実装 | 依頼 | スキル | 読み込まれたスキル | モデル呼び出し | 入力トークン合計 | 所要時間 |
|---|---|---|---|---|---|---|
| ADK | 京都に1泊2日、予算3万円 | あり | 日程の担当が `budget-allocation` | 10 | 17006 | 14.8s |
| ADK | 東京から京都に行きたい。歴史が好き | あり | なし | 8 | 10765 | 12.1s |
| ADK | 京都に1泊2日、予算3万円 | なし | - | 9 | 10861 | 12.0s |
| Genkit | 京都に1泊2日、予算3万円 | あり | 日程の生成が `budget-allocation` | 8 | 6612 | 13.8s |
| Genkit | 東京から京都に行きたい。歴史が好き | あり | なし | 8 | 4723 | 12.0s |
| Genkit | 京都に1泊2日、予算3万円 | なし | - | 7 | 3772 | 13.4s |

- 予算の依頼では、両方の版で日程の担当がスキルを読んだ。予算の担当は読み直さなかった
- 予算の合計はスキルありで 2.9 万円（ADK）と 3 万円（Genkit）、ADK のスキルなしで 5.9 万円だった
- ADK の予算の担当の入力は 4637 から 6452 に増えた。スキルの本文が履歴に残って届くため
- メタデータだけのコストは日程の生成で数百トークン（Genkit は 886 から 1358）
- 時期を示していない依頼なので、`seasonal-caution` はどの回も読まれなかった
