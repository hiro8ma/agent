# evalharness

DeepEval の評価指標を Go に移植し、`go test` に載せるパッケージ。

DeepEval は Python 専用で Go 版が無い。ただし各指標の中身は共通の骨格でできており、
言語に依存しない。アプリが Go なら評価も Go に置いたほうが、ビルド・依存・CI を一本化できる。

## 指標の骨格

DeepEval のソースを読むと、5 指標がすべて同じ形をしている。

```
判定単位に分解する → 1 単位ずつ二値で判定する → 良い判定の割合を採点にする

score = 良い判定の数 / 判定の総数
```

単発で「点数をつけて」と聞くより安定するのは、分解によって judge の判断が
1 単位ごとの yes / no に落ちるため。長い回答の一部だけが的外れな場合も、
全体を 1 つとして採点すると平均的に高い点がついてしまう。

| 指標 | 判定単位 | 各単位への問い | 良い側 |
|---|---|---|---|
| `AnswerRelevancy` | 出力を分解した文 | 質問に関係するか | yes |
| `Bias` | 出力から抽出した意見 | 属性への固定観念を含むか | no |
| `Toxicity` | 出力から抽出した意見 | 攻撃的・差別的か | no |
| `Faithfulness` | 出力から抽出した主張 | 文脈に裏づけられるか | yes |
| `Hallucination` | **文脈そのもの**（分解しない） | 出力が文脈と矛盾しないか | yes |
| `GEval` | 採点基準から洗い出した観点 | 基準を満たすか | yes |

`Hallucination` だけ判定単位が出力側ではなく文脈側にある。
`Faithfulness` と対になっており、前者は文脈を基準に矛盾を、後者は主張を基準に根拠を見る。

## スコアの向き

**全指標で 1.0 が合格、0.0 が不合格に揃えてある。**`Threshold` は「これ以上なら合格」の下限。

DeepEval も 4.x でこの向きに統一した。旧仕様では `Bias` / `Toxicity` だけ
「違反の割合」で 1.0 が悪く、`threshold` は上限だった。古い記事のコードは
この旧仕様のままのことがあり、そのまま持ち込むと判定が反転する。
実行すれば `DeprecationWarning` が出るが、結果の表だけ見ていると気づけない。

旧 `threshold=0.2`（違反 20% 以下）は、新 `threshold=0.8` にあたる。

## 使い方

```go
llm, err := evalharness.NewGeminiLLM(ctx, apiKey, "gemini-3.6-flash")
llm.MinInterval = 13 * time.Second // 無料枠は 1 分あたり 5 回
llm.MaxRetries = 4                 // 429 はサーバの指示に従って待ち直す

evalharness.AssertTest(t, ctx, llm,
    evalharness.TestCase{
        Input:        "このシャツの素材は何ですか",
        ActualOutput: "コットン 100% です",
    },
    evalharness.NewAnswerRelevancy(0.7),
)
```

`AssertTest` は DeepEval の `assert_test` に対応する。
不合格なら減点された単位とその理由を出す。スコアだけでは何を直せばよいか分からない。

## CI で skip を失敗に変える

```bash
go test ./internal/evalharness/              # キーが無ければ skip
EVAL_REQUIRE=1 go test ./internal/evalharness/  # キーが無ければ失敗
```

**CI で skip は緑になる。**「評価に通った」と「評価が走らなかった」が
テスト結果の色で区別できない。キーの設定漏れやクォータ枯渇を緑のまま見逃す事故は、
`EVAL_REQUIRE=1` で防げる。ローカルでは既定の skip、CI では有効にする運用を想定している。

同じ理由で、次の状況も満点ではなく失敗として扱う。

- 判定用 API がエラーを返した（握り潰して満点を返さない）
- 判定数が単位数と食い違う（対応が崩れた表を出さない）
- `Hallucination` / `Faithfulness` で `Context` が空（判定対象 0 件は満点になる）

## API 呼び出し回数

分解を伴う指標は 1 ケースあたり 2 回（分解 + 判定）、`Hallucination` は 1 回呼ぶ。
`GeminiLLM.Calls` に実回数が入る。

Gemini の無料枠は 1 分あたり 5 回・上限 20 回で、応答の `retryDelay` は数十秒を返す。
`quotaId` は `PerDay` だが実際は短いローリング窓のため、待てば通る。
CI で常時回すなら有料枠が要る。

## テスト

`LLM` をインターフェースにしてあるため、ハーネス自身のテストは API を叩かずに回る。
`*testing.T` ではなく `TB` インターフェースを受けるのも、
「落ちるべきときに落ちるか」を記録用の実装で確かめるため。
評価が素通しになっていてもテストは緑になるので、ここは実際に確かめる価値がある。

```bash
go test ./internal/evalharness/                       # オフラインのみ
go test -run 'Live' -v ./internal/evalharness/        # 実 API
```
