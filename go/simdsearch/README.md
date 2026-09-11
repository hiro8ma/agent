# simdsearch

Go 1.27 の `simd/archsimd`（arm64 の Neon）とポータブルな `simd` パッケージで、メモリ上のベクトル全探索を速くする実験

ルーフラインで「いま何が性能の上限か」を測ってから打ち手を選ぶ
題材は Go Conference 2026 のワークショップ [po3rin/gocon2026-simd-search](https://github.com/po3rin/gocon2026-simd-search) で、教材は Codespaces の AMD EPYC（AVX2）で測っている
ここでは同じ流れを Apple M5 Pro の Neon で自前に実装し、教材の主張を測り直した

## 動かし方

Go 1.27 以上が要る。`GOEXPERIMENT=simd` は Makefile が付ける

```bash
cd go/simdsearch
make test     # arm64 で SIMD の経路を通し、amd64 と simd なしのビルドも確かめる
make ceiling  # FMA のピークと DRAM の読み出し帯域
make search   # 10 万件 x 384 次元の全探索を実装ごとに
make batch    # クエリを B 本まとめる
make quant    # int8 と 1bit、1bit + rerank
make recall   # 量子化の Recall@10
make spill    # FMA ループの機械語
```

`GOEXPERIMENT=simd` を付けずに `go test` すると `TestBuiltWithSIMD` が落ちる
フラグを付け忘れたまま、フォールバックの速さを SIMD の結果として測る事故を止めるため

## 構成

| パス | 中身 |
| --- | --- |
| `vec/dot.go` | スカラの内積。`DotNaive`（1 本）と `DotUnroll4`（スカラのままアキュムレータ 4 本の対照実験） |
| `vec/dot_arm64.go` | Neon の内積。`DotSIMD1`（1 本）、`DotSIMD`（4 本、スライスで読む）、`DotSIMD4A` / `DotSIMD8A` / `DotSIMD16A`（配列ポインタで読む） |
| `vec/dot_portable.go` | `simd.Float32s` の内積。レーン数は実行時に決まる |
| `vec/int8*.go` | 対称 int8 量子化と int8 の内積（SMULL で広げて SXTL で足す） |
| `vec/binary*.go` | 符号 1bit 量子化とハミング距離 |
| `vec/ceiling*_test.go` | ルーフラインの天井を測るベンチ |
| `index/` | 全探索、バッチ検索、int8 / 1bit 検索、1bit + rerank、上位 k 件のヒープ、合成データ |

arm64 と `GOEXPERIMENT=simd` 以外では、SIMD の関数はスカラにフォールバックする

## 測定結果

Apple M5 Pro、Go 1.27.0、1 goroutine、`-count 5` の中央値
データは 10 万件 x 384 次元の fp32（153.6MB）で、M5 Pro の L2（16MB）には収まらない

### 天井

| 項目 | 値 |
| --- | --- |
| FMA のピーク（アキュムレータ 16 本、`a = m*c + a` の形） | 117.4 GFLOP/s |
| FMA のピーク（アキュムレータ 24 本、教材と同じ `a = a*m + c` の形） | 19.8 GFLOP/s |
| DRAM の読み出し帯域（256MB を流し読み） | 78.2 GB/s |
| リッジ（演算ピーク ÷ 帯域） | 約 1.5 flop/byte |
| 内積（算術強度 0.5）のメモリ側の上限 | 39 GFLOP/s |

### 全探索（1 クエリ）

| 実装 | ms/query | GB/s | GFLOP/s |
| --- | --- | --- | --- |
| `DotNaive` スカラ 1 本 | 26.1 | 5.9 | 2.9 |
| `DotUnroll4` スカラ 4 本 | 11.2 | 13.7 | 6.8 |
| `DotSIMD1` Neon 1 本 | 7.14 | 21.5 | 10.8 |
| `DotSIMD` Neon 4 本（スライス） | 4.48〜5.34 | 28.8〜34.3 | 14.4〜17.1 |
| `DotPortable` simd.Float32s 4 本 | 5.25 | 29.3 | 14.6 |
| `DotSIMD4A` Neon 4 本（配列ポインタ） | 2.66 | 57.9 | 28.9 |
| `DotSIMD8A` Neon 8 本（配列ポインタ） | 2.52 | 60.9 | 30.4 |
| `DotSIMD16A` Neon 16 本（配列ポインタ） | 3.60 | 42.7 | 21.3 |

`DotSIMD` の幅は 2 回の実行の差。macOS ではコアを固定できないので、実行ごとに 2 割ほど揺れる

### バッチ化（`DotSIMD8A`）

| B | 算術強度 | ms/query | GFLOP/s |
| --- | --- | --- | --- |
| 1 | 0.5 | 2.42 | 31.8 |
| 2 | 1 | 2.05 | 37.4 |
| 8 | 4 | 2.08 | 37.0 |
| 64 | 32 | 2.02 | 37.9 |

### 量子化と rerank

| 方式 | ms/query | Recall@10 |
| --- | --- | --- |
| fp32 `DotSIMD8A` | 2.52 | 1.000 |
| int8 スカラ | 12.05 | 0.986 |
| int8 Neon | 1.97 | 0.986 |
| 1bit スカラ popcount | 0.81 | 0.353 |
| 1bit Neon popcount | 1.17 | 0.353 |
| 1bit で 100 件に絞り fp32 で rerank | 0.76 | 0.999 |

Recall は 2 万件、100 クエリの合成データ（中心の周りに点を散らした分布）で測った。教材のデータとは分布が違うので、値は比べられない

## 分かったこと

**教材の天井の測り方は、arm64 では FMA ではなくコピーの鎖を測る**
arm64 の FMLA は加算側のレジスタに結果を上書きする
`a = a*m + c` だと定数 `c` を毎回新しいレジスタにコピーしてから掛けることになり、コンパイラはそのコピーを直前のコピーから取る鎖にしていた
1 周の時間がアキュムレータの本数に比例して延びるので、4 本でも 24 本でも約 20 GFLOP/s で止まる
`a = m*c + a` と書くとループが FMLA だけになり、16 本で 117.4 GFLOP/s に届いた
x86 の FMA は結果を書くレジスタを選べる形式が 3 つあるので、同じ式でもこの問題は出ない

**M5 Pro では SIMD 化しただけではメモリ律速にならない**
教材の Codespaces では、SIMD 化した時点で帯域の 93% に達していた
M5 Pro の 1 コアの帯域は約 4 倍あるので、スライスで読む `DotSIMD` は帯域の 44% で止まる
機械語を見ると、4 レーン読むたびに境界チェックとスライスの計算が並んでいた
配列ポインタで読んで境界チェックを 1 周 1 回にすると 2 倍になり、8 本で帯域の 78% に届いた

**アキュムレータ 16 本は register spill で遅くなる**
コンパイラが 32 個のロードをループの先頭にまとめて並べるため、アキュムレータ 16 本と合わせて 48 個の値が同時に生きる
arm64 のベクトルレジスタは 32 本なので、スタックへの退避が出る

**バッチ化は 1.2 倍で頭打ちになる**
最速の `DotSIMD8A` では、B=1 の 2.42 ms（帯域の 81%）が B=2 で 2.05 ms になり、その先は B=64 まで約 2.02 ms で変わらない
B=2 以降は 37〜38 GFLOP/s で平らになり、FMA のピークの約 1/3 で止まる
バッチ化で減るのは DRAM からの転送だけで、DB ベクトルは L1 から読み直すので、1 回の FMA に 2 回のロードが要る点は変わらない。この天井はロードの処理量と考えられるが、確かめてはいない
内積が遅い `DotSIMD` と `DotSIMD16A` では、B=1 の時点で帯域が上限でないので、バッチ化はまったく効かなかった

**ポータブルな simd は archsimd より遅い**
同じアキュムレータ 4 本で、内積単体は 26%、全探索は 17% 遅かった
`simd` パッケージには配列ポインタから読む関数が無いので、境界チェックを減らす手も使えない

**int8 は SDOT が使えない**
M5 Pro の CPU は int8 の内積命令（FEAT_DotProd）を持っているが、archsimd の arm64 API にはそれに当たるメソッドが無い
16 要素ごとに広げる加算が 4 回要るので、fp32 の最速版に対して 1.28 倍に留まる

**1bit の SIMD popcount は速くならない**
arm64 のスカラ版 `bits.OnesCount64` は、もともと Neon の `VCNT` と `VUADDLV` にコンパイルされる
`Uint8x16.ReduceSum` はエミュレーションで、しかも uint8 で返すので 255 を超える集計には幅の拡張が要る
