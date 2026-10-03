GO ?= go
export GOEXPERIMENT = simd
BENCH = $(GO) test -run '^$$' -benchtime 2s -count 5

.PHONY: test ceiling dot search batch quant recall spill knn

## test: arm64 で SIMD の経路を通し、amd64 と simd なしでもビルドが通ることを確かめる
test:
	$(GO) test ./...
	GOARCH=amd64 $(GO) vet ./...
	GOEXPERIMENT= $(GO) vet ./...

## ceiling: ルーフラインの天井（FMA のピークと DRAM の読み出し帯域）
ceiling:
	$(BENCH) -bench 'BenchmarkPeak' ./vec

## dot: 内積とハミング距離の単体
dot:
	$(BENCH) -bench 'BenchmarkDot|BenchmarkHamming' ./vec

## search: 10 万件の全探索を実装ごとに
search:
	$(BENCH) -bench 'BenchmarkSearch(Naive|Unroll4|SIMD1|SIMD|Portable|SIMD4A|SIMD8A|SIMD16A)$$' ./index

## batch: クエリを B 本まとめて算術強度を上げる
batch:
	$(BENCH) -bench 'BenchmarkSearchBatch' ./index

## quant: int8 と 1bit の量子化、1bit + rerank
quant:
	$(BENCH) -bench 'BenchmarkSearch(Int8|Binary)' ./index

## knn: 教材の NumPy の実験と同じ 100 万件 x 128 次元、K=20 の L2 全探索（ノルムの展開）と、256 ビット符号のハミング全走査
knn:
	$(BENCH) -bench 'BenchmarkKNN1M|BenchmarkHamming256Scan' ./index

recall:
	$(GO) test -run 'TestRecall' -v ./index

## spill: FMA ループがアキュムレータをスタックへ退避していないかを機械語で見る
spill:
	$(GO) test -c -o $(TMPDIR)/simdsearch-vec.test ./vec
	$(GO) tool objdump -s 'vec.peakFMA24$$' $(TMPDIR)/simdsearch-vec.test
