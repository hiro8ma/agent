package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/hiro8ma/agent/go/internal/gpt"
)

// 動作確認用のフォールバックコーパス（『吾輩は猫である』冒頭、青空文庫・パブリックドメイン）。
// 本格的な学習は -data で全文テキストを指定する。
const sampleText = `吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
この書生の掌の裏でしばらくはよい心持に坐っておったが、しばらくすると非常な速力で運転し始めた。書生が動くのか自分だけが動くのか分らないが無暗に眼が廻る。胸が悪くなる。到底助からないと思っていると、どさりと音がして眼から火が出た。それまでは記憶しているがあとは何の事やらいくら考え出そうとしても分らない。`

func main() {
	var (
		dataPath  = flag.String("data", "", "学習テキストのパス（省略時は内蔵の漱石サンプル）")
		steps     = flag.Int("steps", 2000, "学習ステップ数")
		batchSize = flag.Int("batch", 16, "バッチサイズ")
		blockSize = flag.Int("block", 64, "コンテキスト長")
		embedDim  = flag.Int("embd", 128, "埋め込み次元")
		numLayers = flag.Int("layers", 4, "Transformer ブロック数")
		numHeads  = flag.Int("heads", 4, "アテンションヘッド数")
		maxLR     = flag.Float64("lr", 3e-3, "最大学習率")
		prompt    = flag.String("prompt", "吾輩は", "生成プロンプト")
		genLen    = flag.Int("gen", 200, "生成トークン数")
		temp      = flag.Float64("temp", 0.8, "温度")
		topK      = flag.Int("topk", 20, "top-k（0 で無効）")
		seed      = flag.Int64("seed", 42, "乱数シード")
	)
	flag.Parse()

	text := sampleText
	if *dataPath != "" {
		b, err := os.ReadFile(*dataPath)
		if err != nil {
			log.Fatalf("read %s: %v", *dataPath, err)
		}
		text = string(b)
	}

	tok := gpt.NewCharTokenizer(text)
	ds := &gpt.Dataset{Tokens: tok.Encode(text)}
	fmt.Printf("corpus: %d chars, vocab: %d\n", len(ds.Tokens), tok.VocabSize())

	cfg := gpt.Config{
		VocabSize: tok.VocabSize(),
		BlockSize: *blockSize,
		EmbedDim:  *embedDim,
		NumLayers: *numLayers,
		NumHeads:  *numHeads,
	}
	rng := rand.New(rand.NewSource(*seed))
	model := gpt.NewModel(cfg, rng)
	fmt.Printf("model: %s, params: %d\n", cfg, model.NumParams())

	gpt.Train(model, ds, gpt.TrainConfig{
		BatchSize:   *batchSize,
		Steps:       *steps,
		WarmupSteps: *steps / 20,
		MaxLR:       *maxLR,
		MinLR:       *maxLR / 10,
		WeightDecay: 0.01,
		GradClip:    1.0,
		LogEvery:    100,
		Seed:        *seed,
	}, func(format string, args ...any) { fmt.Printf(format+"\n", args...) })

	fmt.Printf("\n--- prompt: %q ---\n", *prompt)
	out := model.Generate(tok.Encode(*prompt), *genLen, *temp, *topK, rng)
	fmt.Println(tok.Decode(out))
}
