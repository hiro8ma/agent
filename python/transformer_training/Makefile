.PHONY: help setup data model train generate clean

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール（uv sync）
	uv sync

data: ## データセットの動作確認
	uv run python data.py

model: ## モデル構造の動作確認（学習前）
	uv run python model.py

train: ## mini-GPT を学習する
	uv run python train.py

generate: ## 学習済みモデルで文章を生成
	uv run python generate.py

generate-romeo: ## "ROMEO:" プロンプトで生成
	uv run python generate.py --prompt "ROMEO:"

generate-greedy: ## greedy decoding（temperature=0.0 相当）
	uv run python generate.py --temperature 0.01

clean: ## checkpoint と __pycache__ を削除（データは残す）
	rm -rf checkpoints/*.pt __pycache__ .ruff_cache

clean-all: ## データも含めて全削除
	rm -rf data/*.txt checkpoints/*.pt __pycache__ .ruff_cache
