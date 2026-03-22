.PHONY: help setup build run api clean

-include .env
export

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール
	go mod tidy

build: ## ビルド
	go build -o bin/agent .

run: build ## エージェントをCLIモードで起動
	./bin/agent console

api: build ## エージェントをAPIモードで起動
	./bin/agent api

clean: ## ビルド成果物を削除
	rm -rf bin/
