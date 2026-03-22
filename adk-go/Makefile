.PHONY: help setup run web

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール
	go mod tidy

run: ## エージェントをCLIモードで起動
	go run main.go

web: ## エージェントをWebUIモードで起動（http://localhost:8080）
	adk web .
