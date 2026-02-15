APP_NAME := agent
BIN_DIR  := bin

# .env があれば読み込む
ifneq (,$(wildcard .env))
  include .env
  export
endif

.PHONY: help build run demo server clean lint vet test fmt tidy setup chat rag

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ========================================
# ビルド・実行
# ========================================

build: ## バイナリをビルド
	go build -o $(BIN_DIR)/$(APP_NAME) .

run: build ## ビルドして実行（MODE 環境変数に従う）
	./$(BIN_DIR)/$(APP_NAME)

demo: build ## デモモードで実行
	MODE=demo ./$(BIN_DIR)/$(APP_NAME)

server: build ## REST API サーバーを起動
	MODE=server ./$(BIN_DIR)/$(APP_NAME)

# ========================================
# 開発ユーティリティ
# ========================================

fmt: ## コードをフォーマット
	go fmt ./...

vet: ## 静的解析
	go vet ./...

lint: vet ## lint（vet + staticcheck があれば実行）
	@which staticcheck > /dev/null 2>&1 && staticcheck ./... || echo "staticcheck not installed, skipping (go install honnef.co/go/tools/cmd/staticcheck@latest)"

test: ## テストを実行
	go test -v ./...

tidy: ## go mod tidy
	go mod tidy

clean: ## ビルド成果物を削除
	rm -rf $(BIN_DIR)

# ========================================
# セットアップ
# ========================================

setup: ## 初回セットアップ（.env 作成 + 依存取得）
	@test -f .env || (cp .env.example .env && echo ".env を作成しました。API キーを設定してください。")
	@test -f .env && echo ".env は既に存在します。"
	go mod download
	@echo "セットアップ完了"

# ========================================
# API リクエスト（サーバーモード起動中に使用）
# ========================================

chat: ## チャット送信 (例: make chat MSG="こんにちは")
	@curl -s -X POST http://localhost:$${PORT:-8080}/api/chat \
		-H 'Content-Type: application/json' \
		-d '{"message":"$(MSG)"}' | jq .

rag: ## RAG 質問 (例: make rag Q="Genkitとは？")
	@curl -s -X POST http://localhost:$${PORT:-8080}/api/rag \
		-H 'Content-Type: application/json' \
		-d '{"question":"$(Q)"}' | jq .

joke: ## ジョーク生成 (例: make joke TOPIC="Go言語")
	@curl -s -X POST http://localhost:$${PORT:-8080}/api/tellJoke \
		-H 'Content-Type: application/json' \
		-d '{"data":{"topic":"$(TOPIC)"}}' | jq .

sentiment: ## 感情分析 (例: make sentiment TEXT="今日は最高！")
	@curl -s -X POST http://localhost:$${PORT:-8080}/api/analyzeSentiment \
		-H 'Content-Type: application/json' \
		-d '{"data":"$(TEXT)"}' | jq .

health: ## ヘルスチェック
	@curl -s http://localhost:$${PORT:-8080}/health && echo
