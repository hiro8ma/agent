.PHONY: help test lint fmt web api run live

# API キーは .env に置く（git 管理外）。GOOGLE_API_KEY か GEMINI_API_KEY のどちらでもよい。
-include .env
export

UV ?= uv
PORT ?= 8000
AGENTS ?= samples/agents
AGENT ?= travel_planner
Q ?= 東京から京都に2泊3日。予算は1人8万円、歴史的な場所と和食が好き

# 先祖のディレクトリの .env を拾わせない。Vertex AI 用の値が残っていると API キー経路が壊れる。
ADK_ENV = ADK_DISABLE_LOAD_DOTENV=1 GOOGLE_GENAI_USE_VERTEXAI=FALSE \
	GOOGLE_API_KEY=$${GOOGLE_API_KEY:-$$GEMINI_API_KEY}

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test: ## テスト（API キー不要）
	$(UV) run pytest samples -q

lint: ## ruff の検査と整形チェック
	$(UV) run ruff check samples
	$(UV) run ruff format --check samples

fmt: ## ruff で整形
	$(UV) run ruff format samples

## web: ブラウザ UI を起動（http://localhost:$(PORT)）
## 情報パネルは Workflow root と動的 instruction で失敗する。会話とグラフ表示は使える。
web: ## adk web を起動（AGENTS=... PORT=...）
	@echo "http://localhost:$(PORT) で開く"
	$(ADK_ENV) $(UV) run adk web $(AGENTS) --port $(PORT)

api: ## adk api_server を起動（UI なし。curl で確認する用）
	$(ADK_ENV) $(UV) run adk api_server $(AGENTS) --port $(PORT)

run: ## adk run を 1 回実行（AGENT=... Q="..."）。無料枠では途中で 429 になる
	cd $(AGENTS) && $(ADK_ENV) $(UV) run adk run $(AGENT) "$(Q)" --jsonl

live: ## 旅行プランナーを実機で 1 往復（429 を待つリトライ付き）
	$(ADK_ENV) $(UV) run python -m samples.agents.travel_planner.live_run
