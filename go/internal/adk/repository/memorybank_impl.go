// Package repository は internal/adk/domain のポートを実装する。
package repository

import (
	"context"
	"fmt"
	"os"

	"google.golang.org/adk/v2/memory"
	vertexmemory "google.golang.org/adk/v2/memory/vertexai"

	"github.com/hiro8ma/agent/go/internal/adk/domain"
)

// LastUpdateKey は Memory Bank へ渡す差分の基準時刻を置く State の鍵。
//
// この鍵を設定すると、Memory Bank は前回以降のイベントだけを記憶の生成に使う。
// 空にすると毎回セッション全体を送ることになり、書き込みの回数がそのまま増える。
const LastUpdateKey = "memory_last_update_time"

// 環境変数の名前。Vertex を使うかどうかはこの 3 つで決まる。
const (
	EnvProject         = "GOOGLE_CLOUD_PROJECT"
	EnvLocation        = "GOOGLE_CLOUD_LOCATION"
	EnvReasoningEngine = "ADK_REASONING_ENGINE_ID"
)

// NewInMemoryStore はプロセス内に閉じた記憶を返す。課金は発生しない。
func NewInMemoryStore() domain.MemoryStore {
	return memory.InMemoryService()
}

// NewMemoryBankStore は Vertex AI Memory Bank を使う記憶を返す。
//
// 接続先が欠けている場合は、接続を試みる前に落とす。
// 途中まで進んでから失敗すると、どの設定が足りないのか分からなくなる。
func NewMemoryBankStore(ctx context.Context, at domain.BankLocation) (domain.MemoryStore, error) {
	if !at.Complete() {
		return nil, fmt.Errorf("memory bank の接続先が不足している: %v", at.Missing())
	}
	svc, err := vertexmemory.NewService(ctx, &vertexmemory.ServiceConfig{
		ProjectID:                     at.ProjectID,
		Location:                      at.Location,
		ReasoningEngine:               at.ReasoningEngine,
		StateKeySessionLastUpdateTime: LastUpdateKey,
		// 記憶の生成を待つ。待たないと次の対話に前回の記憶が間に合わない。
		WaitForCompletion: true,
	})
	if err != nil {
		return nil, fmt.Errorf("memory bank の作成: %w", err)
	}
	return svc, nil
}

// BankLocationFromEnv は環境変数から接続先を読む。
func BankLocationFromEnv() domain.BankLocation {
	return domain.BankLocation{
		ProjectID:       os.Getenv(EnvProject),
		Location:        os.Getenv(EnvLocation),
		ReasoningEngine: os.Getenv(EnvReasoningEngine),
	}
}

// NewStoreFromEnv は環境変数がそろっていれば Memory Bank を、そうでなければ in-memory を返す。
//
// 第 2 戻り値は Memory Bank を使ったかどうか。
// どちらで動いているかを利用者へ表示するために返す。黙って課金する経路を作らない。
func NewStoreFromEnv(ctx context.Context) (domain.MemoryStore, bool, error) {
	at := BankLocationFromEnv()
	if !at.Complete() {
		return NewInMemoryStore(), false, nil
	}
	store, err := NewMemoryBankStore(ctx, at)
	if err != nil {
		return nil, false, err
	}
	return store, true, nil
}
