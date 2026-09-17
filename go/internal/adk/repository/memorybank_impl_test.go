package repository

import (
	"testing"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/domain"
)

func TestBankLocationReportsMissingFields(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		at          domain.BankLocation
		wantAll     bool
		wantMissing int
	}{
		{
			name:    "3 つそろえば接続できる",
			at:      domain.BankLocation{ProjectID: "p", Location: "us-central1", ReasoningEngine: "1"},
			wantAll: true,
		},
		{
			name:        "エンジン ID が無い",
			at:          domain.BankLocation{ProjectID: "p", Location: "us-central1"},
			wantMissing: 1,
		},
		{
			name:        "何も無い",
			at:          domain.BankLocation{},
			wantMissing: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.at.Complete(); got != tt.wantAll {
				t.Errorf("Complete() = %v, want %v", got, tt.wantAll)
			}
			if got := len(tt.at.Missing()); got != tt.wantMissing {
				t.Errorf("len(Missing()) = %d, want %d（%v）", got, tt.wantMissing, tt.at.Missing())
			}
		})
	}
}

func TestMemoryBankStoreRejectsIncompleteLocation(t *testing.T) {
	t.Parallel()

	// 接続先が欠けている場合は、外部へ出る前に落ちる。
	// 落ちないと、認証情報を探しに行く分だけ失敗が遅く、原因も分かりにくくなる。
	store, err := NewMemoryBankStore(t.Context(), domain.BankLocation{ProjectID: "p"})
	if err == nil {
		t.Fatalf("NewMemoryBankStore() = %v, want error", store)
	}
	if store != nil {
		t.Errorf("NewMemoryBankStore() store = %v, want nil", store)
	}
}

func TestStoreFromEnvFallsBackWithoutSettings(t *testing.T) {
	t.Setenv(EnvProject, "")
	t.Setenv(EnvLocation, "")
	t.Setenv(EnvReasoningEngine, "")

	store, useBank, err := NewStoreFromEnv(t.Context())
	if err != nil {
		t.Fatalf("NewStoreFromEnv() error = %v", err)
	}
	if useBank {
		t.Error("NewStoreFromEnv() useBank = true, want false（課金経路へ黙って入らない）")
	}
	if store == nil {
		t.Error("NewStoreFromEnv() store = nil, want in-memory の実装")
	}
}

func TestStoreFromEnvPicksBankWhenSettingsArePresent(t *testing.T) {
	t.Setenv(EnvProject, "p")
	t.Setenv(EnvLocation, "us-central1")
	t.Setenv(EnvReasoningEngine, "123")

	if at := BankLocationFromEnv(); !at.Complete() {
		t.Fatalf("BankLocationFromEnv() = %+v, want 3 つそろった状態", at)
	}
}

func TestInMemoryStoreRoundTrip(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	store := NewInMemoryStore()

	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "demo", UserID: "u1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	s := created.Session

	ev := session.NewEvent(ctx, "inv-1")
	ev.Author = "user"
	ev.Content = &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "I like udon"}},
	}
	if err := sessions.AppendEvent(ctx, s, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	if err := store.AddSessionToMemory(ctx, s); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	got, err := store.SearchMemory(ctx, &memory.SearchRequest{
		AppName: "demo", UserID: "u1", Query: "udon",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 1 {
		t.Fatalf("SearchMemory() len = %d, want 1", len(got.Memories))
	}

	// 記憶は利用者ごとに閉じる。別の利用者からは引けない。
	other, err := store.SearchMemory(ctx, &memory.SearchRequest{
		AppName: "demo", UserID: "u2", Query: "udon",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(other.Memories) != 0 {
		t.Errorf("別の利用者から引けてしまう len = %d, want 0", len(other.Memories))
	}
}
