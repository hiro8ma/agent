package repository

import (
	"path/filepath"
	"testing"

	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/domain"
)

func TestParseSessionBackendDefaultsToMemory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   string
		want    domain.SessionBackend
		wantErr bool
	}{
		{name: "未設定は memory", value: "", want: domain.SessionInMemory},
		{name: "memory", value: "memory", want: domain.SessionInMemory},
		{name: "database", value: "database", want: domain.SessionDatabase},
		{name: "vertex", value: "vertex", want: domain.SessionVertex},
		{name: "綴り違いは落とす", value: "databse", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := domain.ParseSessionBackend(tt.value)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseSessionBackend(%q) = %v, want error", tt.value, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseSessionBackend(%q) error = %v", tt.value, err)
			}
			if got != tt.want {
				t.Errorf("ParseSessionBackend(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

func TestSessionBackendReportsPersistenceAndBilling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		backend        domain.SessionBackend
		wantPersistent bool
		wantBillable   bool
	}{
		{name: "memory", backend: domain.SessionInMemory},
		{name: "database", backend: domain.SessionDatabase, wantPersistent: true},
		{name: "vertex", backend: domain.SessionVertex, wantPersistent: true, wantBillable: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.backend.Persistent(); got != tt.wantPersistent {
				t.Errorf("Persistent() = %v, want %v", got, tt.wantPersistent)
			}
			if got := tt.backend.Billable(); got != tt.wantBillable {
				t.Errorf("Billable() = %v, want %v", got, tt.wantBillable)
			}
		})
	}
}

func TestSessionsFromEnvDefaultsToMemory(t *testing.T) {
	t.Setenv(EnvSessionBackend, "")

	svc, backend, err := NewSessionsFromEnv(t.Context())
	if err != nil {
		t.Fatalf("NewSessionsFromEnv() error = %v", err)
	}
	if backend != domain.SessionInMemory {
		t.Errorf("backend = %v, want %v", backend, domain.SessionInMemory)
	}
	if svc == nil {
		t.Error("svc = nil, want in-memory の実装")
	}
}

func TestSessionsFromEnvRejectsUnknownBackend(t *testing.T) {
	t.Setenv(EnvSessionBackend, "postgres")

	if _, _, err := NewSessionsFromEnv(t.Context()); err == nil {
		t.Fatal("NewSessionsFromEnv() error = nil, want error（綴り違いを黙って memory へ落とさない）")
	}
}

func TestDatabaseSessionsSurviveReopen(t *testing.T) {
	// ファイルに残ることを、同じ DSN で開き直して確かめる。
	// プロセスを越える保証そのものは測れないが、サービスの再生成では消えない。
	dsn := "file:" + filepath.Join(t.TempDir(), "nested", "sessions.db")
	ctx := t.Context()

	first, err := NewDatabaseSessions(dsn)
	if err != nil {
		t.Fatalf("NewDatabaseSessions() error = %v", err)
	}

	created, err := first.Create(ctx, &session.CreateRequest{AppName: "demo", UserID: "u1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	id := created.Session.ID()

	ev := session.NewEvent(ctx, "inv-1")
	ev.Author = "user"
	ev.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "覚えて"}}}
	if err := first.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	second, err := NewDatabaseSessions(dsn)
	if err != nil {
		t.Fatalf("NewDatabaseSessions() 2 回目 error = %v", err)
	}
	got, err := second.Get(ctx, &session.GetRequest{AppName: "demo", UserID: "u1", SessionID: id})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got.Session == nil {
		t.Fatal("Get() session = nil, want 保存済みの session")
	}
	if n := got.Session.Events().Len(); n != 1 {
		t.Errorf("Events().Len() = %d, want 1", n)
	}
}

func TestVertexSessionsRejectsIncompleteLocation(t *testing.T) {
	t.Parallel()

	svc, err := NewVertexSessions(t.Context(), domain.BankLocation{ProjectID: "p"})
	if err == nil {
		t.Fatalf("NewVertexSessions() = %v, want error", svc)
	}
}
