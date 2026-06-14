package workflow

import (
	"path/filepath"
	"testing"
)

func TestFileJournal_AppendAndLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	j := NewFileJournal(path)

	// 未作成ファイルは空履歴。
	if got, err := j.Load(); err != nil || len(got) != 0 {
		t.Fatalf("empty load: events=%d err=%v", len(got), err)
	}

	if err := j.Append(Event{StepName: ActivityInventory, Attempts: 1}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := j.Append(Event{StepName: ActivityTransportation, Attempts: 2}); err != nil {
		t.Fatalf("append: %v", err)
	}

	// 別インスタンスから読んでも順序通りに復元できる（クラッシュ後の再開を模す）。
	reopened := NewFileJournal(path)
	events, err := reopened.Load()
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	if events[0].StepName != ActivityInventory || events[1].StepName != ActivityTransportation {
		t.Fatalf("unexpected order: %+v", events)
	}
}

func TestMemoryJournal_AppendAndLoad(t *testing.T) {
	j := NewMemoryJournal()
	if err := j.Append(Event{StepName: ActivitySupplier, Attempts: 1}); err != nil {
		t.Fatalf("append: %v", err)
	}
	events, err := j.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(events) != 1 || events[0].StepName != ActivitySupplier {
		t.Fatalf("unexpected events: %+v", events)
	}
}
