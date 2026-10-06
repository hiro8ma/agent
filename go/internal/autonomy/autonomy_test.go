package autonomy_test

import (
	"testing"

	"github.com/hiro8ma/agent/go/internal/autonomy"
)

func newLadder() *autonomy.Ladder {
	return autonomy.New(map[string]autonomy.Policy{
		"triage": {Max: autonomy.Act, Window: 10, MinSamples: 10, PromoteAt: 0.9},
		"refund": {Max: autonomy.Draft, Window: 10, MinSamples: 10, PromoteAt: 0.9},
		"search": {Max: autonomy.Read},
	})
}

func record(l *autonomy.Ladder, kind string, outcomes ...bool) {
	for _, ok := range outcomes {
		l.RecordDraft(kind, ok)
	}
}

func repeat(ok bool, n int) []bool {
	out := make([]bool, n)
	for i := range out {
		out[i] = ok
	}
	return out
}

func TestLevel(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		kind     string
		outcomes []bool
		incident bool
		want     autonomy.Level
	}{
		"始めは Draft": {kind: "triage", want: autonomy.Draft},
		"上限が Read の操作は Read のまま":      {kind: "search", outcomes: repeat(true, 10), want: autonomy.Read},
		"規則のない操作は Read":               {kind: "unknown", want: autonomy.Read},
		"件数が足りないと上げない":                {kind: "triage", outcomes: repeat(true, 9), want: autonomy.Draft},
		"10 件で 9 件承認なら Act に上げる":      {kind: "triage", outcomes: append([]bool{false}, repeat(true, 9)...), want: autonomy.Act},
		"10 件で 8 件承認なら上げない":           {kind: "triage", outcomes: append([]bool{false, false}, repeat(true, 8)...), want: autonomy.Draft},
		"古い却下は窓から外れる":                 {kind: "triage", outcomes: append(repeat(false, 5), repeat(true, 10)...), want: autonomy.Act},
		"上限が Draft の操作は承認が続いても Draft": {kind: "refund", outcomes: repeat(true, 20), want: autonomy.Draft},
		"Act で事故が起きたら Draft に戻す":      {kind: "triage", outcomes: repeat(true, 10), incident: true, want: autonomy.Draft},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			l := newLadder()
			record(l, tc.kind, tc.outcomes...)
			if tc.incident {
				l.RecordIncident(tc.kind)
			}
			if got := l.Level(tc.kind); got != tc.want {
				t.Errorf("Level = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestIncidentResetsTrack は、事故のあとは前の実績を使わず、承認を数え直してから Act に戻すことを確かめる。
func TestIncidentResetsTrack(t *testing.T) {
	t.Parallel()
	l := newLadder()
	record(l, "triage", repeat(true, 10)...)
	l.RecordIncident("triage")
	record(l, "triage", repeat(true, 9)...)
	if got := l.Level("triage"); got != autonomy.Draft {
		t.Fatalf("事故のあと 9 件で Level = %v, want draft", got)
	}
	record(l, "triage", true)
	if got := l.Level("triage"); got != autonomy.Act {
		t.Fatalf("事故のあと 10 件で Level = %v, want act", got)
	}
}
