package agentmemory

import (
	"testing"
	"time"
)

var base = time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)

func fact(subj, stmt, src string, conf float64, at time.Time, ttl time.Duration) Fact {
	return Fact{Subject: subj, Statement: stmt, Source: src,
		Confidence: conf, WrittenAt: at, TTL: ttl}
}

// 検証を通り抜けた 1 件が、以後の読み出し全てに乗る。
// 推論 1 回で終わる誤りとの違いはここにある。
func TestOneBadFactContaminatesEveryRecall(t *testing.T) {
	const reads = 50

	// 検証なしの保管庫。
	loose := NewStore(nil, nil)
	loose.Write(fact("user1", "在宅勤務は週 3 日まで", "就業規則 v2", 0.95, base, 0))
	loose.Write(fact("user1", "在宅勤務は無制限", "", 0.30, base.Add(time.Minute), 0)) // 出典なし・低確度

	bad := 0
	for i := 0; i < reads; i++ {
		for _, f := range loose.Recall("user1") {
			if f.Statement == "在宅勤務は無制限" {
				bad++
			}
		}
	}
	t.Logf("検証なし: %d 回の読み出しのうち %d 回に誤りが乗る（増幅 %d 倍）", reads, bad, bad)
	if bad != reads {
		t.Fatalf("誤りが %d 回。%d 回を期待", bad, reads)
	}

	// 書き込み時に検証する保管庫。
	strict := NewStore([]Rule{RequireSource, MinConfidence(0.6)}, nil)
	strict.Write(fact("user1", "在宅勤務は週 3 日まで", "就業規則 v2", 0.95, base, 0))
	if err := strict.Write(fact("user1", "在宅勤務は無制限", "", 0.30, base.Add(time.Minute), 0)); err == nil {
		t.Fatal("出典の無い Fact を保存した")
	}

	bad = 0
	for i := 0; i < reads; i++ {
		for _, f := range strict.Recall("user1") {
			if f.Statement == "在宅勤務は無制限" {
				bad++
			}
		}
	}
	t.Logf("書き込み検証あり: %d 回の読み出しのうち %d 回", reads, bad)
	if bad != 0 {
		t.Fatalf("誤りが %d 回残った", bad)
	}
	if strict.Rejected != 1 {
		t.Errorf("拒否 %d 件。1 件を期待", strict.Rejected)
	}
}

// 書き込み時に正しかった事実が、時間で誤りになる。
// 書き込み検証では捕まらないため、読み出し側にも検証が要る。
func TestWriteValidationCannotCatchStaleFacts(t *testing.T) {
	f := fact("user1", "現在のプロジェクトは A", "本人の発言", 0.95, base, 24*time.Hour)

	// 書き込み検証は全部通る。書いた時点では正しい。
	for _, r := range []Rule{RequireSource, MinConfidence(0.6), RejectHedged} {
		if err := r(f); err != nil {
			t.Fatalf("書き込み時に弾かれた: %v", err)
		}
	}

	// 読み出しだけを検証する保管庫。時刻を進める。
	now := base
	s := NewStore(nil, []Rule{NotExpired(func() time.Time { return now })})
	s.Write(f)

	if got := len(s.Recall("user1")); got != 1 {
		t.Fatalf("書いた直後に %d 件", got)
	}
	now = base.Add(25 * time.Hour)
	if got := len(s.Recall("user1")); got != 0 {
		t.Fatalf("失効後に %d 件返った", got)
	}
	if s.Filtered != 1 {
		t.Errorf("除いた件数 %d", s.Filtered)
	}
}

// 矛盾は書き込み時には分からない。後から入る Fact と衝突する。
func TestContradictionAppearsLater(t *testing.T) {
	s := NewStore([]Rule{RequireSource, MinConfidence(0.6)}, nil)

	s.Write(fact("user1", "希望勤務地は東京", "面談 1 回目", 0.90, base, 0))
	if len(s.Contradictions()) != 0 {
		t.Fatal("1 件目で矛盾が出た")
	}

	// 2 件目も単体では正しい。組にして初めて矛盾になる。
	s.Write(fact("user1", "希望勤務地は大阪", "面談 2 回目", 0.90, base.Add(time.Hour), 0))

	c := s.Contradictions()
	if len(c["user1"]) != 2 {
		t.Fatalf("矛盾を検出できていない: %v", c)
	}
	t.Logf("矛盾: %v", c["user1"])

	// 読み出しは新しい順。古い方を黙って消さず、両方残して順序で示す。
	got := s.Recall("user1")
	if got[0].Statement != "希望勤務地は大阪" {
		t.Errorf("新しい順になっていない: %v", got)
	}
	if len(got) != 2 {
		t.Errorf("%d 件。両方残すことを期待", len(got))
	}
}

func TestRejectHedgedStatements(t *testing.T) {
	s := NewStore([]Rule{RequireSource, RejectHedged}, nil)
	err := s.Write(fact("user1", "転職を考えているかもしれない", "雑談", 0.5, base, 0))
	if err == nil {
		t.Fatal("推測を断定として保存した")
	}
	t.Logf("拒否理由: %v", err)
	if s.Len() != 0 {
		t.Errorf("%d 件保存された", s.Len())
	}
}
