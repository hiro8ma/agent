package knowledge

import (
	"strings"
	"testing"
)

func TestEfSearch(t *testing.T) {
	// HNSW は ef_search 件の候補を集めてから WHERE を適用する。
	// 選択率 s なら残るのは期待値で ef_search * s 件なので、
	// limit 件を返すには ef_search >= limit / s が要る。
	tests := []struct {
		name        string
		selectivity float64
		cap         int
		limit       int
		want        int
	}{
		// 絞り込みが無ければ既定の 40 で足りる。
		{"選択率 1.0 は既定値", 1.0, 1000, 10, 40},
		// 6.2% は実測した構成。10 / 0.062 * 2 = 322
		{"選択率 6.2% で 322", 0.062, 1000, 10, 322},
		{"選択率 25% で 80", 0.25, 1000, 10, 80},
		// 期待値ちょうどでは半分の確率で足りないため倍を取る。
		{"倍の余裕を取る", 0.5, 1000, 10, 40},
		{"limit が増えれば比例", 0.1, 1000, 50, 1000},
		// 選択率が極端に低いと際限なく増える。厳密検索より遅くなるため頭打ちにする。
		{"上限で頭打ち", 0.001, 500, 10, 500},
		// 未設定は絞り込み無しと同じ扱いにする。過小評価より安全側に倒す。
		{"未設定は 1.0 扱い", 0, 1000, 10, 40},
		{"範囲外も 1.0 扱い", 1.5, 1000, 10, 40},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &PgVector{cfg: PgVectorConfig{Selectivity: tt.selectivity, EfSearchCap: tt.cap}}
			if got := s.EfSearch(tt.limit); got != tt.want {
				t.Errorf("EfSearch(%d) = %d, 期待 %d", tt.limit, got, tt.want)
			}
		})
	}
}

// 選択率が下がるほど ef_search が増えることを確かめる。
// 逆向きになっていると、絞り込みが強い場面ほど取りこぼす。
func TestEfSearchMonotonic(t *testing.T) {
	prev := 0
	for _, sel := range []float64{1.0, 0.5, 0.25, 0.1, 0.05, 0.01} {
		s := &PgVector{cfg: PgVectorConfig{Selectivity: sel, EfSearchCap: 100000}}
		got := s.EfSearch(10)
		if got < prev {
			t.Errorf("選択率 %.2f で ef_search = %d, 直前より小さい (%d)", sel, got, prev)
		}
		prev = got
	}
}

func TestBuildQuery(t *testing.T) {
	t.Run("絞り込みなし", func(t *testing.T) {
		s := &PgVector{cfg: PgVectorConfig{
			Table: "docs", EmbeddingColumn: "emb", TitleColumn: "t", ContentColumn: "c",
		}}
		sql, args := s.buildQuery(toVector([]float32{1, 2, 3}), 5)

		if strings.Contains(sql, "WHERE") {
			t.Errorf("絞り込みが無いのに WHERE がある: %s", sql)
		}
		// <=> はコサイン距離。索引を vector_cosine_ops で作る前提と対になる。
		if !strings.Contains(sql, "emb <=> $1") {
			t.Errorf("コサイン距離で並べていない: %s", sql)
		}
		if len(args) != 2 {
			t.Errorf("引数 %d 個, 期待 2（ベクトルと limit）", len(args))
		}
		if args[1] != 5 {
			t.Errorf("limit = %v, 期待 5", args[1])
		}
	})

	t.Run("絞り込みあり", func(t *testing.T) {
		s := &PgVector{cfg: PgVectorConfig{
			Table: "docs", EmbeddingColumn: "emb", TitleColumn: "t", ContentColumn: "c",
			FilterColumn: "tenant_id", FilterValue: "acme",
		}}
		sql, args := s.buildQuery(toVector([]float32{1, 2, 3}), 5)

		if !strings.Contains(sql, "WHERE tenant_id = $1") {
			t.Errorf("絞り込みが入っていない: %s", sql)
		}
		if !strings.Contains(sql, "emb <=> $2") {
			t.Errorf("ベクトルの位置がずれている: %s", sql)
		}
		if len(args) != 3 || args[0] != "acme" || args[2] != 5 {
			t.Errorf("引数がずれている: %v", args)
		}
	})

	// 値を SQL に埋め込まずプレースホルダで渡すことを確かめる。
	t.Run("値を埋め込まない", func(t *testing.T) {
		s := &PgVector{cfg: PgVectorConfig{
			Table: "docs", EmbeddingColumn: "emb", TitleColumn: "t", ContentColumn: "c",
			FilterColumn: "tenant_id", FilterValue: "'; DROP TABLE docs; --",
		}}
		sql, args := s.buildQuery(toVector([]float32{1}), 5)

		if strings.Contains(sql, "DROP TABLE") {
			t.Errorf("値が SQL に埋め込まれている: %s", sql)
		}
		if args[0] != "'; DROP TABLE docs; --" {
			t.Errorf("値が引数として渡っていない: %v", args[0])
		}
	})
}

func TestApplyDefaults(t *testing.T) {
	c := PgVectorConfig{}
	c.applyDefaults()

	for _, tt := range []struct{ got, want, name string }{
		{c.Table, "documents", "Table"},
		{c.EmbeddingColumn, "embedding", "EmbeddingColumn"},
		{c.TitleColumn, "title", "TitleColumn"},
		{c.ContentColumn, "content", "ContentColumn"},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %q, 期待 %q", tt.name, tt.got, tt.want)
		}
	}
	if c.EfSearchCap != 1000 {
		t.Errorf("EfSearchCap = %d, 期待 1000", c.EfSearchCap)
	}

	// 明示された値は上書きしない。
	c2 := PgVectorConfig{Table: "chunks", EfSearchCap: 200}
	c2.applyDefaults()
	if c2.Table != "chunks" || c2.EfSearchCap != 200 {
		t.Errorf("明示した設定が上書きされた: %+v", c2)
	}
}
