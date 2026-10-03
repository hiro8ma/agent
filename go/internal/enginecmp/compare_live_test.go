package enginecmp_test

import (
	"cmp"
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

// ENGINECMP_LIVE=1 のときだけ、起動済みの Elasticsearch と Meilisearch に繋ぐ。教材の順位の主張を確かめる。
func TestLiveTutorialClaims(t *testing.T) {
	if os.Getenv("ENGINECMP_LIVE") != "1" {
		t.Skip("ENGINECMP_LIVE=1 で Elasticsearch と Meilisearch に繋ぐ")
	}
	ctx := t.Context()
	es := enginecmp.NewElasticsearch(cmp.Or(os.Getenv("ENGINECMP_ES_URL"), "http://localhost:9200"), "kenpou_test", nil)
	meili := enginecmp.NewMeilisearch(cmp.Or(os.Getenv("ENGINECMP_MEILI_URL"), "http://localhost:7700"), "kenpou_test", nil)
	if err := enginecmp.WaitReady(ctx, 2*time.Minute, es.Ping, meili.Health); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx := context.WithoutCancel(ctx)
		if err := es.DeleteIndex(ctx); err != nil {
			t.Error(err)
		}
		if err := meili.DeleteIndex(ctx); err != nil {
			t.Error(err)
		}
	})
	articles, err := enginecmp.Kenpou()
	if err != nil {
		t.Fatal(err)
	}
	engines := []enginecmp.Engine{
		enginecmp.ElasticsearchEngine{Client: es},
		enginecmp.MeilisearchEngine{Client: meili},
		&enginecmp.LocalEngine{Ranking: search.RankingBucket},
	}
	reports, err := enginecmp.Compare(ctx, engines, articles, []string{"すべて国民"}, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Elasticsearch は教材の 26, 11, 13。Meilisearch は「すべて国民」を含む条のうち番号の小さい 3 つで、規則による順位も同じになる。
	want := [][]int{{26, 11, 13}, {13, 14, 25}, {13, 14, 25}}
	for i, r := range reports {
		t.Logf("%s: 索引 %v, 上位 %+v", r.Engine, r.IndexTime, r.Hits[0])
		if got := ids(r.Hits[0]); !reflect.DeepEqual(got, want[i]) {
			t.Errorf("%s の上位 = %v, want %v", r.Engine, got, want[i])
		}
	}
}
