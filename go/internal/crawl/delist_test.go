package crawl_test

import (
	"math"
	"testing"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

// 許す差は相対で3%。数える記事は1万件を超え、平均の標準誤差は1%を下回る。
func TestSimulateDelist(t *testing.T) {
	t.Parallel()
	base := crawl.DelistConfig{Days: 400, Warmup: 100, NewPerDay: 50, EndPerDay: 0.05, CheckCost: 1, Seed: 1}
	hazard := -math.Log1p(-base.EndPerDay)
	testCases := map[string]struct {
		check, list float64
	}{
		"存在の確認を1日ごと、一覧を1時間ごと": {check: 1, list: 1.0 / 24},
		"存在の確認を3日ごと、一覧を1日ごと":  {check: 3, list: 1},
		"存在の確認を6時間ごと、一覧を1日ごと": {check: 0.25, list: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			cfg := base
			cfg.CheckInterval, cfg.ListInterval = tc.check, tc.list
			got := crawl.SimulateDelist(cfg)
			// 掲載の期間は指数分布なので、見つけてから終えるまでも同じ指数分布で、確認の刻みに切り上げた分だけ遅れる。
			wantStale := tc.check/(-math.Expm1(-hazard*tc.check)) - 1/hazard
			wantMissed := 1 - (-math.Expm1(-hazard*tc.list))/(hazard*tc.list)
			// 見つけた記事に限ると遅れは一覧の間隔の半分より少し短いが、この設定では1%未満の差。
			wantDelay := tc.list / 2
			found := base.NewPerDay * (1 - wantMissed)
			wantFetches := 1/tc.list + found + found/(-math.Expm1(-hazard*tc.check))
			checks := map[string][2]float64{
				"出し続けた時間":     {got.StaleDays, wantStale},
				"見つけるまでの遅れ":   {got.DiscoverDays, wantDelay},
				"見つける前に終えた割合": {got.MissedRate, wantMissed},
				"1日の取得の回数":    {got.FetchesPerDay, wantFetches},
			}
			for name, c := range checks {
				if math.Abs(c[0]-c[1]) > 0.03*c[1] {
					t.Errorf("%s = %.4f, want %.4f", name, c[0], c[1])
				}
			}
		})
	}
}

func TestSimulateDelistConditionalCost(t *testing.T) {
	t.Parallel()
	cfg := crawl.DelistConfig{Days: 200, Warmup: 50, NewPerDay: 50, EndPerDay: 0.05, CheckInterval: 1, ListInterval: 1.0 / 24, Seed: 2}
	cfg.CheckCost = 1
	full := crawl.SimulateDelist(cfg)
	cfg.CheckCost = 0.1
	conditional := crawl.SimulateDelist(cfg)
	if full.FetchesPerDay != conditional.FetchesPerDay || full.StaleDays != conditional.StaleDays {
		t.Fatalf("費用だけを変えたのに回数か時間が変わった: %+v, %+v", full, conditional)
	}
	if math.Abs(full.CostPerDay-full.FetchesPerDay) > 1e-9 {
		t.Errorf("全件の取得の費用 %.3f が回数 %.3f と違う", full.CostPerDay, full.FetchesPerDay)
	}
	listAndNew := 24 + cfg.NewPerDay
	others := full.FetchesPerDay - (full.CostPerDay-conditional.CostPerDay)/0.9
	if math.Abs(others-listAndNew) > 0.05*listAndNew {
		t.Errorf("確認以外の取得 %.1f, want %.1f 前後", others, listAndNew)
	}
}
