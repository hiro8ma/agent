package crawl

import (
	"math"
	"math/rand/v2"
)

// DelistConfig は、記事が出ては掲載を終えるサイトを、一覧のページと記事ごとの存在の確認で追う模擬の設定。時間の単位は日。
type DelistConfig struct {
	// Days は模擬の長さで、Warmup より後に起きたことだけを数える。
	Days   float64
	Warmup float64
	// NewPerDay は1日に出る新しい記事の件数の期待値（ポアソン過程）。
	NewPerDay float64
	// EndPerDay は掲載中の記事が1日のうちに掲載を終える確率。
	EndPerDay float64
	// ListInterval は一覧のページを見る間隔。一覧で見つけた記事はその場で1回取る。
	ListInterval float64
	// CheckInterval は見つけた記事の存在を確かめる間隔。掲載を終えた記事は次の確認で404になり、出すのをやめる。
	CheckInterval float64
	// CheckCost は存在の確認1回の費用で、全件の取得を1とする。条件付きの取得で304を返すなら1より小さい。
	CheckCost float64
	Seed      uint64
}

// DelistResult は模擬の結果。
type DelistResult struct {
	// StaleDays は掲載を終えた記事を、まだ出していた時間の平均。
	StaleDays float64
	// DiscoverDays は新しい記事が出てから見つけるまでの遅れの平均。
	DiscoverDays float64
	// MissedRate は見つける前に掲載を終えた記事の割合。
	MissedRate float64
	// FetchesPerDay は一覧、新しい記事、存在の確認を合わせた1日の取得の回数。
	FetchesPerDay float64
	// CostPerDay は FetchesPerDay を、存在の確認だけ CheckCost で数えた1日の費用。
	CostPerDay float64
	Published  int
	Ended      int
}

// SimulateDelist は記事の出る時刻と掲載の期間を乱数で引き、一覧と存在の確認の時刻から、掲載を終えた記事を出し続けた時間と、見つけるまでの遅れを数える。
func SimulateDelist(cfg DelistConfig) DelistResult {
	rng := rand.New(rand.NewPCG(cfg.Seed, 0x6a09e667f3bcc909))
	hazard := -math.Log1p(-cfg.EndPerDay)
	window := cfg.Days - cfg.Warmup
	inWindow := func(t float64) bool { return t >= cfg.Warmup && t < cfg.Days }
	var res DelistResult
	var stale, delay float64
	var found, missed int
	var listings, newFetches, checks float64
	for k := 0.0; k*cfg.ListInterval < cfg.Days; k++ {
		if inWindow(k * cfg.ListInterval) {
			listings++
		}
	}
	for published := rng.ExpFloat64() / cfg.NewPerDay; published < cfg.Days; published += rng.ExpFloat64() / cfg.NewPerDay {
		ended := published + rng.ExpFloat64()/hazard
		discovered := math.Ceil(published/cfg.ListInterval) * cfg.ListInterval
		countPublished := inWindow(published)
		if countPublished {
			res.Published++
		}
		if discovered >= ended {
			if countPublished {
				missed++
			}
			continue
		}
		if countPublished {
			found++
			delay += discovered - published
		}
		if inWindow(discovered) {
			newFetches++
		}
		n := math.Ceil((ended - discovered) / cfg.CheckInterval)
		removed := discovered + n*cfg.CheckInterval
		if inWindow(ended) && removed < cfg.Days {
			res.Ended++
			stale += removed - ended
		}
		first := max(1, math.Ceil((cfg.Warmup-discovered)/cfg.CheckInterval))
		last := min(n, math.Ceil((cfg.Days-discovered)/cfg.CheckInterval)-1)
		if last >= first {
			checks += last - first + 1
		}
	}
	res.StaleDays = stale / float64(res.Ended)
	res.DiscoverDays = delay / float64(found)
	res.MissedRate = float64(missed) / float64(res.Published)
	res.FetchesPerDay = (listings + newFetches + checks) / window
	res.CostPerDay = (listings + newFetches + cfg.CheckCost*checks) / window
	return res
}
