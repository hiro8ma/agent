// enginecmp は日本国憲法の条を Elasticsearch（kuromoji）、Meilisearch、internal/search に入れ、同じクエリの上位を並べて出す。
// 先に Elasticsearch を localhost:9200、Meilisearch を localhost:7700 で起動しておく（起動の手順は internal/enginecmp の README）。
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
	"github.com/hiro8ma/agent/go/internal/search"
)

func main() {
	var (
		esURL    = flag.String("es", "http://localhost:9200", "Elasticsearch の URL")
		meiliURL = flag.String("meili", "http://localhost:7700", "Meilisearch の URL")
		index    = flag.String("index", "kenpou", "索引の名前")
		limit    = flag.Int("limit", 3, "クエリごとに出す件数")
		wait     = flag.Duration("wait", 2*time.Minute, "起動を待つ上限")
		keep     = flag.Bool("keep", false, "終わっても索引を消さない")
	)
	flag.Parse()
	queries := flag.Args()
	if len(queries) == 0 {
		queries = enginecmp.DefaultQueries
	}

	ctx := context.Background()
	es := enginecmp.NewElasticsearch(*esURL, *index, nil)
	meili := enginecmp.NewMeilisearch(*meiliURL, *index, nil)
	meiliJA := enginecmp.NewMeilisearch(*meiliURL, *index, nil)
	meiliJA.Locales = []string{"jpn"}
	if err := enginecmp.WaitReady(ctx, *wait, es.Ping, meili.Health); err != nil {
		log.Fatal(err)
	}
	texts, err := enginecmp.Kenpou()
	if err != nil {
		log.Fatal(err)
	}
	engines := []enginecmp.Engine{
		enginecmp.ElasticsearchEngine{Client: es},
		enginecmp.MeilisearchEngine{Client: meili},
		enginecmp.MeilisearchEngine{Client: meiliJA},
		&enginecmp.LocalEngine{Ranking: search.RankingBM25},
		&enginecmp.LocalEngine{Ranking: search.RankingBucket},
	}
	reports, err := enginecmp.Compare(ctx, engines, texts, queries, *limit)
	if err != nil {
		log.Fatal(err)
	}
	if !*keep {
		defer func() {
			if err := es.DeleteIndex(ctx); err != nil {
				log.Print(err)
			}
			if err := meili.DeleteIndex(ctx); err != nil {
				log.Print(err)
			}
		}()
	}

	w := os.Stdout
	fmt.Fprintf(w, "文書数 %d\n\n## 索引の時間\n", len(texts))
	for _, r := range reports {
		fmt.Fprintf(w, "- %s: %v\n", r.Engine, r.IndexTime.Round(time.Millisecond))
	}
	bigram := search.NewAnalyzer()
	for qi, q := range queries {
		tokens, err := es.Analyze(ctx, q)
		if err != nil {
			log.Print(err)
		}
		fmt.Fprintf(w, "\n## %s\n- kuromoji: %s\n- bigram: %s\n", q, strings.Join(tokens, " / "), strings.Join(bigram.Analyze(q), " / "))
		for _, r := range reports {
			fmt.Fprintf(w, "\n%s\n", r.Engine)
			if len(r.Hits[qi]) == 0 {
				fmt.Fprintln(w, "  （該当なし）")
			}
			for rank, h := range r.Hits[qi] {
				fmt.Fprintf(w, "  %d. 第%d条 score=%.4f %s\n", rank+1, h.ID, h.Score, snippet(h.Text, 120))
			}
		}
	}
}

func snippet(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}
