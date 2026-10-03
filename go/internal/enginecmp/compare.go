package enginecmp

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// Engine は比べる検索エンジン。Index は texts[i] を番号 i+1 の文書として入れ、検索できる状態になるまで戻らない。
type Engine interface {
	Name() string
	Index(ctx context.Context, texts []string) error
	Search(ctx context.Context, query string, limit int) ([]Hit, error)
}

// ElasticsearchEngine は索引を作り直してから _bulk で入れ、_refresh まで済ませる。
type ElasticsearchEngine struct{ Client *Elasticsearch }

func (ElasticsearchEngine) Name() string { return "Elasticsearch (kuromoji, BM25)" }

func (e ElasticsearchEngine) Index(ctx context.Context, texts []string) error {
	if err := e.Client.DeleteIndex(ctx); err != nil {
		return err
	}
	if err := e.Client.CreateIndex(ctx); err != nil {
		return err
	}
	if err := e.Client.Bulk(ctx, texts); err != nil {
		return err
	}
	return e.Client.Refresh(ctx)
}

func (e ElasticsearchEngine) Search(ctx context.Context, query string, limit int) ([]Hit, error) {
	return e.Client.Match(ctx, query, limit)
}

// MeilisearchEngine は索引を作り直してから文書を送り、タスクの完了まで待つ。
type MeilisearchEngine struct{ Client *Meilisearch }

func (m MeilisearchEngine) Name() string {
	if len(m.Client.Locales) > 0 {
		return "Meilisearch (locales=" + strings.Join(m.Client.Locales, ",") + ")"
	}
	return "Meilisearch"
}

func (m MeilisearchEngine) Index(ctx context.Context, texts []string) error {
	if err := m.Client.DeleteIndex(ctx); err != nil {
		return err
	}
	uid, err := m.Client.AddDocuments(ctx, texts)
	if err != nil {
		return err
	}
	_, err = m.Client.WaitTask(ctx, uid)
	return err
}

func (m MeilisearchEngine) Search(ctx context.Context, query string, limit int) ([]Hit, error) {
	return m.Client.Search(ctx, query, limit)
}

// LocalEngine は internal/search を既定の解析器（日本語は文字の bigram）で使う。
type LocalEngine struct {
	Ranking search.Ranking
	ix      *search.Index
}

func (l *LocalEngine) Name() string {
	if l.Ranking == search.RankingBucket {
		return "internal/search (bigram, RankingBucket)"
	}
	return "internal/search (bigram, BM25)"
}

func (l *LocalEngine) Index(ctx context.Context, texts []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	docs := make([]search.Doc, len(texts))
	for i, t := range texts {
		docs[i] = search.Doc{ID: strconv.Itoa(i + 1), Content: t}
	}
	l.ix = search.New(docs)
	return nil
}

func (l *LocalEngine) Search(ctx context.Context, query string, limit int) ([]Hit, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if l.ix == nil {
		return nil, errors.New("enginecmp: Index の前に Search を呼んだ")
	}
	res := l.ix.RankQuery(search.Query{Text: query, Ranking: l.Ranking}, limit)
	hits := make([]Hit, len(res.Hits))
	for i, h := range res.Hits {
		id, err := strconv.Atoi(h.Doc.ID)
		if err != nil {
			return nil, err
		}
		hits[i] = Hit{ID: id, Score: h.Score, Text: h.Doc.Content}
	}
	return hits, nil
}

// Report は 1 つのエンジンの結果。Hits はクエリと同じ順に並ぶ。
type Report struct {
	Engine    string
	IndexTime time.Duration
	Hits      [][]Hit
}

// Compare はエンジンごとに texts を入れてから queries を順に投げ、上位 limit 件を集める。
func Compare(ctx context.Context, engines []Engine, texts, queries []string, limit int) ([]Report, error) {
	reports := make([]Report, 0, len(engines))
	for _, e := range engines {
		start := time.Now()
		if err := e.Index(ctx, texts); err != nil {
			return nil, fmt.Errorf("%s: %w", e.Name(), err)
		}
		r := Report{Engine: e.Name(), IndexTime: time.Since(start), Hits: make([][]Hit, len(queries))}
		for i, q := range queries {
			hits, err := e.Search(ctx, q, limit)
			if err != nil {
				return nil, fmt.Errorf("%s: %q: %w", e.Name(), q, err)
			}
			r.Hits[i] = hits
		}
		reports = append(reports, r)
	}
	return reports, nil
}
