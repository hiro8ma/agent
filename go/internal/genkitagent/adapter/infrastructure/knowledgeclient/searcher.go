// Package knowledgeclient は KnowledgeService を呼んで社内ナレッジを検索する。
package knowledgeclient

import (
	"context"
	"net/http"
	"os"

	"connectrpc.com/connect"

	knowledgev1 "github.com/hiro8ma/agent/go/gen/knowledge/v1"
	"github.com/hiro8ma/agent/go/gen/knowledge/v1/knowledgev1connect"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/inmemory"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

const EnvURL = "KNOWLEDGE_URL"

type Searcher struct {
	client knowledgev1connect.KnowledgeServiceClient
}

var _ externalservice.KnowledgeSearcher = (*Searcher)(nil)

func NewSearcher(httpClient connect.HTTPClient, baseURL string) *Searcher {
	return &Searcher{client: knowledgev1connect.NewKnowledgeServiceClient(httpClient, baseURL,
		connect.WithInterceptors(libconnect.Telemetry(), libconnect.ForwardIdentity()))}
}

func (s *Searcher) Search(ctx context.Context, query string, limit int) ([]model.KnowledgeDoc, error) {
	res, err := s.client.Search(ctx, connect.NewRequest(&knowledgev1.SearchRequest{
		Query: query, Limit: int32(min(limit, 20)), //nolint:gosec // 上限 20 に丸めてから変換する
	}))
	if err != nil {
		return nil, libconnect.FromConnect(err, "knowledge: 検索")
	}
	docs := make([]model.KnowledgeDoc, len(res.Msg.GetDocuments()))
	for i, d := range res.Msg.GetDocuments() {
		docs[i] = model.KnowledgeDoc{Title: d.GetTitle(), Content: d.GetContent()}
	}
	return docs, nil
}

// FromEnv は KNOWLEDGE_URL があれば別プロセスの KnowledgeService を、無ければプロセス内の検索器を返す。
func FromEnv() (searcher externalservice.KnowledgeSearcher, where string) {
	if url := os.Getenv(EnvURL); url != "" {
		return NewSearcher(http.DefaultClient, url), url
	}
	return inmemory.NewKnowledge(), "in-process"
}
