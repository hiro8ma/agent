// Package client はエージェントのサービスから KnowledgeService を呼ぶ。
// agentcore.KnowledgeSearcher を満たすので、ADK と Genkit の検索ツールにそのまま渡せる。
package client

import (
	"context"
	"net/http"
	"os"

	"connectrpc.com/connect"

	knowledgev1 "github.com/hiro8ma/agent/go/gen/knowledge/v1"
	"github.com/hiro8ma/agent/go/gen/knowledge/v1/knowledgev1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/genkitagent/knowledge"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// EnvURL が設定されていれば、別プロセスの KnowledgeService を呼ぶ。
const EnvURL = "KNOWLEDGE_URL"

// Searcher は KnowledgeService を呼ぶ agentcore.KnowledgeSearcher。
type Searcher struct {
	client knowledgev1connect.KnowledgeServiceClient
}

var _ agentcore.KnowledgeSearcher = (*Searcher)(nil)

func NewSearcher(httpClient connect.HTTPClient, baseURL string) *Searcher {
	return &Searcher{client: knowledgev1connect.NewKnowledgeServiceClient(httpClient, baseURL,
		connect.WithInterceptors(libconnect.ForwardIdentity()))}
}

func (s *Searcher) Search(ctx context.Context, query string, limit int) ([]agentcore.KnowledgeDoc, error) {
	res, err := s.client.Search(ctx, connect.NewRequest(&knowledgev1.SearchRequest{
		Query: query, Limit: int32(min(limit, 20)), //nolint:gosec // 上限 20 に丸めてから変換する
	}))
	if err != nil {
		return nil, libconnect.FromConnect(err, "knowledge: 検索")
	}
	docs := make([]agentcore.KnowledgeDoc, len(res.Msg.GetDocuments()))
	for i, d := range res.Msg.GetDocuments() {
		docs[i] = agentcore.KnowledgeDoc{Title: d.GetTitle(), Content: d.GetContent()}
	}
	return docs, nil
}

// FromEnv は KNOWLEDGE_URL があれば別プロセスの KnowledgeService を、無ければプロセス内の検索器を返す。
func FromEnv() (searcher agentcore.KnowledgeSearcher, where string) {
	if url := os.Getenv(EnvURL); url != "" {
		return NewSearcher(http.DefaultClient, url), url
	}
	return knowledge.NewInMemory(), "in-process"
}
