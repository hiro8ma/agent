// Package adapter は KnowledgeService を Connect RPC で公開する。
package adapter

import (
	"context"
	"net/http"
	"strings"

	"connectrpc.com/connect"

	knowledgev1 "github.com/hiro8ma/agent/go/gen/knowledge/v1"
	"github.com/hiro8ma/agent/go/gen/knowledge/v1/knowledgev1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

const (
	defaultLimit  = 5
	maxLimit      = 20
	maxQueryRunes = 500
)

// Handler は検索器を Connect で公開する。検索器の実装（メモリ内 / pgvector / RAG Engine）には依存しない。
type Handler struct {
	searcher agentcore.KnowledgeSearcher
}

var _ knowledgev1connect.KnowledgeServiceHandler = (*Handler)(nil)

// NewHandler は利用者の特定まで含めたハンドラを返す。
func NewHandler(searcher agentcore.KnowledgeSearcher, auth libconnect.Authenticator) (string, http.Handler) {
	return knowledgev1connect.NewKnowledgeServiceHandler(&Handler{searcher: searcher},
		connect.WithInterceptors(libconnect.Telemetry(), libconnect.ServerIdentity(auth)))
}

func (h *Handler) Search(ctx context.Context, req *connect.Request[knowledgev1.SearchRequest]) (*connect.Response[knowledgev1.SearchResponse], error) {
	query := strings.TrimSpace(req.Msg.GetQuery())
	if query == "" {
		return nil, libconnect.Error(liberrors.Newf(liberrors.CodeInvalidArgument, "query が空"))
	}
	if len([]rune(query)) > maxQueryRunes {
		return nil, libconnect.Error(liberrors.Newf(liberrors.CodeInvalidArgument, "query は %d 文字まで", maxQueryRunes))
	}
	limit := int(req.Msg.GetLimit())
	switch {
	case limit == 0:
		limit = defaultLimit
	case limit < 0 || limit > maxLimit:
		return nil, libconnect.Error(liberrors.Newf(liberrors.CodeInvalidArgument, "limit は 1 から %d", maxLimit))
	}

	docs, err := h.searcher.Search(ctx, query, limit)
	if err != nil {
		return nil, libconnect.Error(err)
	}
	out := make([]*knowledgev1.Document, len(docs))
	for i, d := range docs {
		out[i] = &knowledgev1.Document{Title: d.Title, Content: d.Content}
	}
	return connect.NewResponse(&knowledgev1.SearchResponse{Documents: out}), nil
}
