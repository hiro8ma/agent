package enginecmp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// Hit は検索結果の 1 件。ID は 1 から始まる文書の番号で、憲法では条の番号になる。
type Hit struct {
	ID    int
	Score float64
	Text  string
}

// Elasticsearch は 1 つの索引を、kuromoji で形態素に分けて BM25 で採点する設定で扱う。
type Elasticsearch struct {
	http  httpDoer
	index string
}

// NewElasticsearch の client が nil なら 30 秒で打ち切るクライアントを使う。
func NewElasticsearch(baseURL, index string, client *http.Client) *Elasticsearch {
	return &Elasticsearch{http: newHTTPDoer(baseURL, client), index: index}
}

func (e *Elasticsearch) path(suffix string) string {
	return "/" + url.PathEscape(e.index) + suffix
}

func (e *Elasticsearch) Ping(ctx context.Context) error {
	return e.http.do(ctx, http.MethodGet, "/", "", nil, nil)
}

const indexSettings = `{
  "settings": {
    "analysis": {
      "analyzer": {
        "ja": {
          "type": "custom",
          "tokenizer": "kuromoji_tokenizer",
          "filter": ["lowercase", "kuromoji_baseform", "kuromoji_stemmer"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "text": {"type": "text", "analyzer": "ja"}
    }
  }
}`

// CreateIndex は text 項目を kuromoji の形態素に分け、小文字化、活用語の基本形化、カタカナ語末の長音の除去をかける解析器 ja で索引する。
func (e *Elasticsearch) CreateIndex(ctx context.Context) error {
	return e.http.do(ctx, http.MethodPut, e.path(""), "application/json", []byte(indexSettings), nil)
}

// DeleteIndex は索引が無くてもエラーにしない。
func (e *Elasticsearch) DeleteIndex(ctx context.Context) error {
	err := e.http.do(ctx, http.MethodDelete, e.path(""), "", nil, nil)
	if isNotFound(err) {
		return nil
	}
	return err
}

// Bulk は texts[i] を _id が i+1 の文書として _bulk で一度に入れる。
func (e *Elasticsearch) Bulk(ctx context.Context, texts []string) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for i, t := range texts {
		if err := enc.Encode(map[string]any{"index": map[string]any{"_index": e.index, "_id": strconv.Itoa(i + 1)}}); err != nil {
			return err
		}
		if err := enc.Encode(map[string]string{"text": t}); err != nil {
			return err
		}
	}
	var res struct {
		Errors bool `json:"errors"`
		Items  []map[string]struct {
			ID    string          `json:"_id"`
			Error json.RawMessage `json:"error"`
		} `json:"items"`
	}
	if err := e.http.do(ctx, http.MethodPost, "/_bulk", "application/x-ndjson", buf.Bytes(), &res); err != nil {
		return err
	}
	// _bulk は一部の文書が失敗しても 200 を返すので、errors を見る。
	if res.Errors {
		for _, item := range res.Items {
			for _, r := range item {
				if len(r.Error) > 0 {
					return fmt.Errorf("_bulk: _id %s: %s", r.ID, r.Error)
				}
			}
		}
		return fmt.Errorf("_bulk: errors が true")
	}
	return nil
}

// Refresh は入れた文書を検索に見えるようにする。
func (e *Elasticsearch) Refresh(ctx context.Context) error {
	return e.http.do(ctx, http.MethodPost, e.path("/_refresh"), "", nil, nil)
}

// Match は text 項目への match クエリで上位 size 件を返す。
func (e *Elasticsearch) Match(ctx context.Context, query string, size int) ([]Hit, error) {
	body := map[string]any{
		"query": map[string]any{"match": map[string]any{"text": query}},
		"size":  size,
	}
	var res struct {
		Hits struct {
			Hits []struct {
				ID     string  `json:"_id"`
				Score  float64 `json:"_score"`
				Source struct {
					Text string `json:"text"`
				} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := e.http.do(ctx, http.MethodPost, e.path("/_search"), "application/json", body, &res); err != nil {
		return nil, err
	}
	hits := make([]Hit, len(res.Hits.Hits))
	for i, h := range res.Hits.Hits {
		id, err := strconv.Atoi(h.ID)
		if err != nil {
			return nil, fmt.Errorf("_id %q: %w", h.ID, err)
		}
		hits[i] = Hit{ID: id, Score: h.Score, Text: h.Source.Text}
	}
	return hits, nil
}

// Analyze は解析器 ja が text を分けた索引語を返す。
func (e *Elasticsearch) Analyze(ctx context.Context, text string) ([]string, error) {
	var res struct {
		Tokens []struct {
			Token string `json:"token"`
		} `json:"tokens"`
	}
	body := map[string]string{"analyzer": "ja", "text": text}
	if err := e.http.do(ctx, http.MethodPost, e.path("/_analyze"), "application/json", body, &res); err != nil {
		return nil, err
	}
	tokens := make([]string, len(res.Tokens))
	for i, t := range res.Tokens {
		tokens[i] = t.Token
	}
	return tokens, nil
}
