// Package tei は Hugging Face の text-embeddings-inference（TEI）に HTTP で繋ぎ、手元で動かす埋め込みモデルで文字列をベクトルにする。
package tei

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// Ruri v3 の接頭辞。検索ではクエリと文書で別の接頭辞を付け、意味の近さだけを見るときは何も付けない。
const (
	RuriQueryPrefix    = "検索クエリ: "
	RuriDocumentPrefix = "検索文書: "
	RuriTopicPrefix    = "トピック: "
)

const (
	DefaultBatchSize = 32
	DefaultTimeout   = 60 * time.Second
)

// Client は WithPrefixes の接頭辞を、Embed と EmbedQueries ではクエリの側、EmbedDocuments では文書の側を入力の前に付ける。EmbedTexts は何も付けない。
type Client struct {
	baseURL        string
	httpClient     *http.Client
	batchSize      int
	queryPrefix    string
	documentPrefix string
}

var (
	_ search.Embedder      = (*Client)(nil)
	_ search.Reranker      = (*Client)(nil)
	_ search.TokenEmbedder = (*Client)(nil)
)

type Option func(*Client)

func WithHTTPClient(c *http.Client) Option { return func(cl *Client) { cl.httpClient = c } }

// WithBatchSize の n は 1 回の要求に詰める文字列の数。TEI の --max-client-batch-size（既定 32）を超えると 413 で断られる。
func WithBatchSize(n int) Option { return func(cl *Client) { cl.batchSize = n } }

// WithPrefixes はクエリと文書の接頭辞を設定する。Ruri v3 なら RuriQueryPrefix と RuriDocumentPrefix を渡す。
func WithPrefixes(query, document string) Option {
	return func(cl *Client) { cl.queryPrefix, cl.documentPrefix = query, document }
}

func New(baseURL string, opts ...Option) *Client {
	c := &Client{baseURL: strings.TrimRight(baseURL, "/")}
	for _, o := range opts {
		o(c)
	}
	if c.httpClient == nil {
		c.httpClient = &http.Client{Timeout: DefaultTimeout}
	}
	if c.batchSize <= 0 {
		c.batchSize = DefaultBatchSize
	}
	return c
}

// Embed はクエリを 1 件ベクトルにする。search.Flat などの Retriever がクエリに使うのでクエリの接頭辞を付ける。
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := c.EmbedTexts(ctx, []string{c.queryPrefix + text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func (c *Client) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	return c.EmbedTexts(ctx, withPrefix(c.documentPrefix, texts))
}

func (c *Client) EmbedQueries(ctx context.Context, texts []string) ([][]float32, error) {
	return c.EmbedTexts(ctx, withPrefix(c.queryPrefix, texts))
}

func withPrefix(prefix string, texts []string) []string {
	if prefix == "" {
		return texts
	}
	out := make([]string, len(texts))
	for i, t := range texts {
		out[i] = prefix + t
	}
	return out
}

type embedRequest struct {
	Inputs    []string `json:"inputs"`
	Normalize bool     `json:"normalize"`
	Truncate  bool     `json:"truncate"`
}

// EmbedTexts は texts をそのまま batchSize 件ずつ送り、同じ順でベクトルを返す。ベクトルは長さ 1 に正規化してある。
func (c *Client) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += c.batchSize {
		batch := texts[start:min(start+c.batchSize, len(texts))]
		var vecs [][]float32
		if err := c.do(ctx, http.MethodPost, "/embed", embedRequest{Inputs: batch, Normalize: true, Truncate: true}, &vecs); err != nil {
			return nil, err
		}
		if len(vecs) != len(batch) {
			return nil, fmt.Errorf("tei: got %d embeddings for %d inputs", len(vecs), len(batch))
		}
		out = append(out, vecs...)
	}
	return out, nil
}

type rerankRequest struct {
	Query     string   `json:"query"`
	Texts     []string `json:"texts"`
	RawScores bool     `json:"raw_scores"`
	Truncate  bool     `json:"truncate"`
}

type rerankResult struct {
	Index int     `json:"index"`
	Score float32 `json:"score"`
}

// Rerank は交差エンコーダを読み込んだ TEI の /rerank で、texts をクエリとの関連度で採点し、texts と同じ順で返す。点数はシグモイドを通した0から1の値。
// 接頭辞は付けない。TEI は点数の高い順に返すので、index で入力の順に戻す。
func (c *Client) Rerank(ctx context.Context, query string, texts []string) ([]float32, error) {
	out := make([]float32, len(texts))
	for start := 0; start < len(texts); start += c.batchSize {
		batch := texts[start:min(start+c.batchSize, len(texts))]
		var res []rerankResult
		if err := c.do(ctx, http.MethodPost, "/rerank", rerankRequest{Query: query, Texts: batch, Truncate: true}, &res); err != nil {
			return nil, err
		}
		if len(res) != len(batch) {
			return nil, fmt.Errorf("tei: got %d scores for %d texts", len(res), len(batch))
		}
		seen := make([]bool, len(batch))
		for _, r := range res {
			if r.Index < 0 || r.Index >= len(batch) || seen[r.Index] {
				return nil, fmt.Errorf("tei: rerank returned index %d for %d texts", r.Index, len(batch))
			}
			seen[r.Index] = true
			out[start+r.Index] = r.Score
		}
	}
	return out, nil
}

type embedAllRequest struct {
	Inputs   []string `json:"inputs"`
	Truncate bool     `json:"truncate"`
}

// EmbedTokens はクエリの接頭辞を付けた text のトークンごとのベクトルを /embed_all で求める。特殊トークン（Ruri v3 なら先頭の <s> と末尾の </s>）と接頭辞のトークンも含む。
func (c *Client) EmbedTokens(ctx context.Context, text string) ([][]float32, error) {
	vecs, err := c.embedAll(ctx, []string{c.queryPrefix + text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedDocumentTokens は文書の接頭辞を付けた texts のトークンごとのベクトルを、texts と同じ順で返す。
func (c *Client) EmbedDocumentTokens(ctx context.Context, texts []string) ([][][]float32, error) {
	return c.embedAll(ctx, withPrefix(c.documentPrefix, texts))
}

// embedAll はプーリング前のベクトルを返す /embed_all を呼ぶ。TEI はこの経路では正規化しないので、ここで長さ1にする。
func (c *Client) embedAll(ctx context.Context, texts []string) ([][][]float32, error) {
	out := make([][][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += c.batchSize {
		batch := texts[start:min(start+c.batchSize, len(texts))]
		var vecs [][][]float32
		if err := c.do(ctx, http.MethodPost, "/embed_all", embedAllRequest{Inputs: batch, Truncate: true}, &vecs); err != nil {
			return nil, err
		}
		if len(vecs) != len(batch) {
			return nil, fmt.Errorf("tei: got %d token embeddings for %d inputs", len(vecs), len(batch))
		}
		for _, tokens := range vecs {
			for _, v := range tokens {
				normalize(v)
			}
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// Info は /info の一部。読み込んだモデルと、要求の大きさの上限を返す。
type Info struct {
	ModelID            string `json:"model_id"`
	ModelSHA           string `json:"model_sha"`
	ModelDType         string `json:"model_dtype"`
	MaxInputLength     int    `json:"max_input_length"`
	MaxBatchTokens     int    `json:"max_batch_tokens"`
	MaxClientBatchSize int    `json:"max_client_batch_size"`
	Version            string `json:"version"`
}

func (c *Client) Info(ctx context.Context) (Info, error) {
	var info Info
	err := c.do(ctx, http.MethodGet, "/info", nil, &info)
	return info, err
}

// Health はモデルを読み終えて要求を受けられるなら nil を返す。
func (c *Client) Health(ctx context.Context) error {
	return c.do(ctx, http.MethodGet, "/health", nil, nil)
}

// Error は TEI が 2xx 以外で返した応答。413 は件数かトークンの上限超え、429 は同時要求の上限超え。
type Error struct {
	Code    int
	Message string
	Type    string
}

func (e *Error) Error() string {
	if e.Type != "" {
		return fmt.Sprintf("tei: %d %s: %s", e.Code, e.Type, e.Message)
	}
	return fmt.Sprintf("tei: %d %s", e.Code, e.Message)
}

func (c *Client) do(ctx context.Context, method, path string, body, out any) error {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, r)
	if err != nil {
		return err
	}
	if r != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("tei: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		e := &Error{Code: resp.StatusCode}
		var payload struct {
			Error     string `json:"error"`
			ErrorType string `json:"error_type"`
		}
		if json.Unmarshal(data, &payload) == nil && payload.Error != "" {
			e.Message, e.Type = payload.Error, payload.ErrorType
		} else {
			e.Message = strings.TrimSpace(string(data))
		}
		return e
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("tei: decode %s: %w", path, err)
	}
	return nil
}
