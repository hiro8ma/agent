package tei_test

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/embedding/tei"
)

// fakeTEI は入力の文字列の長さを 1 次元目に入れたベクトルを返し、受け取った要求を記録する。
type fakeTEI struct {
	mu       sync.Mutex
	requests [][]string
	flags    [][2]bool
}

func (f *fakeTEI) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/embed" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Inputs    []string `json:"inputs"`
			Normalize bool     `json:"normalize"`
			Truncate  bool     `json:"truncate"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		f.requests = append(f.requests, req.Inputs)
		f.flags = append(f.flags, [2]bool{req.Normalize, req.Truncate})
		f.mu.Unlock()
		out := make([][]float32, len(req.Inputs))
		for i, s := range req.Inputs {
			out[i] = []float32{float32(len([]rune(s))), 1}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	})
}

func TestEmbedTextsBatches(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		batchSize int
		texts     []string
		wantCalls [][]string
	}{
		"件数が上限ちょうどなら 1 回で送る": {
			batchSize: 2,
			texts:     []string{"あ", "いい"},
			wantCalls: [][]string{{"あ", "いい"}},
		},
		"上限を超えると順を保って分けて送る": {
			batchSize: 2,
			texts:     []string{"あ", "いい", "ううう", "ええええ", "お"},
			wantCalls: [][]string{{"あ", "いい"}, {"ううう", "ええええ"}, {"お"}},
		},
		"空なら送らない": {
			batchSize: 2,
			texts:     nil,
			wantCalls: nil,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fake := &fakeTEI{}
			srv := httptest.NewServer(fake.handler(t))
			t.Cleanup(srv.Close)
			c := tei.New(srv.URL+"/", tei.WithBatchSize(tc.batchSize))
			got, err := c.EmbedTexts(t.Context(), tc.texts)
			if err != nil {
				t.Fatalf("EmbedTexts: %v", err)
			}
			if !reflect.DeepEqual(fake.requests, tc.wantCalls) {
				t.Errorf("requests = %q, want %q", fake.requests, tc.wantCalls)
			}
			if len(got) != len(tc.texts) {
				t.Fatalf("len = %d, want %d", len(got), len(tc.texts))
			}
			for i, s := range tc.texts {
				if want := float32(len([]rune(s))); got[i][0] != want {
					t.Errorf("vector %d = %v, want first %v", i, got[i], want)
				}
			}
			for _, f := range fake.flags {
				if f != [2]bool{true, true} {
					t.Errorf("normalize, truncate = %v, want both true", f)
				}
			}
		})
	}
}

func TestPrefixes(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		call func(context.Context, *tei.Client) error
		want []string
	}{
		"Embed はクエリの接頭辞を付ける": {
			call: func(ctx context.Context, c *tei.Client) error { _, err := c.Embed(ctx, "宗教の自由"); return err },
			want: []string{tei.RuriQueryPrefix + "宗教の自由"},
		},
		"EmbedQueries はクエリの接頭辞を付ける": {
			call: func(ctx context.Context, c *tei.Client) error {
				_, err := c.EmbedQueries(ctx, []string{"働く権利"})
				return err
			},
			want: []string{tei.RuriQueryPrefix + "働く権利"},
		},
		"EmbedDocuments は文書の接頭辞を付ける": {
			call: func(ctx context.Context, c *tei.Client) error {
				_, err := c.EmbedDocuments(ctx, []string{"第九条", "第十四条"})
				return err
			},
			want: []string{tei.RuriDocumentPrefix + "第九条", tei.RuriDocumentPrefix + "第十四条"},
		},
		"EmbedTexts は何も付けない": {
			call: func(ctx context.Context, c *tei.Client) error {
				_, err := c.EmbedTexts(ctx, []string{"カレーのお店"})
				return err
			},
			want: []string{"カレーのお店"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fake := &fakeTEI{}
			srv := httptest.NewServer(fake.handler(t))
			t.Cleanup(srv.Close)
			c := tei.New(srv.URL, tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
			if err := tc.call(t.Context(), c); err != nil {
				t.Fatal(err)
			}
			if len(fake.requests) != 1 || !reflect.DeepEqual(fake.requests[0], tc.want) {
				t.Errorf("requests = %q, want [%q]", fake.requests, tc.want)
			}
		})
	}
}

func TestErrors(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		status  int
		body    string
		want    tei.Error
		wantMsg string
	}{
		"TEI の誤りの形なら本文と種類を読む": {
			status:  http.StatusRequestEntityTooLarge,
			body:    `{"error":"batch size 40 > maximum allowed batch size 32","error_type":"Validation"}`,
			want:    tei.Error{Code: 413, Message: "batch size 40 > maximum allowed batch size 32", Type: "Validation"},
			wantMsg: "tei: 413 Validation: batch size 40 > maximum allowed batch size 32",
		},
		"JSON でなければ本文をそのまま持つ": {
			status:  http.StatusBadGateway,
			body:    "upstream down\n",
			want:    tei.Error{Code: 502, Message: "upstream down"},
			wantMsg: "tei: 502 upstream down",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(srv.Close)
			_, err := tei.New(srv.URL).Embed(t.Context(), "x")
			var e *tei.Error
			if !errors.As(err, &e) {
				t.Fatalf("err = %v, want *tei.Error", err)
			}
			if *e != tc.want {
				t.Errorf("err = %+v, want %+v", *e, tc.want)
			}
			if e.Error() != tc.wantMsg {
				t.Errorf("Error() = %q, want %q", e.Error(), tc.wantMsg)
			}
		})
	}
}

func TestEmbedCountMismatch(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[[1,0]]`))
	}))
	t.Cleanup(srv.Close)
	if _, err := tei.New(srv.URL).EmbedTexts(t.Context(), []string{"a", "b"}); err == nil {
		t.Fatal("err = nil, want count mismatch")
	}
}

func TestEmbedHonorsContext(t *testing.T) {
	t.Parallel()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	_, err := tei.New(srv.URL).Embed(ctx, "x")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestInfo(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/info" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"model_id":"cl-nagoya/ruri-v3-310m","model_dtype":"float32","max_input_length":2048,"max_batch_tokens":2048,"max_client_batch_size":32,"version":"1.9.4"}`))
	}))
	t.Cleanup(srv.Close)
	got, err := tei.New(srv.URL).Info(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	want := tei.Info{ModelID: "cl-nagoya/ruri-v3-310m", ModelDType: "float32", MaxInputLength: 2048, MaxBatchTokens: 2048, MaxClientBatchSize: 32, Version: "1.9.4"}
	if got != want {
		t.Errorf("Info = %+v, want %+v", got, want)
	}
}

// fakeRerankServer は本文の文字数を点数にし、TEI と同じく点数の高い順に並べて返す。受け取った要求を記録する。
type fakeRerankServer struct {
	mu       sync.Mutex
	requests []rerankBody
}

type rerankBody struct {
	Query     string   `json:"query"`
	Texts     []string `json:"texts"`
	RawScores bool     `json:"raw_scores"`
	Truncate  bool     `json:"truncate"`
}

func (f *fakeRerankServer) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/rerank" {
			http.NotFound(w, r)
			return
		}
		var req rerankBody
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		f.requests = append(f.requests, req)
		f.mu.Unlock()
		type result struct {
			Index int     `json:"index"`
			Score float32 `json:"score"`
		}
		out := make([]result, len(req.Texts))
		for i, s := range req.Texts {
			out[i] = result{Index: i, Score: float32(len([]rune(s)))}
		}
		slices.SortFunc(out, func(a, b result) int { return cmp.Compare(b.Score, a.Score) })
		_ = json.NewEncoder(w).Encode(out)
	})
}

func TestRerank(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		batchSize int
		texts     []string
		wantCalls [][]string
	}{
		"TEI が点数の高い順に並べ替えて返しても入力の順に戻す": {
			batchSize: 4,
			texts:     []string{"あ", "ううう", "いい"},
			wantCalls: [][]string{{"あ", "ううう", "いい"}},
		},
		"上限を超えると分けて送り、添字を全体の位置に戻す": {
			batchSize: 2,
			texts:     []string{"あ", "ううう", "いい", "ええええ", "お"},
			wantCalls: [][]string{{"あ", "ううう"}, {"いい", "ええええ"}, {"お"}},
		},
		"空なら送らない": {
			batchSize: 2,
			texts:     nil,
			wantCalls: nil,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fake := &fakeRerankServer{}
			srv := httptest.NewServer(fake.handler(t))
			t.Cleanup(srv.Close)
			c := tei.New(srv.URL, tei.WithBatchSize(tc.batchSize), tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
			got, err := c.Rerank(t.Context(), "宗教の自由", tc.texts)
			if err != nil {
				t.Fatalf("Rerank: %v", err)
			}
			want := make([]float32, len(tc.texts))
			for i, s := range tc.texts {
				want[i] = float32(len([]rune(s)))
			}
			if !slices.Equal(got, want) {
				t.Errorf("scores = %v, want %v", got, want)
			}
			var calls [][]string
			for _, r := range fake.requests {
				calls = append(calls, r.Texts)
				if r.Query != "宗教の自由" || r.RawScores || !r.Truncate {
					t.Errorf("request = %+v, want query without prefix, raw_scores false, truncate true", r)
				}
			}
			if !reflect.DeepEqual(calls, tc.wantCalls) {
				t.Errorf("requests = %q, want %q", calls, tc.wantCalls)
			}
		})
	}
}

func TestRerankBadIndex(t *testing.T) {
	t.Parallel()
	testCases := map[string]string{
		"件数が足りない": `[{"index":0,"score":0.5}]`,
		"添字が範囲の外": `[{"index":0,"score":0.5},{"index":2,"score":0.1}]`,
		"同じ添字が2回": `[{"index":1,"score":0.5},{"index":1,"score":0.1}]`,
	}
	for tn, body := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(body))
			}))
			t.Cleanup(srv.Close)
			if _, err := tei.New(srv.URL).Rerank(t.Context(), "q", []string{"a", "b"}); err == nil {
				t.Fatal("err = nil, want error")
			}
		})
	}
}

// fakeEmbedAll は入力の文字数だけトークンを作り、i 番目のトークンに長さ1でないベクトル (3, 4i) を返す。
type fakeEmbedAll struct {
	mu       sync.Mutex
	requests [][]string
}

func (f *fakeEmbedAll) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/embed_all" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Inputs   []string `json:"inputs"`
			Truncate bool     `json:"truncate"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if !req.Truncate {
			t.Errorf("truncate = false, want true")
		}
		f.mu.Lock()
		f.requests = append(f.requests, req.Inputs)
		f.mu.Unlock()
		out := make([][][]float32, len(req.Inputs))
		for i, s := range req.Inputs {
			for j := range len([]rune(s)) {
				out[i] = append(out[i], []float32{3, float32(4 * j)})
			}
		}
		_ = json.NewEncoder(w).Encode(out)
	})
}

func TestEmbedTokens(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		call      func(context.Context, *tei.Client) ([][][]float32, error)
		wantCalls [][]string
	}{
		"EmbedTokens はクエリの接頭辞を付ける": {
			call: func(ctx context.Context, c *tei.Client) ([][][]float32, error) {
				v, err := c.EmbedTokens(ctx, "宗教")
				return [][][]float32{v}, err
			},
			wantCalls: [][]string{{tei.RuriQueryPrefix + "宗教"}},
		},
		"EmbedDocumentTokens は文書の接頭辞を付けて batchSize 件ずつ送る": {
			call: func(ctx context.Context, c *tei.Client) ([][][]float32, error) {
				return c.EmbedDocumentTokens(ctx, []string{"第九条", "第十四条", "第二十条"})
			},
			wantCalls: [][]string{
				{tei.RuriDocumentPrefix + "第九条", tei.RuriDocumentPrefix + "第十四条"},
				{tei.RuriDocumentPrefix + "第二十条"},
			},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fake := &fakeEmbedAll{}
			srv := httptest.NewServer(fake.handler(t))
			t.Cleanup(srv.Close)
			c := tei.New(srv.URL, tei.WithBatchSize(2), tei.WithPrefixes(tei.RuriQueryPrefix, tei.RuriDocumentPrefix))
			got, err := tc.call(t.Context(), c)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(fake.requests, tc.wantCalls) {
				t.Errorf("requests = %q, want %q", fake.requests, tc.wantCalls)
			}
			inputs := slices.Concat(tc.wantCalls...)
			if len(got) != len(inputs) {
				t.Fatalf("len = %d, want %d", len(got), len(inputs))
			}
			for i, tokens := range got {
				if len(tokens) != len([]rune(inputs[i])) {
					t.Errorf("input %d has %d tokens, want %d", i, len(tokens), len([]rune(inputs[i])))
				}
				// (3, 4) は長さ5なので、正規化すると (0.6, 0.8) になる。
				if v := tokens[1]; math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
					t.Errorf("input %d token 1 = %v, want [0.6 0.8]", i, v)
				}
			}
		})
	}
}
