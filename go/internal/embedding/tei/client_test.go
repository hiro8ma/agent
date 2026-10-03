package tei_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
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
