package enginecmp_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
)

type recorded struct {
	method, path, contentType string
	body                      []byte
}

// fakeServer は受けたリクエストを記録し、"METHOD /path" ごとに決めたステータスと本文を返す。
func fakeServer(t *testing.T, routes map[string]func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, func() []recorded) {
	t.Helper()
	var (
		mu  sync.Mutex
		got []recorded
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		got = append(got, recorded{method: r.Method, path: r.URL.RequestURI(), contentType: r.Header.Get("Content-Type"), body: body})
		mu.Unlock()
		h, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			http.Error(w, "no route", http.StatusTeapot)
			return
		}
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []recorded {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(got)
	}
}

func respond(code int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = io.WriteString(w, body)
	}
}

func TestElasticsearchCreateIndexSendsKuromojiAnalyzer(t *testing.T) {
	t.Parallel()
	srv, got := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
		"PUT /kenpou": respond(http.StatusOK, `{"acknowledged":true}`),
	})
	es := enginecmp.NewElasticsearch(srv.URL+"/", "kenpou", nil)
	if err := es.CreateIndex(t.Context()); err != nil {
		t.Fatal(err)
	}
	var body struct {
		Settings struct {
			Analysis struct {
				Analyzer map[string]struct {
					Tokenizer string   `json:"tokenizer"`
					Filter    []string `json:"filter"`
				} `json:"analyzer"`
			} `json:"analysis"`
		} `json:"settings"`
		Mappings struct {
			Properties map[string]struct {
				Analyzer string `json:"analyzer"`
			} `json:"properties"`
		} `json:"mappings"`
	}
	if err := json.Unmarshal(got()[0].body, &body); err != nil {
		t.Fatal(err)
	}
	ja := body.Settings.Analysis.Analyzer["ja"]
	if ja.Tokenizer != "kuromoji_tokenizer" {
		t.Errorf("tokenizer = %q", ja.Tokenizer)
	}
	if want := []string{"lowercase", "kuromoji_baseform", "kuromoji_stemmer"}; !reflect.DeepEqual(ja.Filter, want) {
		t.Errorf("filter = %v, want %v", ja.Filter, want)
	}
	if a := body.Mappings.Properties["text"].Analyzer; a != "ja" {
		t.Errorf("text の analyzer = %q", a)
	}
}

func TestElasticsearchBulk(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		response string
		wantErr  string
	}{
		"errors が false なら成功する": {
			response: `{"errors":false,"items":[]}`,
		},
		"一部の文書が失敗したらその _id とエラーを返す": {
			response: `{"errors":true,"items":[{"index":{"_id":"1"}},{"index":{"_id":"2","error":{"type":"mapper_parsing_exception"}}}]}`,
			wantErr:  "_id 2",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv, got := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
				"POST /_bulk": respond(http.StatusOK, tc.response),
			})
			es := enginecmp.NewElasticsearch(srv.URL, "kenpou", nil)
			err := es.Bulk(t.Context(), []string{"第一条　天皇は", "第二条　皇位は"})
			if tc.wantErr == "" && err != nil {
				t.Fatal(err)
			}
			if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Fatalf("err = %v, want %q を含む", err, tc.wantErr)
			}
			req := got()[0]
			if req.contentType != "application/x-ndjson" {
				t.Errorf("Content-Type = %q", req.contentType)
			}
			var lines []string
			sc := bufio.NewScanner(strings.NewReader(string(req.body)))
			for sc.Scan() {
				lines = append(lines, sc.Text())
			}
			want := []string{
				`{"index":{"_id":"1","_index":"kenpou"}}`,
				`{"text":"第一条　天皇は"}`,
				`{"index":{"_id":"2","_index":"kenpou"}}`,
				`{"text":"第二条　皇位は"}`,
			}
			if !reflect.DeepEqual(lines, want) {
				t.Errorf("NDJSON =\n%s\nwant\n%s", strings.Join(lines, "\n"), strings.Join(want, "\n"))
			}
		})
	}
}

func TestElasticsearchMatchParsesHits(t *testing.T) {
	t.Parallel()
	srv, got := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
		"POST /kenpou/_search": respond(http.StatusOK, `{"hits":{"hits":[
			{"_id":"26","_score":4.67,"_source":{"text":"第二十六条"}},
			{"_id":"11","_score":4.54,"_source":{"text":"第十一条"}}]}}`),
	})
	es := enginecmp.NewElasticsearch(srv.URL, "kenpou", nil)
	hits, err := es.Match(t.Context(), "すべて国民", 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []enginecmp.Hit{{ID: 26, Score: 4.67, Text: "第二十六条"}, {ID: 11, Score: 4.54, Text: "第十一条"}}
	if !reflect.DeepEqual(hits, want) {
		t.Errorf("hits = %+v, want %+v", hits, want)
	}
	if b := string(got()[0].body); b != `{"query":{"match":{"text":"すべて国民"}},"size":2}` {
		t.Errorf("body = %s", b)
	}
}

func TestElasticsearchDeleteIndex(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		code    int
		wantErr bool
	}{
		"消せたら成功する":       {code: http.StatusOK},
		"索引が無くても成功にする":   {code: http.StatusNotFound},
		"それ以外の失敗はエラーにする": {code: http.StatusInternalServerError, wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv, _ := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
				"DELETE /kenpou": respond(tc.code, `{}`),
			})
			err := enginecmp.NewElasticsearch(srv.URL, "kenpou", nil).DeleteIndex(t.Context())
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			var se *enginecmp.StatusError
			if tc.wantErr && (!errors.As(err, &se) || se.Code != tc.code) {
				t.Errorf("err = %v, want StatusError %d", err, tc.code)
			}
		})
	}
}

func TestElasticsearchAnalyze(t *testing.T) {
	t.Parallel()
	srv, _ := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
		"POST /kenpou/_analyze": respond(http.StatusOK, `{"tokens":[{"token":"侵す"},{"token":"ない"}]}`),
	})
	tokens, err := enginecmp.NewElasticsearch(srv.URL, "kenpou", nil).Analyze(t.Context(), "侵さない")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"侵す", "ない"}; !reflect.DeepEqual(tokens, want) {
		t.Errorf("tokens = %v, want %v", tokens, want)
	}
}
