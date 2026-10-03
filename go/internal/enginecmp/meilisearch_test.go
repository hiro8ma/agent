package enginecmp_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/hiro8ma/agent/go/internal/enginecmp"
)

func TestMeilisearchAddDocuments(t *testing.T) {
	t.Parallel()
	srv, got := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
		"POST /indexes/kenpou/documents": respond(http.StatusAccepted, `{"taskUid":7}`),
	})
	uid, err := enginecmp.NewMeilisearch(srv.URL, "kenpou", nil).AddDocuments(t.Context(), []string{"第一条", "第二条"})
	if err != nil {
		t.Fatal(err)
	}
	if uid != 7 {
		t.Errorf("uid = %d", uid)
	}
	req := got()[0]
	if req.path != "/indexes/kenpou/documents?primaryKey=id" {
		t.Errorf("path = %s", req.path)
	}
	if b := string(req.body); b != `[{"id":1,"text":"第一条"},{"id":2,"text":"第二条"}]` {
		t.Errorf("body = %s", b)
	}
}

func TestMeilisearchWaitTask(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		statuses []string
		wantCode string
		wantErr  bool
	}{
		"succeeded になるまで問い合わせる": {
			statuses: []string{"enqueued", "processing", "succeeded"},
		},
		"failed ならエラーの種類を返す": {
			statuses: []string{"processing", "failed"},
			wantCode: "index_not_found",
			wantErr:  true,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var calls atomic.Int32
			srv, _ := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
				"GET /tasks/3": func(w http.ResponseWriter, r *http.Request) {
					status := tc.statuses[min(int(calls.Add(1))-1, len(tc.statuses)-1)]
					task := map[string]any{"uid": 3, "status": status}
					if status == "failed" {
						task["error"] = map[string]string{"code": tc.wantCode, "message": "not found"}
					}
					respond(http.StatusOK, mustJSON(t, task))(w, r)
				},
			})
			task, err := enginecmp.NewMeilisearch(srv.URL, "kenpou", nil).WaitTask(t.Context(), 3)
			if got := int(calls.Load()); got != len(tc.statuses) {
				t.Errorf("問い合わせた回数 = %d, want %d", got, len(tc.statuses))
			}
			if !tc.wantErr {
				if err != nil || task.Status != "succeeded" {
					t.Fatalf("task = %+v, err = %v", task, err)
				}
				return
			}
			var te *enginecmp.TaskError
			if !errors.As(err, &te) || te.Code != tc.wantCode {
				t.Fatalf("err = %v, want TaskError %s", err, tc.wantCode)
			}
		})
	}
}

func TestMeilisearchSearch(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		locales  []string
		wantBody string
	}{
		"Locales が空なら言語を送らない": {
			wantBody: `{"limit":3,"q":"会議","showRankingScore":true}`,
		},
		"Locales を送って言語を固定する": {
			locales:  []string{"jpn"},
			wantBody: `{"limit":3,"locales":["jpn"],"q":"会議","showRankingScore":true}`,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv, got := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
				"POST /indexes/kenpou/search": respond(http.StatusOK, `{"hits":[{"id":57,"text":"第五十七条","_rankingScore":0.99}]}`),
			})
			m := enginecmp.NewMeilisearch(srv.URL, "kenpou", nil)
			m.Locales = tc.locales
			hits, err := m.Search(t.Context(), "会議", 3)
			if err != nil {
				t.Fatal(err)
			}
			if want := []enginecmp.Hit{{ID: 57, Score: 0.99, Text: "第五十七条"}}; !reflect.DeepEqual(hits, want) {
				t.Errorf("hits = %+v, want %+v", hits, want)
			}
			if b := string(got()[0].body); b != tc.wantBody {
				t.Errorf("body = %s, want %s", b, tc.wantBody)
			}
		})
	}
}

func TestMeilisearchDeleteIndex(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		task    string
		wantErr bool
	}{
		"消せたら成功する":       {task: `{"uid":9,"status":"succeeded"}`},
		"索引が無くても成功にする":   {task: `{"uid":9,"status":"failed","error":{"code":"index_not_found"}}`},
		"それ以外の失敗はエラーにする": {task: `{"uid":9,"status":"failed","error":{"code":"internal"}}`, wantErr: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			srv, _ := fakeServer(t, map[string]func(http.ResponseWriter, *http.Request){
				"DELETE /indexes/kenpou": respond(http.StatusAccepted, `{"taskUid":9}`),
				"GET /tasks/9":           respond(http.StatusOK, tc.task),
			})
			err := enginecmp.NewMeilisearch(srv.URL, "kenpou", nil).DeleteIndex(t.Context())
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
