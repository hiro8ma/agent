package enginecmp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Meilisearch は 1 つの索引を扱う。文書の追加と削除は非同期のタスクになるので、完了をタスクの API で待つ。
// Locales を空にすると言語を文字列から推定させる。漢字だけのクエリは中国語と推定されて日本語の文書と一致しなくなるので、["jpn"] で固定できる。
type Meilisearch struct {
	Locales []string

	http     httpDoer
	index    string
	interval time.Duration
}

// NewMeilisearch の client が nil なら 30 秒で打ち切るクライアントを使う。
func NewMeilisearch(baseURL, index string, client *http.Client) *Meilisearch {
	return &Meilisearch{http: newHTTPDoer(baseURL, client), index: index, interval: 50 * time.Millisecond}
}

// Task はタスクの状態。Duration は Meilisearch が処理にかけた時間で、ISO 8601 の文字列（PT0.1S など）のまま持つ。
type Task struct {
	UID      int    `json:"uid"`
	Status   string `json:"status"`
	Duration string `json:"duration"`
	Error    *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// TaskError はタスクが failed か canceled で終わったことを表す。Code は Meilisearch のエラーの種類（index_not_found など）。
type TaskError struct {
	UID     int
	Status  string
	Code    string
	Message string
}

func (e *TaskError) Error() string {
	return fmt.Sprintf("meilisearch: タスク %d が %s: %s（%s）", e.UID, e.Status, e.Message, e.Code)
}

type enqueued struct {
	TaskUID int `json:"taskUid"`
}

func (m *Meilisearch) path(suffix string) string {
	return "/indexes/" + url.PathEscape(m.index) + suffix
}

func (m *Meilisearch) Health(ctx context.Context) error {
	var res struct {
		Status string `json:"status"`
	}
	if err := m.http.do(ctx, http.MethodGet, "/health", "", nil, &res); err != nil {
		return err
	}
	if res.Status != "available" {
		return fmt.Errorf("meilisearch: status %q", res.Status)
	}
	return nil
}

// AddDocuments は texts[i] を id が i+1 の文書として送り、タスクの番号を返す。
func (m *Meilisearch) AddDocuments(ctx context.Context, texts []string) (int, error) {
	docs := make([]map[string]any, len(texts))
	for i, t := range texts {
		docs[i] = map[string]any{"id": i + 1, "text": t}
	}
	var res enqueued
	if err := m.http.do(ctx, http.MethodPost, m.path("/documents?primaryKey=id"), "application/json", docs, &res); err != nil {
		return 0, err
	}
	return res.TaskUID, nil
}

// WaitTask はタスクが succeeded になるまで問い合わせる。failed や canceled ならエラーにする。
func (m *Meilisearch) WaitTask(ctx context.Context, uid int) (Task, error) {
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		var t Task
		if err := m.http.do(ctx, http.MethodGet, "/tasks/"+strconv.Itoa(uid), "", nil, &t); err != nil {
			return t, err
		}
		switch t.Status {
		case "succeeded":
			return t, nil
		case "failed", "canceled":
			te := &TaskError{UID: uid, Status: t.Status}
			if t.Error != nil {
				te.Code, te.Message = t.Error.Code, t.Error.Message
			}
			return t, te
		}
		select {
		case <-ctx.Done():
			return t, ctx.Err()
		case <-ticker.C:
		}
	}
}

// Search は上位 limit 件を返す。Score は _rankingScore（0 から 1）で、BM25 の点数ではない。
func (m *Meilisearch) Search(ctx context.Context, query string, limit int) ([]Hit, error) {
	body := map[string]any{"q": query, "limit": limit, "showRankingScore": true}
	if len(m.Locales) > 0 {
		body["locales"] = m.Locales
	}
	var res struct {
		Hits []struct {
			ID    int     `json:"id"`
			Text  string  `json:"text"`
			Score float64 `json:"_rankingScore"`
		} `json:"hits"`
	}
	if err := m.http.do(ctx, http.MethodPost, m.path("/search"), "application/json", body, &res); err != nil {
		return nil, err
	}
	hits := make([]Hit, len(res.Hits))
	for i, h := range res.Hits {
		hits[i] = Hit{ID: h.ID, Score: h.Score, Text: h.Text}
	}
	return hits, nil
}

// DeleteIndex は削除のタスクの完了まで待つ。索引が無くてもエラーにしない。
func (m *Meilisearch) DeleteIndex(ctx context.Context) error {
	var res enqueued
	if err := m.http.do(ctx, http.MethodDelete, m.path(""), "", nil, &res); err != nil {
		return err
	}
	_, err := m.WaitTask(ctx, res.TaskUID)
	var te *TaskError
	if errors.As(err, &te) && te.Code == "index_not_found" {
		return nil
	}
	return err
}
