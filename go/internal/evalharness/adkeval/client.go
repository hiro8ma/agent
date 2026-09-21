package adkeval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

// Event は ADK の REST が返すイベントのうち、採点に使う部分。
type Event struct {
	Author       string   `json:"author"`
	Partial      bool     `json:"partial,omitempty"`
	Content      *Content `json:"content"`
	ErrorCode    string   `json:"errorCode,omitempty"`
	ErrorMessage string   `json:"errorMessage,omitempty"`
}

// Agent は 1 ターン分を実行し、そのターンのイベントを返す。
type Agent interface {
	NewSession(ctx context.Context, userID string) (string, error)
	Run(ctx context.Context, userID, sessionID string, message Content) ([]Event, error)
}

// RESTAgent は ADK の REST（Python の adk api_server / Go の ADK の web api）を呼ぶ。
type RESTAgent struct {
	BaseURL string // 例 http://localhost:8000 や http://localhost:8080/api
	AppName string
	HTTP    *http.Client
}

func (a *RESTAgent) client() *http.Client {
	if a.HTTP != nil {
		return a.HTTP
	}
	return http.DefaultClient
}

func (a *RESTAgent) NewSession(ctx context.Context, userID string) (string, error) {
	path := fmt.Sprintf("%s/apps/%s/users/%s/sessions", a.BaseURL, url.PathEscape(a.AppName), url.PathEscape(userID))
	var out struct {
		ID string `json:"id"`
	}
	if err := a.post(ctx, path, map[string]any{}, &out); err != nil {
		return "", fmt.Errorf("adkeval: セッションの作成: %w", err)
	}
	return out.ID, nil
}

func (a *RESTAgent) Run(ctx context.Context, userID, sessionID string, message Content) ([]Event, error) {
	body := map[string]any{
		"appName":    a.AppName,
		"userId":     userID,
		"sessionId":  sessionID,
		"newMessage": message,
	}
	var events []Event
	if err := a.post(ctx, a.BaseURL+"/run", body, &events); err != nil {
		return nil, fmt.Errorf("adkeval: 実行: %w", err)
	}
	return events, nil
}

func (a *RESTAgent) post(ctx context.Context, endpoint string, body, out any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := a.client().Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = res.Body.Close() }()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return err
	}
	if res.StatusCode/100 != 2 {
		return fmt.Errorf("%s: %d %s", endpoint, res.StatusCode, bytes.TrimSpace(raw))
	}
	return json.Unmarshal(raw, out)
}
