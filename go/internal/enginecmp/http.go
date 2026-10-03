// Package enginecmp は Elasticsearch（kuromoji）、Meilisearch、internal/search の 3 つの検索エンジンに同じ文書と同じクエリを与え、順位を比べる。
package enginecmp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultTimeout = 30 * time.Second

// StatusError は期待しない HTTP ステータスが返ったことを表す。
type StatusError struct {
	Method string
	URL    string
	Code   int
	Body   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s %s: %d %s", e.Method, e.URL, e.Code, e.Body)
}

func isNotFound(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Code == http.StatusNotFound
}

type httpDoer struct {
	base   string
	client *http.Client
}

func newHTTPDoer(base string, client *http.Client) httpDoer {
	if client == nil {
		client = &http.Client{Timeout: defaultTimeout}
	}
	return httpDoer{base: strings.TrimRight(base, "/"), client: client}
}

// do は body が []byte ならそのまま、それ以外は JSON にして送り、2xx の応答を out に読む。out が nil なら読み捨てる。
func (h httpDoer) do(ctx context.Context, method, path, contentType string, body, out any) error {
	var r io.Reader
	switch b := body.(type) {
	case nil:
	case []byte:
		r = bytes.NewReader(b)
	default:
		buf, err := json.Marshal(b)
		if err != nil {
			return err
		}
		r = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, method, h.base+path, r)
	if err != nil {
		return err
	}
	if r != nil {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return &StatusError{Method: method, URL: req.URL.String(), Code: resp.StatusCode, Body: strings.TrimSpace(string(data))}
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}
