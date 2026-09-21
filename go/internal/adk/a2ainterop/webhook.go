package a2ainterop

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"sync"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

// maxNotificationBytes は 1 件の通知の本文の上限。
const maxNotificationBytes = 1 << 20

// Notifications は Push Notification のトークンを発行し、届いた通知を確かめる。
//
// トークンは最初に届いた通知のタスクに結び付け、以後ほかのタスクの通知には使わせない。
// 送信時に Push の設定を渡すと、発行の時点ではタスクの ID がまだ無いため。
type Notifications struct {
	mu    sync.Mutex
	issue map[string]a2a.TaskID
}

// NewNotifications は空の Notifications を作る。
func NewNotifications() *Notifications {
	return &Notifications{issue: map[string]a2a.TaskID{}}
}

// Issue は推測できないトークンを 1 つ発行する。
func (n *Notifications) Issue() string {
	token := rand.Text()
	n.mu.Lock()
	defer n.mu.Unlock()
	n.issue[token] = ""
	return token
}

// Revoke はトークンを無効にする。タスクが終わったら呼ぶ。
func (n *Notifications) Revoke(token string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	delete(n.issue, token)
}

// bind はトークンを検証し、未使用なら task に結び付ける。
func (n *Notifications) bind(token string, task a2a.TaskID) int {
	n.mu.Lock()
	defer n.mu.Unlock()
	bound, ok := n.issue[token]
	switch {
	case !ok:
		return http.StatusUnauthorized
	case bound == "":
		n.issue[token] = task
		return http.StatusNoContent
	case bound != task:
		return http.StatusForbidden
	default:
		return http.StatusNoContent
	}
}

// Handler は v1.0 の通知（StreamResponse）を受け、トークンが合うときだけ onEvent を呼ぶ。
//
// トークンは v1.0 の A2A-Notification-Token と、v0.3 の X-A2A-Notification-Token の両方から読む。
func (n *Notifications) Handler(onEvent func(context.Context, a2a.Event)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		token := r.Header.Get("A2A-Notification-Token")
		if token == "" {
			token = r.Header.Get("X-A2A-Notification-Token")
		}
		if token == "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		var sr a2a.StreamResponse
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxNotificationBytes)).Decode(&sr); err != nil || sr.Event == nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		task := sr.TaskInfo().TaskID
		if task == "" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if status := n.bind(token, task); status != http.StatusNoContent {
			w.WriteHeader(status)
			return
		}
		onEvent(r.Context(), sr.Event)
		w.WriteHeader(http.StatusNoContent)
	})
}
