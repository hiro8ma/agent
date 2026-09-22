// Package a2aexec は agentcore.Agent（Genkit などの実装）を A2A のサーバーの実行器にする。
//
// ADK のエージェントは adka2a の実行器を使う。こちらは ADK 以外のエージェント向け。
package a2aexec

import (
	"context"
	"iter"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libauth"
	"github.com/hiro8ma/agent/go/internal/toolscope"
)

// Executor は 1 つの agentcore.Agent を A2A で実行する。
type Executor struct {
	Agent agentcore.Agent
}

var _ a2asrv.AgentExecutor = (*Executor)(nil)

// Execute は最後の利用者の発話を Ask に渡し、差分を成果物として、最後に完了か失敗の状態を返す。
func (e *Executor) Execute(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		ctx = toolscope.Enforce(ctx)
		if p, ok := libauth.FromContext(ctx); ok {
			if id, err := identity.ParseUserID(p.Subject); err == nil {
				ctx = identity.With(ctx, id)
			}
		}
		if execCtx.StoredTask == nil && !yield(a2a.NewSubmittedTask(execCtx, execCtx.Message), nil) {
			return
		}
		if !yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateWorking, nil), nil) {
			return
		}
		input := &agentcore.AskInput{SessionID: execCtx.ContextID, UserMessage: text(execCtx.Message)}
		var artifact a2a.ArtifactID
		for chunk, out := range e.Agent.Ask(ctx, input) {
			if chunk != nil && chunk.AnswerDelta != "" {
				part := a2a.NewTextPart(chunk.AnswerDelta)
				var ev *a2a.TaskArtifactUpdateEvent
				if artifact == "" {
					ev = a2a.NewArtifactEvent(execCtx, part)
					artifact = ev.Artifact.ID
				} else {
					ev = a2a.NewArtifactUpdateEvent(execCtx, artifact, part)
				}
				if !yield(ev, nil) {
					return
				}
			}
			if out == nil {
				continue
			}
			if out.ErrorMessage != "" {
				msg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(out.ErrorMessage))
				yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateFailed, msg), nil)
				return
			}
			if artifact == "" && out.Answer != "" {
				if !yield(a2a.NewArtifactEvent(execCtx, a2a.NewTextPart(out.Answer)), nil) {
					return
				}
			}
		}
		yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateCompleted, nil), nil)
	}
}

// Cancel は取り消しの状態を返す。
func (e *Executor) Cancel(_ context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateCanceled, nil), nil)
	}
}

func text(m *a2a.Message) string {
	if m == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range m.Parts {
		if t, ok := p.Content.(a2a.Text); ok {
			b.WriteString(string(t))
		}
	}
	return b.String()
}
