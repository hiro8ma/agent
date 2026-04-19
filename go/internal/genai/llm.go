// Package genai は LLMProvider の google.golang.org/genai 実装。
// このパッケージにのみ genai SDK 依存を閉じ込める。
package genai

import (
	"context"
	"strings"

	"google.golang.org/genai"

	domainsvc "github.com/hiro8ma/agent/go/internal/agent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// LLM は LLMProvider を google.golang.org/genai で実装する。
type LLM struct {
	client    *genai.Client
	modelName string
}

// New は LLM アダプタを生成する。
func New(client *genai.Client, modelName string) *LLM {
	return &LLM{client: client, modelName: modelName}
}

// Generate は会話履歴 + Tool 宣言を genai SDK 形式に変換し、1 回 GenerateContent を叩く。
func (l *LLM) Generate(ctx context.Context, systemPrompt string, history []model.Message, tools []model.ToolSchema) (domainsvc.LLMResponse, error) {
	contents, err := toGenaiContents(history)
	if err != nil {
		return domainsvc.LLMResponse{}, liberrors.Wrap(liberrors.CodeInvalidArgument, err, "history convert")
	}

	cfg := &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromText(systemPrompt, genai.RoleUser),
		Tools:             []*genai.Tool{{FunctionDeclarations: toGenaiFunctionDecls(tools)}},
	}

	resp, err := l.client.Models.GenerateContent(ctx, l.modelName, contents, cfg)
	if err != nil {
		return domainsvc.LLMResponse{}, liberrors.Wrap(liberrors.CodeUnavailable, err, "generate content")
	}
	if len(resp.Candidates) == 0 {
		return domainsvc.LLMResponse{}, liberrors.Newf(liberrors.CodeInternal, "no candidates returned")
	}

	cand := resp.Candidates[0]
	msg := fromGenaiContent(cand.Content)

	out := domainsvc.LLMResponse{
		Message:      msg,
		FinishReason: string(cand.FinishReason),
	}
	if resp.UsageMetadata != nil {
		out.TotalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}
	return out, nil
}

// ---------- 変換 ドメイン → genai ----------

func toGenaiContents(history []model.Message) ([]*genai.Content, error) {
	var contents []*genai.Content
	for _, m := range history {
		switch m.Role {
		case model.RoleUser:
			contents = append(contents, genai.NewContentFromText(m.Text, genai.RoleUser))
		case model.RoleAssistant:
			parts := []*genai.Part{}
			if m.Text != "" {
				parts = append(parts, &genai.Part{Text: m.Text})
			}
			for _, c := range m.ToolCalls {
				parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
					Name: c.Name,
					Args: c.Args,
				}})
			}
			contents = append(contents, &genai.Content{Role: genai.RoleModel, Parts: parts})
		case model.RoleTool:
			if m.ToolResult == nil {
				return nil, liberrors.Newf(liberrors.CodeInvalidArgument, "tool message without ToolResult")
			}
			payload := m.ToolResult.Payload
			if payload == nil {
				payload = map[string]any{}
			}
			if m.ToolResult.Err != "" {
				payload["error"] = m.ToolResult.Err
			}
			part := &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name:     m.ToolResult.Name,
				Response: payload,
			}}
			contents = append(contents, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{part}})
		}
	}
	return contents, nil
}

func toGenaiFunctionDecls(tools []model.ToolSchema) []*genai.FunctionDeclaration {
	out := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		props := map[string]*genai.Schema{}
		for name, p := range t.Parameters.Properties {
			props[name] = &genai.Schema{
				Type:        toGenaiType(p.Type),
				Description: p.Description,
			}
		}
		out = append(out, &genai.FunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
			Parameters: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: props,
				Required:   t.Parameters.Required,
			},
		})
	}
	return out
}

func toGenaiType(t string) genai.Type {
	switch strings.ToLower(t) {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeString
	}
}

// ---------- 変換 genai → ドメイン ----------

func fromGenaiContent(c *genai.Content) model.Message {
	msg := model.Message{Role: model.RoleAssistant}
	var textChunks []string
	for _, p := range c.Parts {
		if p.Text != "" {
			textChunks = append(textChunks, p.Text)
		}
		if p.FunctionCall != nil {
			msg.ToolCalls = append(msg.ToolCalls, model.ToolCall{
				Name: p.FunctionCall.Name,
				Args: p.FunctionCall.Args,
			})
		}
	}
	msg.Text = strings.Join(textChunks, "")
	return msg
}
