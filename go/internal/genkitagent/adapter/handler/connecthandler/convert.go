package connecthandler

import (
	"encoding/json"

	"google.golang.org/protobuf/types/known/structpb"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/genkitagent/usecase"
)

func fromChatRequest(msg *agentv1.ChatRequest) *usecase.ChatRequest {
	return &usecase.ChatRequest{AgentID: msg.GetAgentId(), SessionID: msg.GetSessionId(), Message: msg.GetMessage()}
}

func toListAgentsResponse(res *usecase.ListAgentsResponse) *agentv1.ListAgentsResponse {
	out := &agentv1.ListAgentsResponse{}
	for _, info := range res.Agents {
		out.Agents = append(out.Agents, &agentv1.AgentInfo{Id: info.ID, Description: info.Description})
	}
	return out
}

func toChatResponse(res *usecase.ChatResponse) *agentv1.ChatResponse {
	if res.Output == nil {
		return &agentv1.ChatResponse{Event: &agentv1.ChatResponse_AnswerDelta{AnswerDelta: res.Delta.AnswerDelta}}
	}
	result := toResult(res.Output)
	result.HistorySaved = res.HistorySaved
	return &agentv1.ChatResponse{Event: &agentv1.ChatResponse_Result{Result: result}}
}

func toResult(out *model.ChatOutput) *agentv1.ChatResult {
	result := &agentv1.ChatResult{
		SessionId:    out.SessionID,
		Answer:       out.Answer,
		FinishReason: out.FinishReason,
		ErrorMessage: out.ErrorMessage,
		Usage: &agentv1.TokenUsage{
			InputTokens:  int32(out.Usage.InputTokens),
			OutputTokens: int32(out.Usage.OutputTokens),
			TotalTokens:  int32(out.Usage.TotalTokens),
		},
	}
	for _, tc := range out.ToolCalls {
		result.ToolCalls = append(result.ToolCalls, &agentv1.ToolCall{Name: tc.Name, Input: toStruct(tc.Input)})
	}
	for _, p := range out.PendingToolCalls {
		result.PendingToolCalls = append(result.PendingToolCalls, &agentv1.PendingToolCall{
			ToolCallId: p.ID,
			Name:       p.Name,
			Input:      toStruct(p.Input),
		})
	}
	return result
}

// toStruct は任意の値を JSON 経由で Struct へ変換する。変換できない値は nil。
func toStruct(v any) *structpb.Struct {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	s, err := structpb.NewStruct(m)
	if err != nil {
		return nil
	}
	return s
}
