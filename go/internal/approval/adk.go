package approval

import (
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/tool"
)

// BeforeTool はリスクに応じて、実行、承認依頼、拒否を振り分ける。
//
// 承認済みかどうかは Session の State ではなく Service で確かめる。
// 返す結果はモデルが読むので、承認依頼の ID と待っている理由を入れる。
func BeforeTool(s *Service) llmagent.BeforeToolCallback {
	return func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
		risk := s.Assess(t.Name(), args)
		switch risk {
		case Low:
			return nil, nil
		case Forbidden:
			return map[string]any{"status": "forbidden", "message": "この操作は承認があっても実行できない"}, nil
		case Medium, High:
		}
		ok, err := s.Consume(ctx.UserID(), t.Name(), args)
		if err != nil {
			return nil, err
		}
		if ok {
			return nil, nil
		}
		r, err := s.Open(ctx.UserID(), t.Name(), args, risk)
		if err != nil {
			return nil, err
		}
		who := "依頼者本人の確認"
		if risk == High {
			who = "承認者の承認"
		}
		return map[string]any{
			"status":     "pending_approval",
			"request_id": r.ID,
			"risk":       risk.String(),
			"message":    who + "を待っている。承認されたら同じ内容でもう一度依頼する",
		}, nil
	}
}
