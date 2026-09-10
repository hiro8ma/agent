package chapter02

import (
	"context"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// recorder は各エージェントへ届いた system instruction と履歴を分けて記録する。
//
// 前段の結果が鍵で届けば system instruction に、履歴で届けば contents に出る。
// 分けて持つことで、どちらの経路で届いたかを見分けられる。
type recorder struct {
	mu       sync.Mutex
	sys      map[string]string
	contents map[string]string
	roles    map[string]string // system instruction に含まれる語 → 返す目印
}

func newRecorder(roles map[string]string) *recorder {
	return &recorder{sys: map[string]string{}, contents: map[string]string{}, roles: roles}
}

func (r *recorder) Name() string { return "recorder" }

func (r *recorder) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var sys, hist strings.Builder
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, p := range req.Config.SystemInstruction.Parts {
			sys.WriteString(p.Text)
		}
	}
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			hist.WriteString(p.Text)
		}
	}
	who, reply := "?", "MARK_?"
	for role, mark := range r.roles {
		if strings.Contains(sys.String(), role) {
			who, reply = role, mark
			break
		}
	}
	r.mu.Lock()
	r.sys[who], r.contents[who] = sys.String(), hist.String()
	r.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{{Text: reply}}},
			TurnComplete: true,
		}, nil)
	}
}

func run(t *testing.T, a agent.Agent, text string) error {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "ch02", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	for _, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

var chapterRoles = map[string]string{
	"観光スポットの専門家":    "MARK_SPOTS",
	"グルメの専門家":       "MARK_REST",
	"交通手段の専門家":      "MARK_TRANS",
	"旅行スケジュールの作成担当": "MARK_SCHEDULE",
	"旅行予算の計算担当":     "MARK_BUDGET",
}

// 後段が前段の結果を State の鍵で受け取っているかを見る。
//
// 目印が system instruction に出れば鍵の経路、contents だけなら履歴の経路。
func TestPlannersReadUpstreamResultsByKey(t *testing.T) {
	rec := newRecorder(chapterRoles)
	a, err := NewWithModel(rec)
	if err != nil {
		t.Fatal(err)
	}
	if err := run(t, a, "京都に2泊3日で旅行したい"); err != nil {
		t.Fatalf("run: %v", err)
	}
	for who, want := range map[string][]string{
		"旅行スケジュールの作成担当": {"MARK_SPOTS", "MARK_REST", "MARK_TRANS"},
		"旅行予算の計算担当":     {"MARK_SCHEDULE", "MARK_SPOTS", "MARK_REST", "MARK_TRANS"},
	} {
		for _, mark := range want {
			if !strings.Contains(rec.sys[who], mark) {
				t.Errorf("%s の system instruction に %s が無い。鍵で読んでいない", who, mark)
			}
		}
		if !strings.Contains(rec.contents[who], "2泊3日") {
			t.Errorf("%s に利用者の依頼が届いていない", who)
		}
	}
}

func miniPipeline(t *testing.T, m model.LLM, include llmagent.IncludeContents, planInstr string) agent.Agent {
	t.Helper()
	mk := func(name, role, key string) agent.Agent {
		a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Instruction: role, OutputKey: key})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{
		Name:      "research",
		SubAgents: []agent.Agent{mk("s", "ROLE_SPOTS", "spots"), mk("r", "ROLE_REST", "rest"), mk("t", "ROLE_TRANS", "trans")},
	}})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := llmagent.New(llmagent.Config{Name: "plan", Model: m, Instruction: planInstr, IncludeContents: include})
	if err != nil {
		t.Fatal(err)
	}
	root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
		Name: "root", SubAgents: []agent.Agent{par, plan},
	}})
	if err != nil {
		t.Fatal(err)
	}
	return root
}

var miniRoles = map[string]string{
	"ROLE_SPOTS": "MARK_SPOTS", "ROLE_REST": "MARK_REST",
	"ROLE_TRANS": "MARK_TRANS", "ROLE_PLAN": "MARK_PLAN",
}

// 履歴と鍵の 2 つの経路が、それぞれ何を運ぶかを固定する。
//
// 履歴を切ると、鍵で読まない後段は入力をすべて失う。エラーは出ない。
// 鍵で読めば調査結果は残るが、利用者の依頼は履歴にしか無いので消える。
func TestHistoryAndStateCarryDifferentThings(t *testing.T) {
	const byKey = "ROLE_PLAN 観光={spots} 食={rest} 交通={trans}"
	for _, tc := range []struct {
		name        string
		include     llmagent.IncludeContents
		instr       string
		wantResults bool
		wantRequest bool
	}{
		{"履歴あり・鍵で読まない", "", "ROLE_PLAN", true, true},
		{"履歴なし・鍵で読まない", llmagent.IncludeContentsNone, "ROLE_PLAN", false, false},
		{"履歴あり・鍵で読む", "", byKey, true, true},
		{"履歴なし・鍵で読む", llmagent.IncludeContentsNone, byKey, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRecorder(miniRoles)
			if err := run(t, miniPipeline(t, rec, tc.include, tc.instr), "京都に2泊3日で行きたい"); err != nil {
				t.Fatalf("run: %v", err)
			}
			got := rec.sys["ROLE_PLAN"] + rec.contents["ROLE_PLAN"]
			results := strings.Contains(got, "MARK_SPOTS") && strings.Contains(got, "MARK_REST") && strings.Contains(got, "MARK_TRANS")
			if results != tc.wantResults {
				t.Errorf("調査結果の到達 = %v, 期待 %v", results, tc.wantResults)
			}
			if request := strings.Contains(got, "2泊3日"); request != tc.wantRequest {
				t.Errorf("利用者の依頼の到達 = %v, 期待 %v", request, tc.wantRequest)
			}
		})
	}
}

// 鍵で読むと、前段が欠けたときに実行が止まるかを見る。
//
// ドキュメントは「State に無い鍵を参照するとエラー」と書いている。
// 履歴の経路は、欠けても黙って少ない入力で進む。
func TestMissingKeyFailsLoudly(t *testing.T) {
	rec := newRecorder(miniRoles)
	a := miniPipeline(t, rec, "", "ROLE_PLAN 観光={spots} 宿={hotel_result}")
	if err := run(t, a, "京都に行きたい"); err == nil {
		t.Error("存在しない鍵を読んだのにエラーにならなかった")
	}
}
