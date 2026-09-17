package supportcontext

import (
	"context"
	"iter"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type fixedModel struct{}

func (fixedModel) Name() string { return "fixed" }

func (fixedModel) GenerateContent(
	context.Context,
	*model.LLMRequest,
	bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

type mapState map[string]any

func (m mapState) Get(key string) (any, error) {
	value, ok := m[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return value, nil
}

func (m mapState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for key, value := range m {
			if !yield(key, value) {
				return
			}
		}
	}
}

type readonlyContext struct {
	*agent.ContextMock
	state mapState
}

func (c *readonlyContext) ReadonlyState() session.ReadonlyState { return c.state }

func TestSupportInstructionReadsUserAndSessionState(t *testing.T) {
	ctx := &readonlyContext{
		ContextMock: &agent.ContextMock{},
		state: mapState{
			StateSchema.MustKey("display_name"): "田中",
			StateSchema.MustKey("issue_kind"):   "接続障害",
		},
	}
	instruction, err := SupportInstruction(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"田中", "接続障害"} {
		if !strings.Contains(instruction, want) {
			t.Fatalf("%q が Instruction に無い: %s", want, instruction)
		}
	}
}

func TestInstructionScopesUseDifferentConfigFields(t *testing.T) {
	m := fixedModel{}
	order := orderConfig(m)
	support := supportConfig(m)
	root := rootConfig(m, nil)

	if order.Instruction == "" || order.InstructionProvider != nil {
		t.Fatalf("注文エージェントの固定 Instruction が違う: %#v", order)
	}
	if support.Instruction != "" || support.InstructionProvider == nil {
		t.Fatalf("技術サポートの動的 Instruction が違う: %#v", support)
	}
	if root.GlobalInstruction != GlobalPolicy {
		t.Fatalf("GlobalInstruction = %q", root.GlobalInstruction)
	}
	if root.GlobalInstructionProvider != nil {
		t.Fatal("固定の GlobalInstruction に Provider が設定されている")
	}
}

func TestOnlyRootDeclaresGlobalInstruction(t *testing.T) {
	m := fixedModel{}
	if orderConfig(m).GlobalInstruction != "" {
		t.Fatal("子エージェントに GlobalInstruction がある")
	}
	if supportConfig(m).GlobalInstruction != "" {
		t.Fatal("子エージェントに GlobalInstruction がある")
	}
	if rootConfig(m, nil).GlobalInstruction == "" {
		t.Fatal("ルートに GlobalInstruction が無い")
	}
}

func TestNewBuildsRootWithTwoSpecialists(t *testing.T) {
	root, err := New(fixedModel{})
	if err != nil {
		t.Fatal(err)
	}
	if root.Name() != "support_router" {
		t.Fatalf("root = %q", root.Name())
	}
	children := root.SubAgents()
	if len(children) != 2 || children[0].Name() != "order_agent" || children[1].Name() != "support_agent" {
		t.Fatalf("children = %v", children)
	}
}
