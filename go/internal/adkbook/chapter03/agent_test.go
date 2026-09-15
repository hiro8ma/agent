package chapter03

import (
	"context"
	"errors"
	"io"
	"iter"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

type countingSource struct {
	skill.Source
	instructionLoads int
	resourceLoads    int
}

func (s *countingSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	s.instructionLoads++
	return s.Source.LoadInstructions(ctx, name)
}

func (s *countingSource) LoadResource(
	ctx context.Context,
	name string,
	resourcePath string,
) (io.ReadCloser, error) {
	s.resourceLoads++
	return s.Source.LoadResource(ctx, name, resourcePath)
}

type fixedModel struct{}

func (fixedModel) Name() string { return "fixed" }

func (fixedModel) GenerateContent(
	context.Context,
	*model.LLMRequest,
	bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

func TestFrontmatterPreloadDefersInstructionsAndResources(t *testing.T) {
	ctx := t.Context()
	base, err := NewSkillSource(ctx)
	if err != nil {
		t.Fatal(err)
	}
	counted := &countingSource{Source: base}
	preloaded, _, err := skill.WithFrontmatterPreloadSource(ctx, counted)
	if err != nil {
		t.Fatal(err)
	}

	frontmatters, err := preloaded.ListFrontmatters(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontmatters) != 1 || frontmatters[0].Name != "order-management" {
		t.Fatalf("frontmatters = %#v", frontmatters)
	}
	if counted.instructionLoads != 0 || counted.resourceLoads != 0 {
		t.Fatalf("起動時に本文かリソースを読んだ: %#v", counted)
	}

	instructions, err := preloaded.LoadInstructions(ctx, "order-management")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(instructions, "get_order_status") {
		t.Fatalf("instructions = %q", instructions)
	}
	resource, err := preloaded.LoadResource(
		ctx, "order-management", "references/cancel-policy.md",
	)
	if err != nil {
		t.Fatal(err)
	}
	defer resource.Close()
	if counted.instructionLoads != 1 || counted.resourceLoads != 1 {
		t.Fatalf("必要時の読み込み回数が違う: %#v", counted)
	}
}

func TestToolsetExposesThreeReadTools(t *testing.T) {
	toolset, err := NewSkillToolset(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, current := range tools {
		names = append(names, current.Name())
	}
	slices.Sort(names)
	want := []string{"list_skills", "load_skill", "load_skill_resource"}
	if !slices.Equal(names, want) {
		t.Fatalf("tools = %v, want %v", names, want)
	}
}

func TestProcessRequestInjectsMetadataOnly(t *testing.T) {
	toolset, err := NewSkillToolset(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{}
	if err := toolset.ProcessRequest(nil, request); err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, part := range request.Config.SystemInstruction.Parts {
		text.WriteString(part.Text)
	}
	got := text.String()
	for _, want := range []string{"order-management", "注文状況の確認"} {
		if !strings.Contains(got, want) {
			t.Fatalf("%q がメタデータに無い: %s", want, got)
		}
	}
	if strings.Contains(got, "get_order_status") {
		t.Fatalf("起動時に SKILL.md 本文が入った: %s", got)
	}
}

func TestResourcePathCannotEscapeSkill(t *testing.T) {
	source, err := NewSkillSource(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	_, err = source.LoadResource(t.Context(), "order-management", "../SKILL.md")
	if !errors.Is(err, skill.ErrInvalidResourcePath) {
		t.Fatalf("error = %v", err)
	}
}

func TestAllowedToolsStringIsNotPortableToGoV22(t *testing.T) {
	_, _, err := skill.ParseBytes([]byte(`---
name: order-management
description: 注文を管理する
allowed-tools: get_order_status cancel_order
---
手順
`))
	if err == nil {
		t.Fatal("空白区切りの allowed-tools が Go v2.2.0 で通った")
	}
}

func TestNewConnectsSkillToolsetToAgent(t *testing.T) {
	a, err := New(t.Context(), fixedModel{})
	if err != nil {
		t.Fatal(err)
	}
	if a.Name() != "support_agent" {
		t.Fatalf("agent = %q", a.Name())
	}
}
