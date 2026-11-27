package agentskills

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
	"testing"
	"testing/fstest"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/skilltoolset"
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
	base, err := newSkillSource(ctx)
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
	var names []string
	for _, frontmatter := range frontmatters {
		names = append(names, frontmatter.Name)
	}
	if !slices.Equal(names, []string{"order-cancellation", "order-management"}) {
		t.Fatalf("frontmatters = %v", names)
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
	source, err := newSkillSource(t.Context())
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

type runnableTool interface {
	Run(ctx agent.Context, args any) (map[string]any, error)
}

func runSkillTool(
	t *testing.T,
	toolset *skilltoolset.SkillToolset,
	name string,
	args map[string]any,
) (map[string]any, error) {
	t.Helper()
	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, current := range tools {
		if current.Name() != name {
			continue
		}
		runnable, ok := current.(runnableTool)
		if !ok {
			t.Fatalf("%s を実行できない: %T", name, current)
		}
		return runnable.Run(&agent.ContextMock{}, args)
	}
	t.Fatalf("%s が toolset に無い", name)
	return nil, nil
}

func newTestSource(t *testing.T, metadata map[string]map[string]string) skill.Source {
	t.Helper()
	files := fstest.MapFS{}
	for name, values := range metadata {
		content, err := skill.Build(
			&skill.Frontmatter{Name: name, Description: name + " の説明", Metadata: values},
			name+" の手順\n",
		)
		if err != nil {
			t.Fatal(err)
		}
		files[name+"/SKILL.md"] = &fstest.MapFile{Data: content}
		files[name+"/references/note.md"] = &fstest.MapFile{Data: []byte(name + " の資料")}
	}
	return skill.NewFileSystemSource(files)
}

func checkReadable(t *testing.T, path string, err error, want bool) {
	t.Helper()
	if want && err != nil {
		t.Errorf("%s で読めない: %v", path, err)
	}
	if !want && !errors.Is(err, skill.ErrSkillNotFound) {
		t.Errorf("%s の error = %v, want ErrSkillNotFound", path, err)
	}
}

func TestSkillToolsetHidesHumanOnlySkillFromModel(t *testing.T) {
	t.Parallel()
	toolset, err := NewSkillToolset(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{}
	if err := toolset.ProcessRequest(nil, request); err != nil {
		t.Fatal(err)
	}
	var instruction strings.Builder
	for _, part := range request.Config.SystemInstruction.Parts {
		instruction.WriteString(part.Text)
	}
	if got := instruction.String(); !strings.Contains(got, "order-management") ||
		strings.Contains(got, "order-cancellation") || strings.Contains(got, "利用者へ通知を送る") {
		t.Fatalf("モデルの文脈に載るスキルが違う: %s", got)
	}

	testCases := map[string]struct {
		tool         string
		args         map[string]any
		want         string
		wantNotFound bool
	}{
		"list_skills は人に残したスキルを返さない": {
			tool: "list_skills",
			args: map[string]any{},
			want: "order-management",
		},
		"load_skill は人に残したスキルを名前を当てても読めない": {
			tool:         "load_skill",
			args:         map[string]any{"name": "order-cancellation"},
			wantNotFound: true,
		},
		"load_skill は設定なしのスキルを読める": {
			tool: "load_skill",
			args: map[string]any{"name": "order-management"},
			want: "get_order_status",
		},
		"load_skill_resource は人に残したスキルの資料を読めない": {
			tool:         "load_skill_resource",
			args:         map[string]any{"skill_name": "order-cancellation", "resource_path": "references/cancel-policy.md"},
			wantNotFound: true,
		},
		"load_skill_resource は設定なしのスキルの資料を読める": {
			tool: "load_skill_resource",
			args: map[string]any{"skill_name": "order-management", "resource_path": "references/cancel-policy.md"},
			want: "発送前の注文だけ",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			result, err := runSkillTool(t, toolset, tc.tool, tc.args)
			if tc.wantNotFound {
				if !errors.Is(err, skill.ErrSkillNotFound) {
					t.Fatalf("error = %v, result = %v, want ErrSkillNotFound", err, result)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got := fmt.Sprint(result)
			if !strings.Contains(got, tc.want) || strings.Contains(got, "order-cancellation") {
				t.Fatalf("result = %s, want %q を含み order-cancellation を含まない", got, tc.want)
			}
		})
	}
}

func TestLoadSkillForUserReadsHumanOnlySkill(t *testing.T) {
	t.Parallel()
	instructions, err := LoadSkillForUser(t.Context(), "order-cancellation")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(instructions, "cancel_order") {
		t.Fatalf("instructions = %q", instructions)
	}
}

func TestInvocationSettingsGateModelAndUser(t *testing.T) {
	t.Parallel()
	source := newTestSource(t, map[string]map[string]string{
		"plain":             nil,
		"explicit-defaults": {disableModelInvocationKey: "false", userInvocableKey: "true"},
		"human-only":        {disableModelInvocationKey: "true"},
		"background":        {userInvocableKey: "false"},
	})
	modelOnly := modelSource{source: source}
	listed, err := modelOnly.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var listedNames []string
	for _, frontmatter := range listed {
		listedNames = append(listedNames, frontmatter.Name)
	}

	testCases := map[string]struct {
		skill   string
		byModel bool
		byUser  bool
	}{
		"設定なしはモデルと人の両方で読める":       {skill: "plain", byModel: true, byUser: true},
		"既定と同じ値を書いてもモデルと人の両方で読める": {skill: "explicit-defaults", byModel: true, byUser: true},
		"人に残したスキルはモデルから外れ人は読める":   {skill: "human-only", byModel: false, byUser: true},
		"背景知識はモデルが読めて人は読めない":      {skill: "background", byModel: true, byUser: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			if got := slices.Contains(listedNames, tc.skill); got != tc.byModel {
				t.Errorf("モデル向けの一覧に %s がある = %v, want %v", tc.skill, got, tc.byModel)
			}
			reads := map[string]func() error{
				"LoadFrontmatter": func() error {
					_, err := modelOnly.LoadFrontmatter(ctx, tc.skill)
					return err
				},
				"LoadInstructions": func() error {
					_, err := modelOnly.LoadInstructions(ctx, tc.skill)
					return err
				},
				"ListResources": func() error {
					_, err := modelOnly.ListResources(ctx, tc.skill, "references")
					return err
				},
				"LoadResource": func() error {
					resource, err := modelOnly.LoadResource(ctx, tc.skill, "references/note.md")
					if err != nil {
						return err
					}
					return resource.Close()
				},
			}
			for method, read := range reads {
				checkReadable(t, "モデルの "+method, read(), tc.byModel)
			}
			instructions, err := loadForUser(ctx, source, tc.skill)
			checkReadable(t, "人の経路", err, tc.byUser)
			if tc.byUser && !strings.Contains(instructions, tc.skill+" の手順") {
				t.Errorf("人の経路の本文 = %q", instructions)
			}
		})
	}
}

func TestInvalidInvocationSettingsFailClosed(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		metadata map[string]string
	}{
		"大文字の TRUE は受け付けない":     {metadata: map[string]string{disableModelInvocationKey: "TRUE"}},
		"yes は真偽値として受け付けない":     {metadata: map[string]string{userInvocableKey: "yes"}},
		"空文字は受け付けない":            {metadata: map[string]string{disableModelInvocationKey: ""}},
		"両方で閉じると誰も呼べないので受け付けない": {metadata: map[string]string{disableModelInvocationKey: "true", userInvocableKey: "false"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			source := newTestSource(t, map[string]map[string]string{"broken": tc.metadata})
			modelOnly := modelSource{source: source}
			_, listErr := modelOnly.ListFrontmatters(ctx)
			_, loadErr := modelOnly.LoadInstructions(ctx, "broken")
			_, userErr := loadForUser(ctx, source, "broken")
			for path, err := range map[string]error{"モデル向けの一覧": listErr, "モデルの読み込み": loadErr, "人の経路": userErr} {
				if !errors.Is(err, skill.ErrInvalidFrontmatter) {
					t.Errorf("%s の error = %v, want ErrInvalidFrontmatter", path, err)
				}
			}
		})
	}
}

func TestModelCannotTellHiddenSkillFromMissingSkill(t *testing.T) {
	t.Parallel()
	toolset, err := NewSkillToolset(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	testCases := map[string]struct {
		tool string
		args func(name string) map[string]any
	}{
		"load_skill は隠したスキルと無いスキルに同じエラーを返す": {
			tool: "load_skill",
			args: func(name string) map[string]any { return map[string]any{"name": name} },
		},
		"load_skill_resource は隠したスキルと無いスキルに同じエラーを返す": {
			tool: "load_skill_resource",
			args: func(name string) map[string]any {
				return map[string]any{"skill_name": name, "resource_path": "references/cancel-policy.md"}
			},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			_, hiddenErr := runSkillTool(t, toolset, tc.tool, tc.args("order-cancellation"))
			_, missingErr := runSkillTool(t, toolset, tc.tool, tc.args("no-such-skill"))
			if hiddenErr == nil || missingErr == nil {
				t.Fatalf("hidden = %v, missing = %v", hiddenErr, missingErr)
			}
			hidden := strings.ReplaceAll(hiddenErr.Error(), "order-cancellation", "<name>")
			missing := strings.ReplaceAll(missingErr.Error(), "no-such-skill", "<name>")
			if hidden != missing {
				t.Fatalf("隠したスキルと無いスキルのエラーが違う:\n%s\n%s", hidden, missing)
			}
		})
	}
}
