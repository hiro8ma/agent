// Package agentskills は Agent Skills の段階的な読み込みを扱う。
package agentskills

import (
	"context"
	"embed"
	"fmt"
	"io"
	"io/fs"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

//go:embed skills
var skillFiles embed.FS

// SDK のフロントマターは最上位の未知のキーをエラーにするため、起動の設定は metadata の下に置く。
const (
	disableModelInvocationKey = "disable-model-invocation"
	userInvocableKey          = "user-invocable"
)

func newSkillSource(ctx context.Context) (skill.Source, error) {
	skillFS, err := fs.Sub(skillFiles, "skills")
	if err != nil {
		return nil, fmt.Errorf("スキル用ファイルシステムの作成: %w", err)
	}
	source := skill.NewFileSystemSource(skillFS)
	preloaded, _, err := skill.WithFrontmatterPreloadSource(ctx, source)
	if err != nil {
		return nil, fmt.Errorf("スキルメタデータの読み込み: %w", err)
	}
	return preloaded, nil
}

func NewSkillToolset(ctx context.Context) (*skilltoolset.SkillToolset, error) {
	source, err := newSkillSource(ctx)
	if err != nil {
		return nil, err
	}
	toolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: modelSource{source: source}})
	if err != nil {
		return nil, fmt.Errorf("SkillToolset の作成: %w", err)
	}
	return toolset, nil
}

func LoadSkillForUser(ctx context.Context, name string) (string, error) {
	source, err := newSkillSource(ctx)
	if err != nil {
		return "", err
	}
	return loadForUser(ctx, source, name)
}

func loadForUser(ctx context.Context, source skill.Source, name string) (string, error) {
	frontmatter, err := source.LoadFrontmatter(ctx, name)
	if err != nil {
		return "", err
	}
	_, byUser, err := invocation(frontmatter)
	if err != nil {
		return "", err
	}
	if !byUser {
		return "", skill.ErrSkillNotFound
	}
	return source.LoadInstructions(ctx, name)
}

func invocation(frontmatter *skill.Frontmatter) (byModel, byUser bool, err error) {
	disabled, err := metadataFlag(frontmatter, disableModelInvocationKey, false)
	if err != nil {
		return false, false, err
	}
	byUser, err = metadataFlag(frontmatter, userInvocableKey, true)
	if err != nil {
		return false, false, err
	}
	if disabled && !byUser {
		return false, false, fmt.Errorf(
			"%w: %s は %s と %s の両方で閉じられ、誰も呼べない",
			skill.ErrInvalidFrontmatter, frontmatter.Name, disableModelInvocationKey, userInvocableKey,
		)
	}
	return !disabled, byUser, nil
}

func metadataFlag(frontmatter *skill.Frontmatter, key string, defaultValue bool) (bool, error) {
	value, ok := frontmatter.Metadata[key]
	if !ok {
		return defaultValue, nil
	}
	switch value {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}
	return false, fmt.Errorf(
		"%w: %s の metadata の %s は true か false: %q",
		skill.ErrInvalidFrontmatter, frontmatter.Name, key, value,
	)
}

// 一覧から外すだけでは名前を当てて読めるため、読み込みの経路もすべて閉じ、無いスキルと同じエラーを返す。
type modelSource struct {
	source skill.Source
}

func (s modelSource) ListFrontmatters(ctx context.Context) ([]*skill.Frontmatter, error) {
	frontmatters, err := s.source.ListFrontmatters(ctx)
	if err != nil {
		return nil, err
	}
	var visible []*skill.Frontmatter
	for _, frontmatter := range frontmatters {
		byModel, _, err := invocation(frontmatter)
		if err != nil {
			return nil, err
		}
		if byModel {
			visible = append(visible, frontmatter)
		}
	}
	return visible, nil
}

func (s modelSource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	frontmatter, err := s.source.LoadFrontmatter(ctx, name)
	if err != nil {
		return nil, err
	}
	byModel, _, err := invocation(frontmatter)
	if err != nil {
		return nil, err
	}
	if !byModel {
		return nil, skill.ErrSkillNotFound
	}
	return frontmatter, nil
}

func (s modelSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	if _, err := s.LoadFrontmatter(ctx, name); err != nil {
		return "", err
	}
	return s.source.LoadInstructions(ctx, name)
}

func (s modelSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	if _, err := s.LoadFrontmatter(ctx, name); err != nil {
		return nil, err
	}
	return s.source.ListResources(ctx, name, subpath)
}

func (s modelSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if _, err := s.LoadFrontmatter(ctx, name); err != nil {
		return nil, err
	}
	return s.source.LoadResource(ctx, name, resourcePath)
}

func New(ctx context.Context, m model.LLM) (agent.Agent, error) {
	skills, err := NewSkillToolset(ctx)
	if err != nil {
		return nil, err
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Model:       m,
		Description: "注文に関する問い合わせへ回答する",
		Instruction: "必要な Agent Skill を読み、注文ツールの結果だけを使って回答してください。",
		Toolsets:    []tool.Toolset{skills},
	})
	if err != nil {
		return nil, fmt.Errorf("サポートエージェントの作成: %w", err)
	}
	return a, nil
}
