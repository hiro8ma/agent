// Package agentskills は Agent Skills の段階的な読み込みを扱う。
package agentskills

import (
	"context"
	"embed"
	"fmt"
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

func NewSkillSource(ctx context.Context) (skill.Source, error) {
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
	source, err := NewSkillSource(ctx)
	if err != nil {
		return nil, err
	}
	toolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		return nil, fmt.Errorf("SkillToolset の作成: %w", err)
	}
	return toolset, nil
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
