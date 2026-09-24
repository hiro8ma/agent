package travelplanner

import (
	"context"
	"fmt"
	"os"

	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// NewSkillToolset は dir 直下の SKILL.md を持つディレクトリを Agent Skills として読む。
//
// 起動時に読むのは frontmatter だけで、本文は load_skill が呼ばれたときに読む。
func NewSkillToolset(ctx context.Context, dir string) (tool.Toolset, error) {
	source, _, err := skill.WithFrontmatterPreloadSource(ctx, skill.NewFileSystemSource(os.DirFS(dir)))
	if err != nil {
		return nil, fmt.Errorf("スキルメタデータの読み込み: %w", err)
	}
	toolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		return nil, fmt.Errorf("SkillToolset の作成: %w", err)
	}
	return toolset, nil
}
