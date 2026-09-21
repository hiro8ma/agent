// Package mcpguard は MCP のツールを、注釈に従って絞り込み、確認をかける。
//
// Go の ADK の mcptoolset は MCP のツールを名前と説明だけで包み、注釈（readOnlyHint など）を外に出さない。
// そのため、先に MCP のクライアントでツールの一覧を取り、注釈から名前の一覧を作ってから絞る。
//
// 絞り込みはモデルに見せるツールを減らすだけで、MCP サーバーの権限は変えない。
// 同じサーバーには別のクライアントから除外したツールを呼べるので、最小権限はサーバー側
// （読み取り専用のサーバーを分ける、データベースの権限を絞る）で持つ。
package mcpguard

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ReadOnly は注釈で読み取り専用と宣言されたツールかを返す。注釈の無いツールは読み取り専用とみなさない。
func ReadOnly(t *mcp.Tool) bool {
	return t.Annotations != nil && t.Annotations.ReadOnlyHint
}

// Destructive は破壊的な変更をしうるツールかを返す。
// MCP の仕様の既定に合わせ、注釈が無いか destructiveHint が省かれていれば破壊的とみなす。
func Destructive(t *mcp.Tool) bool {
	if ReadOnly(t) {
		return false
	}
	if t.Annotations == nil || t.Annotations.DestructiveHint == nil {
		return true
	}
	return *t.Annotations.DestructiveHint
}

// Catalog はサーバーのツールを注釈で分けた一覧。
type Catalog struct {
	ReadOnly    []string
	Destructive map[string]bool
}

// Inspect はサーバーのツールの一覧を取り、注釈で分ける。
func Inspect(ctx context.Context, cs *mcp.ClientSession) (Catalog, error) {
	c := Catalog{Destructive: map[string]bool{}}
	for t, err := range cs.Tools(ctx, nil) {
		if err != nil {
			return Catalog{}, fmt.Errorf("mcpguard: ツールの一覧: %w", err)
		}
		if ReadOnly(t) {
			c.ReadOnly = append(c.ReadOnly, t.Name)
		}
		if Destructive(t) {
			c.Destructive[t.Name] = true
		}
	}
	return c, nil
}

// ConfirmDestructive は破壊的なツールの呼び出しに確認を求める。mcptoolset.Config.RequireConfirmationProvider に渡す。
func (c Catalog) ConfirmDestructive(name string, _ any) bool {
	return c.Destructive[name]
}
