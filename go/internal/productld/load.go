package productld

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Result は1ファイルの結果。File はファイル名だけを持つ。
type Result struct {
	File string
	Page
}

// Load は HTML のファイル1つ、またはディレクトリの直下の .html と .htm を名前の順に読む。
func Load(path string) ([]Result, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	fsys, names := os.DirFS(filepath.Dir(path)), []string{filepath.Base(path)}
	if info.IsDir() {
		fsys, names = os.DirFS(path), nil
		entries, err := fs.ReadDir(fsys, ".")
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if ext := strings.ToLower(filepath.Ext(e.Name())); !e.IsDir() && (ext == ".html" || ext == ".htm") {
				names = append(names, e.Name())
			}
		}
	}
	out := make([]Result, 0, len(names))
	for _, name := range names {
		b, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, err
		}
		out = append(out, Result{File: name, Page: Extract(b)})
	}
	return out, nil
}
