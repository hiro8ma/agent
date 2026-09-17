package domain

import "fmt"

// SessionBackend は Session の保管先の種類。
type SessionBackend string

const (
	// SessionInMemory はプロセス内に持つ。プロセスが終われば消える。
	SessionInMemory SessionBackend = "memory"
	// SessionDatabase は RDB に持つ。ローカル検証では sqlite のファイルを使う。
	SessionDatabase SessionBackend = "database"
	// SessionVertex は Vertex AI の Agent Engine に持つ。本番向け。
	SessionVertex SessionBackend = "vertex"
)

// ParseSessionBackend は設定値を種類へ変換する。
//
// 空文字は memory として扱う。設定を忘れたときに課金経路へ入らないようにする。
// 知らない値はエラーにする。綴り違いを黙って memory へ落とすと、
// 永続化したつもりのデータが消えることに気づけない。
func ParseSessionBackend(value string) (SessionBackend, error) {
	switch SessionBackend(value) {
	case "":
		return SessionInMemory, nil
	case SessionInMemory:
		return SessionInMemory, nil
	case SessionDatabase:
		return SessionDatabase, nil
	case SessionVertex:
		return SessionVertex, nil
	default:
		return "", fmt.Errorf("未知の session backend: %q（memory / database / vertex のいずれか）", value)
	}
}

// Persistent はプロセスを越えて残るかを返す。
func (b SessionBackend) Persistent() bool {
	return b == SessionDatabase || b == SessionVertex
}

// Billable は使うと課金が発生しうるかを返す。
//
// 起動時の表示に使う。どちらで動いているかを利用者へ見せるための判定で、
// 料金の計算には使わない。
func (b SessionBackend) Billable() bool {
	return b == SessionVertex
}
