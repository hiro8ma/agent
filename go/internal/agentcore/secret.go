package agentcore

// MaskSecret は API キーなどの秘匿値をログに出せる形に落とす。
//
// 先頭 4 文字だけ残すのは、どのキーが読み込まれたかを識別できるようにするため。
// 伏せ字の長さは固定にしてある。元の長さを出すと、桁数から鍵の種別が絞り込めるうえ、
// 「切り詰められていないか」の確認は先頭の一致で足りる。
func MaskSecret(s string) string {
	if s == "" {
		return "(未設定)"
	}
	if len(s) <= 4 {
		return "****"
	}
	return s[:4] + "****"
}
