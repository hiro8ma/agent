package search_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// ruleGroup は話題名を含む主項目と、主語を主項目に頼る後続の項目の雛形。主項目は subject に rest を続けた文で、rest の %d には lo から hi の数を入れる。
type ruleGroup struct {
	subject, rest string
	lo, hi        int
	deps          []string
}

func rg(subject, rest string, lo, hi int, deps ...string) ruleGroup {
	return ruleGroup{subject: subject, rest: rest, lo: lo, hi: hi, deps: deps}
}

type ruleTopic struct {
	name   string
	groups []ruleGroup
}

// dependentPrefixes は後続の項目の書き出し。どれも主語を書かずに前の項目の条件を足すか外す。
var dependentPrefixes = []string{"ただし", "この場合は", "この場合も", "上記に加えて"}

// ruleTopics は社内規程を模した話題ごとの雛形。後続の項目は自分の話題名も、他の話題名も含まない。
var ruleTopics = []ruleTopic{
	{name: "出張", groups: []ruleGroup{
		rg("出張の申請", "は%d営業日前までに上長へ出す", 2, 5, "ただし急な顧客対応なら事後でよい", "この場合は戻った翌日までに理由を書いて出す"),
		rg("出張の宿泊費", "は1泊%d000円までとする", 9, 14, "ただし東京23区と大阪市内は2000円を上乗せできる", "この場合も朝食代は含めない"),
		rg("出張の日当", "は1日%d00円とする", 15, 30, "ただし片道100キロ未満の日帰りには支給しない", "上記に加えて、海外では現地の物価に合わせて倍額にする"),
		rg("出張の交通手段", "は原則として公共交通機関とする", 0, 0, "ただし始発で間に合わない場合は前泊を認める", "この場合は前泊の理由を申請書に書く", "上記に加えて、タクシーは深夜の移動に限る"),
		rg("出張の報告書", "は帰社後%d日以内に出す", 3, 7, "この場合は訪問先と面談した人の名前を必ず書く", "ただし定例の訪問は月ごとにまとめてよい"),
		rg("海外出張", "は%d週間前までに人事部へ届け出る", 2, 4, "この場合は渡航先の安全情報を確かめる", "上記に加えて、海外旅行保険に入る"),
		rg("出張中の飲食", "は自己負担とする", 0, 0, "ただし顧客との会食は接待の手続きで申請する"),
		rg("出張の航空券", "はエコノミークラスで手配する", 0, 0, "ただし飛行時間が8時間を超える便はプレミアムエコノミーを使える", "この場合は事前に部長の承認を得る"),
		rg("出張の仮払金", "は出発の%d日前までに申請する", 3, 5, "この場合は帰社後1週間以内に残額を返す"),
		rg("出張先での事故", "はすぐに総務部へ連絡する", 0, 0, "上記に加えて、24時間以内に事故報告書を出す", "ただし夜間は翌朝の連絡でよい"),
		rg("出張の行き先", "は社内の予定表に書く", 0, 0),
		rg("出張の精算", "は帰社後%d日以内に済ませる", 5, 10),
		rg("出張中の勤怠", "は通常の出勤として扱う", 0, 0),
		rg("出張の同行者", "は%d人までとする", 2, 3),
		rg("出張のお土産", "は自分で払う", 0, 0),
		rg("出張の移動時間", "は労働時間に含めない", 0, 0),
	}},
	{name: "経費", groups: []ruleGroup{
		rg("経費の申請", "は発生した月の翌月%d日までに行う", 5, 10, "ただし決算月は当月中に締める", "この場合は経理部の案内に従う"),
		rg("経費の領収書", "は原本を%d年間保管する", 5, 7, "ただし電子で受け取ったものはデータのまま保存する", "この場合は印刷した紙を原本として扱わない"),
		rg("経費で買う書籍", "は1冊%d000円までとする", 3, 5, "上記に加えて、読み終えたら部署で回し読みする"),
		rg("経費の立替", "は1回%d万円までとする", 3, 5, "ただし事前に承認を得た場合は上限を超えてよい", "この場合は承認のメールを申請に添える"),
		rg("接待の経費", "は1人%d000円までとする", 5, 10, "ただし役員が同席する場合は上限を設けない", "上記に加えて、参加者の名前と会社名を記録する"),
		rg("経費の勘定科目", "は申請者が選ぶ", 0, 0, "ただし迷う場合は経理部に確かめる"),
		rg("経費の差し戻し", "を受けたら%d日以内に出し直す", 3, 5, "この場合は指摘された点への回答を書く", "ただし期限を過ぎたものは翌月の申請に回す"),
		rg("経費の振込", "は毎月%d日に行う", 20, 25, "ただしその日が休日なら前の営業日に振り込む"),
		rg("経費のカード払い", "は会社のカードを使う", 0, 0, "ただし個人のカードで払った場合も立替として申請できる", "この場合は利用明細を添える"),
		rg("少額の経費", "は%d000円未満なら領収書を省ける", 1, 3, "ただし交通系ICカードの履歴は必ず添える"),
		rg("経費の承認", "は直属の上長が行う", 0, 0),
		rg("経費の申請システム", "は毎月末の%d時に締める", 17, 18),
		rg("経費の不正", "は懲戒の対象とする", 0, 0),
		rg("経費の科目の一覧", "は経理部のページに載せる", 0, 0),
		rg("経費の上限", "は役職ごとに決める", 0, 0),
		rg("経費の監査", "は年%d回行う", 1, 2),
	}},
	{name: "休暇", groups: []ruleGroup{
		rg("休暇の申請", "は取得する%d日前までに出す", 3, 7, "ただし急病の場合は当日の朝に電話で伝えればよい", "この場合は復帰した後に申請を出し直す"),
		rg("夏季休暇", "は7月から9月の間に%d日取る", 3, 5, "ただし繁忙期の8月第2週は避ける", "上記に加えて、チーム内で日程が重ならないよう調整する"),
		rg("休暇の繰り越し", "は翌年度まで%d日を上限とする", 10, 20, "この場合も2年を過ぎた日数は消える"),
		rg("半日単位の休暇", "は年%d回まで取れる", 5, 10, "ただし午前と午後の境目は12時とする"),
		rg("慶弔休暇", "は本人の結婚で%d日とする", 3, 5, "上記に加えて、配偶者の出産では2日を認める", "ただし式の日から6か月以内に使う"),
		rg("休暇中の連絡", "は緊急時に限る", 0, 0, "この場合は電話ではなくチャットで送る"),
		rg("長期の休暇", "は%dか月前までに上長と相談する", 1, 3, "ただし病気による療養は診断書の提出をもって代える", "この場合は産業医の面談を受ける"),
		rg("休暇の取得率", "は毎年%d割以上を目標にする", 6, 8, "上記に加えて、年5日は会社が時季を指定する"),
		rg("休暇の取り消し", "は前日の17時までに行う", 0, 0, "ただし会社の都合で取り消す場合は別の日に振り替える"),
		rg("休暇明けの引き継ぎ", "は初日の午前中に済ませる", 0, 0, "この場合は不在中のメールを先に読む", "上記に加えて、代理を頼んだ人にお礼を伝える"),
		rg("休暇の残日数", "は勤怠システムで確かめる", 0, 0),
		rg("休暇の付与", "は毎年4月1日に行う", 0, 0),
		rg("休暇の理由", "は申請に書かなくてよい", 0, 0),
		rg("休暇の最低取得日数", "は年5日とする", 0, 0),
		rg("休暇中の代理", "は上長が指名する", 0, 0),
		rg("連続した休暇", "は%d日まで取れる", 5, 10),
	}},
	{name: "在宅勤務", groups: []ruleGroup{
		rg("在宅勤務の申請", "は前日の%d時までに行う", 15, 18, "ただし体調不良の場合は当日の朝でよい", "この場合は始業の前にチャットで伝える"),
		rg("在宅勤務の日数", "は週%d日までとする", 2, 3, "ただし育児や介護の事情がある場合は上限を設けない", "この場合は人事部へ事情を届け出る"),
		rg("在宅勤務中の始業と終業", "はチャットで報告する", 0, 0, "上記に加えて、昼休みに入るときも一言送る"),
		rg("在宅勤務の場所", "は自宅に限る", 0, 0, "ただし実家に帰省している間は事前に申請すれば認める", "この場合も公共の場所での作業は認めない"),
		rg("在宅勤務の手当", "は月%d000円とする", 3, 5, "ただし月の半分以上を出社した場合は支給しない"),
		rg("在宅勤務の通信環境", "は社員が自分で整える", 0, 0, "この場合は回線の速度が下り30Mbps以上あることを確かめる", "上記に加えて、公衆無線LANは使わない"),
		rg("在宅勤務中の書類", "は持ち帰らない", 0, 0, "ただし上長が認めた場合は鍵のかかるかばんで運ぶ"),
		rg("在宅勤務の評価", "は出社と同じ基準で行う", 0, 0, "上記に加えて、成果物の提出期限を毎週確かめる"),
		rg("在宅勤務中の離席", "は30分を超えるならチャットで知らせる", 0, 0, "ただし通院などで1時間を超える場合は中抜けとして申請する"),
		rg("在宅勤務の機器", "は会社が貸すノートPCを使う", 0, 0, "この場合は私物のPCで社内システムに入らない", "ただし故障したときは代わりの機器が届くまで休んでよい"),
		rg("在宅勤務の対象者", "は入社%dか月を過ぎた社員とする", 3, 6),
		rg("在宅勤務中の服装", "は自由とする", 0, 0),
		rg("在宅勤務中のビデオ通話", "ではカメラを付ける", 0, 0),
		rg("在宅勤務の光熱費", "は手当に含める", 0, 0),
		rg("在宅勤務の勤怠", "は自己申告とする", 0, 0),
		rg("在宅勤務の取り消し", "は上長が判断する", 0, 0),
	}},
	{name: "会議室", groups: []ruleGroup{
		rg("会議室の予約", "は%d週間先まで入れられる", 2, 4, "ただし役員の予定は1か月先まで押さえてよい", "この場合も使わなくなったらすぐに取り消す"),
		rg("会議室の利用時間", "は1回%d時間までとする", 2, 3, "ただし研修や面接で使う場合は延長できる", "この場合は総務部の承認を得る"),
		rg("会議室の片付け", "は使った人が行う", 0, 0, "上記に加えて、ホワイトボードの文字を消す", "この場合は椅子を元の数に戻す"),
		rg("大会議室", "は%d人以上の集まりに限る", 10, 20, "ただし空いていれば前日から少人数でも使える"),
		rg("会議室での飲食", "は飲み物だけ認める", 0, 0, "ただし昼をまたぐ打ち合わせでは弁当を食べてよい", "この場合はごみを持ち帰る"),
		rg("会議室のプロジェクター", "は予約のときに一緒に申し込む", 0, 0, "この場合は終わったら電源を切って返す"),
		rg("会議室の無断利用", "は禁止とする", 0, 0, "ただし予約の開始から15分を過ぎても誰も来ない場合は使ってよい"),
		rg("会議室の鍵", "は総務部の受付で借りる", 0, 0, "上記に加えて、返すときに貸出簿へ時刻を書く"),
		rg("オンライン会議室", "は社外の人を招くときに使う", 0, 0, "この場合は待合室を有効にする", "ただし社内だけの打ち合わせなら設定を変えなくてよい"),
		rg("会議室の空調", "は26度を目安にする", 0, 0, "ただし冬は20度とする"),
		rg("会議室の定員", "は扉の表示に従う", 0, 0),
		rg("会議室の予約名", "には部署名を入れる", 0, 0),
		rg("会議室の延長", "は次の予約がなければ認める", 0, 0),
		rg("会議室の忘れ物", "は総務部が%d日間預かる", 7, 14),
		rg("役員会議室", "は役員の許可を得て使う", 0, 0),
		rg("会議室の照明", "は退出時に消す", 0, 0),
	}},
	{name: "来客", groups: []ruleGroup{
		rg("来客の受付", "は1階の窓口で行う", 0, 0, "この場合は入館証を渡して名前を記録する", "上記に加えて、担当者が窓口まで迎えに行く"),
		rg("来客の予定", "は前日までに受付へ伝える", 0, 0, "ただし急な訪問は担当者が直接窓口に連絡する"),
		rg("来客へのお茶", "は担当部署が用意する", 0, 0, "ただし10人以上の場合は総務部に頼む"),
		rg("来客の駐車場", "は%d台まで使える", 2, 4, "この場合は車のナンバーを前日までに知らせる", "ただし雨の日は満車になりやすいので電車を勧める"),
		rg("来客の入館証", "は帰るときに回収する", 0, 0, "ただし回収できなかった場合はすぐに警備室へ知らせる"),
		rg("来客の案内", "は担当者が付き添う", 0, 0, "上記に加えて、執務室には入れない", "この場合は打ち合わせ用の部屋だけを使う"),
		rg("来客との写真撮影", "は事前に許可を取る", 0, 0, "ただし公開しない記録用の写真は口頭の確認でよい"),
		rg("来客のWi-Fi", "はゲスト用の回線を使う", 0, 0, "この場合は当日だけ使える接続情報を渡す"),
		rg("来客の手土産", "は受け取ってよい", 0, 0, "ただし5000円を超える品は上長に報告する"),
		rg("来客の喫煙", "は屋外の喫煙所に案内する", 0, 0, "上記に加えて、灰皿の場所を伝える"),
		rg("来客の名簿", "は%d年間保存する", 1, 3),
		rg("来客の対応時間", "は9時から18時とする", 0, 0),
		rg("来客の待ち時間", "は10分以内を目安にする", 0, 0),
		rg("来客の傘", "は入口の傘立てに置いてもらう", 0, 0),
		rg("来客の見送り", "はエレベーターの前までとする", 0, 0),
		rg("来客の情報", "は社外に漏らさない", 0, 0),
	}},
	{name: "社用車", groups: []ruleGroup{
		rg("社用車の予約", "は前日の%d時までに行う", 15, 18, "ただし急な外出で空いている場合は当日でもよい", "この場合は鍵を受け取るときに予約簿に書く"),
		rg("社用車の運転", "は免許を取って%d年以上の社員に限る", 1, 3, "ただし初めて運転する人は事前に講習を受ける"),
		rg("社用車の給油", "は残りが半分を切ったら行う", 0, 0, "この場合は会社のカードで払う", "上記に加えて、給油の領収書をグローブボックスに入れる"),
		rg("社用車での事故", "はすぐに警察と総務部へ連絡する", 0, 0, "上記に加えて、相手の連絡先と保険会社を聞く", "ただしけが人がいる場合は救急への通報を先にする"),
		rg("社用車の私的な利用", "は禁止とする", 0, 0, "ただし災害時の帰宅には使ってよい"),
		rg("社用車の駐車違反", "は運転者が反則金を払う", 0, 0, "この場合は総務部に違反の内容を報告する"),
		rg("社用車の点検", "は月に1回行う", 0, 0, "ただし走行距離が3000キロを超えた月は2回行う"),
		rg("社用車の鍵", "は使い終わったらその日のうちに返す", 0, 0, "ただし直行や直帰の場合は翌朝に返してよい", "この場合も鍵を自宅の外に置かない"),
		rg("社用車内の喫煙", "は禁止とする", 0, 0, "上記に加えて、車内での飲食も控える"),
		rg("社用車の高速料金", "はETCカードで払う", 0, 0, "ただしカードを忘れた場合は現金で払い、立替として申請する"),
		rg("社用車の台数", "は本社に%d台とする", 3, 6),
		rg("社用車の洗車", "は月に1回行う", 0, 0),
		rg("社用車の保険", "は会社がまとめて入る", 0, 0),
		rg("社用車の運転日報", "は毎日つける", 0, 0),
		rg("社用車の駐車場所", "は地下の指定区画とする", 0, 0),
		rg("社用車の乗車人数", "は定員を超えない", 0, 0),
	}},
	{name: "備品", groups: []ruleGroup{
		rg("備品の購入", "は%d万円未満なら部長の承認で足りる", 3, 10, "ただしそれ以上は役員の承認を得る", "この場合は見積書を2社分そろえる"),
		rg("備品の貸し出し", "は総務部の窓口で行う", 0, 0, "この場合は貸出簿に名前と返す予定の日を書く", "上記に加えて、1週間を超える場合は延長を申し出る"),
		rg("備品の故障", "はすぐに総務部へ届け出る", 0, 0, "ただし自分で直そうとしない"),
		rg("備品の持ち出し", "は事前に許可を取る", 0, 0, "ただし同じ建物の中で使うだけなら許可は要らない", "この場合も使い終わったら元の場所に戻す"),
		rg("備品の在庫確認", "は年%d回行う", 1, 2, "上記に加えて、ラベルの剥がれたものを報告する"),
		rg("備品の廃棄", "は総務部がまとめて行う", 0, 0, "ただし個人情報が残る機器はデータを消してから出す"),
		rg("文房具などの備品", "は共用の引き出しから取る", 0, 0, "この場合は使った分を補充の用紙に書く"),
		rg("備品の私的な利用", "は禁止とする", 0, 0, "ただし休憩中に充電器を使う程度は認める"),
		rg("備品の紛失", "は%d日以内に届け出る", 1, 3, "上記に加えて、紛失届に最後に使った場所を書く", "ただし故意や重い過失がなければ弁償は求めない"),
		rg("高額な備品", "は台帳に番号を登録する", 0, 0, "この場合は本体に管理番号のシールを貼る"),
		rg("備品の一覧", "は社内のページで見られる", 0, 0),
		rg("備品の管理者", "は各部署に1人置く", 0, 0),
		rg("備品の修理", "は総務部が業者に頼む", 0, 0),
		rg("備品の予約", "は%d日前までに行う", 1, 3),
		rg("備品の返却", "は使った日のうちに行う", 0, 0),
		rg("中古の備品", "は社内で譲り合う", 0, 0),
	}},
	{name: "パスワード", groups: []ruleGroup{
		rg("パスワードの長さ", "は%d文字以上にする", 10, 14, "ただし管理者のアカウントは16文字以上とする", "この場合は英字と数字と記号をすべて含める"),
		rg("パスワードの使い回し", "は禁止とする", 0, 0, "上記に加えて、社外のサービスと同じものも使わない"),
		rg("パスワードの変更", "は漏えいの疑いがあるときに行う", 0, 0, "ただし定期的な変更は求めない", "この場合は情報システム部に経緯を報告する"),
		rg("パスワードの保管", "は会社が指定した管理ツールで行う", 0, 0, "ただし紙に書いて机に貼ってはならない"),
		rg("初期パスワード", "は最初のログインで変える", 0, 0, "この場合は仮の文字列を他人に見せない"),
		rg("パスワードの共有", "は同僚の間でも禁止とする", 0, 0, "ただし共用のアカウントは管理ツールの共有機能を使う", "上記に加えて、共有した相手を一覧で管理する"),
		rg("パスワードの入力ミス", "が%d回続くとアカウントを止める", 5, 10, "この場合は情報システム部に解除を頼む"),
		rg("パスワードの再発行", "は本人の確認が取れてから行う", 0, 0, "ただし電話だけの依頼には応じない"),
		rg("パスワードと二段階認証", "は全員が設定する", 0, 0, "上記に加えて、認証アプリを入れた端末をなくしたらすぐに届け出る"),
		rg("パスワードの通知", "はメールで送らない", 0, 0, "ただしやむを得ない場合は別の経路で2回に分けて送る"),
		rg("パスワードの入力画面", "は他人から見えない場所で開く", 0, 0),
		rg("パスワードの有効期限", "は設けない", 0, 0),
		rg("パスワードの文字の種類", "は%d種類以上を混ぜる", 3, 4),
		rg("パスワードの自動保存", "はブラウザでは使わない", 0, 0),
		rg("パスワードの紛失", "は情報システム部に届け出る", 0, 0),
		rg("推測されやすいパスワード", "は登録できない", 0, 0),
	}},
	{name: "残業", groups: []ruleGroup{
		rg("残業の申請", "は当日の%d時までに行う", 15, 17, "ただし障害の対応など急ぐ場合は事後でよい", "この場合は翌朝までに理由を報告する"),
		rg("残業の上限", "は月%d時間とする", 30, 45, "ただし繁忙期は労使協定の範囲で延長できる", "上記に加えて、2か月続けて超えた人は産業医と面談する"),
		rg("深夜の残業", "は22時以降を指す", 0, 0, "この場合は割増の手当を付ける"),
		rg("残業の記録", "は勤怠システムに入力する", 0, 0, "ただし入力を忘れた場合は上長の確認を経て直す"),
		rg("休日の残業", "には振替休日を当てる", 0, 0, "ただし振り替えられない場合は割増の手当を付ける"),
		rg("残業中の食事", "は%d時を過ぎたら補助を出す", 19, 20, "この場合は1回1000円までとする"),
		rg("残業の削減", "は部署ごとに目標を立てる", 0, 0, "上記に加えて、毎月の実績を全員に共有する"),
		rg("残業の命令", "は上長が行う", 0, 0, "ただし本人の体調が悪い場合は断ってよい"),
		rg("残業後の帰宅", "は23時を過ぎたらタクシーを使ってよい", 0, 0, "この場合は乗車の記録を翌日に出す", "ただし相乗りできるときは相乗りする"),
		rg("残業代", "は翌月の給与と一緒に払う", 0, 0, "ただし固定の手当に含まれる分は別に払わない"),
		rg("残業の割増率", "は法律で決まった率とする", 0, 0),
		rg("残業の多い部署", "は人事部が面談する", 0, 0),
		rg("残業の事前承認", "はシステムで行う", 0, 0),
		rg("残業時間の集計", "は毎月%d日に締める", 15, 20),
		rg("残業の禁止日", "は毎週水曜日とする", 0, 0),
		rg("管理職の残業", "も記録の対象とする", 0, 0),
	}},
	{name: "名刺", groups: []ruleGroup{
		rg("名刺の発注", "は残りが%d枚を切ったら行う", 20, 50, "この場合は総務部の申込書を使う", "ただし肩書きが変わった場合は残りの枚数によらず作り直す"),
		rg("名刺の受け渡し", "は立って行う", 0, 0, "上記に加えて、相手より低い位置で差し出す"),
		rg("もらった名刺", "は%d日以内に顧客管理システムへ登録する", 3, 7, "ただし個人の連絡先は登録しない", "この場合は登録した後で紙を細断する"),
		rg("名刺の肩書き", "は人事部が決めた表記にする", 0, 0, "ただし社外向けに英語の肩書きを併記してよい"),
		rg("名刺の持ち歩き", "は常に%d枚以上とする", 10, 20, "上記に加えて、入れ物は汚れていないものを使う"),
		rg("退職者の名刺", "は総務部がまとめて処分する", 0, 0, "この場合は個人情報として溶解処理に出す"),
		rg("名刺のデザイン", "は会社の指定から変えない", 0, 0, "ただし採用イベント用は人事部が別に作る"),
		rg("名刺の裏面", "には英語の表記を入れる", 0, 0, "ただし国内だけで使う場合は白紙でよい"),
		rg("名刺の紛失", "は上長に報告する", 0, 0, "この場合は渡した相手の一覧を見直す", "上記に加えて、悪用された疑いがあれば情報システム部に伝える"),
		rg("名刺のQRコード", "は会社のサイトに飛ぶものにする", 0, 0, "ただし個人のSNSへのリンクは載せない"),
		rg("名刺の枚数", "は1回の発注で%d00枚とする", 1, 2),
		rg("名刺の印刷", "は外部の業者に頼む", 0, 0),
		rg("名刺の英語版", "は希望者だけに作る", 0, 0),
		rg("名刺の費用", "は会社が払う", 0, 0),
		rg("名刺の管理", "は顧客管理システムで行う", 0, 0),
		rg("名刺の交換", "は目下の人から差し出す", 0, 0),
	}},
	{name: "郵便物", groups: []ruleGroup{
		rg("郵便物の受け取り", "は総務部が行う", 0, 0, "この場合は午前と午後の2回に分けて各部署に配る", "ただし書留は本人が受付で受け取る"),
		rg("郵便物の発送", "は%d時までに総務部へ出す", 14, 16, "ただしそれを過ぎたものは翌日の発送になる"),
		rg("郵便物の切手", "は総務部の料金計器で付ける", 0, 0, "ただし私用の手紙には使わない"),
		rg("速達の郵便物", "は封筒に赤い線を引く", 0, 0, "この場合は発送の記録に番号を控える"),
		rg("郵便物の開封", "は宛名の本人だけが行う", 0, 0, "ただし部署宛てのものは部署の担当者が開ける", "上記に加えて、親展と書かれたものは代理で開けない"),
		rg("個人宛ての郵便物", "は会社に届けさせない", 0, 0, "ただし仕事で使う書籍の受け取りは認める"),
		rg("郵便物の転送", "は在籍していない人の分を差出人へ戻す", 0, 0, "この場合は受取拒否ではなく宛先不明として返す"),
		rg("郵便物の保管", "は%d日を過ぎたら差出人に返す", 14, 30, "上記に加えて、返す前に本人の上長へ一度知らせる"),
		rg("不審な郵便物", "は開けずに総務部へ渡す", 0, 0, "この場合は触った人が手を洗う", "ただし液体が漏れている場合は警察に通報する"),
		rg("宅配便と郵便物", "は同じ受付で扱う", 0, 0, "ただし冷蔵の荷物はすぐに本人へ連絡する"),
		rg("郵便物の仕分け", "は部署ごとの箱で行う", 0, 0),
		rg("郵便物の受付時間", "は9時から%d時とする", 16, 17),
		rg("郵便物の差出人の名前", "は会社名で書く", 0, 0),
		rg("郵便物の料金", "は部署ごとに集計する", 0, 0),
		rg("大型の郵便物", "は受付の横に置く", 0, 0),
		rg("郵便物の宛名", "は部署名まで書く", 0, 0),
	}},
}

type ruleItem struct {
	section   int
	group     int
	text      string
	dependent bool
}

// ruleSection の項目は items[start:end]。
type ruleSection struct {
	heading    string
	start, end int
}

type ruleCorpus struct {
	sections []ruleSection
	items    []ruleItem
	subjects []string
}

type ruleQuery struct {
	text     string
	relevant []int
}

const (
	ruleSeed    = 1
	querySuffix = "に関するルールは"
	// keepDependent は後続の項目を前から1つずつ残す確率。残さなかった項目より後ろも捨て、頼る先の無い項目を作らない。
	keepDependent = 0.7
)

// newRuleCorpus は話題ごとに1つの節を作り、節の中で群の順を混ぜる。主項目の数と、残す後続の項目は seed で決まる。
func newRuleCorpus(topics []ruleTopic, seed uint64) ruleCorpus {
	r := rand.New(rand.NewPCG(seed, 0))
	var c ruleCorpus
	for s, tp := range topics {
		sec := ruleSection{heading: tp.name, start: len(c.items)}
		for _, gi := range r.Perm(len(tp.groups)) {
			g := tp.groups[gi]
			id := len(c.subjects)
			c.subjects = append(c.subjects, g.subject)
			lead := g.subject + g.rest
			if g.hi > 0 {
				lead = fmt.Sprintf(lead, g.lo+r.IntN(g.hi-g.lo+1))
			}
			c.items = append(c.items, ruleItem{section: s, group: id, text: lead})
			for _, d := range g.deps {
				if r.Float64() >= keepDependent {
					break
				}
				c.items = append(c.items, ruleItem{section: s, group: id, text: d, dependent: true})
			}
		}
		sec.end = len(c.items)
		c.sections = append(c.sections, sec)
	}
	return c
}

func span(start, end int) []int {
	out := make([]int, 0, end-start)
	for i := start; i < end; i++ {
		out = append(out, i)
	}
	return out
}

// topicQueries は節ごとに「話題名に関するルールは」を作り、節の全項目を正解にする。
func (c ruleCorpus) topicQueries() []ruleQuery {
	qs := make([]ruleQuery, len(c.sections))
	for i, s := range c.sections {
		qs[i] = ruleQuery{text: s.heading + querySuffix, relevant: span(s.start, s.end)}
	}
	return qs
}

// groupQueries は後続の項目を1つ以上残した群ごとに「主題に関するルールは」を作り、主項目と後続の項目を正解にする。
func (c ruleCorpus) groupQueries() []ruleQuery {
	var qs []ruleQuery
	for i := 0; i < len(c.items); {
		j := i + 1
		for j < len(c.items) && c.items[j].group == c.items[i].group {
			j++
		}
		if j-i > 1 {
			qs = append(qs, ruleQuery{text: c.subjects[c.items[i].group] + querySuffix, relevant: span(i, j)})
		}
		i = j
	}
	return qs
}

type chunkMode int

const (
	chunkItem chunkMode = iota
	chunkHeading
	chunkPrevious
)

func (m chunkMode) String() string {
	switch m {
	case chunkHeading:
		return "b 見出しを付ける"
	case chunkPrevious:
		return "c 直前の項目を連結する"
	default:
		return "a 項目だけ"
	}
}

// chunks は項目と同じ順に索引する文字列を返す。c は節をまたいで連結しないので、節の先頭の項目は a と同じになる。
func (c ruleCorpus) chunks(mode chunkMode) []string {
	out := make([]string, len(c.items))
	for i, it := range c.items {
		switch {
		case mode == chunkHeading:
			out[i] = c.sections[it.section].heading + "\n" + it.text
		case mode == chunkPrevious && i > c.sections[it.section].start:
			out[i] = c.items[i-1].text + "\n" + it.text
		default:
			out[i] = it.text
		}
	}
	return out
}

func (c ruleCorpus) sectionTexts() []string {
	out := make([]string, len(c.sections))
	for i, s := range c.sections {
		lines := []string{s.heading}
		for _, it := range c.items[s.start:s.end] {
			lines = append(lines, it.text)
		}
		out[i] = strings.Join(lines, "\n")
	}
	return out
}

func TestRuleCorpusIsReproducible(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		seed      uint64
		wantEqual bool
	}{
		"same seed gives the same items":       {seed: ruleSeed, wantEqual: true},
		"different seed changes order or keep": {seed: ruleSeed + 1, wantEqual: false},
	}
	base := newRuleCorpus(ruleTopics, ruleSeed)
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := newRuleCorpus(ruleTopics, tc.seed)
			equal := slices.Equal(got.items, base.items) && slices.Equal(got.sections, base.sections)
			if equal != tc.wantEqual {
				t.Errorf("corpus equal to seed %d = %v, want %v", ruleSeed, equal, tc.wantEqual)
			}
		})
	}
}

func TestRuleCorpusShape(t *testing.T) {
	t.Parallel()
	c := newRuleCorpus(ruleTopics, ruleSeed)
	if len(c.sections) != len(ruleTopics) {
		t.Fatalf("sections = %d, want %d", len(c.sections), len(ruleTopics))
	}
	if len(c.items) < 200 {
		t.Errorf("items = %d, want at least 200", len(c.items))
	}
	var deps int
	for _, s := range c.sections {
		if n := s.end - s.start; n < 20 {
			t.Errorf("section %s has %d items, want at least 20", s.heading, n)
		}
		for _, it := range c.items[s.start:s.end] {
			if it.dependent {
				deps++
			}
		}
	}
	if share := float64(deps) / float64(len(c.items)); share < 0.2 || share > 0.5 {
		t.Errorf("dependent share = %.2f, want between 0.2 and 0.5", share)
	}
}

// TestDependentItemsFollowTheirLead は、主語を前に頼る項目が必ず同じ群の項目の後ろにあり、話題名の索引語を1つも持たないことを確かめる。
func TestDependentItemsFollowTheirLead(t *testing.T) {
	t.Parallel()
	c := newRuleCorpus(ruleTopics, ruleSeed)
	a := search.NewAnalyzer()
	for i, it := range c.items {
		s := c.sections[it.section]
		for j, other := range c.sections {
			if j != it.section && strings.Contains(it.text, other.heading) {
				t.Errorf("item %d %q contains another topic %s", i, it.text, other.heading)
			}
		}
		if !it.dependent {
			if !strings.Contains(it.text, s.heading) {
				t.Errorf("lead %d %q lacks its topic %s", i, it.text, s.heading)
			}
			if !strings.HasPrefix(it.text, c.subjects[it.group]) {
				t.Errorf("lead %d %q does not start with its subject %s", i, it.text, c.subjects[it.group])
			}
			continue
		}
		if i == s.start || c.items[i-1].group != it.group {
			t.Fatalf("dependent %d %q has no previous item in its group", i, it.text)
		}
		if !slices.ContainsFunc(dependentPrefixes, func(p string) bool { return strings.HasPrefix(it.text, p) }) {
			t.Errorf("dependent %d %q does not start with %v", i, it.text, dependentPrefixes)
		}
		terms := a.Analyze(it.text)
		for _, w := range a.Analyze(s.heading) {
			if slices.Contains(terms, w) {
				t.Errorf("dependent %d %q shares the term %q with its topic %s", i, it.text, w, s.heading)
			}
		}
		lead := i - 1
		for c.items[lead].dependent {
			lead--
		}
		if c.items[lead].group != it.group {
			t.Errorf("dependent %d %q is not preceded by the lead of its group", i, it.text)
		}
	}
}

func TestChunkTexts(t *testing.T) {
	t.Parallel()
	c := ruleCorpus{
		sections: []ruleSection{{heading: "カレー", start: 0, end: 3}, {heading: "猫", start: 3, end: 5}},
		items: []ruleItem{
			{section: 0, group: 0, text: "6日に1回はカレーを食べる"},
			{section: 0, group: 0, text: "ただし仏滅には食べない", dependent: true},
			{section: 0, group: 1, text: "レトルトカレーを温めたお湯は飲む"},
			{section: 1, group: 2, text: "猫は撫でる"},
			{section: 1, group: 2, text: "この場合は手を洗う", dependent: true},
		},
	}
	testCases := map[string]struct {
		mode chunkMode
		want []string
	}{
		"a keeps each item as it is": {mode: chunkItem, want: []string{
			"6日に1回はカレーを食べる", "ただし仏滅には食べない", "レトルトカレーを温めたお湯は飲む", "猫は撫でる", "この場合は手を洗う",
		}},
		"b puts the section heading before every item": {mode: chunkHeading, want: []string{
			"カレー\n6日に1回はカレーを食べる", "カレー\nただし仏滅には食べない", "カレー\nレトルトカレーを温めたお湯は飲む", "猫\n猫は撫でる", "猫\nこの場合は手を洗う",
		}},
		"c joins the previous item and stops at the section start": {mode: chunkPrevious, want: []string{
			"6日に1回はカレーを食べる",
			"6日に1回はカレーを食べる\nただし仏滅には食べない",
			"ただし仏滅には食べない\nレトルトカレーを温めたお湯は飲む",
			"猫は撫でる",
			"猫は撫でる\nこの場合は手を洗う",
		}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := c.chunks(tc.mode); !slices.Equal(got, tc.want) {
				t.Errorf("chunks(%v) = %q, want %q", tc.mode, got, tc.want)
			}
		})
	}
	wantSections := []string{
		"カレー\n6日に1回はカレーを食べる\nただし仏滅には食べない\nレトルトカレーを温めたお湯は飲む",
		"猫\n猫は撫でる\nこの場合は手を洗う",
	}
	if got := c.sectionTexts(); !slices.Equal(got, wantSections) {
		t.Errorf("sectionTexts() = %q, want %q", got, wantSections)
	}
}

// TestPreviousChunkGivesFirstDependentItsTopic は、c で連結すると群の最初の後続の項目だけが話題名を持ち、2つ目より後ろは持たないことを確かめる。
func TestPreviousChunkGivesFirstDependentItsTopic(t *testing.T) {
	t.Parallel()
	c := newRuleCorpus(ruleTopics, ruleSeed)
	chunks := c.chunks(chunkPrevious)
	var first, later int
	for i, it := range c.items {
		if !it.dependent {
			continue
		}
		heading := c.sections[it.section].heading
		hasTopic := strings.Contains(chunks[i], heading)
		if wantTopic := !c.items[i-1].dependent; hasTopic != wantTopic {
			t.Errorf("chunk %d %q contains %s = %v, want %v", i, chunks[i], heading, hasTopic, wantTopic)
		}
		if c.items[i-1].dependent {
			later++
		} else {
			first++
		}
	}
	if first == 0 || later == 0 {
		t.Errorf("first dependents = %d, later dependents = %d, want both above 0", first, later)
	}
}

func TestRuleQueries(t *testing.T) {
	t.Parallel()
	c := newRuleCorpus(ruleTopics, ruleSeed)
	testCases := map[string]struct {
		queries      []ruleQuery
		wantQueries  int
		wantRelevant func(q ruleQuery) []int
	}{
		"topic query covers every item of its section": {
			queries:     c.topicQueries(),
			wantQueries: len(c.sections),
			wantRelevant: func(q ruleQuery) []int {
				for _, s := range c.sections {
					if q.text == s.heading+querySuffix {
						return span(s.start, s.end)
					}
				}
				return nil
			},
		},
		"group query covers its lead and dependents only": {
			queries: c.groupQueries(),
			wantQueries: func() int {
				n := 0
				for i, it := range c.items {
					if it.dependent && !c.items[i-1].dependent {
						n++
					}
				}
				return n
			}(),
			wantRelevant: func(q ruleQuery) []int {
				var out []int
				for i, it := range c.items {
					if c.subjects[it.group]+querySuffix == q.text {
						out = append(out, i)
					}
				}
				return out
			},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if len(tc.queries) != tc.wantQueries {
				t.Fatalf("queries = %d, want %d", len(tc.queries), tc.wantQueries)
			}
			for _, q := range tc.queries {
				if want := tc.wantRelevant(q); !slices.Equal(q.relevant, want) {
					t.Errorf("%s relevant = %v, want %v", q.text, q.relevant, want)
				}
				if !slices.ContainsFunc(q.relevant, func(i int) bool { return c.items[i].dependent }) {
					t.Errorf("%s has no dependent item in its relevant set", q.text)
				}
			}
		})
	}
}
