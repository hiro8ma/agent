// Package main は ADK の評価セットを、REST で立てた複数のエージェントに流して採点する。
//
// Python の adk api_server と Go の ADK の web api は同じ /run の口を持つので、
// 同じ評価セットと同じ採点で Python 版と Go 版を並べられる。
// 閾値を割ったケース、禁止した内容を出したケース、前回から悪化したケースがあれば終了コード 1 を返すので、
// CI の品質ゲートに使える。conversation_scenario のケースは LLM が利用者役を演じる。
//
//	go run ./cmd/adk-eval-compare \
//	  -evalset ../python/adk_multi_agent/samples/agents/weather_agent/evals/weather_agent_v1.evalset.json \
//	  -target python=http://localhost:8000#weather_agent \
//	  -target go=http://localhost:8080/api#weather_agent
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/hiro8ma/agent/go/internal/evalharness"
	"github.com/hiro8ma/agent/go/internal/evalharness/adkeval"
)

type targetFlags []adkeval.Target

func (t *targetFlags) String() string { return fmt.Sprint(len(*t)) }

// Set は name=baseURL#appName を受ける。
func (t *targetFlags) Set(v string) error {
	name, rest, ok := strings.Cut(v, "=")
	if !ok {
		return errors.New("-target は name=baseURL#appName")
	}
	base, app, ok := strings.Cut(rest, "#")
	if !ok || name == "" || base == "" || app == "" {
		return errors.New("-target は name=baseURL#appName")
	}
	*t = append(*t, adkeval.Target{Name: name, Agent: &adkeval.RESTAgent{BaseURL: strings.TrimSuffix(base, "/"), AppName: app}})
	return nil
}

func main() {
	var targets targetFlags
	evalset := flag.String("evalset", "", "評価セット（*.evalset.json）")
	flag.Var(&targets, "target", "name=baseURL#appName。複数指定できる")
	minTrajectory := flag.Float64("min-trajectory", 1.0, "ツール呼び出しの一致の合格ライン")
	minResponse := flag.Float64("min-response", 0.3, "応答の文字 2-gram の合格ライン。言い回しの近さしか測れないので低めに置く")
	match := flag.String("match", "exact", "exact / in_order / any_order。呼ばないことを期待するケースは exact でしか見分けられない")
	cases := flag.String("cases", "", "流すケースの eval_id（カンマ区切り）。空なら全件")
	pace := flag.Duration("pace", 0, "ケースの間に空ける時間（無料枠の毎分の上限の対策）")
	out := flag.String("out", "", "結果を JSON で書き出すパス")
	timeout := flag.Duration("timeout", 30*time.Minute, "全体の時間切れ")
	simModel := flag.String("simulator-model", "", "conversation_scenario のケースで利用者役を演じるモデル（例 gemini-3.8-flash）。空ならそのケースは失敗にする")
	maxInvocations := flag.Int("max-invocations", adkeval.DefaultMaxInvocations, "利用者役との対話の上限。最初の発話も数える")
	var forbid regexpFlags
	flag.Var(&forbid, "forbid", "エージェントの応答に出てはいけない正規表現（アンチゴール）。複数指定できる")
	baseline := flag.String("baseline", "", "前回の -out の結果。悪化したケースがあれば失敗にする")
	maxDrop := flag.Float64("max-drop", 0.05, "ベースラインからのスコアの低下の許容幅")
	flag.Parse()

	if *evalset == "" || len(targets) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	mode, ok := map[string]adkeval.MatchType{"exact": adkeval.Exact, "in_order": adkeval.InOrder, "any_order": adkeval.AnyOrder}[*match]
	if !ok {
		log.Fatalf("-match は exact / in_order / any_order（%q）", *match)
	}
	set, err := adkeval.Load(*evalset)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	var only []string
	if *cases != "" {
		only = strings.Split(*cases, ",")
	}
	opts := adkeval.Options{Match: mode, Pace: *pace, Cases: only, MaxInvocations: *maxInvocations, Forbidden: forbid}
	if *simModel != "" {
		llm, err := evalharness.NewGeminiLLM(ctx, os.Getenv("GEMINI_API_KEY"), *simModel)
		if err != nil {
			log.Fatal(err)
		}
		opts.Simulator = &adkeval.LLMSimulator{LLM: llm}
	}
	results := adkeval.Compare(ctx, set, targets, opts)
	th := adkeval.Thresholds{Trajectory: *minTrajectory, Response: *minResponse}

	if err := adkeval.WriteTable(os.Stdout, results, th); err != nil {
		log.Fatal(err)
	}
	if *out != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*out, b, 0o600); err != nil {
			log.Fatal(err)
		}
	}
	passed := adkeval.AllPassed(results, th)
	if *baseline != "" {
		before, err := adkeval.LoadResults(*baseline)
		if err != nil {
			log.Fatal(err)
		}
		for _, r := range adkeval.Regressions(before, results, th, *maxDrop) {
			fmt.Printf("REGRESSION %s %s\n", r.Key, r.Reason)
			passed = false
		}
	}
	if !passed {
		os.Exit(1)
	}
}

type regexpFlags []*regexp.Regexp

func (r *regexpFlags) String() string { return fmt.Sprint(len(*r)) }

func (r *regexpFlags) Set(v string) error {
	re, err := regexp.Compile(v)
	if err != nil {
		return err
	}
	*r = append(*r, re)
	return nil
}
