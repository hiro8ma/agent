// Package main は ADK の評価セットを、REST で立てた複数のエージェントに流して採点する。
//
// Python の adk api_server と Go の ADK の web api は同じ /run の口を持つので、
// 同じ評価セットと同じ採点で Python 版と Go 版を並べられる。
// 閾値を割ったケースがあれば終了コード 1 を返すので、CI の品質ゲートに使える。
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
	"strings"
	"time"

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
	results := adkeval.Compare(ctx, set, targets, adkeval.Options{Match: mode, Pace: *pace, Cases: only})
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
	if !adkeval.AllPassed(results, th) {
		os.Exit(1)
	}
}
