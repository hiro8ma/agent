// a2a-call は、クライアント証明書と、それに結び付いたトークンで A2A のエージェントを呼ぶ。
//
//	go run ./cmd/a2a-call -url https://localhost:19940 -message "注文 A-1 を見せて" \
//	    -cert ../ops/.run/pki/genkit-agent-client.pem -key ../ops/.run/pki/genkit-agent-client.key \
//	    -ca ../ops/.run/pki/ca.pem -token-url https://127.0.0.1:19800/token -client-id genkit-agent
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"

	"github.com/hiro8ma/agent/go/internal/lib/libauth"
)

func main() {
	url := flag.String("url", "https://localhost:19940", "呼ぶエージェントの A2A の口")
	message := flag.String("message", "", "送る文")
	cert := flag.String("cert", "", "クライアント証明書")
	key := flag.String("key", "", "クライアント証明書の鍵")
	ca := flag.String("ca", "", "IdP とエージェントのサーバー証明書を確かめる CA")
	tokenURL := flag.String("token-url", "https://127.0.0.1:19800/token", "IdP のトークンの口")
	clientID := flag.String("client-id", "", "クライアント ID（証明書の CN）")
	scope := flag.String("scope", "", "求めるスコープ（空白区切り。空なら IdP が許すすべて）")
	flag.Parse()
	if err := run(*url, *message, *cert, *key, *ca, *tokenURL, *clientID, *scope); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(url, message, cert, key, ca, tokenURL, clientID, scope string) error {
	ctx := context.Background()
	tlsCfg, err := libauth.ClientTLSConfig(cert, key, ca)
	if err != nil {
		return err
	}
	hc := libauth.ClientCredentials(ctx, libauth.ClientConfig{
		TokenURL: tokenURL, ClientID: clientID, Scopes: strings.Fields(scope), TLS: tlsCfg,
	})
	c, err := a2aclient.NewFromEndpoints(ctx,
		[]*a2a.AgentInterface{a2a.NewAgentInterface(strings.TrimSuffix(url, "/")+"/", a2a.TransportProtocolJSONRPC)},
		a2aclient.WithJSONRPCTransport(hc))
	if err != nil {
		return err
	}
	defer func() { _ = c.Destroy() }()
	res, err := c.SendMessage(ctx, &a2a.SendMessageRequest{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart(message))})
	if err != nil {
		return err
	}
	task, ok := res.(*a2a.Task)
	if !ok {
		return fmt.Errorf("unexpected result %T", res)
	}
	fmt.Println("state:", task.Status.State)
	for _, art := range task.Artifacts {
		for _, p := range art.Parts {
			if t, ok := p.Content.(a2a.Text); ok {
				fmt.Print(string(t))
			}
		}
	}
	fmt.Println()
	return nil
}
