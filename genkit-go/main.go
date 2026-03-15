package main

import (
	"context"

	"github.com/hiro8ma/agent/genkit-go/genkit"
)

func main() {
	ctx := context.Background()
	app := genkit.New(ctx)
	app.Run(ctx)
}
