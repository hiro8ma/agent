# 3 実装を同じベースイメージからビルドする。
# TARGET 環境変数で cmd/{genai,genkit,adk} を切り替える（compose.yaml で設定）。
FROM golang:1.25-bookworm AS builder

WORKDIR /agent
COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ ./cmd/
COPY internal/ ./internal/

ARG TARGET=genai
RUN CGO_ENABLED=0 go build -o /out/agent ./cmd/${TARGET}

FROM gcr.io/distroless/base-debian12:nonroot
COPY --from=builder /out/agent /agent
ENTRYPOINT ["/agent"]
