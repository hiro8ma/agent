#!/usr/bin/env bash
# ホストでサービスを動かし、JSON のログを .run/logs に書く。Collector がそのファイルを読む。
# 使い方: run-services.sh start | stop | status
set -euo pipefail

cd "$(dirname "$0")/.."
OPS_DIR=$(pwd)
GO_DIR="$OPS_DIR/../go"
RUN_DIR="$OPS_DIR/.run"

# 名前:cmd のディレクトリ:ポートの環境変数。ほかのリポジトリへ移すときはここを差し替える。
SERVICES=(
  "knowledge-server:knowledge-server:KNOWLEDGE_PORT"
  "conversation-server:conversation-server:CONVERSATION_PORT"
  "action-server:action-server:ACTION_PORT"
  "ops-demo-agent:ops-demo-agent:AGENT_PORT"
)

start() {
  mkdir -p "$RUN_DIR/bin" "$RUN_DIR/logs" "$RUN_DIR/pids" "$RUN_DIR/data"
  set -a
  # shellcheck disable=SC1091
  source "$OPS_DIR/config/services.env"
  set +a
  export CONVERSATION_DSN="file:$RUN_DIR/data/conversation.db"
  for entry in "${SERVICES[@]}"; do
    IFS=: read -r name dir port_var <<<"$entry"
    (cd "$GO_DIR" && go build -o "$RUN_DIR/bin/$name" "./cmd/$dir")
    PORT="${!port_var}" OTEL_SERVICE_NAME="$name" \
      nohup "$RUN_DIR/bin/$name" >>"$RUN_DIR/logs/$name.log" 2>&1 &
    echo $! >"$RUN_DIR/pids/$name.pid"
    echo "started $name (port ${!port_var})"
  done
}

stop() {
  for pid_file in "$RUN_DIR"/pids/*.pid; do
    [ -e "$pid_file" ] || continue
    kill -TERM "$(cat "$pid_file")" 2>/dev/null || true
    rm -f "$pid_file"
  done
}

status() {
  for pid_file in "$RUN_DIR"/pids/*.pid; do
    [ -e "$pid_file" ] || continue
    name=$(basename "$pid_file" .pid)
    if kill -0 "$(cat "$pid_file")" 2>/dev/null; then echo "$name running"; else echo "$name stopped"; fi
  done
}

"${1:-status}"
