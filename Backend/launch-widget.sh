#!/bin/sh
# LaunchServices gives no useful stderr destination; retain bootstrap and API
# failures beside the backend's persistent logs.
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
OLLMLX="${OLLMLX_HOME:-$HOME/.ollmlx}"
LOG="$OLLMLX/logs/mlx-stack.log"
mkdir -p "$OLLMLX/logs"
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 1048576 ]; then
  mv "$LOG" "$LOG.1"
fi
exec >>"$LOG" 2>&1
printf '\n=== MLX Stack launch %s pid %d ===\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$"
exec /bin/sh "$APP_DIR/local-llm" --widget-api
