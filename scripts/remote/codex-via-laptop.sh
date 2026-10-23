#!/usr/bin/env bash
set -euo pipefail
port="${CODEX_LAPTOP_PROXY_PORT:-17890}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'Invalid CODEX_LAPTOP_PROXY_PORT' >&2; exit 2
fi
export HTTP_PROXY="http://127.0.0.1:$port"
export HTTPS_PROXY="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
export NO_PROXY='localhost,127.0.0.1,::1' no_proxy='localhost,127.0.0.1,::1'
# nvm is usually absent from non-interactive SSH PATH.
if ! command -v codex >/dev/null 2>&1 && [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  set +u
  source "$HOME/.nvm/nvm.sh"
  nvm use --silent default >/dev/null
  set -u
fi
if ! command -v codex >/dev/null 2>&1; then
  echo 'Codex not found. Install with npm install -g @openai/codex after enabling the proxy.' >&2
  exit 127
fi
# Fail explicitly when the laptop tunnel is absent. Never unset the proxy as a fallback.
if ! (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
  echo "Laptop proxy tunnel on port $port is unavailable." >&2; exit 3
fi
exec codex "$@"
