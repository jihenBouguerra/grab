#!/bin/zsh
# Double-click this file in Finder to start Grab.
cd "$(dirname "$0")"
.venv/bin/pip install --quiet --upgrade yt-dlp   # platforms change often; stay current

# Free public link (Cloudflare quick tunnel) — lets you use Grab from
# anywhere, not just home Wi-Fi. Note: the URL changes on every restart.
if command -v cloudflared >/dev/null; then
  pkill -f "cloudflared tunnel" 2>/dev/null
  (cloudflared tunnel --url http://localhost:5001 > tunnel.log 2>&1 &)
  for i in {1..20}; do
    PUB=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1)
    [ -n "$PUB" ] && break
    sleep 1
  done
  [ -n "$PUB" ] && echo "\n  🌍 Public link (anywhere): $PUB\n"
fi

exec .venv/bin/python app.py
