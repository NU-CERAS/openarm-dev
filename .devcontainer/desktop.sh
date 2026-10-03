#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/.vnc"
x11vnc -storepasswd openarm "$HOME/.vnc/passwd" >/dev/null
Xvfb :1 -screen 0 1600x900x24 -ac +extension GLX +render -noreset &
xvfb_pid=$!
trap 'kill "$xvfb_pid" ${wm_pid:-} ${vnc_pid:-} ${web_pid:-} 2>/dev/null || true' EXIT
for attempt in {1..50}; do
  if xdpyinfo -display :1 >/dev/null 2>&1; then break; fi
  sleep 0.2
done
openbox &
wm_pid=$!
x11vnc -display :1 -forever -shared -rfbauth "$HOME/.vnc/passwd" -rfbport 5901 -localhost &
vnc_pid=$!
websockify --web=/usr/share/novnc/ 6080 localhost:5901 &
web_pid=$!
wait -n "$xvfb_pid" "$wm_pid" "$vnc_pid" "$web_pid"
