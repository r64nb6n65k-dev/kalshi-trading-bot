#!/bin/sh
set -eu

if [ -z "${TS_AUTHKEY:-}" ]; then
    echo "Tailscale disabled: TS_AUTHKEY is not set"
    exec "$@"
fi

TS_SOCKET="${TS_SOCKET:-/tmp/tailscaled.sock}"
TS_STATE_DIR="${TS_STATE_DIR:-/tmp/tailscale}"
TS_HOSTNAME="${TS_HOSTNAME:-railway-kalshi-bot}"

mkdir -p "$TS_STATE_DIR"

tailscaled \
    --tun=userspace-networking \
    --socket="$TS_SOCKET" \
    --state="$TS_STATE_DIR/tailscaled.state" &

TAILSCALED_PID=$!

cleanup() {
    kill "$TAILSCALED_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

attempt=0

until tailscale --socket="$TS_SOCKET" status >/dev/null 2>&1; do
    attempt=$((attempt + 1))

    if [ "$attempt" -ge 30 ]; then
        echo "Tailscale failed: daemon did not become ready"
        exit 1
    fi

    sleep 1
done

tailscale --socket="$TS_SOCKET" up \
    --auth-key="$TS_AUTHKEY" \
    --hostname="$TS_HOSTNAME" \
    --accept-dns=false

echo "Tailscale connected as $TS_HOSTNAME"
tailscale --socket="$TS_SOCKET" status

"$@" &
APP_PID=$!
wait "$APP_PID"
