#!/usr/bin/env bash
set -e

latest=$(ls -dt /tmp/ray/session_* | head -n 1)
echo "LATEST_SESSION=$latest"

ls -1 "$latest"/logs | head -n 40

echo "---RAYLET_ERR---"
sed -n '1,140p' "$latest"/logs/raylet.err || true

echo "---DASHBOARD_AGENT_LOG---"
sed -n '1,140p' "$latest"/logs/dashboard_agent.log || true

echo "---GCS_OUT---"
sed -n '1,100p' "$latest"/logs/gcs_server.out || true
