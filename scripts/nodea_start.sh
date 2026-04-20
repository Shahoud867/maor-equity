#!/usr/bin/env bash
# =============================================================
# nodea_start.sh
#
# The Ngrok-based cluster setup has been replaced by Tailscale.
# Use ray_cluster.ps1 instead (in the project root).
#
# From PowerShell on Node A:
#   .\ray_cluster.ps1 -Role A
#
# From PowerShell on Node B:
#   .\ray_cluster.ps1 -Role B -TailscaleIP 100.x.x.x
#
# Why Tailscale?  Ngrok caused Ray heartbeat drops every 30-90s
# because it routes through a public relay with 100-400ms RTT.
# Tailscale creates a direct WireGuard tunnel (10-50ms RTT)
# that is stable indefinitely across different internet connections.
# =============================================================

echo ""
echo "  NOTE: Ngrok-based startup has been removed."
echo ""
echo "  Use ray_cluster.ps1 (PowerShell) instead:"
echo ""
echo "    Node A:  .\\ray_cluster.ps1 -Role A"
echo "    Node B:  .\\ray_cluster.ps1 -Role B -TailscaleIP 100.x.x.x"
echo ""
echo "  Tailscale download: https://tailscale.com/download/windows"
echo "  (install on both PCs, sign into the same account)"
echo ""
