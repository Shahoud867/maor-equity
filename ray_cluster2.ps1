# ==========================================================
# Ray + Tailscale + WSL Cluster Bootstrap Script
# Works for BOTH Node A (Head) and Node B (Worker with GPU)
#
# Your setup:
# - Tailscale installed on Windows host
# - Ray runs inside WSL
# - Run this same script on both machines
#
# Usage:
#   On Node A:
#       .\ray_cluster.ps1 -Role head
#
#   On Node B:
#       .\ray_cluster.ps1 -Role worker -HeadIP 100.x.x.x
#
# ==========================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("head","worker")]
    [string]$Role,

    [string]$HeadIP=""
)

# -----------------------------
# CONFIG
# -----------------------------
$RayPort = 6380   # 6379 is Redis default — avoid conflict
$DashboardPort = 8265
$WorkerGPUCount = 1     # change if multiple GPUs
$WSLCommandPrefix = "bash -lc"

# -----------------------------
# CHECK WSL
# -----------------------------
Write-Host "Checking WSL..."

wsl -e bash -lc "echo WSL OK"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL not available."
    exit
}

# -----------------------------
# CHECK TAILSCALE
# -----------------------------
Write-Host "Checking Tailscale..."

tailscale status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tailscale not connected."
    exit
}

# -----------------------------
# GET THIS NODE TAILSCALE IP
# -----------------------------
$MyIP = (tailscale ip -4 | Select-Object -First 1).Trim()

Write-Host "Detected Tailscale IP: $MyIP"

# -----------------------------
# FIREWALL RULES
# -----------------------------
Write-Host "Opening Windows Firewall ports..."

New-NetFirewallRule -DisplayName "Ray6379" `
    -Direction Inbound -Protocol TCP `
    -LocalPort $RayPort -Action Allow -ErrorAction SilentlyContinue

New-NetFirewallRule -DisplayName "Ray8265" `
    -Direction Inbound -Protocol TCP `
    -LocalPort $DashboardPort -Action Allow -ErrorAction SilentlyContinue

# -----------------------------
# START HEAD NODE
# -----------------------------
if ($Role -eq "head") {

    Write-Host ""
    Write-Host "Starting Ray HEAD node inside WSL..."

    wsl -e bash -lc "
        ray stop;
        ray start --head \
        --port=$RayPort \
        --dashboard-host=0.0.0.0 \
        --node-ip-address=0.0.0.0
    "

    Write-Host ""
    Write-Host "=========================================="
    Write-Host "HEAD NODE READY"
    Write-Host "Use this on worker:"
    Write-Host ".\ray_cluster.ps1 -Role worker -HeadIP $MyIP"
    Write-Host "Dashboard:"
    Write-Host "http://$MyIP:$DashboardPort"
    Write-Host "=========================================="
}

# -----------------------------
# START WORKER NODE (GPU)
# -----------------------------
elseif ($Role -eq "worker") {

    if ($HeadIP -eq "") {
        Write-Host "Please provide -HeadIP"
        exit
    }

    Write-Host ""
    Write-Host "Testing connection to head..."

    Test-NetConnection -ComputerName $HeadIP -Port $RayPort

    Write-Host ""
    Write-Host "Starting Ray WORKER with GPU resources..."

    wsl -e bash -lc "
        ray stop;
        ray start \
        --address='$HeadIP`:$RayPort' \
        --num-gpus=$WorkerGPUCount
    "

    Write-Host ""
    Write-Host "=========================================="
    Write-Host "WORKER CONNECTED TO HEAD"
    Write-Host "GPU resources advertised: $WorkerGPUCount"
    Write-Host "=========================================="
}