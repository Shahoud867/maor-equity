<#
.SYNOPSIS
    Ray Cluster Setup using Tailscale VPN.
    Tailscale gives both WSL2 instances stable 100.x.x.x IPs with direct
    peer-to-peer tunnels -- no heartbeat drops, no reconnect loops needed.

.PREREQUISITES
    - Tailscale installed and connected on BOTH nodes (same Google/GitHub account)
    - Ray installed in the Python venv on each node (pip install ray[default])
    - This script MUST be run as Administrator on both nodes (self-elevates automatically)
    - WSL2 installed on both machines (wsl --install in admin PowerShell)

.HOW TO USE
    STEP 1  Both PCs: install Tailscale from https://tailscale.com/download
            Sign in with GOOGLE or GITHUB -- use the SAME account on BOTH PCs.

    STEP 2  Node A PC:  .\ray_cluster.ps1 -Role A
    STEP 3  Node B PC:  .\ray_cluster.ps1 -Role B -TailscaleIP <NodeA-100.x.x.x>

.NOTES
    Log file: ray_setup_log.txt (written to the script directory)
    Generated WSL scripts: %USERPROFILE%\ray_cluster\nodeA.sh / nodeB.sh
#>

param(
    [ValidateSet("A","B","")]
    [string]$Role        = "",
    [string]$TailscaleIP = ""
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# SELF-ELEVATE: firewall rules require Administrator privileges
# ---------------------------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]"Administrator"
)
if (-not $isAdmin) {
    $relaunch = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($Role)        { $relaunch += "-Role";        $relaunch += $Role }
    if ($TailscaleIP) { $relaunch += "-TailscaleIP"; $relaunch += $TailscaleIP }
    Start-Process powershell -Verb RunAs -ArgumentList $relaunch
    exit
}

# ---------------------------------------------------------------------------
# CONFIGURATION -- Edit the values in this block before running
# ---------------------------------------------------------------------------
$NODE_A_TAILSCALE_IP   = "x.x.x.x"   # Auto-detected on Node A; set manually if auto-detect fails
$NODE_B_TAILSCALE_IP   = "x.x.x.x"   # Node B Tailscale IP; set if known, else prompted at runtime
$RAY_PORT              = 6379
$RAY_DASHBOARD_PORT    = 8265
$NODE_B_SSH_USER       = "username"   # Node B Linux username (reserved for future SSH use)
$RETRY_COUNT           = 3
$TIMEOUT_SECONDS       = 60

# Node A paths (WSL/Linux format) -- local machine
$A_VENV    = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
$A_PROJECT = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity"

# Node B paths (WSL/Linux format) -- partner machine
$B_VENV    = "/mnt/d/University Work/Semester 6/NLP + PDC Project/maor-equity/venv"
$B_PROJECT = "/mnt/d/University Work/Semester 6/NLP + PDC Project/maor-equity"

# Temp dir for generated scripts (Windows path + WSL equivalent)
$WIN_TMP  = "$env:USERPROFILE\ray_cluster"
$_drive   = ($env:USERPROFILE -split ":\\")[0].ToLower()
$_relpath = ($env:USERPROFILE -split ":\\")[1] -replace "\\","/"
$WSL_TMP  = "/mnt/$_drive/$_relpath/ray_cluster"

# Log file -- next to this script (fallback to temp dir if PSScriptRoot is empty)
if ($PSScriptRoot) {
    $LOG_FILE = Join-Path $PSScriptRoot "ray_setup_log.txt"
} else {
    $LOG_FILE = Join-Path $WIN_TMP "ray_setup_log.txt"
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Level] $Message"
    Add-Content -Path $LOG_FILE -Value $line -ErrorAction SilentlyContinue
    switch ($Level) {
        "ERROR" { Write-Host "  [ERROR] $Message" -ForegroundColor Red }
        "WARN"  { Write-Host "  [WARN]  $Message" -ForegroundColor Yellow }
        default { Write-Host "  [LOG]   $Message" -ForegroundColor Gray }
    }
}

function Invoke-WithRetry {
    param(
        [scriptblock]$Action,
        [string]$Name,
        [int]$Retries  = $RETRY_COUNT,
        [int]$DelaySec = 5
    )
    for ($i = 1; $i -le $Retries; $i++) {
        try {
            & $Action
            return
        } catch {
            $msg = "Attempt $i/$Retries failed for '$Name': $($_.Exception.Message)"
            Write-Log -Message $msg -Level "WARN"
            if ($i -lt $Retries) { Start-Sleep -Seconds $DelaySec }
            else {
                Write-Log -Message "All $Retries attempts failed for '$Name'." -Level "ERROR"
                throw
            }
        }
    }
}

function Write-UnixFile {
    param([string]$Path, [string]$Content)
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, ($Content -replace "`r`n","`n"), $utf8)
}

function Open-WslWindow {
    param([string]$Title, [string]$Script)
    if (Get-Command wt -ErrorAction SilentlyContinue) {
        Start-Process wt -ArgumentList "new-tab","--title","""$Title""","--","wsl.exe","bash",$Script
    } else {
        Start-Process cmd -ArgumentList "/k","wsl bash $Script"
    }
}

function Add-RayFirewallRules {
    param([string]$TsIP)
    Write-Host "  [FIREWALL] Configuring Windows Defender Firewall for Ray..." -ForegroundColor Cyan
    Write-Log -Message "Configuring firewall rules for $TsIP"

    $null = netsh advfirewall firewall delete rule name="Ray-Inbound"  2>&1
    $null = netsh advfirewall firewall delete rule name="Ray-Outbound" 2>&1

    $null = netsh advfirewall firewall add rule `
        name="Ray-Inbound" dir=in action=allow protocol=TCP `
        localport="$RAY_PORT,$RAY_DASHBOARD_PORT,10001,20000-29999" profile=any 2>&1

    $null = netsh advfirewall firewall add rule `
        name="Ray-Outbound" dir=out action=allow protocol=TCP `
        localport="$RAY_PORT,$RAY_DASHBOARD_PORT,10001,20000-29999" profile=any 2>&1

    Write-Host "  Firewall rules added (inbound + outbound)." -ForegroundColor Green

    # Portproxy bridges Tailscale IP to localhost for static Ray ports.
    # With WSL2 mirrored networking these are usually not required, but
    # keeping them ensures compatibility with NAT-mode WSL2 as well.
    foreach ($port in @($RAY_PORT, $RAY_DASHBOARD_PORT, 10001)) {
        $null = netsh interface portproxy delete v4tov4 listenaddress=$TsIP listenport=$port 2>&1
        $null = netsh interface portproxy add    v4tov4 listenaddress=$TsIP listenport=$port `
            connectaddress=127.0.0.1 connectport=$port 2>&1
    }

    Set-Service  -Name iphlpsvc -StartupType Automatic -ErrorAction SilentlyContinue
    Start-Service -Name iphlpsvc -ErrorAction SilentlyContinue

    $check = netsh interface portproxy show v4tov4 2>&1
    if ($check -match "$RAY_PORT") {
        Write-Host "  Portproxy verified." -ForegroundColor Green
    } else {
        Write-Log -Message "Portproxy could not be confirmed." -Level "WARN"
    }
}

function Test-TailscaleReachability {
    param([string]$TargetIP, [int]$Retries = $RETRY_COUNT)
    Write-Host "  [TAILSCALE] Checking reachability of $TargetIP..." -ForegroundColor Cyan

    for ($i = 1; $i -le $Retries; $i++) {
        $result = ping -n 2 $TargetIP 2>&1
        if ($result -match "TTL=" -or $result -match "bytes=") {
            Write-Host "  Tailscale ping OK ($TargetIP)." -ForegroundColor Green
            Write-Log -Message "Tailscale reachability confirmed: $TargetIP"
            return $true
        }
        Write-Log -Message "Ping attempt $i/$Retries to $TargetIP failed." -Level "WARN"
        if ($i -lt $Retries) {
            Write-Host "  Ping failed (attempt $i/$Retries). Restarting Tailscale service..." -ForegroundColor Yellow
            try {
                Restart-Service -Name Tailscale -ErrorAction Stop
                Start-Sleep -Seconds 5
            } catch {
                Write-Log -Message "Could not restart Tailscale service: $($_.Exception.Message)" -Level "WARN"
                Start-Sleep -Seconds 3
            }
        }
    }

    Write-Log -Message "Tailscale IP $TargetIP is unreachable after $Retries attempts." -Level "ERROR"
    Write-Host ""
    Write-Host "  DIAGNOSIS: Cannot reach $TargetIP via Tailscale." -ForegroundColor Red
    Write-Host "  Suggested fixes:" -ForegroundColor Yellow
    Write-Host "    1. Open the Tailscale system tray icon and click Connect." -ForegroundColor White
    Write-Host "    2. Ensure both machines are signed into the SAME Tailscale account." -ForegroundColor White
    Write-Host "    3. Check admin.tailscale.com -- both machines should show as Connected." -ForegroundColor White
    Write-Host "    4. In an admin PowerShell: Restart-Service Tailscale" -ForegroundColor White
    return $false
}

# ---------------------------------------------------------------------------
# BANNER
# ---------------------------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "      maor-equity  --  Ray Cluster Setup (Tailscale VPN)     " -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Log: $LOG_FILE" -ForegroundColor Gray
Write-Host ""
Write-Log -Message "=== Script started. Role=$Role ==="

# ---------------------------------------------------------------------------
# PRE-FLIGHT: Tailscale check
# ---------------------------------------------------------------------------
Write-Host "  [CHECK] Tailscale on Windows..." -NoNewline

$tsExe = "$env:ProgramFiles\Tailscale\tailscale.exe"
if (-not (Test-Path $tsExe)) { $tsExe = "C:\Program Files\Tailscale\tailscale.exe" }

if (Test-Path $tsExe) {
    Write-Host " found" -ForegroundColor Green
    try {
        $tsStatus    = & $tsExe status --json 2>$null | ConvertFrom-Json
        $windowsTS_IP = $tsStatus.TailscaleIPs |
                        Where-Object { $_ -match "^100\." } |
                        Select-Object -First 1
        if ($windowsTS_IP) {
            Write-Host "  Windows Tailscale IP: $windowsTS_IP" -ForegroundColor Green
            Write-Log -Message "Windows Tailscale IP: $windowsTS_IP"
        }
    } catch {
        Write-Log -Message "Could not read Tailscale status: $($_.Exception.Message)" -Level "WARN"
    }
} else {
    Write-Host " NOT INSTALLED" -ForegroundColor Red
    Write-Log -Message "Tailscale not found." -Level "ERROR"
    Write-Host ""
    Write-Host "  ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "    1. Download Tailscale: https://tailscale.com/download/windows" -ForegroundColor White
    Write-Host "    2. Install and sign in (use the same account on BOTH PCs)" -ForegroundColor White
    Write-Host "    3. Re-run this script" -ForegroundColor White
    Write-Host ""
    $openBrowser = Read-Host "  Open Tailscale download page now? (Y/N)"
    if ($openBrowser -eq "Y" -or $openBrowser -eq "y") {
        Start-Process "https://tailscale.com/download/windows"
    }
    exit 1
}

# ---------------------------------------------------------------------------
# PRE-FLIGHT: WSL check
# ---------------------------------------------------------------------------
Write-Host "  [CHECK] WSL..." -NoNewline
try {
    $null = wsl echo ok 2>&1
    Write-Host " found" -ForegroundColor Green
} catch {
    Write-Host " NOT FOUND" -ForegroundColor Red
    Write-Log -Message "WSL not found." -Level "ERROR"
    Write-Host "  Run in an admin PowerShell: wsl --install" -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
# ROLE SELECTION
# ---------------------------------------------------------------------------
if (-not $Role) {
    Write-Host ""
    Write-Host "  Which node are you?" -ForegroundColor Yellow
    Write-Host "    A = Node A  (CPU head node, your PC)"
    Write-Host "    B = Node B  (GPU worker, partner PC)"
    Write-Host ""
    $Role = (Read-Host "  Enter A or B").Trim().ToUpper()
}
if ($Role -notin @("A","B")) {
    Write-Log -Message "Invalid role: $Role" -Level "ERROR"
    Write-Host "  ERROR: Role must be A or B." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# TEMP DIR
# ---------------------------------------------------------------------------
if (-not (Test-Path $WIN_TMP)) {
    New-Item -ItemType Directory -Path $WIN_TMP -Force | Out-Null
}

# ===========================================================================
# NODE A -- Head node startup
# ===========================================================================
if ($Role -eq "A") {

    Write-Host ""
    Write-Host "  [NODE A] Detecting Tailscale IP..." -ForegroundColor Cyan

    $TS_IP = ""
    try {
        $s     = & $tsExe status --json 2>$null | ConvertFrom-Json
        $TS_IP = $s.TailscaleIPs |
                 Where-Object { $_ -match "^100\." } |
                 Select-Object -First 1
    } catch {
        Write-Log -Message "Failed to parse Tailscale status: $($_.Exception.Message)" -Level "WARN"
    }

    if (-not $TS_IP -and $NODE_A_TAILSCALE_IP -ne "x.x.x.x") {
        $TS_IP = $NODE_A_TAILSCALE_IP
        Write-Log -Message "Using config value for Node A IP: $TS_IP" -Level "WARN"
    }

    if (-not $TS_IP) {
        Write-Log -Message "Could not detect Node A Tailscale IP." -Level "ERROR"
        Write-Host "  ERROR: Could not get Tailscale IP." -ForegroundColor Red
        Write-Host "  Fix: Open Tailscale in the system tray and click Connect." -ForegroundColor Yellow
        Write-Host "  Or set NODE_A_TAILSCALE_IP at the top of this script." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "  Node A Tailscale IP: $TS_IP" -ForegroundColor Green
    Write-Log -Message "Node A Tailscale IP: $TS_IP"

    try {
        Add-RayFirewallRules -TsIP $TS_IP
    } catch {
        Write-Log -Message "Firewall setup error (non-fatal): $($_.Exception.Message)" -Level "WARN"
        Write-Host "  WARNING: Firewall setup had errors -- continuing." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  [NODE A] Writing nodeA.sh to $WIN_TMP..." -ForegroundColor Cyan

    # -----------------------------------------------------------------------
    # NODE A BASH SCRIPT
    # Single-quoted heredoc: PowerShell does NOT expand anything inside.
    # PLACEHOLDER strings are replaced below via -replace after the heredoc.
    # -----------------------------------------------------------------------
    $scriptA = @'
#!/usr/bin/env bash
# Node A: Ray head startup via Tailscale
# Auto-generated by ray_cluster.ps1 -- do not edit directly

set -uo pipefail

# -- Variables injected by ray_cluster.ps1 --------------------------------
VENV_REAL="PLACEHOLDER_A_VENV"
PROJECT="PLACEHOLDER_A_PROJECT"
TS_IP="PLACEHOLDER_TS_IP"
RAY_PORT="PLACEHOLDER_RAY_PORT"
RAY_DASHBOARD_PORT="PLACEHOLDER_RAY_DASHBOARD_PORT"
RETRY_COUNT="PLACEHOLDER_RETRY_COUNT"
TIMEOUT_SECONDS="PLACEHOLDER_TIMEOUT_SECONDS"
LOG_FILE="$PROJECT/ray_node_a.log"

# Ray spawns workers via bash using sys.executable. Spaces in that path cause
# bash to split at the wrong token. /usr/bin/python3.x has no spaces on Ubuntu.
# PYTHONPATH exposes venv packages to all workers spawned by the raylet.
MAOR_PY=$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
if [ -z "$MAOR_PY" ] || echo "$MAOR_PY" | grep -qF ' '; then
    cp "$VENV_REAL/bin/python3" /tmp/maor_py_a 2>/dev/null && chmod +x /tmp/maor_py_a && MAOR_PY="/tmp/maor_py_a"
fi
PY_LIB=$(ls -d "$VENV_REAL/lib/python3."* 2>/dev/null | head -1)
export PYTHONPATH="$PY_LIB/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# -- Ray environment variables ---------------------------------------------
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=120
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=120

# -- Logging helper --------------------------------------------------------
log_msg() {
    local msg="$1"
    echo "  [$(date '+%H:%M:%S')] $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$LOG_FILE" 2>/dev/null || true
}

# -- kill_port: free a TCP port; tries fuser, then lsof, then ss+kill ------
# fuser is not present on all distros (absent from Ubuntu minimal images).
kill_port() {
    local port="$1"
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:${port}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    else
        ss -tlnp 2>/dev/null \
            | awk -F'pid=' "/:${port} /{print \$2}" \
            | cut -d, -f1 \
            | xargs -r kill -9 2>/dev/null || true
    fi
}

# -- check_tcp: test TCP reachability; falls back to /dev/tcp if nc absent --
# nc (netcat) is not installed on all distros by default.
check_tcp() {
    local host="$1" port="$2" timeout="${3:-5}"
    if command -v nc >/dev/null 2>&1; then
        nc -z -w"$timeout" "$host" "$port" 2>/dev/null
    else
        timeout "$timeout" bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null
    fi
}

echo ""
echo "  ============================================================"
echo "  NODE A -- Ray Head  (Tailscale IP: $TS_IP)"
echo "  ============================================================"
echo "  Log: $LOG_FILE"
echo ""
log_msg "=== Node A startup script begun ==="

# -- Step 1: Tailscale sanity check ---------------------------------------
log_msg "[1/4] Tailscale check..."
if check_tcp "$TS_IP" "$RAY_PORT" 2 2>/dev/null || ping -c1 -W2 "$TS_IP" >/dev/null 2>&1; then
    log_msg "Tailscale IP $TS_IP is reachable."
else
    log_msg "WARNING: Cannot reach own Tailscale IP from WSL. Continuing anyway."
fi

# -- Step 1b: Remove stale GCS portproxy ----------------------------------
# WSL2 mirrored networking makes a Windows portproxy on $TS_IP:$RAY_PORT
# visible inside WSL2 and prevents GCS from binding. Remove it first.
# This runs an elevated Windows PowerShell via the already-admin PS session.
powershell.exe -NoProfile -Command \
    "netsh interface portproxy delete v4tov4 listenaddress=$TS_IP listenport=$RAY_PORT >nul 2>&1; exit 0" \
    2>/dev/null || true
log_msg "Removed stale GCS portproxy (if any)."

# -- Step 1c: Add Tailscale IP as loopback alias ---------------------------
# GCS binds to the loopback alias locally so the raylet connects without
# going through the Tailscale userspace stack -- avoids 60s timeout.
log_msg "Adding $TS_IP as loopback alias..."
sudo ip addr add "${TS_IP}/32" dev lo 2>/dev/null || true

# -- Step 2: Aggressive cleanup --------------------------------------------
log_msg "[2/4] Stopping all Ray processes..."
"$MAOR_PY" "$VENV_REAL/bin/ray" stop --force 2>/dev/null || true
sleep 1
pkill -9 -f gcs_server  2>/dev/null || true
pkill -9 -f raylet       2>/dev/null || true
pkill -9 -f plasma_store 2>/dev/null || true
pkill -9 -f monitor.py   2>/dev/null || true
pkill -9 -f "ray::"      2>/dev/null || true
sleep 2
kill_port "$RAY_PORT"
kill_port "$RAY_DASHBOARD_PORT"
sleep 2
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* 2>/dev/null || true
log_msg "Ray cleanup done."

# Verify GCS port is free (retry up to $RETRY_COUNT times)
PORT_FREE=false
for i in $(seq 1 $RETRY_COUNT); do
    if ! ss -tlnp 2>/dev/null | grep -q ":${RAY_PORT}"; then
        PORT_FREE=true
        break
    fi
    log_msg "Port $RAY_PORT still occupied (attempt $i/$RETRY_COUNT) -- forcing free..."
    kill_port "$RAY_PORT"
    sleep 3
done

if [ "$PORT_FREE" = "false" ]; then
    log_msg "ERROR: Port $RAY_PORT still occupied after cleanup."
    echo ""
    echo "  DIAGNOSIS: Port $RAY_PORT is in use by another process."
    echo "  Suggested fix: sudo pkill -9 gcs_server"
    read -rp "  Press Enter to exit..." && exit 1
fi
log_msg "Port $RAY_PORT is free."

# -- Step 3: Start Ray head ------------------------------------------------
log_msg "[3/4] Starting Ray head on $TS_IP:$RAY_PORT..."
echo ""
echo "  Starting Ray head node..."

"$MAOR_PY" "$VENV_REAL/bin/ray" start \
    --head \
    --port="$RAY_PORT" \
    --include-dashboard=false \
    --node-ip-address="$TS_IP" \
    --disable-usage-stats \
    --num-cpus=2 \
    --object-store-memory=268435456 \
    --plasma-directory=/tmp \
    2>&1

RAY_EXIT=$?
if [ $RAY_EXIT -ne 0 ]; then
    log_msg "ERROR: Ray start failed (exit code $RAY_EXIT)."
    echo ""
    echo "  DIAGNOSIS: Ray head failed to start."
    echo "  Suggested fix: Close this window, open a fresh WSL terminal, and re-run."
    read -rp "  Press Enter..." && exit 1
fi
log_msg "Ray start command returned OK."

# -- Step 3b: Poll for GCS health -----------------------------------------
log_msg "[3b/4] Polling GCS health (timeout: ${TIMEOUT_SECONDS}s)..."
GCS_UP=false
POLL_END=$((SECONDS + TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$POLL_END" ]; do
    if ss -tlnp 2>/dev/null | grep -q ":${RAY_PORT}"; then
        GCS_UP=true
        break
    fi
    sleep 3
done

if [ "$GCS_UP" = "false" ]; then
    log_msg "ERROR: GCS did not bind to port $RAY_PORT within ${TIMEOUT_SECONDS}s."
    echo ""
    echo "  DIAGNOSIS: GCS server timed out during startup."
    echo "  Suggested fix: cat /tmp/ray/session_latest/logs/gcs_server.out"
    read -rp "  Press Enter..." && exit 1
fi

RAYLET_COUNT=$(ps aux 2>/dev/null | grep '[r]aylet' | grep -v defunct | wc -l)
log_msg "GCS is UP on $TS_IP:$RAY_PORT. Raylet processes: $RAYLET_COUNT"
echo "  Ray GCS: UP on $TS_IP:$RAY_PORT"
echo "  Raylet processes: $RAYLET_COUNT"

if [ "$RAYLET_COUNT" -eq 0 ]; then
    log_msg "WARNING: No raylet processes found. Ray may still be initializing."
fi

# -- Step 3c: Detect dynamic raylet ports (log only, no portproxy from bash)
# With WSL2 mirrored networking the Tailscale interface is shared between
# Windows and WSL2, so dynamic raylet ports are reachable via $TS_IP directly
# without any portproxy. We log the ports for diagnostic purposes only.
# grep -o + sed is used instead of grep -oP (Perl regex not portable).
RAYLET_PORT=$(ps aux 2>/dev/null | grep '[r]aylet' \
    | grep -o '\-\-node_manager_port=[0-9]*' | head -1 | sed 's/.*=//')
OBJ_PORT=$(ps aux 2>/dev/null | grep '[r]aylet' \
    | grep -o '\-\-object_manager_port=[0-9]*' | head -1 | sed 's/.*=//')
log_msg "Dynamic ports detected: raylet=$RAYLET_PORT obj=$OBJ_PORT"

# -- Step 4: Print connection info for Node B -----------------------------
echo ""
echo "  ============================================================"
echo "  NODE A IS READY -- Send these commands to Node B:"
echo "  ============================================================"
echo ""
echo "  PowerShell command (run on Node B):"
echo "    .\\ray_cluster.ps1 -Role B -TailscaleIP $TS_IP"
echo ""
echo "  OR manual WSL command (run in Node B WSL):"
echo "    export LD_PRELOAD=''"
echo "    export RAY_DISABLE_JEMALLOC=1"
echo "    export CUDA_VISIBLE_DEVICES=0"
echo "    ray start --address=$TS_IP:$RAY_PORT --num-gpus=1 --num-cpus=4"
echo ""
echo "  Dashboard: http://$TS_IP:$RAY_DASHBOARD_PORT"
echo ""
echo "  ============================================================"
echo "  KEEP THIS WINDOW OPEN -- closing it stops Ray"
echo "  ============================================================"
echo ""

log_msg "[4/4] Entering monitor loop (Ctrl+C stops monitor, Ray stays running)."
echo "  Monitoring every 30s -- waiting for Node B to join..."
echo "  ------------------------------------------------------------------"

NODE_B_SEEN=false

while true; do
    sleep 30
    NOW=$(date '+%H:%M:%S')

    if ss -tlnp 2>/dev/null | grep -q ":${RAY_PORT}"; then
        GCS_STATUS="Ray:UP"
    else
        GCS_STATUS="Ray:DOWN"
        log_msg "WARNING: GCS is no longer bound to port $RAY_PORT."
    fi

    # Count alive nodes via Python. timeout 15 prevents hangs.
    # sys.stdout.flush() + os._exit(0) ensures output is written before exit.
    NODES=$(timeout 15 "$MAOR_PY" -c "
import ray, sys, os
os.environ['RAY_DISABLE_JEMALLOC'] = '1'
os.environ['LD_PRELOAD'] = ''
try:
    ray.init(address='${TS_IP}:${RAY_PORT}', ignore_reinit_error=True, logging_level='ERROR')
    alive = [n for n in ray.nodes() if n.get('Alive')]
    sys.stdout.write(str(len(alive)) + '\n')
except Exception:
    sys.stdout.write('?\n')
finally:
    sys.stdout.flush()
    os._exit(0)
" 2>/dev/null || echo "?")

    echo "  [$NOW] $GCS_STATUS | Alive nodes: $NODES | Head: $TS_IP:$RAY_PORT"

    # Announce the first time Node B is seen (node count reaches 2+)
    if [ "$NODES" != "?" ] && [ "$NODE_B_SEEN" = "false" ]; then
        NODE_COUNT_INT=$(echo "$NODES" | tr -d '[:space:]')
        if [ -n "$NODE_COUNT_INT" ] && [ "$NODE_COUNT_INT" -ge 2 ] 2>/dev/null; then
            NODE_B_SEEN=true
            log_msg "SUCCESS: Node B joined. Alive nodes: $NODE_COUNT_INT"
            echo ""
            echo "  ============================================================"
            echo "  SUCCESS: Node B is connected! Cluster has $NODE_COUNT_INT alive nodes."
            echo "  Dashboard: http://$TS_IP:$RAY_DASHBOARD_PORT"
            echo "  ============================================================"
            echo ""
        fi
    fi
done

echo ""
read -rp "  === Script ended. Press Enter to close ==="
'@

    $scriptA = $scriptA `
        -replace 'PLACEHOLDER_A_VENV',            $A_VENV `
        -replace 'PLACEHOLDER_A_PROJECT',          $A_PROJECT `
        -replace 'PLACEHOLDER_TS_IP',              $TS_IP `
        -replace 'PLACEHOLDER_RAY_PORT',           $RAY_PORT `
        -replace 'PLACEHOLDER_RAY_DASHBOARD_PORT', $RAY_DASHBOARD_PORT `
        -replace 'PLACEHOLDER_RETRY_COUNT',        $RETRY_COUNT `
        -replace 'PLACEHOLDER_TIMEOUT_SECONDS',    $TIMEOUT_SECONDS

    Write-UnixFile -Path "$WIN_TMP\nodeA.sh" -Content $scriptA
    wsl chmod +x "$WSL_TMP/nodeA.sh" 2>$null

    Open-WslWindow -Title "NODE A - Ray Head" -Script "$WSL_TMP/nodeA.sh"

    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Host "  Node A startup window opened!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Your Tailscale IP for Node B: $TS_IP" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Tell your partner (Node B) to run:" -ForegroundColor Cyan
    Write-Host "    .\ray_cluster.ps1 -Role B -TailscaleIP $TS_IP" -ForegroundColor White
    Write-Host ""
    Write-Host "  After Node B connects, verify with:" -ForegroundColor Cyan
    Write-Host "    wsl bash -c `"source '$A_VENV/bin/activate' && ray status`"" -ForegroundColor White
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Log -Message "Node A setup complete. WSL window opened. TS_IP=$TS_IP"
}

# ===========================================================================
# NODE B -- Worker node startup
# ===========================================================================
if ($Role -eq "B") {

    if (-not $TailscaleIP -and $NODE_B_TAILSCALE_IP -ne "x.x.x.x") {
        $TailscaleIP = $NODE_B_TAILSCALE_IP
    }
    if (-not $TailscaleIP) {
        Write-Host ""
        Write-Host "  Enter Node A's Tailscale IP (shown in Node A's terminal)." -ForegroundColor Yellow
        Write-Host "  It looks like: 100.x.x.x" -ForegroundColor Gray
        Write-Host ""
        $TailscaleIP = (Read-Host "  Node A Tailscale IP").Trim()
    }

    if ($TailscaleIP -notmatch '^100\.\d+\.\d+\.\d+$') {
        Write-Log -Message "Invalid Tailscale IP: $TailscaleIP" -Level "ERROR"
        Write-Host "  ERROR: Tailscale IPs start with 100. Got: $TailscaleIP" -ForegroundColor Red
        exit 1
    }

    Write-Log -Message "Node B will connect to head at $TailscaleIP"

    $reachable = Test-TailscaleReachability -TargetIP $TailscaleIP -Retries $RETRY_COUNT
    if (-not $reachable) {
        Write-Log -Message "Aborting: Node A at $TailscaleIP is unreachable." -Level "ERROR"
        exit 1
    }

    Write-Host "  [NODE B] Configuring WSL2 mirrored networking..." -ForegroundColor Cyan
    $wslCfg     = Join-Path $env:USERPROFILE ".wslconfig"
    $cfgContent = "[wsl2]`nnetworkingMode=mirrored`n"
    if (-not (Test-Path $wslCfg) -or (Get-Content $wslCfg -Raw) -notmatch 'mirrored') {
        Set-Content -Path $wslCfg -Value $cfgContent -Encoding UTF8
        Write-Host "  .wslconfig written -- restarting WSL to apply..." -ForegroundColor Yellow
        wsl --shutdown 2>$null
        Start-Sleep -Seconds 3
        Write-Host "  WSL restarted with mirrored networking." -ForegroundColor Green
        Write-Log -Message "WSL2 mirrored networking enabled."
    } else {
        Write-Host "  Already configured." -ForegroundColor Green
    }

    try {
        Add-RayFirewallRules -TsIP $TailscaleIP
    } catch {
        Write-Log -Message "Firewall setup error on Node B (non-fatal): $($_.Exception.Message)" -Level "WARN"
        Write-Host "  WARNING: Firewall setup had errors -- continuing." -ForegroundColor Yellow
    }

    # Locate Ray binary via a temp bash script to avoid -lc quoting issues
    # with paths that contain spaces.
    Write-Host ""
    Write-Host "  [NODE B] Locating Ray binary in WSL..." -NoNewline

    $detectSh = "$WIN_TMP\_detect_ray.sh"
    # PS double-quoted string: $B_VENV is expanded here; single quotes in the
    # generated bash file protect the spaces within the path.
    $detectContent  = "#!/usr/bin/env bash`n"
    $detectContent += "if [ -x '$B_VENV/bin/ray' ]; then echo '$B_VENV/bin/ray'; exit 0; fi`n"
    $detectContent += 'if [ -x "$HOME/maor-equity/venv/bin/ray" ]; then echo "$HOME/maor-equity/venv/bin/ray"; exit 0; fi' + "`n"
    $detectContent += 'if [ -x "$HOME/venv/bin/ray" ]; then echo "$HOME/venv/bin/ray"; exit 0; fi' + "`n"
    $detectContent += "which ray 2>/dev/null || true`n"
    Write-UnixFile -Path $detectSh -Content $detectContent
    wsl chmod +x "$WSL_TMP/_detect_ray.sh" 2>$null

    $rayBinRaw = wsl bash "$WSL_TMP/_detect_ray.sh" 2>$null
    $rayBin    = ($rayBinRaw | Where-Object { $_ -match '/ray' } | Select-Object -First 1)
    if ($rayBin) { $rayBin = $rayBin.Trim() }

    if (-not $rayBin) {
        Write-Host " NOT FOUND" -ForegroundColor Red
        Write-Log -Message "Ray binary not found in WSL on Node B." -Level "ERROR"
        Write-Host ""
        Write-Host "  Ray not found in WSL. Run these commands inside WSL to install:" -ForegroundColor Yellow
        Write-Host "    cd '$B_PROJECT'" -ForegroundColor White
        Write-Host "    python3 -m venv venv" -ForegroundColor White
        Write-Host "    source venv/bin/activate" -ForegroundColor White
        Write-Host "    pip install ray[default]" -ForegroundColor White
        exit 1
    }

    Write-Host " $rayBin" -ForegroundColor Green
    Write-Log -Message "Ray binary found at: $rayBin"

    Write-Host "  [NODE B] Writing nodeB.sh..." -ForegroundColor Cyan

    # -----------------------------------------------------------------------
    # NODE B BASH SCRIPT
    # Single-quoted heredoc -- PLACEHOLDER strings replaced below.
    # -----------------------------------------------------------------------
    $scriptB = @'
#!/usr/bin/env bash
# Node B: Ray worker -- connects to Node A via Tailscale
# Auto-generated by ray_cluster.ps1 -- do not edit directly

set -uo pipefail

# -- Variables injected by ray_cluster.ps1 --------------------------------
HEAD_IP="PLACEHOLDER_HEAD_IP"
RAY_PORT="PLACEHOLDER_RAY_PORT"
VENV_REAL="PLACEHOLDER_B_VENV"
PROJECT="PLACEHOLDER_B_PROJECT"
RETRY_COUNT="PLACEHOLDER_RETRY_COUNT"
TIMEOUT_SECONDS="PLACEHOLDER_TIMEOUT_SECONDS"
LOG_FILE="$PROJECT/ray_node_b.log"

# Same sys.executable fix as Node A: copy the binary to /tmp so the path
# Ray stores for worker spawning contains no spaces.
MAOR_PY=$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
if [ -z "$MAOR_PY" ] || echo "$MAOR_PY" | grep -qF ' '; then
    cp "$VENV_REAL/bin/python3" /tmp/maor_py_b 2>/dev/null && chmod +x /tmp/maor_py_b && MAOR_PY="/tmp/maor_py_b"
fi
PY_LIB=$(ls -d "$VENV_REAL/lib/python3."* 2>/dev/null | head -1)
export PYTHONPATH="$PY_LIB/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# -- Ray environment variables ---------------------------------------------
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0

# -- Logging helper --------------------------------------------------------
log_msg() {
    local msg="$1"
    echo "  [$(date '+%H:%M:%S')] $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$LOG_FILE" 2>/dev/null || true
}

# -- kill_port: fuser -> lsof -> ss fallback --------------------------------
kill_port() {
    local port="$1"
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:${port}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    else
        ss -tlnp 2>/dev/null \
            | awk -F'pid=' "/:${port} /{print \$2}" \
            | cut -d, -f1 \
            | xargs -r kill -9 2>/dev/null || true
    fi
}

# -- check_tcp: nc -> /dev/tcp fallback ------------------------------------
check_tcp() {
    local host="$1" port="$2" timeout="${3:-5}"
    if command -v nc >/dev/null 2>&1; then
        nc -z -w"$timeout" "$host" "$port" 2>/dev/null
    else
        timeout "$timeout" bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null
    fi
}

echo ""
echo "  ============================================================"
echo "  NODE B -- Ray Worker  (connecting to $HEAD_IP:$RAY_PORT)"
echo "  ============================================================"
echo "  Log: $LOG_FILE"
echo ""
log_msg "=== Node B startup script begun ==="

# -- Step 1: TCP connectivity check ---------------------------------------
log_msg "[1/3] Checking TCP connectivity to $HEAD_IP:$RAY_PORT..."
CONNECTED=false
for i in $(seq 1 $RETRY_COUNT); do
    if check_tcp "$HEAD_IP" "$RAY_PORT" 5; then
        CONNECTED=true
        log_msg "TCP connection to $HEAD_IP:$RAY_PORT confirmed OK."
        break
    fi
    log_msg "Attempt $i/$RETRY_COUNT: $HEAD_IP:$RAY_PORT unreachable. Waiting 10s..."
    sleep 10
done

if [ "$CONNECTED" = "false" ]; then
    log_msg "WARNING: Cannot reach $HEAD_IP:$RAY_PORT after $RETRY_COUNT attempts."
    echo ""
    echo "  DIAGNOSIS: Cannot connect to Node A at $HEAD_IP:$RAY_PORT"
    echo "  Checklist:"
    echo "    1. Is Node A WSL terminal open and showing Ray:UP?"
    echo "    2. Is Tailscale connected on BOTH machines (system tray icon)?"
    echo "    3. Are both machines signed into the SAME Tailscale account?"
    echo "    4. Did Node A run ray_cluster.ps1 -Role A and accept the UAC prompt?"
    echo "  Attempting to join anyway -- it may still succeed..."
fi

# -- Step 2: Cleanup -------------------------------------------------------
log_msg "[2/3] Stopping existing Ray processes..."
"$MAOR_PY" "$VENV_REAL/bin/ray" stop --force 2>/dev/null || true
sleep 1
pkill -9 -f raylet       2>/dev/null || true
pkill -9 -f plasma_store 2>/dev/null || true
sleep 2
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* 2>/dev/null || true
log_msg "Cleanup done."

# -- connect_worker: stop stale Ray then join the head --------------------
connect_worker() {
    "$MAOR_PY" "$VENV_REAL/bin/ray" stop --force 2>/dev/null || true
    sleep 2
    "$MAOR_PY" "$VENV_REAL/bin/ray" start \
        --address="$HEAD_IP:$RAY_PORT" \
        --num-gpus=1 \
        --num-cpus=4 \
        2>&1
}

# -- Step 3: Join the cluster with retries --------------------------------
log_msg "[3/3] Joining Ray cluster at $HEAD_IP:$RAY_PORT..."
JOIN_OK=false
for i in $(seq 1 $RETRY_COUNT); do
    log_msg "Join attempt $i/$RETRY_COUNT..."
    connect_worker
    sleep 5
    if pgrep -f raylet >/dev/null 2>&1; then
        JOIN_OK=true
        log_msg "SUCCESS: Raylet process confirmed alive (attempt $i)."
        break
    fi
    log_msg "Raylet not found after attempt $i. Retrying..."
    sleep 5
done

if [ "$JOIN_OK" = "false" ]; then
    log_msg "ERROR: Failed to join Ray cluster after $RETRY_COUNT attempts."
    echo ""
    echo "  DIAGNOSIS: Raylet did not start after $RETRY_COUNT join attempts."
    echo "  Suggested fixes:"
    echo "    1. Verify Ray versions match on both nodes: ray --version"
    echo "    2. Check Node B raylet logs:"
    echo "       cat /tmp/ray/session_latest/logs/raylet.out"
    echo "    3. Ensure Node A is running and $HEAD_IP:$RAY_PORT is reachable."
    read -rp "  Press Enter to exit..." && exit 1
fi

echo ""
echo "  ============================================================"
echo "  CONNECTED! Worker joined $HEAD_IP:$RAY_PORT"
echo "  Monitoring every 10s -- auto-reconnects on disconnect."
echo "  ============================================================"
echo ""

# -- Monitor loop: auto-reconnect if raylet dies or TCP drops --------------
while true; do
    sleep 10
    NOW=$(date '+%H:%M:%S')

    RAYLET_ALIVE=false
    pgrep -f raylet >/dev/null 2>&1 && RAYLET_ALIVE=true

    TCP_OK=false
    check_tcp "$HEAD_IP" "$RAY_PORT" 4 && TCP_OK=true

    if [ "$RAYLET_ALIVE" = "true" ] && [ "$TCP_OK" = "true" ]; then
        echo "  [$NOW] OK -- raylet alive, connected to $HEAD_IP:$RAY_PORT"
    elif [ "$RAYLET_ALIVE" = "false" ]; then
        log_msg "[$NOW] RAYLET DIED -- reconnecting..."
        connect_worker
    else
        log_msg "[$NOW] LOST CONNECTION to $HEAD_IP:$RAY_PORT -- reconnecting..."
        connect_worker
    fi
done

echo ""
read -rp "  === Script ended. Press Enter to close ==="
'@

    $scriptB = $scriptB `
        -replace 'PLACEHOLDER_HEAD_IP',        $TailscaleIP `
        -replace 'PLACEHOLDER_RAY_PORT',        $RAY_PORT `
        -replace 'PLACEHOLDER_B_VENV',          $B_VENV `
        -replace 'PLACEHOLDER_B_PROJECT',       $B_PROJECT `
        -replace 'PLACEHOLDER_RETRY_COUNT',     $RETRY_COUNT `
        -replace 'PLACEHOLDER_TIMEOUT_SECONDS', $TIMEOUT_SECONDS

    Write-UnixFile -Path "$WIN_TMP\nodeB.sh" -Content $scriptB
    wsl chmod +x "$WSL_TMP/nodeB.sh" 2>$null

    # Generate a standalone Node B PS1 script (zero-config, just run it on Node B).
    # Uses $PSCommandPath (the actual full path of this script) so the standalone
    # script works regardless of what directory Node B runs it from.
    Write-Host "  [NODE B] Generating standalone node_b_standalone.ps1..." -ForegroundColor Cyan
    $standalonePath    = Join-Path $WIN_TMP "node_b_standalone.ps1"
    $mainScriptPath    = if ($PSCommandPath) { $PSCommandPath } else { Join-Path $PSScriptRoot "ray_cluster.ps1" }
    $standaloneContent = "# node_b_standalone.ps1 -- Run this on Node B. All values are pre-filled.`r`n"
    $standaloneContent += "# Prerequisites: Tailscale connected, Ray installed in venv on Node B.`r`n"
    $standaloneContent += "`$TailscaleIP = `"$TailscaleIP`"`r`n"
    $standaloneContent += "& `"$mainScriptPath`" -Role B -TailscaleIP `$TailscaleIP`r`n"
    Set-Content -Path $standalonePath -Value $standaloneContent -Encoding UTF8
    Write-Host "  Standalone script: $standalonePath" -ForegroundColor Gray
    Write-Log -Message "Standalone Node B script written to: $standalonePath"

    Open-WslWindow -Title "NODE B - Ray Worker" -Script "$WSL_TMP/nodeB.sh"

    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Host "  Node B connection window opened!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Watch that window -- it prints every 10s:" -ForegroundColor Cyan
    Write-Host "    OK -- raylet alive, connected to ${TailscaleIP}:${RAY_PORT}" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  RECONNECTING messages are normal and automatic." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Standalone script for Node B (zero-config, copy to Node B):" -ForegroundColor Cyan
    Write-Host "    $standalonePath" -ForegroundColor White
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Log -Message "Node B setup complete. WSL window opened. HEAD=$TailscaleIP"
}

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  Done. Keep the WSL terminal window open." -ForegroundColor Cyan
Write-Host "  Log file: $LOG_FILE" -ForegroundColor Gray
Write-Host ""
