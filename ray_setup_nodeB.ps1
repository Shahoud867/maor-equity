<#
.SYNOPSIS
    Ray Cluster Worker Setup - Node B (Tailscale + WSL2 + GPU)

.PREREQUISITES
    - Tailscale installed and signed in (same account as Node A)
    - WSL2 Ubuntu with Python venv containing Ray, transformers, bitsandbytes
    - CUDA-capable GPU (T1000 or similar) visible in WSL2
    - Node A has already run ray_setup_nodeA.ps1 and is showing "NODE A IS READY"

.USAGE
    Option 1 - with parameter (recommended):
        powershell -ExecutionPolicy Bypass -File .\ray_setup_nodeB.ps1 -NodeAIP 100.88.99.30

    Option 2 - edit CONFIG block and run:
        powershell -ExecutionPolicy Bypass -File .\ray_setup_nodeB.ps1

.WHAT THIS SCRIPT DOES
    Step 1  Validate Tailscale, ping Node A
    Step 2  Kill stale Ray processes in WSL2 (by PID, no ray stop)
    Step 3  Clean Ray temp files
    Step 4  Configure Windows Firewall (inbound + outbound)
    Step 5  Start Ray worker in WSL2 with GPU, PYTHONPATH, and watchdog loop
    Step 6  Verify this node appears in the cluster
#>

param(
    [string]$NodeAIP = ""   # Node A's Tailscale IP - pass via -NodeAIP or set in CONFIG
)

# ===========================================================================
#  CONFIG - edit before running (or pass -NodeAIP parameter)
# ===========================================================================
if (-not $NodeAIP) {
    $NodeAIP = "100.88.99.30"   # Node A's Tailscale IP
}
$NODE_A_TAILSCALE_IP = $NodeAIP
$RAY_PORT            = 6379
$RETRY_COUNT         = 3
$TIMEOUT_SECONDS     = 90
$LOG_FILE            = "$env:USERPROFILE\ray_setup_nodeB.log"

# WSL2 paths on Node B - adjust to where the repo is cloned
$WSL_VENV    = "/home/$env:USERNAME/maor-equity/venv"
$WSL_PROJECT = "/home/$env:USERNAME/maor-equity"

# Tailscale executable
$TAILSCALE_EXE = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path $TAILSCALE_EXE)) {
    $TAILSCALE_EXE = "$env:ProgramFiles\Tailscale\tailscale.exe"
}

# ===========================================================================
#  SELF-ELEVATION
# ===========================================================================
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
               [Security.Principal.WindowsBuiltInRole]"Administrator")
if (-not $isAdmin) {
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"",
           "-NodeAIP", $NODE_A_TAILSCALE_IP)
    Start-Process powershell -Verb RunAs -ArgumentList $a
    exit
}

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

# ===========================================================================
#  LOGGING
# ===========================================================================
function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}
function Write-Ok   { param([string]$M) Write-Log "  OK  $M" "Green"  }
function Write-Warn { param([string]$M) Write-Log "  !!  $M" "Yellow" }
function Write-Err  { param([string]$M) Write-Log "  XX  $M" "Red"    }
function Write-Step {
    param([string]$T)
    Write-Log ""
    Write-Log ("=" * 64) "Cyan"
    Write-Log "  $T" "Cyan"
    Write-Log ("=" * 64) "Cyan"
}
function Write-UnixFile {
    param([string]$Path, [string]$Content)
    $enc = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, ($Content -replace "`r`n", "`n"), $enc)
}

"" | Out-File $LOG_FILE -Encoding UTF8
Write-Log "Ray Node B Setup - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Connecting to Node A at ${NODE_A_TAILSCALE_IP}:${RAY_PORT}"

$WIN_TMP = "$env:TEMP\ray_cluster_setup"
if (-not (Test-Path $WIN_TMP)) { New-Item -ItemType Directory -Path $WIN_TMP -Force | Out-Null }

$_drive   = ($env:TEMP -split ":\\")[0].ToLower()
$_relpath = ($env:TEMP -split ":\\")[1] -replace "\\", "/"
$WSL_TMP  = "/mnt/$_drive/$_relpath/ray_cluster_setup"


# ===========================================================================
#  STEP 1: Tailscale validation - ping Node A
# ===========================================================================
Write-Step "STEP 1 - Tailscale Validation"

$ts = Get-Service -Name Tailscale -ErrorAction SilentlyContinue
if ($ts -and $ts.Status -ne "Running") {
    Write-Warn "Tailscale service stopped - restarting..."
    Start-Service Tailscale -ErrorAction SilentlyContinue
    Start-Sleep 4
}

# Get this node's Tailscale IP for display
try {
    $s = & $TAILSCALE_EXE status --json 2>$null | ConvertFrom-Json
    $myIP = $s.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
    Write-Ok "Node B Tailscale IP: $myIP"
} catch { Write-Warn "Could not read Node B Tailscale IP" }

# Ping Node A
$nodeAReachable = $false
for ($i = 1; $i -le $RETRY_COUNT; $i++) {
    $r = ping -n 2 -w 3000 $NODE_A_TAILSCALE_IP 2>&1
    if ($r -match "bytes=") { $nodeAReachable = $true; break }
    Write-Warn "Ping attempt $i/$RETRY_COUNT to Node A failed..."
    Start-Sleep 3
}
if ($nodeAReachable) {
    Write-Ok "Node A ($NODE_A_TAILSCALE_IP) is reachable"
} else {
    Write-Err "Cannot ping Node A at $NODE_A_TAILSCALE_IP"
    Write-Err "Make sure:"
    Write-Err "  1. Node A has run ray_setup_nodeA.ps1 and is showing READY"
    Write-Err "  2. Both machines are on Tailscale with the same account"
    Write-Err "  3. Tailscale shows both machines as 'Connected' in the systray"
    Read-Host "Press Enter to exit"; exit 1
}


# ===========================================================================
#  STEP 2: Kill stale WSL2 Ray processes (by PID - no ray stop)
# ===========================================================================
Write-Step "STEP 2 - WSL2 Ray Cleanup"

$wslCleanup = @'
pgrep -f 'gc[s]_server'   2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'rayle[t]'        2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'plasma_stor[e]'  2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'monitor\.p[y]'   2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* 2>/dev/null || true
echo WSLCLEAN_OK
'@
$r = wsl -u root bash -c $wslCleanup 2>&1
if ($r -match "WSLCLEAN_OK") { Write-Ok "WSL2 Ray processes killed, /tmp/ray cleared" }
else { Write-Warn "WSL2 cleanup: $r" }


# ===========================================================================
#  STEP 3: Windows Firewall
# ===========================================================================
Write-Step "STEP 3 - Windows Firewall"

netsh advfirewall firewall delete rule name="Ray-Cluster-In"  >$null 2>&1
netsh advfirewall firewall delete rule name="Ray-Cluster-Out" >$null 2>&1
netsh advfirewall firewall add rule name="Ray-Cluster-In" `
    dir=in action=allow protocol=TCP `
    localport="6379,8265,10001,20000-29999" profile=any | Out-Null
netsh advfirewall firewall add rule name="Ray-Cluster-Out" `
    dir=out action=allow protocol=TCP `
    remoteport="6379,8265,10001,20000-29999" profile=any | Out-Null
Write-Ok "Firewall rules set"


# ===========================================================================
#  STEP 4: Verify Ray venv and GPU in WSL2
# ===========================================================================
Write-Step "STEP 4 - WSL2 Environment Check"

$pyBin  = "$WSL_VENV/bin/python3"
$rayVer = wsl bash -c "'$pyBin' -c 'import ray; print(ray.__version__)' 2>&1" 2>&1
if ($rayVer -match "^\d") { Write-Ok "Ray version: $rayVer" }
else {
    Write-Err "Ray not found in venv: $WSL_VENV"
    Write-Err "Activate venv and run: pip install ray[default]"
    Read-Host "Press Enter to exit"; exit 1
}

$gpuCheck = wsl bash -c "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1" 2>&1
if ($gpuCheck -match "MiB") { Write-Ok "GPU detected: $gpuCheck" }
else { Write-Warn "nvidia-smi check: $gpuCheck (continuing - GPU may still be available)" }


# ===========================================================================
#  STEP 5: Start Ray worker with watchdog
# ===========================================================================
Write-Step "STEP 5 - Start Ray Worker (with watchdog)"

$workerScript = @'
#!/usr/bin/env bash
# Ray Worker Watchdog - reconnects automatically if raylet dies
HEAD_ADDR=PLACEHOLDER_HEAD_ADDR
VENV="PLACEHOLDER_VENV"
PROJECT="PLACEHOLDER_PROJECT"
RAY="$VENV/bin/ray"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0
# PYTHONPATH lets Ray worker processes (spawned by ray start) import project modules.
# Without this, remote actor classes are not found by the worker runtime.
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

now() { date '+%H:%M:%S'; }

connect_worker() {
    echo "[$(now)] Stopping any existing Ray worker..."
    # Kill by PID only - ray stop --force kills our own process group
    pgrep -f 'gc[s]_server'  2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f 'rayle[t]'       2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f 'plasma_stor[e]' 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 2
    echo "[$(now)] Connecting to head at $HEAD_ADDR..."
    "$RAY" start \
        --address="$HEAD_ADDR" \
        --num-gpus=1 \
        --num-cpus=4 \
        2>&1 | tail -8
    echo "[$(now)] Worker started. Watchdog active (checks every 15s)."
}

# Verify head is reachable before starting
echo "[$(now)] Checking connectivity to $HEAD_ADDR..."
HEAD_HOST=${HEAD_ADDR%:*}
HEAD_PORT=${HEAD_ADDR#*:}
for i in 1 2 3 4 5; do
    if nc -z -w4 "$HEAD_HOST" "$HEAD_PORT" 2>/dev/null; then
        echo "[$(now)] Head reachable."
        break
    fi
    echo "[$(now)] Waiting for head ($i/5)..."
    sleep 5
done

trap 'echo "Watchdog stopped."; exit 0' INT TERM

connect_worker

# Watchdog loop
while true; do
    sleep 15
    if ! pgrep -f 'rayle[t]' >/dev/null 2>&1; then
        echo "[$(now)] RAYLET DIED - reconnecting..."
        connect_worker
        continue
    fi
    if ! nc -z -w4 "$HEAD_HOST" "$HEAD_PORT" 2>/dev/null; then
        echo "[$(now)] LOST CONNECTION to $HEAD_ADDR - reconnecting..."
        connect_worker
        continue
    fi
    echo "[$(now)] OK - connected to $HEAD_ADDR"
done
'@

$workerScript = $workerScript `
    -replace "PLACEHOLDER_HEAD_ADDR", "${NODE_A_TAILSCALE_IP}:${RAY_PORT}" `
    -replace "PLACEHOLDER_VENV",      $WSL_VENV `
    -replace "PLACEHOLDER_PROJECT",   $WSL_PROJECT

Write-UnixFile -Path "$WIN_TMP\ray_worker_watchdog.sh" -Content $workerScript

$wslScript = "$WSL_TMP/ray_worker_watchdog.sh"
wsl chmod +x $wslScript 2>$null

# Open a persistent WSL terminal window running the watchdog
Write-Log "  Opening persistent WSL2 worker window..."
if (Get-Command wt -ErrorAction SilentlyContinue) {
    Start-Process wt -ArgumentList "new-tab", "--title", """Node B - Ray Worker""", "--", "wsl.exe", "bash", $wslScript
} else {
    Start-Process cmd -ArgumentList "/k", "wsl bash $wslScript"
}

Write-Ok "Worker watchdog window opened."
Write-Log "  KEEP THAT WINDOW OPEN - closing it stops the Ray worker." "Yellow"

# Also attempt a direct first connect to verify quickly
Write-Log "  Verifying initial connection (30s)..." "Gray"
Start-Sleep 10

for ($i = 1; $i -le 6; $i++) {
    $nc = wsl bash -c "nc -z -w3 ${NODE_A_TAILSCALE_IP} ${RAY_PORT} 2>/dev/null && echo OK || echo FAIL" 2>&1
    if ($nc -match "OK") {
        Write-Ok "TCP connection to ${NODE_A_TAILSCALE_IP}:${RAY_PORT} verified"
        break
    }
    Write-Log "    Waiting for GCS connection ($i/6)..." "Gray"
    Start-Sleep 5
}


# ===========================================================================
#  STEP 6: Verify this node appears in the cluster
# ===========================================================================
Write-Step "STEP 6 - Cluster Membership Verification"

$verifyPy = @'
import ray, sys, os, time
os.environ['RAY_DISABLE_JEMALLOC'] = '1'
os.environ['LD_PRELOAD'] = ''
try:
    ray.init(address='PLACEHOLDER_ADDR', ignore_reinit_error=True, logging_level='ERROR')
    deadline = time.time() + PLACEHOLDER_TIMEOUT
    while time.time() < deadline:
        nodes = [n for n in ray.nodes() if n.get('Alive')]
        gpus  = sum(n.get('Resources', {}).get('GPU', 0) for n in nodes)
        if len(nodes) >= 2 and gpus >= 1:
            print(f"OK: {len(nodes)} nodes alive | GPU={gpus:.0f} | cluster healthy")
            sys.exit(0)
        print(f"  Waiting... alive={len(nodes)} gpu={gpus}", flush=True)
        time.sleep(5)
    nodes = [n for n in ray.nodes() if n.get('Alive')]
    gpus  = sum(n.get('Resources', {}).get('GPU', 0) for n in nodes)
    print(f"PARTIAL: {len(nodes)} node(s), {gpus:.0f} GPU(s)")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(2)
finally:
    os._exit(0)
'@

$verifyPy = $verifyPy `
    -replace "PLACEHOLDER_ADDR",    "${NODE_A_TAILSCALE_IP}:${RAY_PORT}" `
    -replace "PLACEHOLDER_TIMEOUT", $TIMEOUT_SECONDS

$verifyFile = "$WIN_TMP\verify_nodeb.py"
$enc = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($verifyFile, ($verifyPy -replace "`r`n", "`n"), $enc)

Write-Log "  Waiting up to ${TIMEOUT_SECONDS}s for this node to register..."
$wslVerify = "$WSL_TMP/verify_nodeb.py"
$pyBin  = "$WSL_VENV/bin/python3"
$verOut = wsl bash -c "'$pyBin' '$wslVerify' 2>&1" 2>&1

foreach ($line in $verOut) {
    $col = if ($line -match "^OK:") { "Green" } elseif ($line -match "^PARTIAL|^ERROR") { "Yellow" } else { "Gray" }
    Write-Log "    $line" $col
}

Write-Host ""
if ($verOut -match "^OK:") {
    Write-Host "  ================================================================" -ForegroundColor Green
    Write-Host "  |   NODE B JOINED - CLUSTER FULLY OPERATIONAL               |" -ForegroundColor Green
    Write-Host "  ================================================================" -ForegroundColor Green
} elseif ($verOut -match "PARTIAL") {
    Write-Host "  Node B is joining - check the worker window for progress." -ForegroundColor Yellow
    Write-Host "  Run verify_cluster.py on Node A to confirm when ready." -ForegroundColor Yellow
} else {
    Write-Host "  Could not verify cluster membership." -ForegroundColor Yellow
    Write-Host "  Check the worker watchdog window for error messages." -ForegroundColor Yellow
    Write-Host "  Common causes:" -ForegroundColor Yellow
    Write-Host "    - Node A GCS not yet ready (wait 30s and check)" -ForegroundColor Gray
    Write-Host "    - Wrong NODE_A_TAILSCALE_IP (check Tailscale app)" -ForegroundColor Gray
    Write-Host "    - GPU not visible in WSL2 (run: nvidia-smi)" -ForegroundColor Gray
}

Write-Host ""
Write-Log "Node B setup complete. Log: $LOG_FILE"
Read-Host "  Press Enter to close (worker window stays open)"
