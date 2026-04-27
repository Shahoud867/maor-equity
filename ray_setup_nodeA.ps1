<#
.SYNOPSIS
    Ray Cluster Head Node Setup - Node A (Tailscale + WSL2)

.PREREQUISITES
    - Tailscale installed and signed in on BOTH machines (same account)
      Download: https://tailscale.com/download/windows
    - WSL2 Ubuntu with Python venv containing Ray, transformers, bitsandbytes
    - Node B has this repo cloned at the path in ray_setup_nodeB.ps1

.USAGE
    1. Edit the CONFIG block below (at minimum set $NODE_B_TAILSCALE_IP)
    2. Run AS ADMINISTRATOR (script self-elevates if needed):
         powershell -ExecutionPolicy Bypass -File .\ray_setup_nodeA.ps1
    3. After "NODE A IS READY" appears, tell Node B to run ray_setup_nodeB.ps1

.WHAT THIS SCRIPT DOES
    Step 1  Auto-detect Node A Tailscale IP, validate Tailscale, ping Node B
    Step 2  Kill all stale Ray processes (Windows + WSL2) without ray stop
    Step 3  Clean Ray temp files and directories
    Step 4  Add Tailscale IP as WSL2 loopback alias (CRITICAL: prevents GCS timeout)
    Step 5  Configure Windows Firewall (inbound + outbound)
    Step 6  Reset portproxy - add client port 10001 only (NOT 6379 before ray start)
    Step 7  Start Ray head in WSL2, wait for GCS to bind
    Step 8  Add portproxy for 6379 and dynamic raylet ports AFTER ray start
    Step 9  Display Node B join command and wait for cluster to become healthy
    Step 10 Verify 2-node cluster and print next-step pipeline command

.KNOWN ISSUES HANDLED
    - portproxy for 6379 before ray start causes "Address already in use" (fixed: add AFTER)
    - ray stop --force kills parent bash via process group (fixed: kill by PID only)
    - Tailscale userspace routing adds latency to self-connections (fixed: loopback alias)
    - WSL2 sudo requires interactive TTY (fixed: wsl -u root for all root ops)
    - pkill -f patterns match own script text (fixed: character-class obfuscation)
    - CRLF line endings break bash scripts (fixed: Write-UnixFile with LF only)
#>

# ===========================================================================
#  CONFIG - edit before running
# ===========================================================================
$NODE_A_TAILSCALE_IP = ""               # Leave empty to auto-detect
$NODE_B_TAILSCALE_IP = "100.95.214.76"  # Node B's Tailscale IP (100.x.x.x)
$RAY_PORT            = 6379
$RAY_CLIENT_PORT     = 10001
$RETRY_COUNT         = 3
$TIMEOUT_SECONDS     = 120
$LOG_FILE            = "$env:USERPROFILE\ray_setup_nodeA.log"

# WSL2 paths - adjust if your project/venv is elsewhere
$WSL_VENV    = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
$WSL_PROJECT = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity"

# Tailscale executable
$TAILSCALE_EXE = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path $TAILSCALE_EXE)) {
    $TAILSCALE_EXE = "$env:ProgramFiles\Tailscale\tailscale.exe"
}

# ===========================================================================
#  SELF-ELEVATION  (re-launch as admin if needed)
# ===========================================================================
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
               [Security.Principal.WindowsBuiltInRole]"Administrator")
if (-not $isAdmin) {
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    Start-Process powershell -Verb RunAs -ArgumentList $a
    exit
}

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

# ===========================================================================
#  LOGGING HELPERS
# ===========================================================================
function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}
function Write-Ok    { param([string]$M) Write-Log "  OK  $M" "Green"  }
function Write-Warn  { param([string]$M) Write-Log "  !!  $M" "Yellow" }
function Write-Err   { param([string]$M) Write-Log "  XX  $M" "Red"    }
function Write-Step  {
    param([string]$T)
    Write-Log ""
    Write-Log ("=" * 64) "Cyan"
    Write-Log "  $T" "Cyan"
    Write-Log ("=" * 64) "Cyan"
}
function Write-UnixFile {
    param([string]$Path, [string]$Content)
    $enc = [System.Text.UTF8Encoding]::new($false)  # UTF-8 no BOM
    [System.IO.File]::WriteAllText($Path, ($Content -replace "`r`n", "`n"), $enc)
}
function Wait-ForCondition {
    param([scriptblock]$Test, [int]$TimeoutSec, [int]$PollSec = 3, [string]$Label)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (& $Test) { return $true }
        Write-Log "    Waiting for $Label..." "Gray"
        Start-Sleep $PollSec
    }
    return $false
}

# Init log
"" | Out-File $LOG_FILE -Encoding UTF8
Write-Log "Ray Node A Setup - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Log file: $LOG_FILE"

$WIN_TMP = "$env:TEMP\ray_cluster_setup"
if (-not (Test-Path $WIN_TMP)) { New-Item -ItemType Directory -Path $WIN_TMP -Force | Out-Null }


# ===========================================================================
#  STEP 1: Tailscale validation
# ===========================================================================
Write-Step "STEP 1 - Tailscale Validation"

# Auto-detect Node A IP
if (-not $NODE_A_TAILSCALE_IP) {
    try {
        $s = & $TAILSCALE_EXE status --json 2>$null | ConvertFrom-Json
        $NODE_A_TAILSCALE_IP = $s.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
    } catch { }
}
if (-not $NODE_A_TAILSCALE_IP) {
    Write-Err "Cannot detect Tailscale IP. Is Tailscale installed and connected?"
    Write-Err "Download: https://tailscale.com/download/windows"
    Read-Host "Press Enter to exit"; exit 1
}
Write-Ok "Node A Tailscale IP: $NODE_A_TAILSCALE_IP"

# Ensure Tailscale service is running
$ts = Get-Service -Name Tailscale -ErrorAction SilentlyContinue
if ($ts -and $ts.Status -ne "Running") {
    Write-Warn "Tailscale service stopped - restarting..."
    Start-Service Tailscale -ErrorAction SilentlyContinue
    Start-Sleep 4
}

# Ping Node B
Write-Log "  Pinging Node B ($NODE_B_TAILSCALE_IP)..."
$nodeBReachable = $false
for ($i = 1; $i -le $RETRY_COUNT; $i++) {
    $r = ping -n 2 -w 3000 $NODE_B_TAILSCALE_IP 2>&1
    if ($r -match "bytes=") { $nodeBReachable = $true; break }
    Write-Warn "Ping attempt $i/$RETRY_COUNT failed..."
    Start-Sleep 3
}
if ($nodeBReachable) {
    Write-Ok  "Node B ($NODE_B_TAILSCALE_IP) is reachable"
} else {
    Write-Warn "Node B unreachable - it may not be online yet."
    Write-Warn "Continuing... Node B must run ray_setup_nodeB.ps1 after this completes."
}


# ===========================================================================
#  STEP 2: Kill stale Ray processes - Windows
# ===========================================================================
Write-Step "STEP 2 - Kill Stale Windows Ray Processes"

$winProcs = @("gcs_server", "raylet", "plasma_store")
foreach ($p in $winProcs) {
    try { Stop-Process -Name $p -Force -ErrorAction SilentlyContinue } catch { }
}
# Kill python processes holding Ray ports
$portUsers = @()
try {
    $portUsers = Get-NetTCPConnection -LocalPort $RAY_PORT -ErrorAction SilentlyContinue |
                 Select-Object -ExpandProperty OwningProcess -Unique
} catch { }
foreach ($pid in $portUsers) {
    if ($pid -gt 0) {
        try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch { }
        Write-Warn "Killed Windows PID $pid holding port $RAY_PORT"
    }
}
Write-Ok "Windows Ray process cleanup done"


# ===========================================================================
#  STEP 3: Kill stale Ray processes - WSL2  +  clean temp dirs
# ===========================================================================
Write-Step "STEP 3 - WSL2 Ray Cleanup"

# IMPORTANT: Use 'wsl -u root' to avoid:
#   (a) sudo password prompt (no TTY in non-interactive wsl calls)
#   (b) ray stop --force killing our own process via process-group SIGKILL
#   (c) pkill -f matching our own script text
# Character classes in pgrep patterns prevent self-match.
$wslCleanup = @'
pgrep -f 'gc[s]_server'   2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'rayle[t]'        2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'plasma_stor[e]'  2>/dev/null | xargs -r kill -9 2>/dev/null || true
pgrep -f 'monitor\.p[y]'   2>/dev/null | xargs -r kill -9 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true
sleep 1
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* 2>/dev/null || true
echo WSLCLEAN_OK
'@
$r = wsl -u root bash -c $wslCleanup 2>&1
if ($r -match "WSLCLEAN_OK") { Write-Ok "WSL2 Ray processes and /tmp/ray cleared" }
else { Write-Warn "WSL2 cleanup output: $r" }

# Windows temp dirs
foreach ($d in @("$env:TEMP\ray", "$env:LOCALAPPDATA\ray", "$env:USERPROFILE\.ray")) {
    if (Test-Path $d) { Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue }
}
Write-Ok "Windows temp dirs cleared"


# ===========================================================================
#  STEP 4: WSL2 loopback alias  (THE critical fix)
# ===========================================================================
Write-Step "STEP 4 - WSL2 Loopback Alias for Tailscale IP"

# WHY: When Ray binds GCS to the Tailscale IP (100.x.x.x) in WSL2, the raylet
# on the same machine connects back to that IP. Without this alias, the connection
# routes through Tailscale's userspace WireGuard stack - which adds ~100-400ms
# RTT even for local connections, causing the raylet to miss the 60s GCS timeout.
# Adding the IP to 'lo' (loopback) makes WSL2 handle it as a pure kernel operation
# with <1ms latency. 'wsl -u root' avoids the sudo-password-in-no-TTY hang.

$addAlias = wsl -u root bash -c "ip addr add ${NODE_A_TAILSCALE_IP}/32 dev lo 2>&1; echo ALIAS_DONE" 2>&1
$verAlias = wsl -u root bash -c "ip addr show lo 2>/dev/null | grep '${NODE_A_TAILSCALE_IP}'" 2>&1
if ($verAlias -match $NODE_A_TAILSCALE_IP) {
    Write-Ok "Loopback alias ${NODE_A_TAILSCALE_IP}/32 active on WSL2 lo"
} else {
    Write-Err "Failed to add loopback alias. GCS may timeout. Output: $addAlias"
}


# ===========================================================================
#  STEP 5: Windows Firewall
# ===========================================================================
Write-Step "STEP 5 - Windows Firewall"

netsh advfirewall firewall delete rule name="Ray-Cluster-In"  >$null 2>&1
netsh advfirewall firewall delete rule name="Ray-Cluster-Out" >$null 2>&1
netsh advfirewall firewall add rule name="Ray-Cluster-In" `
    dir=in action=allow protocol=TCP `
    localport="6379,8265,10001,20000-29999" profile=any | Out-Null
netsh advfirewall firewall add rule name="Ray-Cluster-Out" `
    dir=out action=allow protocol=TCP `
    remoteport="6379,8265,10001,20000-29999" profile=any | Out-Null
Write-Ok "Firewall rules set (inbound + outbound: 6379, 10001, 20000-29999)"


# ===========================================================================
#  STEP 6: Portproxy - clean slate, add 10001 only (NOT 6379 yet)
# ===========================================================================
Write-Step "STEP 6 - Port Proxy (pre-start)"

# WHY: Adding portproxy for port 6379 BEFORE ray start causes GCS to fail with
# "Address already in use" because WSL2 mirrored networking makes the Windows
# portproxy listener visible inside WSL2 - exactly where GCS tries to bind.
# We add 10001 now (safe), and add 6379 in Step 8 AFTER ray start.
netsh interface portproxy reset >$null 2>&1

netsh interface portproxy add v4tov4 `
    listenaddress=$NODE_A_TAILSCALE_IP listenport=$RAY_CLIENT_PORT `
    connectaddress=127.0.0.1 connectport=$RAY_CLIENT_PORT | Out-Null

Set-Service -Name iphlpsvc -StartupType Automatic -ErrorAction SilentlyContinue
Start-Service -Name iphlpsvc -ErrorAction SilentlyContinue

Write-Ok "Portproxy: ${NODE_A_TAILSCALE_IP}:${RAY_CLIENT_PORT} -> 127.0.0.1:${RAY_CLIENT_PORT}"
Write-Log "  (6379 portproxy added AFTER ray start in Step 8)" "Gray"


# ===========================================================================
#  STEP 7: Start Ray head node in WSL2
# ===========================================================================
Write-Step "STEP 7 - Start Ray Head Node"

# Write bash startup script with LF line endings (CRLF breaks bash)
# Placeholders are replaced after the heredoc - avoids PowerShell/bash
# variable-substitution conflicts inside the double-quoted PS string.
$headScript = @'
#!/usr/bin/env bash
TS_IP=PLACEHOLDER_TS_IP
VENV="PLACEHOLDER_VENV"
RAY="$VENV/bin/ray"
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=300
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60

echo "[head] Checking port 6379..."
if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    fuser -k 6379/tcp 2>/dev/null || true
    sleep 2
fi
if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "[head] ERROR: port 6379 still occupied after fuser"
    ss -tlnp | grep ':6379'
    exit 1
fi
echo "[head] Port 6379 free. Starting Ray head on ${TS_IP}:6379..."

"$RAY" start \
    --head \
    --port=6379 \
    --node-ip-address="$TS_IP" \
    --disable-usage-stats \
    --include-dashboard=false \
    --num-cpus=2 \
    2>&1
RC=$?
echo "[head] ray start exit code: $RC"

if [ "$RC" -ne 0 ]; then
    echo "[head] FAILED"
    exit "$RC"
fi

sleep 2
if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "[head] GCS_UP"
else
    echo "[head] WARNING: GCS not on :6379 after start"
fi
'@

$headScript = $headScript `
    -replace "PLACEHOLDER_TS_IP",   $NODE_A_TAILSCALE_IP `
    -replace "PLACEHOLDER_VENV",    $WSL_VENV `
    -replace "PLACEHOLDER_TIMEOUT", $TIMEOUT_SECONDS

Write-UnixFile -Path "$WIN_TMP\ray_head_start.sh" -Content $headScript

# Convert Windows path to WSL path for this user
$_drive   = ($env:TEMP -split ":\\")[0].ToLower()
$_relpath = ($env:TEMP -split ":\\")[1] -replace "\\", "/"
$WSL_TMP  = "/mnt/$_drive/$_relpath/ray_cluster_setup"
$wslScript = "$WSL_TMP/ray_head_start.sh"

wsl chmod +x $wslScript 2>$null

Write-Log "  Running Ray head startup (timeout 300s)..." "Gray"
$startOutput = wsl bash $wslScript 2>&1

$failed = $false
foreach ($line in $startOutput) {
    $col = if ($line -match "ERROR|FAILED|Exception|Traceback") { "Red" }
           elseif ($line -match "GCS_UP|READY|UP")             { "Green" }
           elseif ($line -match "WARNING|warn")                 { "Yellow" }
           else { "Gray" }
    Write-Log "    [WSL] $line" $col
    if ($line -match "FAILED|Exception: The current node") { $failed = $true }
}

if ($failed) {
    Write-Err "Ray head startup FAILED. Check log: $LOG_FILE"
    Write-Err "Common causes:"
    Write-Err "  - Loopback alias not set (Step 4 failed)"
    Write-Err "  - Port 6379 still occupied (rerun script)"
    Write-Err "  - WSL2 venv path wrong (check WSL_VENV in CONFIG)"
    Read-Host "  Press Enter to exit"; exit 1
}


# ===========================================================================
#  STEP 8: Verify GCS up, then add post-start portproxy
# ===========================================================================
Write-Step "STEP 8 - GCS Verification + Post-Start Portproxy"

$gcsUp = Wait-ForCondition -Test {
    $r = wsl bash -c "ss -tlnp 2>/dev/null | grep ':6379'" 2>&1
    $r -match ":6379"
} -TimeoutSec 30 -PollSec 2 -Label "GCS on :6379"

if (-not $gcsUp) {
    Write-Err "GCS did not bind to :6379 within 30s"
    Read-Host "Press Enter to exit"; exit 1
}
Write-Ok "GCS is UP on ${NODE_A_TAILSCALE_IP}:6379"

# NOW safe to add portproxy for 6379 (ray already owns the port)
netsh interface portproxy delete v4tov4 listenaddress=$NODE_A_TAILSCALE_IP listenport=$RAY_PORT >$null 2>&1
netsh interface portproxy add    v4tov4 listenaddress=$NODE_A_TAILSCALE_IP listenport=$RAY_PORT `
    connectaddress=127.0.0.1 connectport=$RAY_PORT | Out-Null
Write-Ok "Portproxy added: ${NODE_A_TAILSCALE_IP}:${RAY_PORT} -> 127.0.0.1:${RAY_PORT}"

# Dynamic raylet / object-store ports (give raylet a moment to bind)
Start-Sleep 4
$rayletPort = (wsl bash -c "ps aux 2>/dev/null | grep '[r]aylet' | grep -oP '\-\-node_manager_port=\K[0-9]+' | head -1" 2>&1).Trim()
$objPort    = (wsl bash -c "ps aux 2>/dev/null | grep '[r]aylet' | grep -oP '\-\-object_manager_port=\K[0-9]+' | head -1" 2>&1).Trim()

foreach ($port in @($rayletPort, $objPort)) {
    if ($port -match "^\d+$") {
        netsh interface portproxy delete v4tov4 listenaddress=$NODE_A_TAILSCALE_IP listenport=$port >$null 2>&1
        netsh interface portproxy add    v4tov4 listenaddress=$NODE_A_TAILSCALE_IP listenport=$port `
            connectaddress=127.0.0.1 connectport=$port | Out-Null
        Write-Ok "Portproxy added: ${NODE_A_TAILSCALE_IP}:${port} -> 127.0.0.1:${port}"
    }
}

Write-Log ""
Write-Log "  Current portproxy table:" "Cyan"
netsh interface portproxy show v4tov4 2>&1 | ForEach-Object { Write-Log "    $_" "Gray" }


# ===========================================================================
#  STEP 9: Node B join instructions
# ===========================================================================
Write-Step "STEP 9 - Node B Join Instructions"

Write-Host ""
Write-Host "  ================================================================" -ForegroundColor Green
Write-Host "  |           NODE A IS READY - WAITING FOR NODE B          |" -ForegroundColor Green
Write-Host "  ================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Node A Tailscale IP : $NODE_A_TAILSCALE_IP" -ForegroundColor Yellow
Write-Host "  GCS Address         : ${NODE_A_TAILSCALE_IP}:${RAY_PORT}" -ForegroundColor Yellow
Write-Host ""
Write-Host "  On Node B - run this PowerShell script:" -ForegroundColor Cyan
Write-Host "    .\ray_setup_nodeB.ps1 -NodeAIP $NODE_A_TAILSCALE_IP" -ForegroundColor White
Write-Host ""
Write-Host "  Or in WSL2 on Node B (manual):" -ForegroundColor Cyan
Write-Host "    export PYTHONPATH=/path/to/maor-equity" -ForegroundColor Gray
Write-Host "    ray start --address=${NODE_A_TAILSCALE_IP}:${RAY_PORT} --num-gpus=1 --num-cpus=4" -ForegroundColor White
Write-Host ""


# ===========================================================================
#  STEP 10: Wait for Node B and verify cluster health
# ===========================================================================
Write-Step "STEP 10 - Cluster Verification"

$verifyPy = @'
import ray, sys, os, time
os.environ['RAY_DISABLE_JEMALLOC'] = '1'
os.environ['LD_PRELOAD'] = ''
try:
    ray.init(address='PLACEHOLDER_ADDR', ignore_reinit_error=True, logging_level='ERROR')
    deadline = time.time() + PLACEHOLDER_TIMEOUT
    while time.time() < deadline:
        nodes  = [n for n in ray.nodes() if n.get('Alive')]
        gpus   = sum(n.get('Resources', {}).get('GPU', 0) for n in nodes)
        print(f"  Alive={len(nodes)} GPU={gpus}", flush=True)
        if len(nodes) >= 2 and gpus >= 1:
            cr = ray.cluster_resources()
            print(f"OK: {len(nodes)} nodes | GPU={gpus} | CPU={cr.get('CPU',0):.0f}")
            sys.exit(0)
        time.sleep(5)
    nodes = [n for n in ray.nodes() if n.get('Alive')]
    gpus  = sum(n.get('Resources', {}).get('GPU', 0) for n in nodes)
    print(f"PARTIAL: {len(nodes)} node(s) alive, {gpus} GPU(s) - expected 2 nodes + 1 GPU")
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

$verifyFile = "$WIN_TMP\verify_cluster.py"
$enc = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($verifyFile, ($verifyPy -replace "`r`n", "`n"), $enc)

Write-Log "  Waiting up to ${TIMEOUT_SECONDS}s for Node B to join..."
$wslVerify = "$WSL_TMP/verify_cluster.py"
$pyBin     = "$WSL_VENV/bin/python3"
$verOut    = wsl bash -c "'$pyBin' '$wslVerify' 2>&1" 2>&1

foreach ($line in $verOut) {
    $col = if ($line -match "^OK:") { "Green" } elseif ($line -match "^PARTIAL|^ERROR") { "Yellow" } else { "Gray" }
    Write-Log "    $line" $col
}

Write-Host ""
if ($verOut -match "^OK:") {
    Write-Host "  ================================================================" -ForegroundColor Green
    Write-Host "  |        RAY CLUSTER FULLY OPERATIONAL                       |" -ForegroundColor Green
    Write-Host "  ================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Run the pipeline:" -ForegroundColor Cyan
    Write-Host "    wsl bash -c `"cd '$WSL_PROJECT' && source venv/bin/activate && python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json`"" -ForegroundColor White
} elseif ($verOut -match "PARTIAL") {
    Write-Host "  Cluster partially ready - Node B may still be joining." -ForegroundColor Yellow
    Write-Host "  Run verify_cluster.py manually to recheck:" -ForegroundColor Yellow
    Write-Host "    wsl bash -c `"cd '$WSL_PROJECT' && source venv/bin/activate && python verify_cluster.py`"" -ForegroundColor White
} else {
    Write-Host "  Node B has not joined yet or verification failed." -ForegroundColor Yellow
    Write-Host "  Ensure Node B runs: .\ray_setup_nodeB.ps1 -NodeAIP $NODE_A_TAILSCALE_IP" -ForegroundColor Yellow
}

Write-Host ""
Write-Log "Setup complete. Full log: $LOG_FILE"
Read-Host "  Press Enter to close"
