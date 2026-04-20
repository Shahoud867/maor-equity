<#
.SYNOPSIS
    Ray Cluster Setup using Tailscale VPN.
    Tailscale gives both WSL2 instances stable 100.x.x.x IPs with direct
    peer-to-peer tunnels - no heartbeat drops, no reconnect loops needed.

.HOW TO USE
    STEP 1  Both PCs: install Tailscale for Windows from https://tailscale.com/download
            Sign in with GOOGLE or GITHUB - use the SAME account on both PCs.

    STEP 2  Node A PC:  .\ray_cluster.ps1 -Role A
    STEP 3  Node B PC:  .\ray_cluster.ps1 -Role B   (will prompt for Node A IP)
#>

param(
    [ValidateSet("A","B","")]
    [string]$Role        = "",
    [string]$TailscaleIP = ""   # Node B: paste Node A's 100.x.x.x IP here
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
#  CONFIG  (Node A paths - do not change)
# ---------------------------------------------------------------------------
$A_VENV    = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
$A_PROJECT = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity"

# Temp dir - user-writable, no admin needed
$WIN_TMP = "C:\Users\shaho\ray_cluster"
$WSL_TMP = "/mnt/c/Users/shaho/ray_cluster"

# ---------------------------------------------------------------------------
#  BANNER
# ---------------------------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "      maor-equity  --  Ray Cluster Setup (Tailscale VPN)     " -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Tailscale VPN: stable peer-to-peer tunnel, no heartbeat drops." -ForegroundColor Gray
Write-Host "  Tailscale gives both machines stable 100.x.x.x IPs with"   -ForegroundColor Gray
Write-Host "  direct encrypted tunnels. No more disconnections."          -ForegroundColor Gray
Write-Host ""

# ---------------------------------------------------------------------------
#  PRE-FLIGHT: Tailscale Windows check
# ---------------------------------------------------------------------------
Write-Host "  [CHECK] Tailscale on Windows..." -NoNewline
$tsExe = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path $tsExe)) {
    # Try common alternate path
    $tsExe = "$env:ProgramFiles\Tailscale\tailscale.exe"
}
if (Test-Path $tsExe) {
    Write-Host " found" -ForegroundColor Green
    try {
        $tsStatus = & $tsExe status --json 2>$null | ConvertFrom-Json
        $windowsTS_IP = $tsStatus.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
        if ($windowsTS_IP) {
            Write-Host "  [CHECK] Windows Tailscale IP: $windowsTS_IP" -ForegroundColor Green
        }
    } catch {}
} else {
    Write-Host " NOT INSTALLED" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "    1. Download Tailscale: https://tailscale.com/download/windows" -ForegroundColor White
    Write-Host "    2. Install and sign in (use same Google/GitHub account on BOTH PCs)" -ForegroundColor White
    Write-Host "    3. Re-run this script" -ForegroundColor White
    Write-Host ""
    $open = Read-Host "  Open Tailscale download page now? (Y/N)"
    if ($open -eq "Y" -or $open -eq "y") {
        Start-Process "https://tailscale.com/download/windows"
    }
    exit 1
}

# ---------------------------------------------------------------------------
#  WSL CHECK
# ---------------------------------------------------------------------------
Write-Host "  [CHECK] WSL..." -NoNewline
try {
    $null = wsl echo ok 2>&1
    Write-Host " found" -ForegroundColor Green
} catch {
    Write-Host " NOT FOUND" -ForegroundColor Red
    Write-Host "  Run in PowerShell Admin: wsl --install" -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
#  ROLE SELECTION
# ---------------------------------------------------------------------------
if (-not $Role) {
    Write-Host ""
    Write-Host "  Which node are you?" -ForegroundColor Yellow
    Write-Host "    A = Node A  (i232515 - CPU head node, your PC)"
    Write-Host "    B = Node B  (i232634 - GPU worker, partner PC)"
    Write-Host ""
    $Role = (Read-Host "  Enter A or B").Trim().ToUpper()
}
if ($Role -notin @("A","B")) {
    Write-Host "  ERROR: must be A or B" -ForegroundColor Red; exit 1
}

# ---------------------------------------------------------------------------
#  TEMP DIR
# ---------------------------------------------------------------------------
if (-not (Test-Path $WIN_TMP)) {
    New-Item -ItemType Directory -Path $WIN_TMP -Force | Out-Null
}

# ---------------------------------------------------------------------------
#  HELPER: write file with Unix line endings
# ---------------------------------------------------------------------------
function Write-UnixFile {
    param([string]$Path, [string]$Content)
    $utf8 = [System.Text.UTF8Encoding]::new($false)   # no BOM
    [System.IO.File]::WriteAllText($Path, ($Content -replace "`r`n","`n"), $utf8)
}

# ---------------------------------------------------------------------------
#  HELPER: open a new persistent WSL terminal window
# ---------------------------------------------------------------------------
function Open-WslWindow {
    param([string]$Title, [string]$Script)
    # Run the script file directly - no bash -c quoting issues
    if (Get-Command wt -ErrorAction SilentlyContinue) {
        Start-Process wt -ArgumentList "new-tab","--title","""$Title""","--","wsl.exe","bash",$Script
    } else {
        Start-Process cmd -ArgumentList "/k","wsl bash $Script"
    }
}

# ===========================================================================
#  NODE A
# ===========================================================================
if ($Role -eq "A") {

    Write-Host ""
    Write-Host "  [NODE A] Getting Tailscale IP from Windows..." -ForegroundColor Cyan

    # Get Node A's Windows Tailscale IP
    $TS_IP = ""
    try {
        $s = & $tsExe status --json 2>$null | ConvertFrom-Json
        $TS_IP = $s.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
    } catch {}

    if (-not $TS_IP) {
        Write-Host "  ERROR: Could not get Tailscale IP." -ForegroundColor Red
        Write-Host "  Make sure Tailscale is running and you are logged in." -ForegroundColor Yellow
        Write-Host "  Open Tailscale system tray icon and click 'Connect'." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "  Node A Tailscale IP: $TS_IP" -ForegroundColor Green
    Write-Host ""
    Write-Host "  [NODE A] Writing startup script to $WIN_TMP\nodeA.sh..." -ForegroundColor Cyan

    # -----------------------------------------------------------------------
    #  NODE A BASH SCRIPT
    # -----------------------------------------------------------------------
    $scriptA = @'
#!/usr/bin/env bash
# Node A: Ray head startup via Tailscale
# Auto-generated by ray_cluster.ps1

set -uo pipefail

VENV="PLACEHOLDER_VENV"
RAY="$VENV/bin/ray"
TS_IP="PLACEHOLDER_TS_IP"
WSL_TMP="PLACEHOLDER_WSL_TMP"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=60
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60

echo ""
echo "  ============================================================"
echo "  NODE A -- Ray Head (Tailscale IP: $TS_IP)"
echo "  ============================================================"
echo ""

# ── Verify Tailscale reachable ──────────────────────────────────────────
echo "  [1/4] Checking Tailscale..."
if ! ping -c1 -W2 "$TS_IP" >/dev/null 2>&1; then
    echo "  WARNING: Cannot ping Tailscale IP $TS_IP from WSL."
    echo "  This is OK on some systems -- continuing anyway."
fi
echo "  Tailscale IP: $TS_IP"

# ── Kill everything ─────────────────────────────────────────────────────
echo ""
echo "  [2/4] Stopping existing Ray..."
"$RAY" stop --force 2>/dev/null || true
sleep 1
pkill -9 -f gcs_server   2>/dev/null || true
pkill -9 -f raylet        2>/dev/null || true
pkill -9 -f plasma_store  2>/dev/null || true
sleep 2
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* \
       /tmp/ray_cluster_state.json 2>/dev/null || true
sleep 1

if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "  ERROR: port 6379 still occupied. Run: sudo fuser -k 6379/tcp"
    read -rp "  Press Enter to exit..." && exit 1
fi
echo "  Port 6379 free."

# ── Start Ray head ──────────────────────────────────────────────────────
echo ""
echo "  [3/4] Starting Ray head on $TS_IP:6379..."

"$RAY" start \
    --head \
    --port=6379 \
    --node-ip-address="$TS_IP" \
    --disable-usage-stats \
    --include-dashboard=false \
    --num-cpus=2 \
    --object-store-memory=268435456 \
    --plasma-directory=/tmp \
    2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: Ray failed to start."
    echo "  Close this window, reopen a fresh WSL terminal, and re-run."
    read -rp "  Press Enter..." && exit 1
fi

sleep 3

if ! ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "  ERROR: GCS did not bind to :6379"
    read -rp "  Press Enter..." && exit 1
fi

# Check raylet alive
RAYLET_COUNT=$(ps aux 2>/dev/null | grep '[r]aylet' | grep -v defunct | wc -l)
echo "  Ray GCS: UP on $TS_IP:6379"
echo "  Raylet processes: $RAYLET_COUNT"

# ── Print connection info ────────────────────────────────────────────────
echo ""
echo "  ============================================================"
echo "  NODE A IS READY -- Send this to Node B:"
echo "  ============================================================"
echo ""
echo "  PowerShell command for Node B:"
echo "    .\\ray_cluster.ps1 -Role B -TailscaleIP $TS_IP"
echo ""
echo "  OR manual WSL command for Node B:"
echo "    export LD_PRELOAD=''"
echo "    export RAY_DISABLE_JEMALLOC=1"
echo "    export CUDA_VISIBLE_DEVICES=0"
echo "    ray start --address=$TS_IP:6379 --num-gpus=1 --num-cpus=4"
echo ""
echo "  ============================================================"
echo "  KEEP THIS WINDOW OPEN -- closing it stops Ray"
echo "  ============================================================"
echo ""

# ── Monitor loop ────────────────────────────────────────────────────────
echo "  [4/4] Monitoring every 30s (Ctrl+C stops monitor, Ray stays up):"
echo "  ------------------------------------------------------------------"

while true; do
    sleep 30
    NOW=$(date '+%H:%M:%S')

    # Check GCS
    if ss -tlnp 2>/dev/null | grep -q ':6379'; then
        GCS_STATUS="Ray:UP"
    else
        GCS_STATUS="Ray:DOWN"
    fi

    # Count nodes + write state file
    NODES=$("$VENV/bin/python3" -c "
import ray, os, sys, json, time
os.environ['RAY_DISABLE_JEMALLOC']='1'
os.environ['LD_PRELOAD']=''
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']='0'
try:
    ray.init(address='${TS_IP}:6379', ignore_reinit_error=True, logging_level='ERROR')
    nodes = ray.nodes()
    cr    = dict(ray.cluster_resources())
    alive = [n for n in nodes if n.get('Alive')]
    state = {
        'ts':      time.time(),
        'alive':   [{'ip': n.get('NodeManagerAddress',''), 'cpu': n.get('Resources',{}).get('CPU',0), 'gpu': n.get('Resources',{}).get('GPU',0)} for n in alive],
        'dead_ips':[n.get('NodeManagerAddress','') for n in nodes if not n.get('Alive')],
        'cr':      {k: float(v) for k, v in cr.items() if isinstance(v,(int,float))},
        'cr_keys': list(cr.keys()),
    }
    with open('/tmp/ray_cluster_state.json','w') as f: json.dump(state,f)
    sys.stdout.write(str(len(alive))+'\n')
    sys.stdout.flush()
except Exception as e:
    sys.stdout.write('?\n')
    sys.stdout.flush()
finally:
    os._exit(0)
" 2>/dev/null)

    echo "  [$NOW] $GCS_STATUS | Alive nodes: $NODES | Head: $TS_IP:6379"
done

echo ""
read -rp "  === Script ended. Press Enter to close ==="
'@

    # Inject variables
    $scriptA = $scriptA `
        -replace 'PLACEHOLDER_VENV',    $A_VENV `
        -replace 'PLACEHOLDER_TS_IP',   $TS_IP `
        -replace 'PLACEHOLDER_WSL_TMP', $WSL_TMP

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
    Write-Host "  After Node B connects, verify (in a new PowerShell tab):" -ForegroundColor Cyan
    Write-Host "    wsl bash -c `"cd '$A_PROJECT' && source venv/bin/activate && python verify_cluster.py`"" -ForegroundColor White
    Write-Host "  ============================================================" -ForegroundColor Green
}

# ===========================================================================
#  NODE B
# ===========================================================================
if ($Role -eq "B") {

    # ── Get Node A's Tailscale IP ──────────────────────────────────────────
    if (-not $TailscaleIP) {
        Write-Host ""
        Write-Host "  Enter Node A's Tailscale IP (shown in Node A's terminal)." -ForegroundColor Yellow
        Write-Host "  It looks like: 100.x.x.x" -ForegroundColor Gray
        Write-Host ""
        $TailscaleIP = (Read-Host "  Node A Tailscale IP").Trim()
    }

    if ($TailscaleIP -notmatch '^100\.\d+\.\d+\.\d+$') {
        Write-Host "  ERROR: Tailscale IPs start with 100. Got: $TailscaleIP" -ForegroundColor Red
        exit 1
    }

    # ── Find Ray binary in WSL ─────────────────────────────────────────────
    Write-Host ""
    Write-Host "  [NODE B] Finding Ray in WSL..." -NoNewline

    $rayBin = wsl bash -c @"
for p in ~/maor-equity/venv/bin/ray ~/venv/bin/ray /usr/local/bin/ray; do
    [ -x "`$p" ] && echo "`$p" && exit 0
done
which ray 2>/dev/null || true
"@ 2>$null
    $rayBin = ($rayBin | Where-Object { $_ -match '/ray' } | Select-Object -First 1).Trim()

    if (-not $rayBin) {
        Write-Host " NOT FOUND" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Ray not found. In WSL, run:" -ForegroundColor Yellow
        Write-Host "    git clone https://github.com/Shahoud867/maor-equity ~/maor-equity" -ForegroundColor White
        Write-Host "    cd ~/maor-equity" -ForegroundColor White
        Write-Host "    python3 -m venv venv && source venv/bin/activate" -ForegroundColor White
        Write-Host "    pip install ray[default]" -ForegroundColor White
        exit 1
    }

    Write-Host " $rayBin" -ForegroundColor Green

    Write-Host "  [NODE B] Writing connection script..." -ForegroundColor Cyan

    # -----------------------------------------------------------------------
    #  NODE B BASH SCRIPT
    # -----------------------------------------------------------------------
    $scriptB = @"
#!/usr/bin/env bash
# Node B: Ray worker - connects to Node A via Tailscale
# Auto-generated by ray_cluster.ps1

HEAD_IP="$TailscaleIP"
RAY="$rayBin"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "  ============================================================"
echo "  NODE B -- Ray Worker (connecting to `$HEAD_IP:6379)"
echo "  ============================================================"
echo ""

# ── Check Tailscale connectivity ─────────────────────────────────────────
echo "  [1/3] Checking connectivity to Node A (`$HEAD_IP)..."
if ping -c2 -W3 "`$HEAD_IP" >/dev/null 2>&1; then
    echo "  Ping OK - Tailscale tunnel is up"
else
    echo "  WARNING: Cannot ping `$HEAD_IP"
    echo "  Make sure:"
    echo "    1. Tailscale is running on BOTH PCs (system tray icon)"
    echo "    2. Both are signed into the SAME Tailscale account"
    echo "    3. Node A's ray_cluster.ps1 is still running"
    read -rp "  Press Enter to try anyway, or Ctrl+C to cancel..."
fi

# ── Stop existing Ray ────────────────────────────────────────────────────
echo ""
echo "  [2/3] Stopping existing Ray..."
"`$RAY" stop --force 2>/dev/null || true
sleep 2

# ── Connect to head ──────────────────────────────────────────────────────
echo "  [3/3] Connecting to Ray head at `$HEAD_IP:6379..."

connect_worker() {
    "`$RAY" stop --force 2>/dev/null || true
    sleep 2
    "`$RAY" start \
        --address="`$HEAD_IP:6379" \
        --num-gpus=1 \
        --num-cpus=4 \
        2>&1
}

connect_worker

echo ""
echo "  ============================================================"
echo "  CONNECTED! Watching for disconnects every 10s..."
echo "  ============================================================"
echo ""

# Watch loop - NEVER calls ray.init() or ray.shutdown()
# Only checks: (1) is raylet process alive? (2) can we TCP to head?
while true; do
    sleep 10
    NOW=`$(date '+%H:%M:%S')

    # Check 1: raylet process
    if ! pgrep -f raylet >/dev/null 2>&1; then
        echo "  [`$NOW] RAYLET DIED - reconnecting..."
        connect_worker
        continue
    fi

    # Check 2: TCP to Node A Tailscale IP
    if ! nc -z -w4 "`$HEAD_IP" 6379 2>/dev/null; then
        echo "  [`$NOW] LOST CONNECTION to `$HEAD_IP:6379 - reconnecting..."
        connect_worker
        continue
    fi

    echo "  [`$NOW] OK - raylet alive, connected to `$HEAD_IP:6379"
done

echo ""
read -rp "  === Script ended. Press Enter to close ==="
"@

    Write-UnixFile -Path "$WIN_TMP\nodeB.sh" -Content $scriptB
    wsl chmod +x "$WSL_TMP/nodeB.sh" 2>$null

    Open-WslWindow -Title "NODE B - Ray Worker" -Script "$WSL_TMP/nodeB.sh"

    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Host "  Node B connection window opened!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Watch that window -- it will print every 10s:" -ForegroundColor Cyan
    Write-Host "    OK - raylet alive, connected to $TailscaleIP:6379" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  If it shows RECONNECTING that is normal and automatic." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Tell Node A to run verify_cluster.py after ~30s." -ForegroundColor Cyan
    Write-Host "  ============================================================" -ForegroundColor Green
}

Write-Host ""
Write-Host "  Done. Keep the WSL terminal window open." -ForegroundColor Cyan
Write-Host ""
