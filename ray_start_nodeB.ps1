<#
.SYNOPSIS
    Ray Worker Node (Node B) -- fully automated startup script.

.DESCRIPTION
    Joins the Ray cluster started by Node A (ray_start_nodeA.ps1) on Node B (GPU worker).
    Handles all known failure modes:

    FIX-1  PYTHONPATH for Ray workers.
           Ray spawns worker processes that don't inherit the calling shell's PATH.
           The project root must be in PYTHONPATH so workers can import 'agents.*'.

    FIX-2  PID-based Ray cleanup.
           "ray stop --force" kills its own process group (including the parent shell).
           We kill Ray processes individually by PID instead.

    FIX-3  Prevents duplicate Node B connection (GPU: 2.0 symptom).
           Ensures all previous Ray worker processes are dead before connecting.
           A second connection from the same machine causes GPU resource doubling
           and OOM errors on the T1000.

    FIX-4  CUDA_VISIBLE_DEVICES=0 forces GPU assignment.

.USAGE
    # Run from PowerShell on Node B (will self-elevate to Admin):
        .\ray_start_nodeB.ps1 -HeadIP 100.88.99.30

    # If you don't know the head IP, just run without arguments (will prompt):
        .\ray_start_nodeB.ps1

.PREREQUISITES (one-time)
    1. Tailscale installed and signed into the SAME account as Node A.
    2. WSL2 with Ubuntu, CUDA drivers, and project venv set up at B_VENV below.
    3. bitsandbytes, transformers, ray[default] installed in venv.
    4. Run once as Admin for firewall rules.

.NOTES
    Node B machine : i232634 (GPU worker, NVIDIA T1000 4GB)
    GPU memory     : 4096 MB  --  FinBERTBundle (~525 MB) + Phi-3-mini NF4 (~2736 MB)
                                   = ~3261 MB used, ~835 MB headroom
#>

param(
    [string]$HeadIP = ""   # Node A Tailscale IP, e.g. 100.88.99.30
)

# ============================================================
#  SELF-ELEVATE
# ============================================================
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]"Administrator")
if (-not $isAdmin) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($HeadIP) { $argList += "-HeadIP"; $argList += $HeadIP }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

# ============================================================
#  CONFIG  --  edit these if the project moves on Node B
# ============================================================
$CFG = @{
    B_VENV    = "/mnt/d/University Work/Semester 6/NLP + PDC Project/maor-equity/venv"
    B_PROJECT = "/mnt/d/University Work/Semester 6/NLP + PDC Project/maor-equity"

    TailscaleExe  = "C:\Program Files\Tailscale\tailscale.exe"
    WinTemp       = "$env:TEMP\ray_cluster"
    LogFile       = "$env:TEMP\ray_cluster\nodeB_startup.log"
    FirewallPorts = "6380,8265,10001,20000-29999"
}

# ============================================================
#  LOGGING
# ============================================================
if (-not (Test-Path $CFG.WinTemp)) { New-Item -ItemType Directory -Path $CFG.WinTemp -Force | Out-Null }

function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Msg"
    Add-Content -Path $CFG.LogFile -Value $line -Encoding UTF8
    Write-Host $line -ForegroundColor $Color
}
function Write-Step { param([string]$Msg) Write-Log ""; Write-Log "  ── $Msg" "Cyan" }

$Results = [ordered]@{}
function Set-Result { param([string]$Key, [bool]$Ok, [string]$Detail = "") $Results[$Key] = @{Ok=$Ok; Detail=$Detail} }

# ============================================================
#  BANNER
# ============================================================
Clear-Host
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "   maor-equity  ·  Node B (GPU Worker) Startup  ·  v2.0      " -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Log "Log: $($CFG.LogFile)" "Gray"

# ============================================================
#  STEP 1 -- TAILSCALE VALIDATION
# ============================================================
Write-Step "STEP 1/6: Tailscale validation"

$MY_IP = ""
try {
    if (-not (Test-Path $CFG.TailscaleExe)) { throw "Tailscale not found at $($CFG.TailscaleExe)" }
    $tsJson = & $CFG.TailscaleExe status --json 2>$null | ConvertFrom-Json
    $MY_IP = $tsJson.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
    if (-not $MY_IP) { throw "Could not read Tailscale IP. Is Tailscale connected?" }
    Write-Log "  My Tailscale IP (Node B): $MY_IP" "Green"
    Set-Result "Tailscale" $true $MY_IP
} catch {
    Write-Log "  ERROR: $_" "Red"
    Write-Log "  ACTION: Open Tailscale tray -> Connect, then re-run." "Yellow"
    Set-Result "Tailscale" $false "$_"
    exit 1
}

# ============================================================
#  STEP 2 -- GET + VALIDATE HEAD IP
# ============================================================
Write-Step "STEP 2/6: Locate Ray head (Node A)"

if (-not $HeadIP) {
    Write-Host ""
    Write-Host "  Enter Node A's Tailscale IP (shown in Node A's terminal, looks like 100.x.x.x):" -ForegroundColor Yellow
    $HeadIP = (Read-Host "  Head IP").Trim()
}

if ($HeadIP -notmatch '^100\.\d+\.\d+\.\d+$') {
    Write-Log "  ERROR: '$HeadIP' is not a valid Tailscale IP (expected 100.x.x.x)" "Red"
    exit 1
}

Write-Log "  Head IP: $HeadIP" "Green"

# Ping test
$pingOk = Test-Connection -ComputerName $HeadIP -Count 3 -Quiet -ErrorAction SilentlyContinue
if ($pingOk) {
    Write-Log "  Node A reachable via ping." "Green"
    Set-Result "PingNodeA" $true
} else {
    Write-Log "  WARNING: Node A not responding to ping. Trying TCP check..." "Yellow"
    # Fallback: TCP probe on port 6380
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $async = $tcp.BeginConnect($HeadIP, 6380, $null, $null)
        $wait = $async.AsyncWaitHandle.WaitOne(3000, $false)
        if ($wait -and -not $tcp.Client.Connected) { throw "Connect timed out" }
        $tcp.Close()
        Write-Log "  TCP port 6380 on Node A is open -- proceeding." "Green"
        Set-Result "PingNodeA" $true "TCP:6380 open"
    } catch {
        Write-Log "  WARNING: Node A not reachable on TCP:6380 either. Continuing anyway." "Yellow"
        Write-Log "  If connection fails, ensure Node A ran ray_start_nodeA.ps1 first." "Yellow"
        Set-Result "PingNodeA" $false "no ping, no TCP:6380"
    }
}

# ============================================================
#  STEP 3 -- FIREWALL RULES
# ============================================================
Write-Step "STEP 3/6: Windows Firewall rules"
try {
    netsh advfirewall firewall delete rule name="Ray-Cluster-Ports" 2>$null | Out-Null
    netsh advfirewall firewall add rule `
        name="Ray-Cluster-Ports" `
        dir=in action=allow protocol=TCP `
        localport=$($CFG.FirewallPorts) `
        profile=any | Out-Null
    Write-Log "  Firewall: TCP $($CFG.FirewallPorts) inbound allowed." "Green"
    Set-Result "Firewall" $true
} catch {
    Write-Log "  WARNING: Firewall rule failed: $_" "Yellow"
    Set-Result "Firewall" $false "$_"
}

# ============================================================
#  STEP 4 -- FLUSH DNS
# ============================================================
Write-Step "STEP 4/6: Flush DNS"
try {
    ipconfig /flushdns | Out-Null
    Write-Log "  DNS cache flushed." "Green"
    Set-Result "FlushDNS" $true
} catch {
    Write-Log "  WARNING: DNS flush failed: $_" "Yellow"
    Set-Result "FlushDNS" $false "$_"
}

# ============================================================
#  STEP 5 -- WRITE + RUN WSL2 BASH STARTUP SCRIPT
# ============================================================
Write-Step "STEP 5/6: Join Ray cluster from WSL2"

$bashScript = @"
#!/usr/bin/env bash
# Auto-generated by ray_start_nodeB.ps1
HEAD_IP="$HeadIP"
VENV="$($CFG.B_VENV)"
PROJECT="$($CFG.B_PROJECT)"
RAY="`$VENV/bin/ray"
LOG="/tmp/ray_nodeB_startup.log"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="`$PROJECT:`${PYTHONPATH:-}"

log() { echo "[`$(date '+%H:%M:%S')] `$*" | tee -a "`$LOG"; }
log "=== Node B Ray Worker Startup (v2.0) ==="
log "Head IP    : `$HEAD_IP:6380"
log "PYTHONPATH : `$PROJECT"
log "Ray binary : `$RAY"

# ── 5a: Kill any stale Ray processes by PID (FIX-2 / FIX-3) ──
log "[1/4] Killing stale Ray processes (prevents GPU:2.0 duplicate)..."
for PAT in 'gc[s]_server' 'rayle[t]' 'plasma_stor[e]' 'monitor\.p[y]' 'ray/scri[p]ts'; do
    PIDS=`$(pgrep -f "`$PAT" 2>/dev/null) || true
    if [ -n "`$PIDS" ]; then
        log "  Killing `$PAT: `$PIDS"
        echo "`$PIDS" | xargs -r kill -9 2>/dev/null || true
    fi
done
sleep 3
log "  Stale processes cleared."

# ── 5b: Verify Ray binary exists ──
log "[2/4] Checking Ray binary..."
if [ ! -x "`$RAY" ]; then
    log "ERROR: Ray binary not found at `$RAY"
    log "       Run: source `$VENV/bin/activate && pip install ray[default]"
    exit 1
fi
RAY_VER=`$(`$RAY --version 2>/dev/null | head -1)
log "  Ray: `$RAY_VER"

# ── 5c: Verify head reachable ──
log "[3/4] Testing TCP connection to head `${HEAD_IP}:6380..."
for i in 1 2 3; do
    if nc -z -w4 "`$HEAD_IP" 6380 2>/dev/null; then
        log "  Head `${HEAD_IP}:6380 is reachable."
        break
    fi
    log "  Attempt `$i/3: head not reachable yet, waiting 5s..."
    sleep 5
done
# nc check is advisory only -- ray start will give its own clear error if head is down

# ── 5d: Join the cluster ──
log "[4/4] Connecting to head `${HEAD_IP}:6380..."
"`$RAY" start \
    --address="`${HEAD_IP}:6380" \
    --num-gpus=1 \
    --num-cpus=4 \
    2>&1 | tee -a "`$LOG"
RC=`${PIPESTATUS[0]}

if [ "`$RC" -ne 0 ]; then
    log "ERROR: ray start failed (exit `$RC). Common causes:"
    log "  - Node A not started yet (run ray_start_nodeA.ps1 first)"
    log "  - Tailscale not connected or Node A IP wrong"
    log "  - Stale Ray state on Node A (rerun Node A script)"
    exit `$RC
fi

log ""
log "=== Node B CONNECTED ==="
log "Worker registered with head at `${HEAD_IP}:6380"
log "GPU resources: 1.0 (NVIDIA T1000)"
log ""
log "Verify from Node A:"
log "  python verify_cluster.py"
"@

$bashPath = "$($CFG.WinTemp)\nodeB_ray_start.sh"
# WSL /mnt/... path: derive from WinTemp
$drive    = ($env:TEMP -split ":\\")[0].ToLower()
$relPath  = ($env:TEMP -split ":\\")[1] -replace "\\", "/"
$wslPath  = "/mnt/$drive/$relPath/ray_cluster/nodeB_ray_start.sh"

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($bashPath, ($bashScript -replace "`r`n", "`n"), $utf8NoBom)
wsl chmod +x $wslPath 2>$null

Write-Log "  Bash script written: $bashPath" "Gray"
Write-Log "  Connecting to head $HeadIP`:6380..." "Cyan"

$output = wsl bash $wslPath 2>&1
$rc     = $LASTEXITCODE

$output | ForEach-Object {
    Add-Content -Path $CFG.LogFile -Value $_ -Encoding UTF8
    $col = if ($_ -match "ERROR|FAIL") { "Red" } elseif ($_ -match "CONNECTED|SUCCESS|OK") { "Green" } else { "Gray" }
    Write-Host "  $_" -ForegroundColor $col
}

if ($rc -eq 0) {
    Write-Log "  Ray worker connected successfully." "Green"
    Set-Result "RayConnect" $true
} else {
    Write-Log "  Ray connect failed (exit $rc). See: $($CFG.LogFile)" "Red"
    Set-Result "RayConnect" $false "exit code $rc"
}

# ============================================================
#  STEP 6 -- VERIFY GPU RESOURCES VISIBLE
# ============================================================
Write-Step "STEP 6/6: Verify GPU resources in cluster"
try {
    Start-Sleep -Seconds 5  # give Ray time to register this node

    # Write probe to a file to avoid quote-nesting issues
    $pyProbeB = @"
import ray, os
os.environ['RAY_DISABLE_JEMALLOC'] = '1'
os.environ['LD_PRELOAD'] = ''
try:
    ray.init(address='HEADIP:6380', ignore_reinit_error=True, logging_level='ERROR')
    cr  = ray.cluster_resources()
    gpu = cr.get('GPU', 0)
    cpu = cr.get('CPU', 0)
    alive = len([n for n in ray.nodes() if n.get('Alive')])
    print('GPU=' + str(gpu) + ' CPU=' + str(cpu) + ' ALIVE_NODES=' + str(alive))
    ray.shutdown()
except Exception as e:
    print('PROBE_ERROR:' + str(e))
"@ -replace 'HEADIP', $HeadIP

    $pyPathB  = "$($CFG.WinTemp)\probe_nodeB.py"
    $drive2   = ($env:TEMP -split ":\\")[0].ToLower()
    $rel2     = ($env:TEMP -split ":\\")[1] -replace "\\", "/"
    $wslPyB   = "/mnt/$drive2/$rel2/ray_cluster/probe_nodeB.py"
    $utf8NoBom2 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($pyPathB, ($pyProbeB -replace "`r`n","`n"), $utf8NoBom2)

    $probe = wsl bash -c "source '$($CFG.B_VENV)/bin/activate' 2>/dev/null && python3 '$wslPyB' 2>/dev/null" 2>&1

    if ($probe -match "GPU=") {
        Write-Log "  Cluster resources: $probe" "Green"
        if ($probe -match "GPU=1") {
            Write-Log "  GPU: 1.0 (single worker, correct)" "Green"
            Set-Result "GPUVerify" $true $probe
        } elseif ($probe -match "GPU=2") {
            Write-Log "  WARNING: GPU=2.0 - Node B connected TWICE. Rerun this script to clean up." "Yellow"
            Set-Result "GPUVerify" $false "GPU=2.0 duplicate connection"
        } else {
            Write-Log "  GPU not yet visible (Ray may still be registering)." "Yellow"
            Set-Result "GPUVerify" $false "GPU not yet visible: $probe"
        }
    } else {
        Write-Log "  Could not probe cluster (Ray may still be starting): $probe" "Yellow"
        Set-Result "GPUVerify" $false "probe inconclusive: $probe"
    }
} catch {
    Write-Log "  Probe skipped: $_" "Yellow"
    Set-Result "GPUVerify" $false "$_"
}

# ============================================================
#  FINAL STATUS SUMMARY
# ============================================================
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "   STARTUP SUMMARY  --  Node B" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
foreach ($key in $Results.Keys) {
    $r     = $Results[$key]
    $icon  = if ($r.Ok) { "[PASS]" } else { "[FAIL]" }
    $color = if ($r.Ok) { "Green" } else { "Red" }
    $detail = if ($r.Detail) { "  ->  $($r.Detail)" } else { "" }
    Write-Host "    $icon  $key$detail" -ForegroundColor $color
}
Write-Host ""

$critFailed = @("Tailscale","RayConnect") | Where-Object { $Results.ContainsKey($_) -and -not $Results[$_].Ok }
if ($critFailed.Count -eq 0) {
    Write-Host "  [OK] Node B is CONNECTED to the cluster." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next: from Node A, run:" -ForegroundColor Cyan
    Write-Host "    python verify_cluster.py" -ForegroundColor White
    Write-Host "    python run_pipeline.py --ticker AAPL --filing 8-K" -ForegroundColor White
} else {
    Write-Host "  FAIL Critical steps failed: $($critFailed -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "  KNOWN FIXES:" -ForegroundColor Yellow
    Write-Host "    Tailscale    -> Open Tailscale tray -> Connect" -ForegroundColor White
    Write-Host "    PingNodeA    -> Ensure Node A ran ray_start_nodeA.ps1 first" -ForegroundColor White
    Write-Host "    RayConnect   -> Check that HeadIP is correct and head is running" -ForegroundColor White
    Write-Host "    GPUVerify    -> Re-run this script to clear duplicate connection" -ForegroundColor White
}
Write-Host ""
Write-Host "  Log: $($CFG.LogFile)" -ForegroundColor Gray
Write-Host "  WSL log: /tmp/ray_nodeB_startup.log" -ForegroundColor Gray
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
