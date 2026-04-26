<#
.SYNOPSIS
    Ray Head Node (Node A) -- fully automated startup script.

.DESCRIPTION
    Starts the Ray head node on Node A (i232515 / shahoud) over Tailscale + WSL2.
    Incorporates every fix found during debugging:

    FIX-1  No portproxy for port 6379.
           WSL2 mirrored networking makes the Tailscale IP directly visible inside
           WSL2. Adding a portproxy BEFORE ray start causes "Address already in use".

    FIX-2  Loopback alias for the Tailscale IP.
           Even with mirrored networking, the raylet tries to reach GCS via the
           Tailscale userspace stack (slow), hitting the 60-second startup timeout.
           Adding 100.x.x.x/32 to WSL2's loopback interface makes all self-connections
           fast (kernel loopback instead of Tailscale tunnel).

    FIX-3  PID-based Ray cleanup instead of "ray stop --force".
           "ray stop --force" sends SIGKILL to its own process group, which includes
           the parent bash shell, terminating the startup script itself.

    FIX-4  All portproxy entries removed (not just port 6379).
           With WSL2 mirrored networking, ALL WSL2 ports are already reachable from
           external machines via the Windows Tailscale IP.  Portproxy is unnecessary
           and can conflict.

    FIX-5  Bash startup script written to a temp FILE (not passed via -c "...").
           Inline -c scripts embed all pattern strings in the process command-line.
           pkill -f patterns then match the script process itself.

.USAGE
    # Run from PowerShell (will self-elevate to Admin if needed):
        .\ray_start_nodeA.ps1

    # Optionally pass Node B's Tailscale IP to auto-ping it after startup:
        .\ray_start_nodeA.ps1 -NodeBIP 100.95.214.76

.PREREQUISITES (one-time, both machines)
    1. Install Tailscale for Windows: https://tailscale.com/download/windows
       Sign into the SAME Google/GitHub account on both machines.
    2. WSL2 with Ubuntu installed (wsl --install from Admin PowerShell if missing).
    3. maor-equity project cloned to the WSL paths configured in CONFIG below,
       with Python venv created (python -m venv venv && pip install -r requirements.txt).
    4. Run this script at least once as Admin so firewall rules are set permanently.

.NOTES
    Script version: 2.0 (rewrite incorporating all cluster debugging fixes)
    Node A machine : i232515 (shahoud)
    Node B machine : i232634 (GPU worker)
#>

param(
    [string]$NodeBIP = ""   # Optional: Node B Tailscale IP for connectivity check
)

# ============================================================
#  SELF-ELEVATE TO ADMIN (firewall rules require it)
# ============================================================
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]"Administrator")
if (-not $isAdmin) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($NodeBIP) { $argList += "-NodeBIP"; $argList += $NodeBIP }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

# ============================================================
#  CONFIG  --  edit these paths if the project moves
# ============================================================
$CFG = @{
    # WSL2 paths (forward slashes, /mnt/c/... style)
    A_VENV    = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
    A_PROJECT = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity"

    # Tailscale exe path
    TailscaleExe = "C:\Program Files\Tailscale\tailscale.exe"

    # Windows temp dir for generated bash scripts
    WinTemp = "$env:TEMP\ray_cluster"

    # Log file (Windows path)
    LogFile = "$env:TEMP\ray_cluster\nodeA_startup.log"

    # Ray ports to open in Windows Firewall
    # NOTE: 6379 is Redis default — using 6380 to avoid conflict
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

function Write-Step {
    param([string]$Msg)
    Write-Log ""
    Write-Log "  ── $Msg" "Cyan"
}

# Track pass/fail for summary
$Results = [ordered]@{}
function Set-Result { param([string]$Key, [bool]$Ok, [string]$Detail = "") $Results[$Key] = @{Ok=$Ok; Detail=$Detail} }

# ============================================================
#  BANNER
# ============================================================
Clear-Host
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "   maor-equity  ·  Node A (Head) Startup  ·  v2.0            " -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Log "Log: $($CFG.LogFile)" "Gray"

# ============================================================
#  STEP 1 -- TAILSCALE VALIDATION
# ============================================================
Write-Step "STEP 1/7: Tailscale validation"

$TS_IP = ""
try {
    if (-not (Test-Path $CFG.TailscaleExe)) { throw "Tailscale not found at $($CFG.TailscaleExe)" }
    $tsJson = & $CFG.TailscaleExe status --json 2>$null | ConvertFrom-Json
    $TS_IP = $tsJson.TailscaleIPs | Where-Object { $_ -match "^100\." } | Select-Object -First 1
    if (-not $TS_IP) { throw "Could not read Tailscale IP. Is Tailscale connected?" }
    Write-Log "  Tailscale IP (Node A): $TS_IP" "Green"
    Set-Result "Tailscale" $true $TS_IP
} catch {
    Write-Log "  ERROR: $_" "Red"
    Write-Log "  ACTION: Open Tailscale tray icon -> Connect, then re-run." "Yellow"
    Set-Result "Tailscale" $false "$_"
    exit 1
}

# Optional: ping Node B
if ($NodeBIP) {
    Write-Log "  Pinging Node B ($NodeBIP)..." "Gray"
    $ping = Test-Connection -ComputerName $NodeBIP -Count 2 -Quiet -ErrorAction SilentlyContinue
    if ($ping) {
        Write-Log "  Node B reachable." "Green"
        Set-Result "PingNodeB" $true
    } else {
        Write-Log "  WARNING: Node B ($NodeBIP) not responding to ping. Continuing anyway." "Yellow"
        Set-Result "PingNodeB" $false "No ping response"
    }
}

# ============================================================
#  STEP 2 -- FIREWALL RULES
# ============================================================
Write-Step "STEP 2/7: Windows Firewall rules"
try {
    # Remove old rule then add fresh
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
    # Non-fatal -- continue
}

# ============================================================
#  STEP 3 -- REMOVE ALL PORTPROXY  (WSL2 mirrored networking)
# ============================================================
Write-Step "STEP 3/7: Clear portproxy table"
# With WSL2 mirrored networking, ALL WSL2 ports are directly reachable via the
# Tailscale IP from external machines -- portproxy is not only unnecessary but
# actively harmful (a portproxy on TS_IP:6379 before ray start = "Address in use").
try {
    $before = (netsh interface portproxy show v4tov4 2>&1) -join ""
    netsh interface portproxy reset 2>&1 | Out-Null
    Write-Log "  All portproxy entries cleared (mirrored networking needs none)." "Green"
    Set-Result "ClearPortproxy" $true
} catch {
    Write-Log "  WARNING: portproxy reset failed: $_" "Yellow"
    Set-Result "ClearPortproxy" $false "$_"
}

# ============================================================
#  STEP 4 -- FLUSH DNS + WINDOWS STATE
# ============================================================
Write-Step "STEP 4/7: Flush DNS and Windows cache"
try {
    ipconfig /flushdns | Out-Null
    Write-Log "  DNS cache flushed." "Green"
    Set-Result "FlushDNS" $true
} catch {
    Write-Log "  WARNING: DNS flush failed: $_" "Yellow"
    Set-Result "FlushDNS" $false "$_"
}

# ============================================================
#  STEP 5 -- WSL2 LOOPBACK ALIAS  (FIX-2)
# ============================================================
Write-Step "STEP 5/7: WSL2 loopback alias for Tailscale IP"
# The raylet connects to GCS via the node-ip-address (TS_IP). Without this alias,
# that connection routes through the Tailscale userspace stack inside WSL2, which
# is slow enough to hit the 60-second startup timeout. The alias makes WSL2's kernel
# treat TS_IP as a local address so the connection is instant (loopback speed).
try {
    # wsl -u root requires no password from Windows -- safe and non-interactive
    $aliasCheck = wsl -u root bash -c "ip addr show lo 2>/dev/null | grep -c '$TS_IP'" 2>$null
    if ($aliasCheck -match "^[1-9]") {
        Write-Log "  Loopback alias $TS_IP/32 already present on lo." "Green"
    } else {
        wsl -u root ip addr add "${TS_IP}/32" dev lo 2>$null
        $verify = wsl -u root bash -c "ip addr show lo 2>/dev/null | grep '$TS_IP'" 2>$null
        if ($verify) {
            Write-Log "  Loopback alias $TS_IP/32 added to WSL2 lo." "Green"
        } else {
            throw "ip addr add reported success but alias not found in ip addr show"
        }
    }
    Set-Result "LoopbackAlias" $true
} catch {
    Write-Log "  ERROR: Failed to add loopback alias: $_" "Red"
    Write-Log "  Ray head will likely timeout (60s). Try: wsl -u root ip addr add ${TS_IP}/32 dev lo" "Yellow"
    Set-Result "LoopbackAlias" $false "$_"
    # This is critical -- warn but don't exit; the user may be able to fix it
}

# ============================================================
#  STEP 6 -- WRITE + RUN WSL2 BASH STARTUP SCRIPT  (FIX-5)
# ============================================================
Write-Step "STEP 6/7: Start Ray head inside WSL2"

# Write the bash script to a FILE to avoid:
# (a) escape hell in -c "..." strings
# (b) pkill -f patterns matching the script's own command-line text
$bashScript = @"
#!/usr/bin/env bash
# Auto-generated by ray_start_nodeA.ps1 -- do not edit
TS_IP="$TS_IP"
VENV="$($CFG.A_VENV)"
PROJECT="$($CFG.A_PROJECT)"
RAY="`$VENV/bin/ray"
LOG="/tmp/ray_nodeA_startup.log"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=120
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=120
export PYTHONPATH="`$PROJECT:`${PYTHONPATH:-}"

log() { echo "[`$(date '+%H:%M:%S')] `$*" | tee -a "`$LOG"; }
log "=== Node A Ray Head Startup (v2.0) ==="
log "Tailscale IP : `$TS_IP"
log "PYTHONPATH   : `$PROJECT"
log "Ray binary   : `$RAY"

# ── 6a: Kill stale Ray processes by PID (FIX-3: no ray stop --force) ──
log "[1/4] Killing stale Ray processes by PID..."
# Character-class patterns prevent pkill matching this script file path
for PAT in 'gc[s]_server' 'rayle[t]' 'plasma_stor[e]' 'monitor\.p[y]' 'ray/scri[p]ts'; do
    PIDS=`$(pgrep -f "`$PAT" 2>/dev/null) || true
    if [ -n "`$PIDS" ]; then
        log "  Killing `$PAT: `$PIDS"
        echo "`$PIDS" | xargs -r kill -9 2>/dev/null || true
    fi
done
# Release ports 6379 AND 6380 if still held (6379=Redis default, 6380=Ray)
fuser -k 6379/tcp 2>/dev/null || true
fuser -k 6380/tcp 2>/dev/null || true
sleep 3

# ── 6b: Clear Ray temp state ──
log "[2/4] Clearing Ray tmp state..."
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* 2>/dev/null || true
sleep 1

# ── 6c: Confirm port free ──
if ss -tlnp 2>/dev/null | grep -q ':6380'; then
    log "ERROR: Port 6380 still occupied after cleanup:"
    ss -tlnp | grep ':6380'
    exit 1
fi
log "  Port 6380 free."

# ── 6d: Start Ray head ──
log "[3/4] Starting Ray head on `${TS_IP}:6380..."
"`$RAY" start \
    --head \
    --port=6380 \
    --node-ip-address="`$TS_IP" \
    --disable-usage-stats \
    --include-dashboard=false \
    --num-cpus=2 \
    --object-store-memory=268435456 \
    --plasma-directory=/tmp \
    2>&1 | tee -a "`$LOG"
RC=`${PIPESTATUS[0]}

if [ "`$RC" -ne 0 ]; then
    log "ERROR: ray start failed (exit `$RC). See `$LOG for details."
    exit `$RC
fi

sleep 3

# ── 6e: Verify GCS is listening ──
log "[4/4] Verifying GCS..."
if ss -tlnp 2>/dev/null | grep -q ':6380'; then
    log "GCS UP on ${TS_IP}:6380 - OK"
else
    log "WARNING: GCS not found on :6380 -- ray may still be initialising"
fi

log ""
log "=== Node A READY ==="
log "Head address: `${TS_IP}:6380"
log ""
log "Node B command (run ray_start_nodeB.ps1 on partner machine):"
log "  .\ray_start_nodeB.ps1 -HeadIP `$TS_IP"
log ""
log "Or manual WSL command on Node B:"
log "  export PYTHONPATH=/path/to/maor-equity"
log "  ray start --address=`${TS_IP}:6380 --num-gpus=1 --num-cpus=4"
"@

# Write with Unix line endings (no BOM)
$bashPath = "$($CFG.WinTemp)\nodeA_ray_start.sh"
$wslPath  = "/mnt/c/Users/$env:USERNAME/AppData/Local/Temp/ray_cluster/nodeA_ray_start.sh"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($bashPath, ($bashScript -replace "`r`n", "`n"), $utf8NoBom)
wsl chmod +x $wslPath 2>$null

Write-Log "  Bash script written: $bashPath" "Gray"
Write-Log "  Running in WSL2..." "Cyan"

# Run the bash script synchronously (PowerShell waits for completion)
$output = wsl bash $wslPath 2>&1
$rc = $LASTEXITCODE

# Echo output to console + log
$output | ForEach-Object {
    Add-Content -Path $CFG.LogFile -Value $_ -Encoding UTF8
    $col = if ($_ -match "ERROR|FAIL") { "Red" } elseif ($_ -match "UP|READY|SUCCESS|OK") { "Green" } else { "Gray" }
    Write-Host "  $_" -ForegroundColor $col
}

if ($rc -eq 0) {
    Write-Log "  Ray head started successfully (exit 0)." "Green"
    Set-Result "RayStart" $true
} else {
    Write-Log "  Ray start failed (exit $rc). See: $($CFG.LogFile)" "Red"
    Set-Result "RayStart" $false "exit code $rc"
}

# ============================================================
#  STEP 7 -- VERIFY + OPTIONAL NODE B PING
# ============================================================
Write-Step "STEP 7/7: Final verification"

try {
    # Write the probe to a file so we avoid all PS/bash quote-nesting issues
    $pyProbe = @"
import ray, os, sys
os.environ['RAY_DISABLE_JEMALLOC'] = '1'
os.environ['LD_PRELOAD'] = ''
try:
    ray.init(address='HEADIP:6380', ignore_reinit_error=True, logging_level='ERROR')
    cr  = ray.cluster_resources()
    n   = len(ray.nodes())
    cpu = cr.get('CPU', 0)
    print('NODES=' + str(n) + ' CPU=' + str(cpu))
    ray.shutdown()
except Exception as e:
    print('PROBE_ERROR:' + str(e))
"@ -replace 'HEADIP', $TS_IP

    $pyPath  = "$($CFG.WinTemp)\probe_nodeA.py"
    $wslPy   = $pyPath -replace "\\","/" -replace "^C:","" | ForEach-Object { "/mnt/c$_" }
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($pyPath, ($pyProbe -replace "`r`n","`n"), $utf8NoBom)

    $probe = wsl bash -c "source '$($CFG.A_VENV)/bin/activate' 2>/dev/null && python3 '$wslPy' 2>/dev/null" 2>&1

    if ($probe -match "NODES=") {
        Write-Log "  Cluster probe OK: $($probe -join ' ')" "Green"
        Set-Result "ClusterProbe" $true ($probe -join " | ")
    } else {
        Write-Log "  Cluster probe inconclusive (Ray may still be initialising): $probe" "Yellow"
        Set-Result "ClusterProbe" $false "probe: $probe"
    }
} catch {
    Write-Log "  Probe skipped: $_" "Yellow"
    Set-Result "ClusterProbe" $false "$_"
}

# ============================================================
#  FINAL STATUS SUMMARY
# ============================================================
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "   STARTUP SUMMARY" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
foreach ($key in $Results.Keys) {
    $r = $Results[$key]
    $icon  = if ($r.Ok) { "[PASS]" } else { "[FAIL]" }
    $color = if ($r.Ok) { "Green" } else { "Red" }
    $detail = if ($r.Detail) { "  ->  $($r.Detail)" } else { "" }
    Write-Host "    $icon  $key$detail" -ForegroundColor $color
}
Write-Host ""

$allOk = ($Results.Values | Where-Object { -not $_.Ok }).Count -eq 0
if ($allOk) {
    Write-Host "  [OK] Node A is READY." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Head address: $TS_IP`:6380" -ForegroundColor Yellow
    Write-Host "  Tell Node B: .\ray_start_nodeB.ps1 -HeadIP $TS_IP" -ForegroundColor Yellow
} else {
    $failed = ($Results.GetEnumerator() | Where-Object { -not $_.Value.Ok } | ForEach-Object { $_.Key }) -join ", "
    Write-Host "  FAIL Some steps failed: $failed" -ForegroundColor Red
    Write-Host "  Check log: $($CFG.LogFile)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  KNOWN FIXES:" -ForegroundColor Yellow
    Write-Host "    Tailscale      -> Open Tailscale tray, click Connect" -ForegroundColor White
    Write-Host "    LoopbackAlias  -> Run: wsl -u root ip addr add $TS_IP/32 dev lo" -ForegroundColor White
    Write-Host "    RayStart       -> Check $($CFG.LogFile) and /tmp/ray_nodeA_startup.log (in WSL)" -ForegroundColor White
}
Write-Host ""
Write-Host "  Log saved: $($CFG.LogFile)" -ForegroundColor Gray
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
