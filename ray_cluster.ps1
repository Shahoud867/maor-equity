<#
.SYNOPSIS
    Ray Cluster Setup â€” maor-equity project
    Run on Node A (CPU/head) OR Node B (GPU/worker)

.EXAMPLE
    Node A:  .\ray_cluster.ps1 -Role A
    Node B:  .\ray_cluster.ps1 -Role B -Address "0.tcp.ap.ngrok.io:12345"
    (If you omit parameters the script will prompt you)
#>

param(
    [ValidateSet("A","B","")]
    [string]$Role    = "",
    [string]$Address = ""      # Node B only: Ngrok address from Node A
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CONFIG  (Node A paths â€” do not change)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$A_VENV_WSL   = "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
$A_PROJECT_WSL= "/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity"
$A_NODE_IP    = "172.26.19.0"

# Temp dir with NO spaces â€” accessible from WSL as /mnt/c/ProgramData/ray_cluster/
$WIN_TMP = "C:\ProgramData\ray_cluster"
$WSL_TMP = "/mnt/c/ProgramData/ray_cluster"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  BANNER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Clear-Host
Write-Host ""
Write-Host "  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—" -ForegroundColor Cyan
Write-Host "  â•‘      maor-equity  â€”  Ray Cluster Setup              â•‘" -ForegroundColor Cyan
Write-Host "  â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•" -ForegroundColor Cyan
Write-Host ""

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  ROLE SELECTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (-not $Role) {
    Write-Host "  Which node are you?" -ForegroundColor Yellow
    Write-Host "    A = Node A  (CPU/head â€” runs Ray + Ngrok, i232515's PC)"
    Write-Host "    B = Node B  (GPU/worker â€” connects to Node A, i232634's PC)"
    Write-Host ""
    $Role = (Read-Host "  Enter A or B").Trim().ToUpper()
}

if ($Role -notin @("A","B")) {
    Write-Host "  ERROR: Enter A or B." -ForegroundColor Red
    exit 1
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  WSL CHECK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Write-Host "  Checking WSL..." -NoNewline
try {
    $null = wsl echo ok 2>&1
    Write-Host " OK" -ForegroundColor Green
} catch {
    Write-Host " MISSING" -ForegroundColor Red
    Write-Host "  Install WSL2:  wsl --install" -ForegroundColor Yellow
    exit 1
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CREATE TEMP DIR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (-not (Test-Path $WIN_TMP)) {
    New-Item -ItemType Directory -Path $WIN_TMP -Force | Out-Null
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  HELPER: open a new persistent WSL window
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function Open-WslWindow {
    param([string]$Title, [string]$WslScript)

    # Try Windows Terminal first, fall back to cmd
    $hasWT = Get-Command wt -ErrorAction SilentlyContinue
    if ($hasWT) {
        Start-Process wt -ArgumentList `
            "new-tab", "--title", "`"$Title`"", "--", `
            "wsl.exe", "bash", $WslScript
    } else {
        Start-Process cmd -ArgumentList "/k", "wsl bash $WslScript"
    }
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  NODE A
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if ($Role -eq "A") {

    Write-Host ""
    Write-Host "  [NODE A] Writing startup script..." -ForegroundColor Cyan

    # â”€â”€ Write the Node A bash script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    $nodeAScript = @'
#!/usr/bin/env bash
# Node A startup â€” Ray head + Ngrok + state-file monitor
# Written by ray_cluster.ps1 â€” do not edit manually

set -uo pipefail

VENV="__A_VENV__"
RAY="$VENV/bin/ray"
PROJECT="__A_PROJECT__"
NODE_IP="__A_NODE_IP__"
WSL_TMP="__WSL_TMP__"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=60
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60

echo ""
echo "  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—"
echo "  â•‘       NODE A  â€”  Ray + Ngrok Startup            â•‘"
echo "  â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
echo ""

# â”€â”€ Step 1: Kill everything â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "  [1/4] Killing existing Ray / Ngrok..."
"$RAY" stop --force 2>/dev/null || true
sleep 1
pkill -9 -f gcs_server  2>/dev/null || true
pkill -9 -f raylet       2>/dev/null || true
pkill -9 -f plasma_store 2>/dev/null || true
pkill -9 -f ngrok        2>/dev/null || true
sleep 2
rm -rf /tmp/ray /tmp/ray_* /tmp/plasma_store_socket* /tmp/session_* \
       /tmp/ray_cluster_state.json /tmp/ray_ngrok_address.txt 2>/dev/null || true
sleep 1

if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "  ERROR: port 6379 still in use. Run: sudo fuser -k 6379/tcp"
    read -rp "Press Enter to exit..." && exit 1
fi
echo "  Port 6379 free."

# â”€â”€ Step 2: Start Ray head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo "  [2/4] Starting Ray head on $NODE_IP:6379..."

"$RAY" start \
    --head \
    --port=6379 \
    --node-ip-address="$NODE_IP" \
    --disable-usage-stats \
    --include-dashboard=false \
    --num-cpus=2 \
    --object-store-memory=268435456 \
    --plasma-directory=/tmp \
    2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: Ray failed to start."
    echo "  Tip: close this window, reopen a fresh WSL terminal, and run again."
    read -rp "  Press Enter to exit..." && exit 1
fi

sleep 3

if ! ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "  ERROR: GCS did not bind to :6379"
    read -rp "  Press Enter to exit..." && exit 1
fi
echo "  Ray GCS listening on :6379  âœ“"

# â”€â”€ Step 3: Start Ngrok â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo "  [3/4] Starting Ngrok TCP tunnel..."
nohup ngrok tcp 6379 --log=stdout > /tmp/ngrok.log 2>&1 &
sleep 2

TUNNEL=""
for i in $(seq 1 30); do
    TUNNEL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin); t=d.get('tunnels',[])
    print(t[0]['public_url'] if t else '')
except: print('')
" 2>/dev/null || true)
    [ -n "$TUNNEL" ] && break
    printf "  Waiting for tunnel... %d/30\r" "$i"
    sleep 1
done

if [ -z "$TUNNEL" ]; then
    echo ""
    echo "  ERROR: Ngrok did not start."
    echo "  Log:"
    cat /tmp/ngrok.log | tail -20
    read -rp "  Press Enter to exit..." && exit 1
fi

NGROK_HOST=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f1)
NGROK_PORT=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f2)

# Write address to file so PowerShell can read it
echo "$NGROK_HOST:$NGROK_PORT" > /tmp/ray_ngrok_address.txt

echo ""
echo "  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—"
echo "  â•‘                   NODE A IS READY  âœ“                       â•‘"
echo "  â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£"
echo "  â•‘                                                              â•‘"
echo "  â•‘  Tell Node B to run (PowerShell):                           â•‘"
printf "  â•‘    .\\ray_cluster.ps1 -Role B -Address %s\n" "$NGROK_HOST:$NGROK_PORT"
echo "  â•‘                                                              â•‘"
echo "  â•‘  OR in WSL on Node B:                                       â•‘"
echo "  â•‘    export LD_PRELOAD=''                                      â•‘"
echo "  â•‘    export RAY_DISABLE_JEMALLOC=1                             â•‘"
echo "  â•‘    export CUDA_VISIBLE_DEVICES=0                             â•‘"
printf "  â•‘    ray start --address=%s \\\\\n" "$NGROK_HOST:$NGROK_PORT"
echo "  â•‘          --num-gpus=1 --num-cpus=4                           â•‘"
echo "  â•‘                                                              â•‘"
printf "  â•‘  Full tunnel: %s\n" "$TUNNEL"
echo "  â•‘                                                              â•‘"
echo "  â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£"
echo "  â•‘  KEEP THIS WINDOW OPEN â€” closing it stops Ray               â•‘"
echo "  â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
echo ""

# â”€â”€ Step 4: Monitor loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "  [4/4] Monitoring every 30s (Ctrl+C stops monitor, Ray stays up):"
echo "  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"

while true; do
    sleep 30
    NOW=$(date '+%H:%M:%S')

    if ss -tlnp 2>/dev/null | grep -q ':6379'; then GCS="Ray:UP"; else GCS="Ray:DOWN!"; fi

    CURR=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin); t=d.get('tunnels',[])
    print(t[0]['public_url'].replace('tcp://','') if t else 'NO_TUNNEL')
except: print('?')
" 2>/dev/null || echo "?")

    NODES=$("$VENV/bin/python3" -c "
import ray, os, sys, json, time
os.environ['RAY_DISABLE_JEMALLOC']='1'
os.environ['LD_PRELOAD']=''
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']='0'
try:
    ray.init(address='${NODE_IP}:6379', ignore_reinit_error=True, logging_level='ERROR')
    nodes=ray.nodes(); cr=dict(ray.cluster_resources())
    alive=[n for n in nodes if n.get('Alive')]
    state={
        'ts':time.time(),
        'alive':[{'ip':n.get('NodeManagerAddress',''),'cpu':n.get('Resources',{}).get('CPU',0),'gpu':n.get('Resources',{}).get('GPU',0)} for n in alive],
        'dead_ips':[n.get('NodeManagerAddress','') for n in nodes if not n.get('Alive')],
        'cr':{k:float(v) for k,v in cr.items() if isinstance(v,(int,float))},
        'cr_keys':list(cr.keys())
    }
    with open('/tmp/ray_cluster_state.json','w') as f: json.dump(state,f)
    sys.stdout.write(str(len(alive))+'\n'); sys.stdout.flush()
except:
    sys.stdout.write('?\n'); sys.stdout.flush()
finally:
    os._exit(0)
" 2>/dev/null)

    echo "  [$NOW] $GCS | Ngrok: $CURR | Alive nodes: $NODES"
done
'@

    # Replace placeholders
    $nodeAScript = $nodeAScript `
        -replace '__A_VENV__',    $A_VENV_WSL `
        -replace '__A_PROJECT__', $A_PROJECT_WSL `
        -replace '__A_NODE_IP__', $A_NODE_IP `
        -replace '__WSL_TMP__',   $WSL_TMP

    # Write with Unix line endings
    $nodeAScriptLf = $nodeAScript -replace "`r`n","`n"
    [System.IO.File]::WriteAllText("$WIN_TMP\nodeA.sh", $nodeAScriptLf)

    wsl chmod +x "$WSL_TMP/nodeA.sh" 2>$null

    Write-Host "  [NODE A] Opening WSL window..." -ForegroundColor Cyan
    Open-WslWindow -Title "NODE A â€” Ray Head" -WslScript "$WSL_TMP/nodeA.sh"

    Write-Host ""
    Write-Host "  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•" -ForegroundColor Green
    Write-Host "  A new terminal window opened running Ray + Ngrok." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Wait ~15 seconds for startup, then look in that window" -ForegroundColor Yellow
    Write-Host "  for the NODE A IS READY box â€” it shows the Ngrok address." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Send that address to Node B so they can run:" -ForegroundColor Yellow
    Write-Host '    .\ray_cluster.ps1 -Role B -Address "0.tcp.ap.ngrok.io:XXXXX"' -ForegroundColor White
    Write-Host ""
    Write-Host "  After Node B connects and the monitor shows 'Alive nodes: 2'," -ForegroundColor Yellow
    Write-Host "  wait one more 30s cycle then run in a new PowerShell:" -ForegroundColor Yellow
    Write-Host "    cd `"$($A_PROJECT_WSL -replace '/mnt/c','C:' -replace '/','\' )`"" -ForegroundColor White
    Write-Host "    wsl bash -c `"source $A_VENV_WSL/bin/activate && python $A_PROJECT_WSL/verify_cluster.py`"" -ForegroundColor White
    Write-Host "  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•" -ForegroundColor Green
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  NODE B
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if ($Role -eq "B") {

    # â”€â”€ Get Ngrok address â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (-not $Address) {
        Write-Host ""
        Write-Host "  Paste the Ngrok address from Node A's terminal." -ForegroundColor Yellow
        Write-Host "  Format:  0.tcp.ap.ngrok.io:XXXXX" -ForegroundColor Gray
        Write-Host ""
        $Address = (Read-Host "  Ngrok address").Trim()
    }

    if ($Address -notmatch '^\S+:\d+$') {
        Write-Host "  ERROR: Address must be HOST:PORT (e.g. 0.tcp.ap.ngrok.io:12345)" -ForegroundColor Red
        exit 1
    }

    # â”€â”€ Find Ray binary in WSL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Write-Host ""
    Write-Host "  [NODE B] Locating Ray in WSL..." -NoNewline

    $rayFound = wsl bash -c "
        for p in ~/maor-equity/venv/bin/ray ~/venv/bin/ray ./venv/bin/ray /usr/local/bin/ray; do
            [ -x `"`$p`" ] && echo `"`$p`" && exit 0
        done
        which ray 2>/dev/null || echo ''
    " 2>$null

    $rayFound = $rayFound.Trim()

    if (-not $rayFound) {
        Write-Host " NOT FOUND" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Ray was not found in common locations." -ForegroundColor Yellow
        Write-Host "  Make sure you have cloned the repo and installed requirements:" -ForegroundColor Yellow
        Write-Host "    git clone https://github.com/Shahoud867/maor-equity.git ~/maor-equity"
        Write-Host "    cd ~/maor-equity"
        Write-Host "    python3 -m venv venv"
        Write-Host "    source venv/bin/activate"
        Write-Host "    pip install ray[default] torch --index-url https://download.pytorch.org/whl/cu121"
        exit 1
    }

    Write-Host " Found: $rayFound" -ForegroundColor Green
    $venvBin = Split-Path $rayFound -Parent   # e.g. ~/maor-equity/venv/bin

    # â”€â”€ Write Node B bash script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Write-Host "  [NODE B] Writing connection script..." -ForegroundColor Cyan

    $nodeBScript = @"
#!/usr/bin/env bash
# Node B auto-connect watchdog â€” written by ray_cluster.ps1

ADDRESS="$Address"
RAY="$rayFound"

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0

NGROK_HOST=`$(echo "`$ADDRESS" | cut -d: -f1)
NGROK_PORT=`$(echo "`$ADDRESS" | cut -d: -f2)

now() { date '+%H:%M:%S'; }

echo ""
echo "  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—"
echo "  â•‘       NODE B  â€”  Ray Worker Connect             â•‘"
echo "  â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
echo "  Address : `$ADDRESS"
echo "  Ray     : `$RAY"
echo ""

connect() {
    echo "  [`$(now)] Stopping existing Ray..."
    "`$RAY" stop --force 2>/dev/null || true
    sleep 2
    echo "  [`$(now)] Connecting to head at `$ADDRESS..."
    "`$RAY" start \\
        --address="`$ADDRESS" \\
        --num-gpus=1 \\
        --num-cpus=4 \\
        2>&1 | tail -8
    echo "  [`$(now)] Worker started. Watching for disconnects..."
}

trap 'echo "  Watcher stopped by user."; "`$RAY" stop 2>/dev/null; exit 0' INT TERM

# Initial connect
connect

# Watch loop â€” NEVER calls ray.init() or ray.shutdown()
while true; do
    sleep 10

    # Check 1: local raylet process alive?
    if ! pgrep -f raylet >/dev/null 2>&1; then
        echo "  [`$(now)] RAYLET DIED â€” reconnecting..."
        connect
        continue
    fi

    # Check 2: TCP path to Node A still open?
    if ! nc -z -w4 "`$NGROK_HOST" "`$NGROK_PORT" 2>/dev/null; then
        echo "  [`$(now)] NGROK UNREACHABLE â€” reconnecting..."
        connect
        continue
    fi

    echo "  [`$(now)] OK â€” raylet alive, tunnel reachable"
done
"@

    $nodeBScriptLf = $nodeBScript -replace "`r`n","`n"
    [System.IO.File]::WriteAllText("$WIN_TMP\nodeB.sh", $nodeBScriptLf)

    wsl chmod +x "$WSL_TMP/nodeB.sh" 2>$null

    Write-Host "  [NODE B] Opening WSL window..." -ForegroundColor Cyan
    Open-WslWindow -Title "NODE B â€” Ray Worker" -WslScript "$WSL_TMP/nodeB.sh"

    Write-Host ""
    Write-Host "  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•" -ForegroundColor Green
    Write-Host "  A new terminal window opened connecting Node B to Ray." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Watch that window â€” it will show:" -ForegroundColor Yellow
    Write-Host "    OK â€” raylet alive, tunnel reachable    (every 10s)" -ForegroundColor Gray
    Write-Host "    RECONNECTING...                         (on drops, auto-fixed)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  On Node A â€” after monitor shows 'Alive nodes: 2'," -ForegroundColor Yellow
    Write-Host "  wait one 30s cycle then run verify_cluster.py." -ForegroundColor Yellow
    Write-Host "  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•" -ForegroundColor Green
}

Write-Host ""
Write-Host "  Done. Keep both terminal windows open." -ForegroundColor Cyan
Write-Host ""
