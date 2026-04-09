# =============================================================================
#  NODE B — PARTNER SETUP SCRIPT
#  Run this ONCE after cloning the repository.
#  It installs all GPU dependencies and connects the Ray worker.
# =============================================================================
#
#  USAGE:
#    # Install only (Node A not started yet):
#    .\setup_nodeB.ps1
#
#    # Install + connect Ray worker:
#    .\setup_nodeB.ps1 -NgrokHost "0.tcp.ngrok.io" -NgrokPort 12345
#
#  PREREQUISITES:
#    - Windows 10/11 with WSL2 + Ubuntu 22.04
#    - NVIDIA T1000 drivers on Windows + CUDA WSL2 support
#      https://developer.nvidia.com/cuda/wsl
# =============================================================================

param(
    [string]$NgrokHost        = "",
    [int]   $NgrokPort        = 0,
    [switch]$SkipSystemUpdate,
    [switch]$SkipModelDownload,
    [switch]$VirtualBoxFallback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Project root = folder containing this script (repo root)
$WIN_ROOT = $PSScriptRoot
function ConvertTo-WSLPath([string]$p) {
    "/mnt/$($p.Substring(0,1).ToLower())$($p.Substring(2).Replace('\','/'))"
}
$WSL_ROOT = ConvertTo-WSLPath $WIN_ROOT
$VBOX_HEAD = "192.168.56.10"

function Write-Banner([string]$t,[string]$c="Cyan"){
    $l="="*72; Write-Host "`n$l" -ForegroundColor $c
    Write-Host "  $t" -ForegroundColor $c; Write-Host "$l`n" -ForegroundColor $c
}
function Write-Step([string]$t){ Write-Host "[....] $t" -ForegroundColor Yellow }
function Write-OK  ([string]$t){ Write-Host "[ OK ] $t" -ForegroundColor Green  }
function Write-Warn([string]$t){ Write-Host "[WARN] $t" -ForegroundColor Magenta}
function Write-Err ([string]$t){ Write-Host "[ERR ] $t" -ForegroundColor Red    }
function Invoke-WSL([string]$c,[string]$d=""){
    if($d){Write-Step $d}
    $o=wsl bash -lc $c 2>&1
    if($LASTEXITCODE -ne 0){Write-Err "Failed (exit $LASTEXITCODE): $c"; Write-Host $o; return $false}
    Write-OK "Done"; return $true
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
Write-Banner "NODE B PRE-FLIGHT CHECKS"
try{wsl --status 2>&1|Out-Null}catch{Write-Err "WSL2 not available";exit 1}
Write-OK "WSL2 available"

$distros=(wsl --list --quiet 2>&1) -join " "
if($distros -notmatch "Ubuntu"){
    Write-Warn "No Ubuntu — installing Ubuntu-22.04..."
    wsl --install -d Ubuntu-22.04
    Write-Host "Rerun after Ubuntu setup."; exit 0
}
Write-OK "Ubuntu WSL distro found"

try{$nv=&"C:\Windows\System32\nvidia-smi.exe" 2>&1; if($LASTEXITCODE-ne 0){throw}}
catch{Write-Err "NVIDIA driver not found. Install from https://www.nvidia.com/Download/index.aspx"; exit 1}
Write-OK "NVIDIA driver detected"

$gpuWSL=wsl bash -lc "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null"
if($LASTEXITCODE -ne 0 -or -not $gpuWSL){
    Write-Err "GPU not visible in WSL. Install CUDA WSL2: https://developer.nvidia.com/cuda/wsl"; exit 1
}
Write-OK "GPU in WSL: $gpuWSL"

$cap=[float](wsl bash -lc "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '\r'")
if($cap -lt 7.5){
    Write-Warn "Compute cap $cap < 7.5 — use load_in_8bit=True fallback in agent code"
}else{ Write-OK "Compute capability $cap >= 7.5 (4-bit OK)" }

# ── System deps ───────────────────────────────────────────────────────────────
Write-Banner "SYSTEM DEPENDENCIES"
if(-not $SkipSystemUpdate){
    Invoke-WSL "sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y" "apt update/upgrade"
}
Invoke-WSL "sudo apt-get install -y python3.10 python3.10-venv python3-pip python3.10-dev build-essential" "Python 3.10 + build tools"

# ── Virtual environment inside the repo ───────────────────────────────────────
Write-Banner "VIRTUAL ENVIRONMENT"
Invoke-WSL "cd '$WSL_ROOT' && python3.10 -m venv venv && source venv/bin/activate && pip install --upgrade pip setuptools wheel" "Creating venv"

# ── Install CUDA torch FIRST (critical ordering) ──────────────────────────────
Write-Banner "INSTALLING CUDA PYTORCH (must come first)"
Invoke-WSL "cd '$WSL_ROOT' && source venv/bin/activate && pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121" "torch==2.2.0+cu121 (CUDA 12.1)"

# ── Install the rest from requirements-nodeB.txt ──────────────────────────────
Write-Banner "INSTALLING GPU PYTHON DEPENDENCIES"
Invoke-WSL "cd '$WSL_ROOT' && source venv/bin/activate && pip install -r requirements-nodeB.txt" "pip install -r requirements-nodeB.txt"

# Verify bitsandbytes
$bnb=wsl bash -lc "cd '$WSL_ROOT' && source venv/bin/activate && python3 -c 'import bitsandbytes as bnb; print(\"bnb OK\")' 2>&1"
if($bnb -match "bnb OK"){ Write-OK "bitsandbytes verified" }
else{ Write-Warn "bitsandbytes check: $bnb  (check CUDA version if errors appear)" }

# Verify CUDA torch
$tc=wsl bash -lc "cd '$WSL_ROOT' && source venv/bin/activate && python3 -c 'import torch; print(torch.cuda.is_available())' 2>&1"
if($tc -match "True"){ Write-OK "PyTorch CUDA: True" }
else{ Write-Warn "PyTorch CUDA returned: $tc" }

# ── Pre-download models ───────────────────────────────────────────────────────
if(-not $SkipModelDownload){
    Write-Banner "PRE-DOWNLOADING HUGGINGFACE MODELS"
    $dlScript=@"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
print('Downloading ProsusAI/finbert...')
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('Downloading yiyanghkust/finbert-tone...')
AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
print('Downloading Phi-3-mini tokenizer...')
AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
print('All models downloaded.')
"@
    # Write temp script to repo, run it, delete it
    $tmp=Join-Path $WIN_ROOT "_dl_models_tmp.py"
    [System.IO.File]::WriteAllText($tmp,$dlScript,[System.Text.UTF8Encoding]::new($false))
    $wslTmp=ConvertTo-WSLPath $tmp
    Invoke-WSL "cd '$WSL_ROOT' && source venv/bin/activate && python3 '$wslTmp'" "Downloading FinBERT + Phi-3-mini"
    Remove-Item $tmp -Force
}

# ── Connect Ray worker ────────────────────────────────────────────────────────
Write-Banner "RAY WORKER CONNECTION"
if($NgrokHost -ne "" -and $NgrokPort -ne 0){
    if($VirtualBoxFallback){
        Invoke-WSL "cd '$WSL_ROOT' && source venv/bin/activate && export LD_PRELOAD='' && RAY_DISABLE_JEMALLOC=1 ray start --address=${VBOX_HEAD}:6379" "Connecting Ray worker (VirtualBox)"
    }else{
        Invoke-WSL "cd '$WSL_ROOT' && source venv/bin/activate && export LD_PRELOAD='' && RAY_DISABLE_JEMALLOC=1 ray start --address=${NgrokHost}:${NgrokPort}" "Connecting Ray worker via Ngrok"
    }
    Write-OK "Ray worker connected. Ask partner to run: python verify_cluster.py"
    Write-Host "  For resilient reconnects on Node B, prefer running:" -ForegroundColor White
    Write-Host "  cd '$WSL_ROOT'" -ForegroundColor Gray
    Write-Host "  source venv/bin/activate" -ForegroundColor Gray
    Write-Host "  export LD_PRELOAD=''" -ForegroundColor Gray
    Write-Host "  export RAY_DISABLE_JEMALLOC=1" -ForegroundColor Gray
    Write-Host "  bash scripts/nodeb_connect_watch.sh --address=${NgrokHost}:${NgrokPort}" -ForegroundColor Gray
}else{
    Write-Warn "NgrokHost/NgrokPort not supplied — Ray worker NOT connected yet."
    Write-Host "  Once Node A is running, execute in WSL:" -ForegroundColor White
    Write-Host "  cd '$WSL_ROOT'" -ForegroundColor Gray
    Write-Host "  source venv/bin/activate" -ForegroundColor Gray
    Write-Host "  export LD_PRELOAD=''" -ForegroundColor Gray
    Write-Host "  RAY_DISABLE_JEMALLOC=1 ray start --address=<ngrok_host>:<ngrok_port>" -ForegroundColor Gray
}

Write-Banner "NODE B SETUP COMPLETE" "Green"
Write-Host "  Repo dir (WSL)  : $WSL_ROOT" -ForegroundColor Cyan
Write-Host "  See README.md for day-by-day commands." -ForegroundColor Cyan