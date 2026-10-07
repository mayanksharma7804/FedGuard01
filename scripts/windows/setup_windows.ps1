# =============================================================================
# FedGuard - Windows setup (no Linux, no WSL, no Docker)
# -----------------------------------------------------------------------------
# Run from the repo root in a normal (not admin) PowerShell window:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
# Options:
#   -CpuOnly                 install the CPU build of PyTorch even if an NVIDIA GPU exists
#   -CudaIndex <url>         PyTorch CUDA wheel index (check pytorch.org > Get Started)
# Safe to run again: it only adds what is missing.
# =============================================================================
param(
    [switch]$CpuOnly,
    # cu126 works with NVIDIA drivers >= 527 (CUDA 12.x). cu130/cu132 need drivers >= 580.
    # torch 2.14.1 (our pinned version) has no cu128 build.
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu126"
)
$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Invoke-Checked {
    # Runs a native program and stops the script if it fails (PowerShell 5.1 does not do this itself).
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

# 1. Sanity checks that save hours later -------------------------------------
if ($Repo -match '\s') {
    Write-Warning "The project path has spaces: $Repo. Move the repo to a path like D:\fedguard to avoid tool errors."
}
$lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($lp -ne 1) {
    Write-Warning ("Windows long paths are OFF. Once, in an ADMIN PowerShell, run:`n" +
        "  New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force")
}

# 2. uv: installs Python 3.11 inside the project (the Store Python 3.14 is not used) ----
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not on PATH. Close and reopen PowerShell, then run this script again." }

# 3. Virtual environment ------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Say "Creating .venv with Python 3.11"
    Invoke-Checked uv @("venv", "--python", "3.11", ".venv")
}
$Py = Join-Path $Repo ".venv\Scripts\python.exe"

$Req = if (Test-Path "requirements.txt") { "requirements.txt" } elseif (Test-Path "requirements.in") { "requirements.in" } else { $null }
if (-not $Req) { throw "No requirements.txt or requirements.in in the repo root." }
Say "Installing packages from $Req"
Invoke-Checked uv @("pip", "install", "--python", $Py, "-r", $Req)
if (Test-Path "pyproject.toml") { Invoke-Checked uv @("pip", "install", "--python", $Py, "-e", ".") }

# 4. PyTorch with CUDA (PyPI only has the CPU build for Windows) --------------
$HasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($HasGpu -and -not $CpuOnly) {
    $TorchVer = Select-String -Path $Req -Pattern '^torch==([0-9.]+)' -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1
    $Spec = if ($TorchVer) { "torch==$TorchVer" } else { "torch" }
    Say "NVIDIA GPU found - installing the CUDA build: $Spec from $CudaIndex"
    Invoke-Checked uv @("pip", "install", "--python", $Py, "--reinstall-package", "torch", $Spec, "--index-url", $CudaIndex)
} else {
    Say "Using the CPU build of PyTorch"
}

# 5. Verify -----------------------------------------------------------------
Say "Verifying the setup"
if (Test-Path "scripts\verify_setup.py") {
    Invoke-Checked $Py @("scripts\verify_setup.py")
} else {
    Invoke-Checked $Py @("-c", "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())")
}

Write-Host "`nSetup finished. Activate the environment in new terminals with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "(If activation is blocked, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)"
