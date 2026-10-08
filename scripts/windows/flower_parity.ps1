# =============================================================================
# Phase 4 engine-parity check: run FedAvg through Flower (1 SuperLink + 5 SuperNodes on this laptop,
# deployment mode, no Ray) using our AggregatorStrategy, then compare with our own engine.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\flower_parity.ps1
# =============================================================================
param([int]$Clients = 5)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Bin = Join-Path $Repo ".venv\Scripts"
$Logs = Join-Path $Repo "logs\flower_parity"
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:Path = "$Bin;$env:Path"                       # SuperLink/SuperNode spawn flower-superexec by name

$Cfg = Join-Path $env:USERPROFILE ".flwr\config.toml"
if (-not (Test-Path $Cfg)) { & (Join-Path $Bin "flwr.exe") config list | Out-Null }
$text = Get-Content $Cfg -Raw
if ($text -notmatch '\[superlink\.local-deployment\]') {
    Add-Content -Path $Cfg -Value "`n[superlink.local-deployment]`naddress = `"127.0.0.1:8000`"`ninsecure = true"
}

$procs = @()
try {
    Write-Host "==> Starting SuperLink"
    $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath (Join-Path $Bin "flower-superlink.exe") `
        -ArgumentList "--insecure" -RedirectStandardOutput "$Logs\superlink.out" -RedirectStandardError "$Logs\superlink.err"
    Start-Sleep -Seconds 6
    for ($i = 0; $i -lt $Clients; $i++) {
        Write-Host "==> Starting SuperNode $i"
        $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath (Join-Path $Bin "flower-supernode.exe") `
            -ArgumentList "--insecure", "--superlink", "127.0.0.1:9092", "--port", (9094 + $i), `
                          "--node-config", "`"partition-id=$i num-partitions=$Clients`"" `
            -RedirectStandardOutput "$Logs\supernode$i.out" -RedirectStandardError "$Logs\supernode$i.err"
    }
    Start-Sleep -Seconds 10
    Write-Host "==> flwr run (5 FedAvg rounds through AggregatorStrategy)"
    Push-Location (Join-Path $Repo "flower_app")
    & (Join-Path $Bin "flwr.exe") run . local-deployment --stream
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "flwr run failed with exit code $code (see $Logs)" }
}
finally {
    Write-Host "==> Stopping Flower processes"
    foreach ($p in $procs) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    Get-Process flower-superlink, flower-supernode, flwr-serverapp, flwr-clientapp, flower-superexec -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

Write-Host "==> Running the same config in our engine and comparing"
& (Join-Path $Bin "python.exe") (Join-Path $Repo "scripts\compare_flower_parity.py")
exit $LASTEXITCODE
