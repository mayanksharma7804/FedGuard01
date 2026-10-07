# =============================================================================
# Phase 1 check: run Flower in DEPLOYMENT mode on native Windows (no Ray, no WSL).
# Starts 1 SuperLink + 2 SuperNodes as background processes, runs flower_check/,
# prints the result, then stops everything.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\flower_check.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Bin = Join-Path $Repo ".venv\Scripts"
$Logs = Join-Path $Repo "logs\flower_check"
New-Item -ItemType Directory -Force $Logs | Out-Null
# SuperLink and SuperNode start a helper called flower-superexec by name, so .venv\Scripts must be on PATH.
$env:Path = "$Bin;$env:Path"

# A named SuperLink connection for local deployment (stored in %USERPROFILE%\.flwr\config.toml).
$Cfg = Join-Path $env:USERPROFILE ".flwr\config.toml"
if (-not (Test-Path $Cfg)) { & (Join-Path $Bin "flwr.exe") config list | Out-Null }
# Flower 1.39: the Control API is HTTP on the SuperLink's --host/--port (default 127.0.0.1:8000), not gRPC 9093.
$text = Get-Content $Cfg -Raw
if ($text -notmatch '\[superlink\.local-deployment\]') {
    Add-Content -Path $Cfg -Value "`n[superlink.local-deployment]`naddress = `"127.0.0.1:8000`"`ninsecure = true"
} elseif ($text -match '127\.0\.0\.1:9093') {
    Set-Content -Path $Cfg -Value ($text -replace '127\.0\.0\.1:9093', '127.0.0.1:8000') -NoNewline
}

$procs = @()
try {
    Write-Host "==> Starting SuperLink"
    $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath (Join-Path $Bin "flower-superlink.exe") `
        -ArgumentList "--insecure" `
        -RedirectStandardOutput "$Logs\superlink.out" -RedirectStandardError "$Logs\superlink.err"
    Start-Sleep -Seconds 6
    foreach ($i in 0, 1) {
        Write-Host "==> Starting SuperNode $i"
        $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath (Join-Path $Bin "flower-supernode.exe") `
            -ArgumentList "--insecure", "--superlink", "127.0.0.1:9092", "--port", (9094 + $i), `
                          "--node-config", "`"partition-id=$i num-partitions=2`"" `
            -RedirectStandardOutput "$Logs\supernode$i.out" -RedirectStandardError "$Logs\supernode$i.err"
    }
    Start-Sleep -Seconds 8
    Write-Host "==> flwr run (streams the server log)"
    Push-Location (Join-Path $Repo "flower_check")
    & (Join-Path $Bin "flwr.exe") run . local-deployment --stream
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "flwr run failed with exit code $code (see $Logs)" }
    Write-Host "`nFlower deployment mode works on this Windows laptop." -ForegroundColor Green
}
finally {
    Write-Host "==> Stopping Flower processes"
    foreach ($p in $procs) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    Get-Process flower-superlink, flower-supernode, flwr-serverapp, flwr-clientapp -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}
