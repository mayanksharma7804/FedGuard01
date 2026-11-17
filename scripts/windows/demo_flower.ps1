# =============================================================================
# Live FedGuard demo on ONE laptop (plan 13.3, one-laptop fallback): 1 SuperLink + 5 SuperNodes on 127.0.0.1,
# our AggregatorStrategy, and the live trust-weight dashboard in its own window.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\demo_flower.ps1                       # FedGuard
#   powershell -ExecutionPolicy Bypass -File scripts\windows\demo_flower.ps1 -Aggregator autogm    # AutoGM
# Output (live): results\demo_<aggregator>\  - replay later with
#   python scripts\demo_dashboard.py --dir results\demo_<aggregator>
# Multi-laptop version: docs\demo.md. Never run this next to training jobs (RAM, Findings F25).
# =============================================================================
param([ValidateSet("fedguard", "autogm", "fedavg")][string]$Aggregator = "fedguard", [int]$Rounds = 10,
      [string]$Device = "auto", [switch]$NoDashboard)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Bin = Join-Path $Repo ".venv\Scripts"
$Logs = Join-Path $Repo "logs\demo"
$Out = "results/demo_$Aggregator"
$Config = "configs/demo_$Aggregator.yaml"
if ($Aggregator -eq "fedavg") { $Config = "configs/demo_autogm.yaml" }   # same setting, aggregator overridden below
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:Path = "$Bin;$env:Path"                       # SuperLink/SuperNode spawn flower-superexec by name

$training = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_experiments\.py' }
if ($training) {
    Write-Host "Training jobs are running. Pause them first (RAM, Findings F25):" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\queue_pause.ps1"
    exit 1
}
$Clients = 5

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
        Write-Host "==> Starting SuperNode $i (organisation $i)"
        $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath (Join-Path $Bin "flower-supernode.exe") `
            -ArgumentList "--insecure", "--superlink", "127.0.0.1:9092", "--port", (9094 + $i), `
                          "--node-config", "`"partition-id=$i num-partitions=$Clients`"" `
            -RedirectStandardOutput "$Logs\supernode$i.out" -RedirectStandardError "$Logs\supernode$i.err"
    }
    if (-not $NoDashboard) {
        Write-Host "==> Opening the live dashboard"
        $procs += Start-Process -PassThru -FilePath (Join-Path $Bin "python.exe") -WorkingDirectory $Repo `
            -ArgumentList "scripts\demo_dashboard.py", "--dir", $Out
    }
    Start-Sleep -Seconds 10
    Write-Host "==> flwr run: $Aggregator, $Rounds rounds, 5 organisations (client 0 = attacker, client 1 = eps-1 hospital)"
    $rc = "config='$Config' num-rounds=$Rounds out-dir='$Out' device='$Device'"
    if ($Aggregator -eq "fedavg") { $rc += " aggregator='fedavg'" }
    Push-Location (Join-Path $Repo "flower_app")
    & (Join-Path $Bin "flwr.exe") run . local-deployment --stream --run-config $rc
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "flwr run failed with exit code $code (see $Logs)" }
    Write-Host "==> Done. Results in $Out. Close the dashboard window when finished." -ForegroundColor Green
}
finally {
    Write-Host "==> Stopping Flower processes"
    foreach ($p in $procs) {
        if ($p -and -not $p.HasExited -and $p.ProcessName -ne "python") { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Get-Process flower-superlink, flower-supernode, flwr-serverapp, flwr-clientapp, flower-superexec -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}
