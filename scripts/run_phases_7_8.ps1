# Run Phase 7 then Phase 8 sequentially - ONE SHARD AT A TIME to avoid GPU deadlock (F41).
# Launch: powershell -ExecutionPolicy Bypass -File scripts\run_phases_7_8.ps1
$root   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\python.exe"

function Run-Shard {
    param($sweep, $shard, $logFile)
    Write-Host "  Shard $shard ..." -ForegroundColor Yellow
    $p = Start-Process -FilePath $python -ArgumentList "scripts/run_experiments.py --sweep $sweep --shard $shard" `
        -WorkingDirectory $root -RedirectStandardOutput $logFile `
        -RedirectStandardError ($logFile -replace '\.log','_err.log') -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    Write-Host "  Shard $shard done (exit $($p.ExitCode))" -ForegroundColor Yellow
}

function Run-Phase {
    param($name, $sweep, $logPrefix)
    Write-Host "=== $name ===" -ForegroundColor Cyan
    Run-Shard $sweep "0/2" "logs\${logPrefix}_s0.log"
    Run-Shard $sweep "1/2" "logs\${logPrefix}_s1.log"
    Write-Host "=== $name DONE ===" -ForegroundColor Green
}

Run-Phase "Phase 7 (B5 FedGuard)" "configs/sweeps/p7_fedguard_main.yaml" "p7"
Run-Phase "Phase 8 ablation (E5)" "configs/sweeps/p8_e5_ablation.yaml" "p8"

Write-Host "=== Collecting results ===" -ForegroundColor Cyan
& $python scripts/collect_results.py
& $python scripts/make_plots.py --b1 --b2 --attacks --e4 --fedguard --ablation
Write-Host "=== ALL DONE ===" -ForegroundColor Green
