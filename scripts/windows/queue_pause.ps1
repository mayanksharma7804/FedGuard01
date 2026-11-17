# Pause the experiment queue safely.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_pause.ps1          # after the current ROUND (~1-2 min)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_pause.ps1 -AfterRun  # after the current RUN
# Every round is checkpointed, so a round-level pause loses nothing. Resume with queue_start.ps1.
param([switch]$AfterRun)
$Runs = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path "results\runs"
$file = if ($AfterRun) { "PAUSE_AFTER_RUN" } else { "PAUSE" }
New-Item -ItemType File -Force (Join-Path $Runs $file) | Out-Null
Write-Host "Created results\runs\$file - the jobs stop at the next safe point."
Write-Host "Wait until queue_status.ps1 shows 'no training process' before closing the lid / shutting down."
