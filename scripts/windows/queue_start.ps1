# Start (or resume) the experiment queue in the background, logs in logs\queue_s<i>.log.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_start.ps1           # 1 job (default, cool)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_start.ps1 -Jobs 2   # 2 jobs (faster, hotter)
# Each job uses at most -Threads CPU threads, and the thermal guard (thermal_guard.ps1) makes the jobs wait
# whenever the CPU or GPU reaches -High C (default 75, checked every 25 s) until they are back at -Low C (default 68).
# Safe to run again at any time: finished runs are skipped and interrupted runs resume from their checkpoint.
#
# Several machines (docs/multi_machine.md): every machine runs ONE job on its own share of the SAME queue,
#   main laptop:     queue_start.ps1 -Machine 0 -Machines 3
#   helper laptops:  queue_start.ps1 -Machine 1 -Machines 3      and      queue_start.ps1 -Machine 2 -Machines 3
# Runs listed in results\done_runs.txt (finished on any machine, shared through git) are skipped everywhere.
param([int]$Jobs = 1, [int]$Threads = 2, [int]$High = 75, [int]$Low = 68, [int]$Machine = 0, [int]$Machines = 1)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Runs = Join-Path $Repo "results\runs"
New-Item -ItemType Directory -Force (Join-Path $Repo "logs") | Out-Null
New-Item -ItemType Directory -Force $Runs | Out-Null
if ($Machines -lt 1 -or $Machine -lt 0 -or $Machine -ge $Machines) { throw "-Machine must be 0 ... (-Machines - 1)" }

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_experiments\.py' }
if ($running) {
    Write-Host "The queue is already running (PID $($running.ProcessId -join ', ')). No new jobs started."
} else {
    foreach ($f in "PAUSE", "PAUSE_AFTER_RUN") { Remove-Item (Join-Path $Runs $f) -ErrorAction SilentlyContinue }
    $Sweeps = Get-Content (Join-Path $Repo "configs\sweeps\QUEUE.txt") | Where-Object { $_ -and -not $_.StartsWith("#") }
    powercfg /change standby-timeout-ac 0 | Out-Null          # no sleep on AC power while training
    $env:FEDGUARD_THREADS = "$Threads"; $env:OMP_NUM_THREADS = "$Threads"; $env:MKL_NUM_THREADS = "$Threads"
    $skip = @()
    if (Test-Path (Join-Path $Repo "results\done_runs.txt")) { $skip = @("--skip-file", "results\done_runs.txt") }
    if ($Machines -gt 1) { $Jobs = 1 }
    for ($i = 0; $i -lt $Jobs; $i++) {
        $log = "logs\queue_s$i.log"
        $shard = if ($Machines -gt 1) { "$Machine/$Machines" } else { "$i/$Jobs" }
        $argList = @("-u", "scripts\run_experiments.py", "--sweep") + $Sweeps + @("--shard", $shard) + $skip
        Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Repo -WindowStyle Hidden `
            -RedirectStandardOutput $log -RedirectStandardError "logs\queue_s${i}_err.log"
        Write-Host "Started shard $shard -> $log"
        Write-Host "  Its first line shows the 'queue id': it must be the SAME on every machine."
    }
}
$guard = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'thermal_guard\.ps1' }
if (-not $guard) {
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ('"' + (Join-Path $PSScriptRoot "thermal_guard.ps1") + '"'), "-High", "$High", "-Low", "$Low")
    Write-Host "Thermal guard started (pause jobs at $High C, resume at $Low C) -> logs\thermal.log"
}
Write-Host "Pause: scripts\windows\queue_pause.ps1    Status: scripts\windows\queue_status.ps1"
