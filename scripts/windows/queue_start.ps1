# Start (or resume) the experiment queue in the background, logs in logs\queue_s<i>.log.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_start.ps1           # 1 job (default, cool)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_start.ps1 -Jobs 2   # 2 jobs (faster, hotter)
# Each job uses at most -Threads CPU threads and runs its own thermal guard (inside the training process): it waits
# before the next client whenever the CPU or GPU reaches -High C (default 75, read every -Every = 25 s) until both
# are back at -Low C (default 68). Jobs are started through Windows (WMI), so closing this window or ending a
# Claude Code session does not stop them; stop them with queue_pause.ps1.
# Safe to run again at any time: finished runs are skipped and interrupted runs resume from their checkpoint.
#
# Several machines (docs/multi_machine.md): every machine runs ONE job on its own share of the SAME queue,
#   main laptop:     queue_start.ps1 -Machine 0 -Machines 3
#   helper laptops:  queue_start.ps1 -Machine 1 -Machines 3      and      queue_start.ps1 -Machine 2 -Machines 3
# Runs listed in results\done_runs.txt (finished on any machine, shared through git) are skipped everywhere.
# A laptop WITHOUT an NVIDIA GPU takes only the runs without DP (nearly as fast on a CPU, F06), the GPU laptop the rest:
#   GPU laptop:      queue_start.ps1 -Only dp
#   CPU-only laptop: queue_start.ps1 -Only nodp -Threads 4
param([int]$Jobs = 1, [int]$Threads = 2, [int]$High = 75, [int]$Low = 68, [int]$Every = 25, [int]$Machine = 0,
      [int]$Machines = 1, [ValidateSet('all', 'dp', 'nodp')][string]$Only = 'all')
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
    $skip = ""
    if (Test-Path (Join-Path $Repo "results\done_runs.txt")) { $skip = "--skip-file results\done_runs.txt" }
    if ($Machines -gt 1) { $Jobs = 1 }
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    for ($i = 0; $i -lt $Jobs; $i++) {
        $shard = if ($Machines -gt 1) { "$Machine/$Machines" } else { "$i/$Jobs" }
        $pyArgs = "-u scripts\run_experiments.py --sweep $($Sweeps -join ' ') --shard $shard $skip --threads $Threads " +
                  "--thermal-high $High --thermal-low $Low --thermal-every $Every --only $Only"
        $cmd = "cmd.exe /s /c `"set OMP_NUM_THREADS=$Threads&& set MKL_NUM_THREADS=$Threads&& `"$Py`" $pyArgs " +
               "> logs\queue_s$i.log 2> logs\queue_s${i}_err.log`""
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
            -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = $startup }
        if ($r.ReturnValue -ne 0) { throw "could not start shard $shard (WMI error $($r.ReturnValue))" }
        Write-Host "Started shard $shard (runs: $Only) -> logs\queue_s$i.log   (thermal guard: pause at $High C, resume at $Low C, every $Every s)"
        Write-Host "  Its first line shows the 'queue id': it must be the SAME on every machine."
    }
}
Write-Host "Pause: scripts\windows\queue_pause.ps1    Status: scripts\windows\queue_status.ps1"
