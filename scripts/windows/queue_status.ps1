# Show queue progress, running jobs and CPU / GPU temperature.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\queue_status.ps1
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Sweeps = Get-Content "configs\sweeps\QUEUE.txt" | Where-Object { $_ -and -not $_.StartsWith("#") }
$skip = @()
if (Test-Path "results\done_runs.txt") { $skip = @("--skip-file", "results\done_runs.txt") }
$list = & $Py scripts\run_experiments.py --sweep @Sweeps --dry-run @skip
Write-Host ($list | Select-Object -First 1)
$done = ($list | Select-String "\[done\]").Count
$elsewhere = ($list | Select-String "\[done-elsewhere\]").Count
$todo = ($list | Select-String "\[todo\]").Count
Write-Host "Queue (all machines): $done done here, $elsewhere done elsewhere, $todo to do"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_experiments\.py' }
if ($procs) { Write-Host "Training processes: $($procs.Count)" } else { Write-Host "no training process (safe to shut down)" }
foreach ($f in "PAUSE", "PAUSE_AFTER_RUN") { if (Test-Path "results\runs\$f") { Write-Host "pause requested: results\runs\$f" } }
Get-ChildItem logs\queue_s?.log -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "--- $($_.Name)"; Get-Content $_.FullName -Tail 2 }
$cpu = try { $s = (Get-Counter "\Thermal Zone Information(*)\Temperature" -ErrorAction Stop).CounterSamples
    [math]::Round((($s | Measure-Object CookedValue -Maximum).Maximum) - 273.15, 1) } catch { "n/a" }
$gpu = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader
$throttle = Test-Path (Join-Path $Repo "results\runs\THROTTLE")
Write-Host "CPU $cpu C   GPU $gpu   cooling pause active: $throttle"
if (Test-Path logs\thermal.log) { Get-Content logs\thermal.log -Tail 3 }
