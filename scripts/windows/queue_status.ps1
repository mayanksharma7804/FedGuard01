# Show queue progress, running jobs and GPU temperature.
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Sweeps = Get-Content "configs\sweeps\QUEUE.txt" | Where-Object { $_ -and -not $_.StartsWith("#") }
$list = & $Py scripts\run_experiments.py --sweep @Sweeps --dry-run
$done = ($list | Select-String "\[done\]").Count; $todo = ($list | Select-String "\[todo\]").Count
Write-Host "Queue: $done done, $todo to do"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_experiments\.py' }
if ($procs) { Write-Host "Training processes: $($procs.Count)" } else { Write-Host "no training process (safe to shut down)" }
foreach ($f in "PAUSE", "PAUSE_AFTER_RUN") { if (Test-Path "results\runs\$f") { Write-Host "pause requested: results\runs\$f" } }
Get-ChildItem logs\queue_s?.log -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "--- $($_.Name)"; Get-Content $_.FullName -Tail 2 }
$cpu = try { $s = (Get-Counter "\Thermal Zone Information(*)\Temperature" -ErrorAction Stop).CounterSamples
    [math]::Round((($s | Measure-Object CookedValue -Maximum).Maximum) - 273.15, 1) } catch { "n/a" }
$gpu = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader
Write-Host "CPU $cpu C   GPU $gpu   throttle: $(Test-Path 'results
uns\THROTTLE')"
if (Test-Path logs\thermal.log) { Get-Content logs\thermal.log -Tail 3 }
