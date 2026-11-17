# Thermal guard for long training runs on the laptop (keeps CPU and GPU at or below ~80 C).
# Every 25 s it reads the CPU thermal zone and the GPU temperature. When either reaches -High, it creates
# results\runs\THROTTLE: the training jobs then wait before their next client (GPU and CPU idle) until the
# hottest sensor is back at -Low, when the file is removed. Waiting never changes a result.
# Started automatically by queue_start.ps1; it exits by itself when no training job is left.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\thermal_guard.ps1 [-High 75] [-Low 68] [-Every 25]
param([int]$High = 75, [int]$Low = 68, [int]$Every = 25)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Throttle = Join-Path $Repo "results\runs\THROTTLE"
$Log = Join-Path $Repo "logs\thermal.log"

function Get-CpuTemp {
    try {
        $s = (Get-Counter "\Thermal Zone Information(*)\Temperature" -ErrorAction Stop).CounterSamples
        return [math]::Round((($s | Measure-Object CookedValue -Maximum).Maximum) - 273.15, 1)
    } catch { return $null }
}
function Get-GpuTemp {
    try { return [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) } catch { return $null }
}
function Write-Log($msg) { Add-Content -Path $Log -Value ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg) }

Write-Log "thermal guard started (high $High C, low $Low C)"
$idle = 0; $tick = 0
while ($true) {
    $cpu = Get-CpuTemp; $gpu = Get-GpuTemp
    $hot = (@($cpu, $gpu) | Where-Object { $_ -ne $null } | Measure-Object -Maximum).Maximum
    if ($hot -ne $null) {
        if ($hot -ge $High -and -not (Test-Path $Throttle)) {
            New-Item -ItemType File -Force $Throttle | Out-Null; Write-Log "HOT  cpu $cpu C  gpu $gpu C -> throttle on"
        } elseif ($hot -le $Low -and (Test-Path $Throttle)) {
            Remove-Item $Throttle -ErrorAction SilentlyContinue; Write-Log "OK   cpu $cpu C  gpu $gpu C -> throttle off"
        }
    }
    Write-Log ("temps cpu {0} C  gpu {1} C  throttle {2}" -f $cpu, $gpu, (Test-Path $Throttle))
    $tick++
    $jobs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_experiments\.py' }
    if ($jobs) { $idle = 0 } else { $idle++ }
    if ($idle -ge 3) {
        Remove-Item $Throttle -ErrorAction SilentlyContinue
        Write-Log "no training job left - thermal guard stopped"; break
    }
    Start-Sleep -Seconds $Every
}
