# Laptop 2 (RTX 4050) - steps

Laptop 2 has no Claude Code. Do these steps exactly, then copy the outputs back to laptop 1
(paste them to Claude, or commit them) so Claude can check them.

## Phase 1 - one-time setup (about 15-20 minutes, mostly downloading)

1. Install **Git for Windows** if it is missing: https://git-scm.com/download/win (default options).
2. Update the NVIDIA driver (nvidia.com > Drivers, or the NVIDIA App). Any driver from 2024 or later is fine.
3. Get the project. Use a path **without spaces** on this laptop, e.g. `D:\Project_Final`:
   ```powershell
   git clone https://github.com/<your-username>/<repo-name>.git D:\Project_Final
   cd D:\Project_Final
   ```
   (No GitHub? Copy the whole `Project_Final` folder by USB **without** the `.venv` folder.)
4. One-command setup (installs Python 3.11, all packages, CUDA PyTorch):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
   ```
   It must end with `setup OK` and `cuda True (NVIDIA GeForce RTX 4050 ...)`.
5. Check Flower works on this laptop too:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\flower_check.ps1
   ```
   It must end with `Flower deployment mode works on this Windows laptop.`
6. Measure training speed on the RTX 4050:
   ```powershell
   .\.venv\Scripts\python.exe scripts\check_gpu_dp.py
   ```
7. Put the dataset where the code expects it: copy the NF-UNSW-NB15-v2 CSV into `data\raw\`.
   Then record its hash:
   ```powershell
   Get-FileHash data\raw\*.csv -Algorithm SHA256
   ```

### Bring back to laptop 1
- The full output of steps 4 (last ~15 lines), 5 (last line) and 6 (the whole table).
- The output of step 7 and the CSV file name + size.

## Later phases
Each phase will add its own section here with the exact commands to run on laptop 2.
