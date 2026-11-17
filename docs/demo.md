# Live demo - FedGuard on Flower (plan section 13.3)

**What the audience sees:** five organisations train one intrusion detector together. Client 0 is an
attacker (it sends random noise ten times the size of a real update). Client 1 is a hospital that uses the
strongest privacy (epsilon 1), so its honest update is very noisy.

- With **AutoGM**, the attacker's weight drops to 0, but so does the hospital's: the server cannot tell
  honest privacy noise from an attack.
- With **FedGuard**, the attacker still drops to 0, and the hospital keeps a normal weight, because the
  server forgives exactly the noise the hospital declared.

The dashboard shows each client's trust weight per round (attacker red and dashed, hospital blue). The
dotted line is the fair share 1/K; below the dashed line (0.5/K) a client counts as rejected. A 10-round
demo is about the weights, not the accuracy: under DP the model needs ~8 rounds before it predicts
anything except Benign (Findings F46).

## Before the demo

1. Stop all training: `powershell -ExecutionPolicy Bypass -File scripts\windows\queue_pause.ps1`, then
   `scripts\windows\queue_status.ps1` must show `no training process`. Flower next to training runs out
   of RAM (Findings F25).
2. Rehearse twice (plan 13.3, step 5). One run takes about 3-5 minutes.
3. Keep the backup ready (below).

## One laptop (fallback, also used for rehearsal)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\demo_flower.ps1 -Aggregator autogm     # the problem
powershell -ExecutionPolicy Bypass -File scripts\windows\demo_flower.ps1 -Aggregator fedguard   # our fix
```

Each command starts 1 SuperLink + 5 SuperNodes on 127.0.0.1, opens the dashboard window, and runs 10
rounds. Results stay in `results\demo_autogm\` and `results\demo_fedguard\`; logs in `logs\demo\`.

## Several laptops on the same Wi-Fi

Every laptop needs the repo, `.venv` (run `scripts\windows\setup_windows.ps1`) and `data\processed\nf_unsw.npz`
(copy it over; data is never in Git).

**Laptop A (server + dashboard)** - once, in an Administrator PowerShell, open the Fleet port:

```powershell
New-NetFirewallRule -DisplayName "FedGuard Flower" -Direction Inbound -Protocol TCP -LocalPort 9092 -Action Allow
ipconfig          # note laptop A's IPv4 address, e.g. 192.168.1.20
```

Then, in three PowerShell windows on laptop A (after `.\.venv\Scripts\Activate.ps1`):

```powershell
flower-superlink --insecure
python scripts\demo_dashboard.py --dir results\demo_fedguard
cd flower_app; flwr run . local-deployment --stream --run-config "config='configs/demo_fedguard.yaml' num-rounds=10 out-dir='results/demo_fedguard' device='auto'"
```

Start `flwr run` only after all five SuperNodes are connected.

**Laptops B-F (one organisation each)**, with `<i>` = 0 ... 4 (0 is the attacker, 1 the hospital):

```powershell
.\.venv\Scripts\Activate.ps1
flower-supernode --insecure --superlink 192.168.1.20:9092 --port 9094 --node-config "partition-id=<i> num-partitions=5"
```

With fewer laptops, run several SuperNodes on one laptop with different `--port` values (9094, 9095, ...).

## Backup (no live run possible)

- Replay a finished run: `python scripts\demo_dashboard.py --dir results\demo_fedguard` (or `--run
  results\runs\<run>` for any engine run - the same picture from the real 40-round experiments).
- Keep a screen recording of a successful rehearsal (Win + Alt + R starts and stops Xbox Game Bar recording).

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: [WinError 2]` from SuperLink/SuperNode | activate `.venv` first (`flower-superexec` must be on PATH) |
| `flwr run` cannot connect | `%USERPROFILE%\.flwr\config.toml` needs `[superlink.local-deployment]` with `address = "127.0.0.1:8000"` (the script adds it) |
| SuperNodes on other laptops never connect | firewall rule on laptop A, same Wi-Fi, correct IP, port 9092 |
| `numpy._ArrayMemoryError` | training jobs are still running - pause them (F25) |
| Dashboard stays empty | it fills after the first round; check `logs\demo\superlink.err` |
