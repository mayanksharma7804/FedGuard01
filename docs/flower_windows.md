# Flower on native Windows (no Ray, no WSL) - what worked

Tested on 7 Oct 2026 with **flwr 1.39.0**, Python 3.11.17, Windows 11 Home (laptop 1).

One command runs the whole check (1 SuperLink + 2 SuperNodes + 3 FedAvg rounds on synthetic data):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\flower_check.ps1
```

Expected ending: `Flower deployment mode works on this Windows laptop.`
Logs of each process: `logs\flower_check\`.

## Three Windows gotchas (all handled inside the script)

1. **`.venv\Scripts` must be on PATH.** SuperLink and SuperNode start a helper program called
   `flower-superexec` by name. Without PATH you get `FileNotFoundError: [WinError 2]`.
2. **The Control API is HTTP on port 8000 in flwr 1.39**, not gRPC on 9093 as in older docs.
   The SuperLink connection in `%USERPROFILE%\.flwr\config.toml` must be:
   ```toml
   [superlink.local-deployment]
   address = "127.0.0.1:8000"
   insecure = true
   ```
3. **Quote `--node-config` when using `Start-Process`.** It joins arguments with spaces, so
   `partition-id=0 num-partitions=2` must be passed as one quoted string.

## Ports used

| Port | Who | What |
|---|---|---|
| 9092 | SuperLink | Fleet API (SuperNodes connect here) - open this in the firewall for the multi-laptop demo |
| 8000 | SuperLink | Runtime + Control HTTP API (`flwr run` talks to this) |
| 9094, 9095, ... | each SuperNode | its own Runtime API - must be different per SuperNode on one laptop |

## Manual version (for the live demo in Phase 8)

```powershell
.\.venv\Scripts\Activate.ps1                       # puts flower-superexec on PATH
flower-superlink --insecure                        # window 1
flower-supernode --insecure --superlink 127.0.0.1:9092 --port 9094 --node-config "partition-id=0 num-partitions=2"   # window 2
flower-supernode --insecure --superlink 127.0.0.1:9092 --port 9095 --node-config "partition-id=1 num-partitions=2"   # window 3
cd flower_check; flwr run . local-deployment --stream                                                                # window 4
```

For the multi-laptop demo, the SuperNodes on other laptops use `--superlink <laptop-A-IP>:9092`.

## API notes for the FedGuard strategy (Phase 7)

- Message API: `flwr.serverapp.strategy.FedAvg`, `flwr.clientapp.ClientApp` with `@app.train()` / `@app.evaluate()`.
- Model weights travel as `ArrayRecord` (`ArrayRecord(model.state_dict())`, `.to_torch_state_dict()`).
- To plug FedGuard in: subclass `FedAvg` and override
  `aggregate_train(self, server_round, replies) -> (ArrayRecord | None, MetricRecord | None)`.
  Each reply's `msg.content` holds the client's `arrays` and `metrics` (where the declared sigma, C, B, T, lr will go).
