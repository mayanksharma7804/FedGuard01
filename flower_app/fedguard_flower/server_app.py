"""Flower ServerApp: runs our aggregator through Flower and records validation macro-F1 per round.

Used for the engine-parity check (Phase 4) and the live demo (Phase 8). Everything the demo dashboard
(scripts/demo_dashboard.py) shows is written to `out-dir` while the run is going:
  clients.json        who each client is (data size, epsilon, attacker or not, declared noise radius)
  flower_weights.csv  trust weight of every client in every round (rewritten after each round)
  flower_rounds.csv   validation macro-F1 / accuracy of the global model per round (rewritten after each round)
"""
import json

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedguard import dp
from fedguard.flower_strategy import AggregatorStrategy, is_attacker, write_csv_atomic
from fedguard.train import evaluate
from fedguard.utils import REPO_ROOT
from fedguard_flower.common import initial_model, setup

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg, data, parts, device, _ = setup(context.run_config)
    model = initial_model(cfg, data)
    rounds = int(context.run_config["num-rounds"])
    out = REPO_ROOT / context.run_config["out-dir"]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("flower_weights.csv", "flower_rounds.csv"):          # a fresh run starts with empty live files
        (out / f).unlink(missing_ok=True)

    k = cfg["fl"]["clients"]
    d = sum(p.numel() for p in model.parameters())
    clients = []
    for cid in range(k):
        plan = dp.client_privacy_plan(len(parts[cid]), cfg, cid)
        clients.append({"client": cid, "n": len(parts[cid]), "epsilon": plan.get("target_epsilon"),
                        "attacker": is_attacker(cfg, cid),
                        "declared_radius": dp.noise_radius({**plan, "lr": cfg["train"]["lr"]}, d)
                        if dp.is_dp(cfg) else 0.0})
    json.dump({"clients": clients, "aggregator": cfg["fl"]["aggregator"], "attack": cfg["attack"],
               "rounds": rounds, "config": context.run_config["config"]}, open(out / "clients.json", "w"), indent=2)

    history = []

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        val = evaluate(model, data.X_va, data.y_va, data.classes, device, cfg["train"]["eval_batch"])
        history.append({"round": server_round, "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                        "val_benign_fpr": val["benign_fpr"], "val_attack_detection_rate": val["attack_detection_rate"]})
        write_csv_atomic([h for h in history if h["round"] > 0] or history, out / "flower_rounds.csv")
        return MetricRecord({"val_macro_f1": val["macro_f1"]})

    strategy = AggregatorStrategy(initial_model(cfg, data), cfg["fl"]["aggregator"]["name"], cfg["fl"]["aggregator"],
                                  live_csv=out / "flower_weights.csv",
                                  fraction_train=1.0, fraction_evaluate=0.0,
                                  min_train_nodes=k, min_available_nodes=k)
    strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()), num_rounds=rounds,
                   train_config=ConfigRecord({}), evaluate_fn=evaluate_fn)

    write_csv_atomic([h for h in history if h["round"] > 0], out / "flower_rounds.csv")
    write_csv_atomic(strategy.weights_log, out / "flower_weights.csv")
    json.dump({"rounds": rounds, "seed": cfg["seed"], "config": context.run_config["config"]},
              open(out / "flower_run.json", "w"), indent=2)
    print("FLOWER RUN FINISHED:", out)
