"""Flower ServerApp: runs our aggregator through Flower and records validation macro-F1 per round."""
import json

import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedguard.flower_strategy import AggregatorStrategy
from fedguard.train import evaluate
from fedguard.utils import REPO_ROOT
from fedguard_flower.common import initial_model, setup

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg, data, parts, device, _ = setup(context.run_config)
    model = initial_model(cfg, data)
    rounds = int(context.run_config["num-rounds"])
    history = []

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        val = evaluate(model, data.X_va, data.y_va, data.classes, device, cfg["train"]["eval_batch"])
        history.append({"round": server_round, "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
        return MetricRecord({"val_macro_f1": val["macro_f1"]})

    k = cfg["fl"]["clients"]
    strategy = AggregatorStrategy(initial_model(cfg, data), cfg["fl"]["aggregator"]["name"], cfg["fl"]["aggregator"],
                                  fraction_train=1.0, fraction_evaluate=0.0,
                                  min_train_nodes=k, min_available_nodes=k)
    strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()), num_rounds=rounds,
                   train_config=ConfigRecord({}), evaluate_fn=evaluate_fn)

    out = REPO_ROOT / context.run_config["out-dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([h for h in history if h["round"] > 0]).to_csv(out / "flower_rounds.csv", index=False)
    pd.DataFrame(strategy.weights_log).to_csv(out / "flower_weights.csv", index=False)
    json.dump({"rounds": rounds, "seed": cfg["seed"], "config": context.run_config["config"]},
              open(out / "flower_run.json", "w"), indent=2)
    print("FLOWER PARITY RUN FINISHED:", out)
