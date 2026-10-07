"""ServerApp: plain FedAvg for a few rounds, then prints the per-round results."""
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_check.task import make_model

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rounds = int(context.run_config["num-server-rounds"])
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=2)
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(make_model().state_dict()),
        num_rounds=rounds,
        train_config=ConfigRecord({}),
    )
    print("FLOWER CHECK RESULT:", result)
