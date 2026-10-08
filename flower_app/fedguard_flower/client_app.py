"""Flower ClientApp: one organisation. Its partition id comes from the SuperNode's --node-config."""
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedguard.flower_strategy import client_round
from fedguard_flower.common import initial_model, setup

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, data, parts, device, loss_fn = setup(context.run_config)
    cid = int(context.node_config["partition-id"])
    rnd = int(msg.content["config"]["server-round"])
    model = initial_model(cfg, data).to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    idx = parts[cid]
    loss, meta = client_round(model, data.X_tr[idx], data.y_tr[idx], cfg, cid, rnd, loss_fn, device)
    metrics = {"num-examples": meta["n"], "cid": cid, "train_loss": loss,
               "lr": meta["lr"], "sigma": meta["sigma"], "C": meta["C"], "B": meta["B"], "T": meta["T"]}
    return Message(content=RecordDict({"arrays": ArrayRecord(model.state_dict()),
                                       "metrics": MetricRecord(metrics)}), reply_to=msg)
