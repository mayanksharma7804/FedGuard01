"""ClientApp: receives global weights, trains locally on its own synthetic partition, replies."""
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_check.task import accuracy, make_data, make_model, train

app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    x, y = make_data(int(context.node_config["partition-id"]))
    loss = train(model, x, y, int(context.run_config["local-epochs"]), float(context.run_config["lr"]))
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(x)}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    x, y = make_data(int(context.node_config["partition-id"]) + 100)   # unseen samples
    content = RecordDict({"metrics": MetricRecord({"eval_acc": accuracy(model, x, y), "num-examples": len(x)})})
    return Message(content=content, reply_to=msg)
