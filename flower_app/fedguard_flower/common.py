"""Shared setup for the Flower ClientApp / ServerApp: config, data, partition, model."""
import torch
import torch.nn as nn

from fedguard.config import load_config
from fedguard.data import load_processed
from fedguard.model import build_model
from fedguard.partition import load_partition, partition_path
from fedguard.train import class_weights, get_device
from fedguard.utils import REPO_ROOT


def setup(run_config):
    cfg = load_config(REPO_ROOT / run_config["config"],
                      {"seed": int(run_config["seed"]), "train": {"device": run_config["device"]}})
    data = load_processed(cfg["dataset"])
    parts, _ = load_partition(partition_path(cfg["dataset"], cfg["fl"]["alpha"], cfg["fl"]["clients"], cfg["seed"]),
                              fingerprint=data.fingerprint)
    device = get_device(cfg["train"]["device"])
    w = class_weights(data.y_tr, data.n_classes, cfg["train"]["class_weight"])
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    return cfg, data, parts, device, loss_fn


def initial_model(cfg, data):
    torch.manual_seed(cfg["seed"])                    # same initial weights as FLSimulation
    return build_model(data.n_features, data.n_classes, **cfg["model"])
