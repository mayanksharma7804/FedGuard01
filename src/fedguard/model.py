"""Local intrusion-detection model (plan Figure 8.1, paper Fig. 4).

One CNN-LSTM, identical in every client and every baseline (B0-B5):
  input (B, 1, F) -> Conv1d(32) -> Conv1d(64) + MaxPool(2) -> read as a sequence of F//2 steps
  -> DPLSTM(64) -> last step -> Dropout(0.3) -> Linear(n_classes)

Why so small: DP noise length grows with sqrt(number of parameters); a big model drowns its signal.
Why DPLSTM everywhere: Opacus cannot train nn.LSTM, and using nn.LSTM for non-DP runs would make the
baselines two different models. No BatchNorm (Opacus cannot use it).
"""
import torch
import torch.nn as nn
from opacus.layers import DPLSTM


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, conv1: int = 32, conv2: int = 64,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(conv1, conv2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = DPLSTM(conv2, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:                       # (B, F) -> (B, 1, F)
            x = x.unsqueeze(1)
        h = self.conv(x).permute(0, 2, 1)      # (B, F//2, conv2): features read as a sequence
        out, _ = self.lstm(h)
        return self.head(self.drop(out[:, -1]))


def build_model(n_features: int, n_classes: int, **kwargs) -> CNNLSTM:
    return CNNLSTM(n_features, n_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
