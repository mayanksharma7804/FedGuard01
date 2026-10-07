"""Prints the versions the team must keep identical, and checks that the key libraries work."""
import platform
import sys

import numpy
import pandas
import sklearn
import torch

print(f"{'os':10s} {platform.system()} {platform.release()}")
print(f"{'python':10s} {sys.version.split()[0]}")
for mod in (torch, numpy, pandas, sklearn):
    print(f"{mod.__name__:10s} {mod.__version__}")

for name in ("opacus", "flwr"):
    try:
        mod = __import__(name)
        print(f"{name:10s} {mod.__version__}")
    except ImportError:
        print(f"{name:10s} NOT INSTALLED")

print(f"{'cuda':10s} {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

x = torch.randn(4, 1, 40)
conv = torch.nn.Conv1d(1, 8, 3, padding=1)
assert conv(x).shape == (4, 8, 40), "torch Conv1d check failed"
if sys.version_info[:2] != (3, 11):
    print("WARNING: the project is pinned to Python 3.11; this is", sys.version.split()[0])
print("setup OK")
