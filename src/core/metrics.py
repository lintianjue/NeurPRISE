from typing import Dict
import torch
from .types import Labels

def metric_obj(labels: Labels) -> torch.Tensor:
    return labels.obj.mean()

def metric_feas(labels: Labels) -> torch.Tensor:
    return labels.feasible.float().mean()

METRICS = {
    "obj": metric_obj,
    "feas": metric_feas,
}

def compute(metrics: list, labels: Labels) -> Dict[str, float]:
    out = {}
    for m in metrics:
        out[m] = float(METRICS[m](labels).item())
    return out