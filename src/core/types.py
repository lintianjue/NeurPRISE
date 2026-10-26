from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import numpy as np
import torch


@dataclass
class Instance:
    data: Dict[str, np.ndarray]
    cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # safety for any weird constructions
        if getattr(self, "cache", None) is None:
            self.cache = {}

    def __setstate__(self, state):
        # backward compat for older pickles missing "cache"
        self.__dict__.update(state)
        if getattr(self, "cache", None) is None:
            self.cache = {}

    def cache_get(self, key: str, default: Any = None) -> Any:
        cache = getattr(self, "cache", None)
        if cache is None:
            self.cache = {}
            cache = self.cache
        return cache.get(key, default)

    def cache_set(self, key: str, value: Any) -> None:
        cache = getattr(self, "cache", None)
        if cache is None:
            self.cache = {}
            cache = self.cache
        cache[key] = value

    def to(self, device):
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        self.data = new_data
        return self


@dataclass
class ForwardTrace:
    K: int
    S: int
    selected_idx: np.ndarray              # [T]
    candidate_scores: List[np.ndarray]    # list of [S] arrays, one per step

    # NEW: plateau / stop supervision (trace-only, no need exact obj)
    best_scores: np.ndarray              # [T]
    gains: np.ndarray                    # [T]  (gains[0] = best_scores[0])
    stop_step: int                       # in [0..T], stop at t>=stop_step
    step_mask: np.ndarray                # [T] bool, True => include CE loss
    step_cumulative_time_sec: Optional[np.ndarray] = None  # [T], non-decreasing



@dataclass
class ForwardTraceBatch:
    """
    Batched/padded version of ForwardTrace.

    Contains the full set of ForwardTrace fields, plus `T` (per-instance lengths).
    """
    # copied (and validated to be consistent across batch)
    K: int
    S: int

    # ForwardTrace fields (batched/padded)
    selected_idx: torch.Tensor        # [B, T_max] long, padded with -1
    candidate_scores: torch.Tensor    # [B, T_max, S] float, padded with -inf
    best_scores: torch.Tensor         # [B, T_max] float, padded with nan
    gains: torch.Tensor              # [B, T_max] float, padded with nan
    stop_step: torch.Tensor          # [B] long
    step_mask: torch.Tensor           # [B, T_max] bool, padded with False

    # extra batch metadata
    T: torch.Tensor                   # [B] long, actual length per instance

@dataclass
class Labels:
    # ------------- core solver outputs (per instance) -------------
    # Note: these can be scalar (float/bool) per instance or [B]-shaped after batching.
    obj: Optional[np.ndarray] = None        # robust objective
    feasible: Optional[np.ndarray] = None   # feasibility flag or obj-as-feasible

    # For some problems you may still want the full decision x*
    x: Optional[np.ndarray] = None          # shape depends on problem (e.g. [N])

    # ------------- scenario-level labels/metadata -------------
    # Index of the "target" scenario (e.g. worst scenario); used by scenario_ce_loss
    sce_target_idx: Optional[np.ndarray] = None   # e.g. () or [1] per instance, [B] after batching

    # Full ranking of scenarios (0 = worst, etc.), if you want to log or use top-K later
    sce_ranking: Optional[np.ndarray] = None      # e.g. [S]

    # Scenario scores (recourse costs, regret, etc.)
    sce_scores: Optional[np.ndarray] = None       # e.g. [S]

    # Top-K scenario indices, if you care about multiple worst scenarios
    sce_top_idx: Optional[np.ndarray] = None      # e.g. [K]

    obj_bound: Optional[np.ndarray] = None

    mip_gap: Optional[np.ndarray] = None

    # ------------- catch-all metadata -------------
    meta: Dict[str, Any] = field(default_factory=dict)

    forward_trace: Optional[ForwardTrace] = None


Batch = Dict[str, Any]

@dataclass
class DecoderOutput:
    scores: torch.Tensor                    # [B, S]
    logits: Optional[torch.Tensor] = None   # [B, S] (for RL, policy gradients, etc.)
    probs: Optional[torch.Tensor] = None    # [B, S]
    ranking: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = None              # e.g. attention weights, masks, etc.

