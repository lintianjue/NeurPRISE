import os
from pathlib import Path
from typing import List, Optional

import torch

from src.core.types import ForwardTraceBatch
from src.utils.utils import _find_project_root
from src.utils.vertex_cover.vertex_cover_utils import get_instance_file_name
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
import inspect



def build_problem_from_cfg(problem_cfg):
    """Construct a problem from its configured class and matching constructor arguments.

    Read ``problem_cfg.module`` and ``problem_cfg.instance``; instantiate a configured
    scenario generator with Hydra when accepted. Returns the problem instance.
    """
    ProblemCls = get_class(problem_cfg.module)
    inst_cfg = problem_cfg.instance

    # Convert to a plain dict (so we can do normal Python ops)
    raw_kwargs = OmegaConf.to_container(inst_cfg, resolve=True) or {}

    # Inspect the constructor signature
    sig = inspect.signature(ProblemCls.__init__)
    # Ignore *args/**kwargs when deriving required/optional init params
    param_names = set()
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        param_names.add(p.name)

    init_kwargs = {}

    # Handle scenario_generator specially (Hydra instantiate)
    if "scenario_generator" in raw_kwargs and "scenario_generator" in param_names:
        init_kwargs["scenario_generator"] = instantiate(inst_cfg.scenario_generator)

    # Fill other parameters from cfg if they appear in the signature
    for name in param_names:
        if name == "scenario_generator":
            continue  # already handled
        if name in raw_kwargs:
            init_kwargs[name] = raw_kwargs[name]
        else:
            # Required arg with no default?
            param = sig.parameters[name]
            if param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Required argument '{name}' for {ProblemCls.__name__} "
                    f"is missing from cfg.problem.instance"
                )
            # else: rely on the constructor's default

    problem = ProblemCls(**init_kwargs)
    return problem

def _resolve_data_folder(cfg: DictConfig) -> str:
    """Resolve the absolute data folder path (project-root relative). Pure: an input
    root is read-only, so this only computes a path and creates nothing."""
    project_root = _find_project_root()
    cfg_data_path = cfg.problem.data.path if "data" in cfg.problem and "path" in cfg.problem.data else "data"
    data_path = os.path.join(project_root, cfg_data_path)
    return data_path


def _resolve_instance_spec_path(cfg: DictConfig, data_folder: str, problem_name: str) -> str:
    """Construct the input instance-spec file path from inst_spec_pattern."""
    inst_cfg = cfg.problem.instance
    data_cfg = cfg.problem.data

    # Prefer problem-specific probability parameter:
    # - selection: p_select
    # - vertex_cover: p_edge
    p_select = getattr(inst_cfg, "p_select", None)
    p_edge = getattr(inst_cfg, "p_edge", None)

    # If only one exists, mirror it into the other name so patterns using either key work.
    if p_select is None and p_edge is not None:
        p_select = p_edge
    if p_edge is None and p_select is not None:
        p_edge = p_select

    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)

    file_name = get_instance_file_name(
        seed=cfg.seed,
        instance_cnt=cfg.data.get("instance_cnt", 1000),
        n_nodes=n_nodes,
        n_scenarios=inst_cfg.n_scenarios,
        n_facilities=n_facilities,
        n_customers=n_customers,
        pattern=data_cfg.get("inst_spec_pattern", None),
        # optional extras; only used if pattern references them
        p_edge=p_edge,
        p_select=p_select,
        scenario_name=getattr(inst_cfg, "scenario_name", "default"),
    )
    return os.path.join(data_folder, problem_name, file_name)


def _resolve_label_path(
    cfg: DictConfig,
    data_folder: str,
    problem_name: str,
    pattern_param: str = "inst_spec_label_pattern",
    label_type: str = 'prise'
) -> str:
    """Construct the label (or other) file path from the given pattern name."""
    inst_cfg = cfg.problem.instance
    data_cfg = cfg.problem.data

    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)

    file_name = get_instance_file_name(
        seed=cfg.seed,
        instance_cnt=cfg.data.get("instance_cnt", 1000),
        n_nodes=n_nodes,
        n_scenarios=inst_cfg.n_scenarios,
        n_facilities=n_facilities,
        n_customers=n_customers,
        pattern=data_cfg.get(pattern_param, None),
        p_edge=getattr(inst_cfg, "p_edge", None),
        p_select=getattr(inst_cfg, "p_select", None),
        scenario_name=getattr(inst_cfg, "scenario_name", "default"),
        label_type=label_type
    )

    out_folder = os.path.join(data_folder, problem_name)
    return os.path.join(out_folder, file_name)


def label_filename(base: str, output_mode: str, shard: tuple[int, int] | None) -> str:
    """The label file 02_prise_label.py writes: ``<base>`` for output_mode "both", else
    ``<stem>_<output_mode><suffix>``, then ``<stem>_shard{a}-{b}<suffix>`` when sharded. The
    shard suffix comes last because scripts/pipeline/merge_label_shards.py's SHARD_RE anchors it at
    the end.
    """
    path = Path(base)
    if output_mode != "both":
        path = path.with_name(f"{path.stem}_{output_mode}{path.suffix}")
    if shard is not None:
        start_idx0, end_idx0 = shard
        path = path.with_name(f"{path.stem}_shard{start_idx0}-{end_idx0}{path.suffix}")
    return str(path)


def _collate_forward_trace_from_traces(traces, to_tensor_fn) -> Optional[ForwardTraceBatch]:
    """
    Collate a list of ForwardTrace objects into a ForwardTraceBatch.

    Separated from _collate_forward_trace(labels_list, ...) so it is easy to unit test
    without constructing full label objects.
    """
    if all(t is None for t in traces):
        return None
    if any(t is None for t in traces):
        raise ValueError("Some labels have forward_trace while others don't.")

    def _get_T_from_trace(tr, idx: int) -> int:
        sm = getattr(tr, "step_mask", None)
        if sm is not None:
            Ti = int(len(sm))
        else:
            Ti = int(len(tr.selected_idx))
        if Ti <= 0:
            raise ValueError(f"Empty forward_trace at index {idx}: T={Ti}")
        return Ti

    def _as_1d(name: str, arr, b: int, *, dtype: torch.dtype) -> torch.Tensor:
        if arr is None:
            raise ValueError(f"forward_trace.{name} is None; expected for collation.")
        t = to_tensor_fn(arr).to(dtype=dtype).view(-1)
        return t

    def _require_len(name: str, t: torch.Tensor, exp_len: int, b: int) -> torch.Tensor:
        if int(t.numel()) != int(exp_len):
            raise ValueError(f"Trace {name} length mismatch at b={b}: {t.numel()} vs T={exp_len}")
        return t

    # lengths: prefer step_mask length when available (source-of-truth for stop/plateau supervision)
    T_list: List[int] = []
    for i, t in enumerate(traces):
        T_list.append(_get_T_from_trace(t, i))
    T_max = max(T_list)

    # scenario count S must match across batch
    S_list = [int(t.S) for t in traces]
    if len(set(S_list)) != 1:
        raise ValueError(f"Inconsistent forward_trace.S in batch: {S_list}")
    S = S_list[0]

    # K should match too (optional sanity check)
    K_list = [int(getattr(t, "K", T_max)) for t in traces]
    if len(set(K_list)) != 1:
        raise ValueError(f"Inconsistent forward_trace.K in batch: {K_list}")
    K = K_list[0]

    B = len(traces)

    selected_pad = torch.full((B, T_max), -1, dtype=torch.long)
    scores_pad = torch.full((B, T_max, S), float("-inf"), dtype=torch.float32)
    step_mask_pad = torch.zeros((B, T_max), dtype=torch.bool)
    best_scores_pad = torch.full((B, T_max), float("nan"), dtype=torch.float32)
    gains_pad = torch.full((B, T_max), float("nan"), dtype=torch.float32)
    stop_step_tensor = torch.zeros((B,), dtype=torch.long)

    T_tensor = torch.tensor(T_list, dtype=torch.long)

    for b, tr in enumerate(traces):
        Ti = T_list[b]

        # selected_idx (may be shorter than Ti if stop supervision keeps extra steps)
        sel = _as_1d("selected_idx", tr.selected_idx, b, dtype=torch.long)
        if sel.numel() != len(tr.selected_idx):
            raise ValueError(
                f"Trace selected_idx length mismatch at b={b}: {sel.numel()} vs {len(tr.selected_idx)}"
            )
        if sel.numel() > Ti:
            raise ValueError(
                f"Trace selected_idx longer than T at b={b}: {sel.numel()} vs T={Ti}"
            )
        selected_pad[b, : sel.numel()] = sel

        # candidate_scores: list length must match Ti (step-wise)
        cs = getattr(tr, "candidate_scores", None)
        if cs is None:
            raise ValueError("forward_trace.candidate_scores is None; step-wise training expects it.")
        if len(cs) != Ti:
            raise ValueError(f"Trace candidate_scores length mismatch at b={b}: {len(cs)} vs T={Ti}")

        cs_stack = []
        for t in range(Ti):
            tt = to_tensor_fn(cs[t]).float().view(-1)
            if tt.numel() != S:
                raise ValueError(f"candidate_scores[{t}] has {tt.numel()} elems, expected S={S}")
            cs_stack.append(tt)
        scores_pad[b, :Ti, :] = torch.stack(cs_stack, dim=0)  # [Ti,S]

        # step_mask: MUST come from trace.step_mask if present; otherwise default to all-True for Ti
        sm = getattr(tr, "step_mask", None)
        if sm is None:
            step_mask_pad[b, :Ti] = True
        else:
            sm_t = _require_len("step_mask", _as_1d("step_mask", sm, b, dtype=torch.bool), Ti, b)
            step_mask_pad[b, :Ti] = sm_t
            # Keep selected_idx consistent with step_mask: masked steps must be -1.
            selected_pad[b, :Ti][~sm_t] = -1

        # best_scores / gains
        bs_t = _require_len("best_scores", _as_1d("best_scores", getattr(tr, "best_scores", None), b, dtype=torch.float32), Ti, b)
        best_scores_pad[b, :Ti] = bs_t

        gn_t = _require_len("gains", _as_1d("gains", getattr(tr, "gains", None), b, dtype=torch.float32), Ti, b)
        gains_pad[b, :Ti] = gn_t

        # stop_step
        stop_step = int(getattr(tr, "stop_step"))
        if stop_step < 0 or stop_step > Ti:
            raise ValueError(f"Invalid stop_step at b={b}: {stop_step} not in [0..{Ti}]")
        stop_step_tensor[b] = stop_step

    return ForwardTraceBatch(
        K=int(K),
        S=int(S),
        selected_idx=selected_pad,
        candidate_scores=scores_pad,
        step_mask=step_mask_pad,
        T=T_tensor,
        best_scores=best_scores_pad,
        gains=gains_pad,
        stop_step=stop_step_tensor,
    )


def _collate_forward_trace(labels_list, to_tensor_fn):
    traces = [getattr(lab, "forward_trace", None) for lab in labels_list]
    return _collate_forward_trace_from_traces(traces, to_tensor_fn)
