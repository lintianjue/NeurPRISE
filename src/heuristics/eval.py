"""PRISE traces used to supervise NeurPRISE training."""
import logging
from typing import Optional
from src.heuristics.search_framework import StepwiseGreedySearch, Incremental2ROSolver
from src.utils.batch_utils import _instance_to_numpy_dict
from src.core.types import Instance
from src.utils.eval_utils import _compute_K_from_ratio
logger = logging.getLogger(__name__)

def _compute_effective_K(S: int, k_ratio: float) -> int:
    """Compute K from ratio and enforce at least 1."""
    K_i = _compute_K_from_ratio(S, k_ratio)
    return max(1, K_i)

def _forward_with_trace(
    problem,
    data_np: dict,
    full_obj_val: float,
    K: int,
    milp_time_limit=None,
    log_prefix: str = "",
    threads: int = 1,
    verbose: bool = False,
    *,
    stop_M: int = 1,
    eps_gain: float = 0.0001,
    milp_mip_gap: Optional[float] = None,
):
    """
    Forward greedy heuristic producing a full step-wise trace.
    Uses the optimized StepwiseGreedySearch framework.

    milp_mip_gap: A3 cheap-label knob B. Relaxes the inner subset-solve MIP gap
    (None => exact Gurobi default). Governs supervision fidelity, not BigMILP.
    """
    # 1. Wrap data_np into an Instance (for solver compatibility)
    instance_wrapper = Instance(data_np)
    sce = problem.get_scenarios(data_np)
    S = sce.shape[0]

    # 2. Initialize Solver
    # If the problem class doesn't support incremental, we might need a fallback,
    # but based on current requirements, we assume it does.
    try:
        solver = Incremental2ROSolver(
            problem,
            instance_wrapper,
            time_limit=milp_time_limit,
            threads=threads,
            verbose=verbose,
            mip_gap=milp_mip_gap,
        )
    except NotImplementedError:
        logger.warning(f"{log_prefix} Problem class missing incremental methods. "
                       "Ensure build_base_model/add_scenario_to_model are implemented.")
        raise

    # 3. Run Search
    searcher = StepwiseGreedySearch(
        solver=solver,
        S=S,
        stop_M=stop_M,
        eps_gain=eps_gain,
        log_prefix=log_prefix
    )

    try:
        trace = searcher.search(K)
        return trace
    finally:
        solver.cleanup()
