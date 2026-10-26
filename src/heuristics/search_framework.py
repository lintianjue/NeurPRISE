# src/heuristics/search_framework.py

import logging
import time
import numpy as np
from gurobipy import GRB
from typing import List, Optional, Tuple, Iterable


from src.core.types import Instance, ForwardTrace
from src.utils.eval_utils import _compute_plateau_stop_step, _compute_step_mask
from src.utils.gurobi_utils import copy_model_with_payload

logger = logging.getLogger(__name__)


class Incremental2ROSolver:
    """
    Manages a persistent Gurobi model for 2RO evaluation.
    Supports incremental constraints via model cloning or permanent addition (commit).
    """

    def __init__(self, problem, instance: Instance, time_limit: Optional[float], threads: int, verbose: bool,
                 mip_gap: Optional[float] = None,
                 require_converged: bool = True,
                 unconverged_retry_time_limit: Optional[float] = 600.0):
        self.problem = problem
        self.instance = instance
        # Cheap-label knob: relax the MIP optimality gap of the inner
        # reduced-2RO subset solves. None => Gurobi default (exact behavior).
        self.mip_gap = mip_gap

        # Unconverged minimization incumbents can inflate gains and bias candidate argmax.
        # GRB.OPTIMAL certifies the requested MIP gap, including approximate-label settings.
        self.require_converged = require_converged
        self.unconverged_retry_time_limit = unconverged_retry_time_limit
        # Observability: counters so the incidence of retries and exclusions can be read.
        self.n_unconverged_retries = 0
        self.n_retry_rescued = 0
        self.n_excluded_unconverged = 0

        # Verify problem class supports incremental building
        if not hasattr(problem, "build_base_model") or not hasattr(problem, "add_scenario_to_model"):
            raise NotImplementedError(
                f"Problem class {type(problem).__name__} does not support incremental solving "
                "(missing 'build_base_model' or 'add_scenario_to_model')."
            )

        # Build the base model (x + t only)
        self.master_model = problem.build_base_model(
            instance, time_limit=time_limit, threads=threads, verbose=verbose
        )
        self.master_model.update()

    def evaluate_subset(self, sce_indices: Iterable[int],
                        return_solution: bool = False):
        """Solve a copy of the master with the requested scenarios added.

        Returns ``(obj, obj_bound, mip_gap, feasible)``. With ``return_solution=True``,
        append the first-stage vector as a tuple, or None if no solution exists.
        """
        # 1. Fast Clone (Note: Custom python attributes on master_model are NOT copied)
        cand_model = copy_model_with_payload(self.master_model)

        # 2. Add Candidate Constraints
        for s_idx in sce_indices:
            self.problem.add_scenario_to_model(cand_model, self.instance, int(s_idx))

        # 3. Solve
        if self.mip_gap is not None:
            cand_model.Params.MIPGap = float(self.mip_gap)
        cand_model.optimize()

        # Retry an unconverged candidate before using its objective to rank scenarios.
        if self.require_converged and cand_model.Status != GRB.OPTIMAL:
            self.n_unconverged_retries += 1
            cand_model.Params.TimeLimit = (
                float(self.unconverged_retry_time_limit)
                if self.unconverged_retry_time_limit is not None
                else GRB.INFINITY
            )
            cand_model.optimize()
            if cand_model.Status == GRB.OPTIMAL:
                self.n_retry_rescued += 1
            else:
                # Still not converged. It must not be able to WIN the argmax, and the caller
                # already maps not-feasible to -inf. Loud, because silently dropping a
                # candidate changes the search and must never look like a clean step.
                self.n_excluded_unconverged += 1
                logger.warning(
                    "Convergence gate: candidate solve did not converge even uncapped "
                    "(status=%s, gap=%s); excluding it from the argmax",
                    cand_model.Status,
                    getattr(cand_model, "MIPGap", None),
                )
                return (float("inf"), 0.0, 0.0, False)

        # 4. Extract Metrics
        status = cand_model.Status
        feasible = (cand_model.SolCount > 0)

        # SUBOPTIMAL and TIME_LIMIT remain accepted only when require_converged is False,
        # which exists so the non-converged-accepting behaviour stays reproducible for comparison. It is
        # not a production path.
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and feasible:
            out = (
                float(cand_model.ObjVal),
                float(cand_model.ObjBound),
                float(cand_model.MIPGap),
                True
            )
            if return_solution:
                # Read x by NAME rather than by position: the copied model's variable order
                # is not part of any contract, and a positional read would silently return
                # recourse variables if the build order ever changed.
                xs = [v for v in cand_model.getVars() if v.VarName.startswith("x[")]
                out = out + (tuple(float(v.X) for v in xs) if xs else None,)
            return out
        else:
            out = (float("inf"), 0.0, 0.0, False)
            return out + (None,) if return_solution else out

    def commit_subset(self, sce_indices: Iterable[int]):
        """
        Permanently adds a subset of scenarios to the master model.
        """
        for s_idx in sce_indices:
            self.problem.add_scenario_to_model(self.master_model, self.instance, int(s_idx))
        self.master_model.update()

    def cleanup(self):
        """Dispose of the master model to free resources."""
        if self.master_model:
            self.master_model.dispose()
            self.master_model = None


class StepwiseGreedySearch:
    """
    Framework for stepwise forward selection.
    Decouples the search loop, stopping logic, and trace recording from the solver.
    """

    def __init__(
            self,
            solver: Incremental2ROSolver,
            S: int,
            stop_M: int = 1,
            eps_gain: float = 0.0001,
            log_prefix: str = ""
    ):
        self.solver = solver
        self.S = S
        self.stop_M = stop_M
        self.eps_gain = eps_gain
        self.log_prefix = log_prefix

        # State
        self.selected: List[int] = []
        self.candidate_scores_steps: List[np.ndarray] = []
        self.best_scores_list: List[float] = []
        self.step_cumulative_time_sec: List[float] = []
        self.plateau_count = 0
        self.stop_step: Optional[int] = None

    def search(self, K: int) -> ForwardTrace:
        """
        Run the greedy search for K steps.
        """
        for t in range(1, K + 1):
            if self.stop_step is not None:
                self._record_dummy_step()
                continue

            step_t0 = time.perf_counter()
            # --- Evaluation Step ---
            step_stats = self._evaluate_candidates(t, K)

            if step_stats is None:
                logger.warning(f"{self.log_prefix} Step {t}: No candidates left.")
                self._fill_remaining_dummy(K, t)
                break

            best_j, best_score, avg_gap = step_stats

            # --- Update State ---
            self.selected.append(best_j)
            self.best_scores_list.append(best_score)

            # Commit choice to solver
            self.solver.commit_subset([best_j])

            # --- Calculate Gain ---
            if len(self.best_scores_list) == 1:
                gain = self.best_scores_list[0]
            else:
                gain = self.best_scores_list[-1] - self.best_scores_list[-2]

            # --- Check Plateau ---
            self._check_stopping_condition(t, gain)
            # 1. Calculate Gain
            if len(self.best_scores_list) == 1:
                gain = self.best_scores_list[0]
            else:
                gain = self.best_scores_list[-1] - self.best_scores_list[-2]
            # --- Log ---
            logger.info(
                f"{self.log_prefix} Step {t}/{K}: best_j={best_j}, "
                f"score={best_score:.4f}, gain={gain:.4f}, gap={avg_gap:.4f}"
            )
            prev_cum = self.step_cumulative_time_sec[-1] if self.step_cumulative_time_sec else 0.0
            self.step_cumulative_time_sec.append(prev_cum + (time.perf_counter() - step_t0))

        return self._build_trace(K)

    def _evaluate_candidates(self, t: int, K: int) -> Optional[Tuple[int, float, float]]:
        """
        Iterates over all unused scenarios, evaluates them using the solver,
        and returns best candidate stats.
        """
        used = set(self.selected)
        scores_t = np.full((self.S,), -np.inf, dtype=float)
        gaps = []

        # Heuristic: Evaluate adding single scenarios (Beam width=1)
        for j in range(self.S):
            if j in used:
                continue

            # Solver evaluates 'current committed' + 'j'
            obj, _, gap, feas = self.solver.evaluate_subset([j])

            if feas:
                scores_t[j] = obj
                gaps.append(gap)
            else:
                gaps.append(0.0)

        if not np.any(np.isfinite(scores_t)):
            return None

        # Record full step scores
        self.candidate_scores_steps.append(scores_t)

        best_j = int(np.nanargmax(scores_t))
        best_score = float(scores_t[best_j])
        avg_gap = np.mean(gaps) if gaps else 0.0

        return best_j, best_score, avg_gap

    def _check_stopping_condition(self, t: int, gain: float):
        """
        Checks if gain is below threshold for stop_M consecutive steps.
        """
        if gain <= self.eps_gain:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        if self.stop_step is None and self.plateau_count >= self.stop_M:
            self.stop_step = _compute_plateau_stop_step(
                self.best_scores_list, stop_M=self.stop_M, eps_gain=self.eps_gain
            )
            logger.info(f"{self.log_prefix} Plateau at step {t} (stop_step={self.stop_step})")

    def _record_dummy_step(self):
        """Fills state with dummy values for skipped steps."""
        self.selected.append(-1)
        self.candidate_scores_steps.append(np.full((self.S,), -1.0))
        # Repeat the last valid score so gain is 0
        if self.best_scores_list:
            self.best_scores_list.append(self.best_scores_list[-1])
        else:
            self.best_scores_list.append(0.0)
        if self.step_cumulative_time_sec:
            self.step_cumulative_time_sec.append(self.step_cumulative_time_sec[-1])
        else:
            self.step_cumulative_time_sec.append(0.0)

    def _fill_remaining_dummy(self, K: int, current_t: int):
        """Fills from current_t up to K with dummy steps."""
        remaining = K - len(self.selected)
        for _ in range(remaining):
            self._record_dummy_step()

    def _build_trace(self, K: int) -> ForwardTrace:
        T = len(self.best_scores_list)
        best_scores = np.asarray(self.best_scores_list, dtype=float)
        gains = np.empty((T,), dtype=float)
        if T > 0: gains[0] = best_scores[0]
        if T > 1: gains[1:] = best_scores[1:] - best_scores[:-1]

        final_stop = self.stop_step if self.stop_step is not None else T
        step_mask = _compute_step_mask(T, stop_step=int(final_stop))
        selected_idx = np.asarray(self.selected, dtype=int)
        if selected_idx.size > 0 and final_stop < selected_idx.size:
            selected_idx[int(final_stop):] = -1

        return ForwardTrace(
            K=int(K), S=int(self.S),
            selected_idx=selected_idx,
            candidate_scores=self.candidate_scores_steps,
            best_scores=best_scores,
            gains=gains,
            stop_step=int(final_stop),
            step_mask=step_mask,
            step_cumulative_time_sec=np.asarray(self.step_cumulative_time_sec, dtype=float),
        )
