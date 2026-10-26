"""Golden per-instance record sets for the ``save_results``/``summarize`` fixture.

Shaped like the rows ``run_neurprise_regret`` (``scripts/eval/eval_regret.py``)
appends to its ``results`` list: ``instance_id, full_obj, reduced_obj, eval_on_xi,
regret, reduced_obj_bound, reduced_mip_gap, reduced_status,
reduced_solve_sec``, plus ``selection_sec`` / ``reduced_eval_wall_sec`` on some rows
so ``_time_components`` has data to aggregate.

Frozen by ``gen.py`` — see that file's docstring.
"""
import math

CASES: dict[str, list[dict]] = {
    "all_feasible": [
        {
            "instance_id": 0, "full_obj": 100.0, "reduced_obj": 95.0,
            "eval_on_xi": 102.0, "regret": 2.0,
            "reduced_obj_bound": 94.5, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.234,
            "selection_sec": 0.01, "reduced_eval_wall_sec": 1.5,
        },
        {
            "instance_id": 1, "full_obj": 200.0, "reduced_obj": 190.0,
            "eval_on_xi": 205.0, "regret": 2.5,
            "reduced_obj_bound": 189.0, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 2.345,
            "selection_sec": 0.02, "reduced_eval_wall_sec": 2.5,
        },
        {
            "instance_id": 2, "full_obj": 150.0, "reduced_obj": 148.0,
            "eval_on_xi": 151.0, "regret": 0.6667,
            "reduced_obj_bound": 147.5, "reduced_mip_gap": 0.0002,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.0,
            "selection_sec": 0.015, "reduced_eval_wall_sec": 1.2,
        },
        {
            "instance_id": 3, "full_obj": 80.0, "reduced_obj": 75.0,
            "eval_on_xi": 84.0, "regret": 5.0,
            "reduced_obj_bound": 74.8, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 0.9,
            "selection_sec": 0.01, "reduced_eval_wall_sec": 1.1,
        },
    ],
    "with_inf": [
        {
            "instance_id": 0, "full_obj": 100.0, "reduced_obj": 95.0,
            "eval_on_xi": 102.0, "regret": 2.0,
            "reduced_obj_bound": 94.5, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.234,
            "selection_sec": 0.01, "reduced_eval_wall_sec": 1.5,
        },
        {
            "instance_id": 1, "full_obj": 200.0, "reduced_obj": 190.0,
            "eval_on_xi": 205.0, "regret": 2.5,
            "reduced_obj_bound": 189.0, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 2.345,
            "selection_sec": 0.02, "reduced_eval_wall_sec": 2.5,
        },
        {
            "instance_id": 2, "full_obj": 150.0, "reduced_obj": 148.0,
            "eval_on_xi": 151.0, "regret": 0.6667,
            "reduced_obj_bound": 147.5, "reduced_mip_gap": 0.0002,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.0,
        },
        {
            "instance_id": 3, "full_obj": 80.0, "reduced_obj": 75.0,
            "eval_on_xi": math.inf, "regret": math.inf,
            "reduced_obj_bound": 74.8, "reduced_mip_gap": 0.0001,
            "reduced_status": "INFEASIBLE", "reduced_solve_sec": 0.9,
        },
    ],
    "with_nan": [
        {
            "instance_id": 0, "full_obj": 100.0, "reduced_obj": 95.0,
            "eval_on_xi": 102.0, "regret": 2.0,
            "reduced_obj_bound": 94.5, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.234,
            "selection_sec": 0.01, "reduced_eval_wall_sec": 1.5,
        },
        {
            "instance_id": 1, "full_obj": 0.0, "reduced_obj": 0.0,
            "eval_on_xi": 0.0, "regret": math.nan,
            "reduced_obj_bound": 0.0, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 0.5,
        },
        {
            "instance_id": 2, "full_obj": 150.0, "reduced_obj": 148.0,
            "eval_on_xi": 151.0, "regret": 0.6667,
            "reduced_obj_bound": 147.5, "reduced_mip_gap": 0.0002,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.0,
            "selection_sec": 0.015, "reduced_eval_wall_sec": 1.2,
        },
    ],
    "all_infeasible": [
        {
            "instance_id": 0, "full_obj": 100.0, "reduced_obj": 95.0,
            "eval_on_xi": math.inf, "regret": math.inf,
            "reduced_obj_bound": 94.5, "reduced_mip_gap": 0.0001,
            "reduced_status": "INFEASIBLE", "reduced_solve_sec": 1.234,
        },
        {
            "instance_id": 1, "full_obj": 200.0, "reduced_obj": 190.0,
            "eval_on_xi": math.inf, "regret": math.inf,
            "reduced_obj_bound": 189.0, "reduced_mip_gap": 0.0001,
            "reduced_status": "INFEASIBLE", "reduced_solve_sec": 2.345,
        },
        {
            "instance_id": 2, "full_obj": 150.0, "reduced_obj": 148.0,
            "eval_on_xi": math.inf, "regret": math.inf,
            "reduced_obj_bound": 147.5, "reduced_mip_gap": 0.0002,
            "reduced_status": "INFEASIBLE", "reduced_solve_sec": 1.0,
        },
    ],
    "single": [
        {
            "instance_id": 0, "full_obj": 100.0, "reduced_obj": 95.0,
            "eval_on_xi": 102.0, "regret": 2.0,
            "reduced_obj_bound": 94.5, "reduced_mip_gap": 0.0001,
            "reduced_status": "OPTIMAL", "reduced_solve_sec": 1.234,
            "selection_sec": 0.01, "reduced_eval_wall_sec": 1.5,
        },
    ],
    "empty": [],
}
