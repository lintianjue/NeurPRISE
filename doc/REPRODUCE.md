# Reproducing NeurPRISE

`scripts/reproduce.sh` is the single entry point. Run it from anywhere; it sets
the default directories itself; no directory environment variables are required.

| Mode | What it does | Rough time on our machines |
|---|---|---|
| `download` | fetch the released models and data into `checkpoints/` and `data/` | minutes on a fast connection (~1.4 GB) |
| `eval [--table N ...] [--threads N] [--device cuda]` | rerun evaluations and save per-instance results and summary statistics | hours per table, days for all |
| `data` / `label` / `train <args>` | regenerate instances / PRISE labels / a model (stage wrappers 01–03) | see below |

Add `--dry-run` before the mode to print the commands instead of running them.

Default paths are relative to the repository: `data/` for instances and labels,
`checkpoints/` for released models, and `runs/` for new models, logs and results.
The exact-objective cache is downloaded to `eval_regret/full_objs_cache.json`.
Advanced overrides are covered under [Custom directories](#custom-directories).

## 1. Released files

`scripts/reproduce.sh download` places them under the input root:

```
checkpoints/<experiment>/<tag>[_<variant>].pt   NeurPRISE models, e.g. main/SEL-20-50.pt,
                                      seeds/VC-20-50_s1.pt, transfer/SEL2VC_ft_b5_s42.pt
data/<problem>/inst_*.pkl             test instances                                655 MB
data/<problem>/label_prise_*.pkl      exact labels V(Ξ), x* for the evaluated instances   475 MB
eval_regret/full_objs_cache.json      V(Ξ) of every test instance
```

The problem names in paths are `selection`, `vertex_cover` and `cflp_ebp`. Distribution
variants add a `_normal` or `_mm` suffix.

`<experiment>` is one of `main`, `distribution`, `train_size`, `seeds`, `architecture`,
`loss`, `cheap_labels`, `transfer` and `supervision`; `<tag>` is the paper's instance tag (`SEL-20-50`,
`VC-50-100`, `CFLP-70-70-50`, ...). A model used by several tables is stored once.
The released checkpoints are the paper's models. `repro/checkpoints.json` lists each
model's training config, model and loss (`train`), original training name
(`trained_as`) and SHA-256 prefix.

## 2. Reproduction inputs

The download contains instances, labels, NeurPRISE checkpoints and the exact
full-objective cache required to compute regret. It does not download saved paper results,
paper tables or training-measurement summaries. Experiments generate their own result JSONs.

## 3. Rerun the evaluations (`eval`)

```bash
scripts/reproduce.sh eval --table 1 --threads 8
scripts/reproduce.sh eval --table 2 --threads 8
```

`repro/runs.json` holds the runnable experiment recipes. `repro/runs.py`
turns the selected ones into `runs/reproduce_eval.sh`, one command per
line:

Each NeurPRISE recipe runs two steps:

1. `scripts/04_05_eval_ml_2ro_e2e.py` loads the checkpoint, selects the top-k
   scenarios and solves the reduced problem for each test instance.
2. `scripts/eval/eval_regret.py` computes the regret of that selection.

Each command writes to `runs/repro/<recipe id>/`. Result JSONs contain
per-instance objectives and regret, feasible-only mean regret and infeasibility counts.
The command stops if an experiment fails. It does not compare against paper values.
Table numbers select experiment groups; they do not trigger table generation.
Supported NeurPRISE evaluation groups are **1–6, 10–11, 13–14, 18–20 and 22–26**.
These rerun only the NeurPRISE evaluations associated with those tables, not every
row or metric in a table. Tables 7, 15 and 16 are computed in full; see below.

`--threads N` sets the thread count in both evaluation stages. If omitted, it keeps
the original paper setting of 40 threads. The MIP gap is 1e-4 and the reduced-solve
limit is 900 s. `--device` sets where NeurPRISE scores scenarios (default `cpu`); the paper's
times were measured with `--device cuda`. Regret does not depend on it.
For custom MIP gaps or time limits, use `python repro/runs.py` to generate commands.
You need your own Gurobi license. Results can differ with solver settings and hardware.
New runtime measurements describe your hardware and load; they do not replace the
paper's reported times.

### Table 7: compression budget

```bash
scripts/reproduce.sh eval --table 7 --threads 1
```

runs `scripts/eval/compression_budget.py`, which computes the compression budget
k̂ = min{k ≤ 8 : |V(Ξ) − V(R^(k))| / V(Ξ) ≤ 1%} on the 250 SEL-20-50 and VC-20-50 test
instances. PRISE's V(R^(k)) are read from its labels; MaxSum (scenarios by Σ|ξ|
descending) and Random (one permutation per instance, seed 42) solve up to 8
reduced problems per instance, without a time limit. It takes about 10 minutes on one
thread. `python scripts/eval/compression_budget.py --run-id <name> --methods PRISE`
computes the PRISE rows alone in seconds, without solving. The result,
`runs/repro/compression_budget/compression_budget.json`, holds the table rows and each
instance's k̂ and V(R^(k)). It also reports the fraction of consecutive PRISE steps whose
marginal gain does not increase, pooled over all 2,500 labelled instances (Appendix,
"93–99%").

### Tables 15–16: scenario-count runtime

```bash
scripts/reproduce.sh eval --table 15 16
```

runs `scripts/eval/scenario_runtime.py --scale medium` (Table 15) and `--scale large`
(Table 16). The 50 VC-50-50 / VC-50-100 test instances are truncated to their first
S ∈ {10, 20, 50} (medium) or {20, 50, 100} (large) scenarios; the VC-20-50 model scores them on `--device`, the top 6 are kept
and the reduced MILP is solved. Each result,
`runs/repro/scenario_runtime_med/VC-50-50_truncated_k6.json` and
`runs/repro/scenario_runtime_lrg/VC-50-100_truncated_k6.json`, holds each instance's
inference and solve times and the ratios Σ t(S) / Σ t(20). It takes about 7 minutes at 40 threads; the ratios depend on the
host and its load.

### Table 14: time-budgeted MILP

`scripts/reproduce.sh eval --table 14` also runs `scripts/eval/time_budget.py`, which solves the
full-scenario MILP of the 50 VC-50-50 / VC-50-100 test instances under a fixed limit of 1 s (k = 4)
or 2 s (k = 6) and scores the incumbent against `eval_regret/full_objs_cache.json`. Results:
`runs/repro/VC-50-{50,100}_fullmilp_k{4_budget1s,6_budget2s}/`, about 8 minutes at 40 threads in all.

## 4. Regenerate data, labels and models

The stage wrappers take positional arguments; `-h` prints them. Generated instances
and labels are saved in `data/<problem>/`, which the next stage reads automatically.
New models go to `runs/03_train/<run>/model/`; the training log prints the checkpoint path.

If you downloaded the released data, skip generation and labelling and start with
`train`. To generate a fresh dataset, use a configuration whose files do not already
exist. The following 100-instance SEL example demonstrates the pipeline; it is not
the paper's full training configuration, which uses 2,500 instances:

```bash
# Generate 100 SEL-20-50 instances, then label and train using the same data/ folder.
scripts/reproduce.sh data selection 20 50 100
scripts/reproduce.sh label selection 20 50 prise_label 100 0.4 8
scripts/reproduce.sh train selection 20 50 100 model=det_source_sce_mean \
    losses=score_weighted_loss_flat cuda:0 32
```

Labelling is the expensive step: PRISE solves about K·|Ξ| reduced MILPs per instance.
Estimate the cost on a few instances before you start a full run.

## Custom directories

These overrides are optional; leave them unset for the default workflow.

| Variable | Default | Meaning |
|---|---|---|
| `NEURPRISE_INPUT_ROOT` | repository root | input tree containing `data/` and `checkpoints/` |
| `NEURPRISE_RUN_ROOT` | repository `runs/` | new models, logs and evaluation results |
| `HF_REPO` | script default | Hugging Face repository for `download` |

With a custom input root, generated datasets stay in their individual run directories
instead of writing into that input tree. In this advanced mode, set the next stage's
input root to the previous stage's run directory. Custom run directories must not
overlap input directories; the default repository `runs/` is the supported exception.
Existing runs in other locations are not moved.

## 5. No guarantee of identical results

The code, checkpoints and data are provided "as
is", without warranty of any kind (Apache 2.0 license, sections 7 and 8). A rerun may give
numbers that differ from those in the paper. Differences can come from the GPU model, the
CUDA, cuDNN, PyTorch and Gurobi versions, thread counts, the CPU and operating system,
nondeterministic floating-point arithmetic, and random seeds. When any of these change, a
MILP solver may return a different optimal solution, or a different solution within the
MIP gap. Retrained models and regenerated labels are affected the most. We do not
guarantee that a rerun reproduces the paper's numbers exactly, and small differences of
this kind do not indicate an error in the code. Use the per-instance results and summary
statistics to assess each rerun.

## Tests

```bash
PYTHONPATH=. pytest -q          # no released files needed
```
